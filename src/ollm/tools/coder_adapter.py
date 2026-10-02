"""Schema-aware Qwen Coder XML-like protocol; no XML entity evaluation."""
from __future__ import annotations
import copy
import re
from .adapters import QwenAdapter, PreparedPrompt, _qwen_stop_ids, _split_thinking
from .types import (AssistantTurn, ToolCall, ToolCallParseError, IncompleteGeneration,
                    strict_json, normalize_messages, valid_name)


class QwenCoderAdapter(QwenAdapter):
    """Native Coder-Next / Coder-30B parameter tags, not JSON tool blocks.

    The advertised parameter schema determines string vs JSON interpretation.
    A raw string such as '001' must not silently become a number. Delimiter-like
    content is rejected as ambiguous. Complete calls only; never partial execute.
    """
    def __init__(self):
        self.schemas = {}

    def prepare(self, messages, tools, tokenizer, **kwargs):
        self.schemas = {t['function']['name']: copy.deepcopy(t['function']['parameters']) for t in tools}
        from .registry import _validator
        for name, schema in self.schemas.items():
            valid_name(name)
            _validator(schema)  # no remote schema retrieval
        return super().prepare(messages, tools, tokenizer, **kwargs)

    def parse(self, tokens, tokenizer):
        if not tokens or tokens[-1] not in _qwen_stop_ids(tokenizer):
            raise IncompleteGeneration("Coder did not finish its assistant turn; no tools executed")
        return self.parse_text(tokenizer.decode(tokens[:-1], skip_special_tokens=False))

    def parse_text(self, text):
        thinking, text = _split_thinking(text)
        masked = re.sub(r'(`{3,}|~{3,})[^\n]*\n.*?\1', lambda m: ' ' * len(m[0]), text, flags=re.DOTALL)
        matches = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', masked, re.DOTALL))
        remainder = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', masked, flags=re.DOTALL)
        if re.search(r'</?(?:tool_call|function|parameter)\b', remainder):
            raise ToolCallParseError("Malformed Coder action delimiters")
        if not matches:
            return AssistantTurn(content=text, thinking=thinking)
        calls, cursor = [], matches[0].start()
        for match in matches:
            if text[cursor:match.start()].strip():
                raise ToolCallParseError("Unexpected content between calls")
            body = text[match.start(1):match.end(1)]
            fn = re.fullmatch(r'<function=([A-Za-z_][A-Za-z0-9_]{0,63})>\s*(.*?)\s*</function>', body, re.DOTALL)
            if not fn or fn[1] not in self.schemas:
                raise ToolCallParseError("Malformed or unadvertised Coder function")
            schema = self.schemas[fn[1]]
            params, pos, args = fn[2], 0, {}
            for item in re.finditer(r'<parameter=([A-Za-z_][A-Za-z0-9_]{0,63})>(.*?)</parameter>', params, re.DOTALL):
                name, value = item[1], item[2]
                if params[pos:item.start()].strip() or name in args or name not in schema.get('properties', {}):
                    raise ToolCallParseError("Unknown, duplicate or malformed parameter")
                if re.search(r'</?(?:tool_call|function|parameter)\b', value):
                    raise ToolCallParseError("Ambiguous delimiters in parameter data")
                # Remove the protocol's single framing newline, preserving code
                # indentation and all other user data (do NOT .strip strings).
                if value.startswith('\r\n'):
                    value = value[2:]
                elif value.startswith('\n'):
                    value = value[1:]
                if value.endswith('\r\n'):
                    value = value[:-2]
                elif value.endswith('\n'):
                    value = value[:-1]
                field = schema['properties'][name]
                kind = field.get('type')
                kinds = [kind] if isinstance(kind, str) else kind
                try:
                    if kinds and 'string' in kinds:
                        args[name] = None if 'null' in kinds and value in ('null', 'None') else value
                    elif kinds:
                        # Official Jinja templates stringify scalar bool/None
                        # using Python spelling, while containers use tojson.
                        scalar = value.strip()
                        if 'boolean' in kinds and scalar in ('True', 'False'):
                            args[name] = scalar == 'True'
                        elif 'null' in kinds and scalar == 'None':
                            args[name] = None
                        else:
                            args[name] = strict_json(value)
                    else:
                        raise ValueError("Parameter requires an explicit JSON Schema type")
                except (ValueError, TypeError) as exc:
                    raise ToolCallParseError(f"Invalid Coder parameter {name}: {exc}") from exc
                pos = item.end()
            if params[pos:].strip():
                raise ToolCallParseError("Unparsed Coder parameter content")
            try:
                from .registry import _validator
                _validator(schema).validate(args)
            except Exception as exc:
                raise ToolCallParseError(f"Coder arguments failed schema validation: {exc}") from exc
            calls.append(ToolCall(fn[1], args))
            cursor = match.end()
        if text[cursor:].strip():
            raise ToolCallParseError("Suffix after Coder tool calls is not allowed")
        return AssistantTurn(content=text[:matches[0].start()].strip(), thinking=thinking, tool_calls=calls)


class TextOnlyAdapter:
    """Reasoning/text generation for DeepSeek checkpoints without qualified tools."""
    family = 'text'

    def __init__(self, eos_ids=None):
        self.eos_ids = list(eos_ids) if eos_ids is not None else None

    def prepare(self, messages, tools, tokenizer, **kwargs):
        if tools or any(m.get('role') == 'tool' or m.get('tool_calls') for m in messages):
            raise ValueError("This checkpoint has text inference, not qualified native function calling")
        history = normalize_messages(messages)
        ids = tokenizer.apply_chat_template(history, tokenize=True, add_generation_prompt=True)
        eos = self.eos_ids if self.eos_ids is not None else tokenizer.eos_token_id
        stops = [eos] if isinstance(eos, int) else list(eos or [])
        if not stops:
            raise ValueError("Text tokenizer has no EOS token")
        return PreparedPrompt(ids, stops)

    def parse(self, tokens, tokenizer):
        stops = self.eos_ids if self.eos_ids is not None else tokenizer.eos_token_id
        stops = [stops] if isinstance(stops, int) else list(stops or [])
        if not tokens or tokens[-1] not in stops:
            raise IncompleteGeneration("Text model did not complete the turn")
        thinking, text = _split_thinking(tokenizer.decode(tokens[:-1], skip_special_tokens=False))
        return AssistantTurn(content=text, thinking=thinking)

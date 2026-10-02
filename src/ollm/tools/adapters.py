"""Native Qwen tool-call and OpenAI Harmony adapters.

Only explicitly framed, complete calls are returned to the dispatcher. Model
output is never evaluated as Python, and reasoning is not treated as an action.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Protocol

from .types import (AssistantTurn, IncompleteGeneration, ToolCall,
                    ToolCallParseError, arguments_dict, json_dumps,
                    normalize_messages, strict_json)


@dataclass(frozen=True)
class PreparedPrompt:
    input_ids: list[int]
    stop_ids: list[int]


class ModelAdapter(Protocol):
    family: str

    def prepare(self, messages: list[dict], tools: list[dict], tokenizer: Any,
                *, reasoning_effort: str = "low", enable_thinking: bool = False) -> PreparedPrompt: ...

    def parse(self, tokens: list[int], tokenizer: Any) -> AssistantTurn: ...


def _qwen_stop_ids(tokenizer) -> list[int]:
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" not in vocab:
        raise ValueError("Qwen adapter requires a ChatML tokenizer containing <|im_end|>")
    ids = [vocab["<|im_end|>"]]
    eos = tokenizer.eos_token_id
    if eos is not None:
        ids.extend(eos if isinstance(eos, list) else [eos])
    return list(dict.fromkeys(ids))


def _split_thinking(text: str) -> tuple[str, str]:
    text = text.strip()
    if "</think>" in text:
        prefix, text = text.split("</think>", 1)
        # A thinking-enabled template can already end with the opening tag.
        thinking = prefix.removeprefix("<think>").strip()
        if "<think>" in thinking or "</think>" in text or text.lstrip().startswith("<think>"):
            raise ToolCallParseError("Malformed reasoning block")
        return thinking, text.strip()
    if text.startswith("<think>"):
        raise ToolCallParseError("Unclosed reasoning block")
    return "", text


def parse_qwen_text(text: str) -> AssistantTurn:
    """Parse a completed Qwen assistant turn, not an arbitrary document.

    Code-fenced examples are inert. A tool-call block must occupy the tail of
    the turn, optionally after a prose preamble. Any malformed block aborts the
    entire turn, including earlier otherwise-valid calls.
    """
    thinking, text = _split_thinking(text)
    # Preserve positions while removing fenced code from consideration.
    masked = re.sub(r"(`{3,}|~{3,})[^\n]*\n.*?\1", lambda m: " " * len(m[0]),
                    text, flags=re.DOTALL)
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", masked, re.DOTALL))
    remainder = re.sub(r"<tool_call>\s*.*?\s*</tool_call>", "", masked, flags=re.DOTALL)
    if re.search(r"</?tool_call\b", remainder):
        raise ToolCallParseError("Incomplete or malformed Qwen tool-call delimiters")
    if not matches:
        return AssistantTurn(content=text, thinking=thinking)
    calls = []
    cursor = matches[0].start()
    for match in matches:
        if text[cursor:match.start()].strip():
            raise ToolCallParseError("Unexpected text between tool-call blocks")
        try:
            payload = strict_json(text[match.start(1):match.end(1)])
            if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
                raise ValueError("Expected exactly the name and arguments fields")
            calls.append(ToolCall(payload["name"], arguments_dict(payload["arguments"])))
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            raise ToolCallParseError(f"Invalid Qwen tool call: {exc}") from exc
        cursor = match.end()
    if text[cursor:].strip():
        raise ToolCallParseError("Unexpected content after a tool call; refusing ambiguous execution")
    return AssistantTurn(content=text[:matches[0].start()].strip(), tool_calls=calls, thinking=thinking)


class QwenAdapter:
    family = "qwen"

    def prepare(self, messages, tools, tokenizer, *, reasoning_effort="low", enable_thinking=False):
        history = normalize_messages(messages)
        for msg in history:
            if "thinking" in msg:
                msg["reasoning_content"] = msg.pop("thinking")
        kwargs = dict(tokenize=True, add_generation_prompt=True, enable_thinking=enable_thinking)
        if tools:
            kwargs["tools"] = copy.deepcopy(tools)
        ids = tokenizer.apply_chat_template(history, **kwargs)
        if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
            raise ValueError("Expected a single, tokenized Qwen conversation")
        return PreparedPrompt(ids, _qwen_stop_ids(tokenizer))

    def parse(self, tokens, tokenizer):
        if not tokens or tokens[-1] not in _qwen_stop_ids(tokenizer):
            raise IncompleteGeneration("Qwen did not produce an end-of-turn token; no tools were executed")
        # Strip the actual terminal token, not arbitrary matching text in arguments.
        return parse_qwen_text(tokenizer.decode(tokens[:-1], skip_special_tokens=False))


class GPTOSSAdapter:
    """Render and parse token IDs with the official openai-harmony package.

    This deliberately avoids depending on a repacked checkpoint's Jinja
    template or decoding Harmony tokens through a different tokenizer.
    """
    family = "gpt-oss"

    def __init__(self):
        try:
            import openai_harmony as harmony
        except ImportError as exc:
            raise ImportError("Install Harmony support with: pip install --no-build-isolation -e '.[gpt-oss-tools]' (from the patched checkout)") from exc
        self.harmony = harmony
        self.encoding = harmony.load_harmony_encoding(harmony.HarmonyEncodingName.HARMONY_GPT_OSS)

    def prepare(self, messages, tools, tokenizer=None, *, reasoning_effort="low", enable_thinking=False):
        h = self.harmony
        history = normalize_messages(messages)
        efforts = {"low": h.ReasoningEffort.LOW, "medium": h.ReasoningEffort.MEDIUM,
                   "high": h.ReasoningEffort.HIGH}
        if reasoning_effort not in efforts:
            raise ValueError("reasoning_effort must be low, medium or high")
        system = h.SystemContent.new().with_reasoning_effort(efforts[reasoning_effort])
        rendered = [h.Message.from_role_and_content(h.Role.SYSTEM, system)]
        instructions = ""
        if history[0]["role"] == "system":
            instructions = history.pop(0)["content"]
        if instructions or tools:
            developer = h.DeveloperContent.new().with_instructions(instructions)
            if tools:
                descriptions = [h.ToolDescription(name=t["function"]["name"],
                    description=t["function"].get("description", ""),
                    parameters=t["function"]["parameters"]) for t in tools]
                developer = developer.with_function_tools(descriptions)
            rendered.append(h.Message.from_role_and_content(h.Role.DEVELOPER, developer))
        for msg in history:
            if msg["role"] == "user":
                rendered.append(h.Message.from_role_and_content(h.Role.USER, msg["content"]))
            elif msg["role"] == "tool":
                author = h.Author(role=h.Role.TOOL, name="functions." + msg["name"])
                rendered.append(h.Message.from_author_and_content(author, msg["content"]))
            elif msg.get("tool_calls"):
                if len(msg["tool_calls"]) != 1:
                    raise ValueError("Native gpt-oss supports one tool handoff per assistant action")
                if msg.get("thinking"):
                    rendered.append(h.Message.from_role_and_content(h.Role.ASSISTANT, msg["thinking"])
                                    .with_channel("analysis"))
                if msg["content"]:
                    rendered.append(h.Message.from_role_and_content(h.Role.ASSISTANT, msg["content"])
                                    .with_channel("commentary"))
                fn = msg["tool_calls"][0]["function"]
                rendered.append(h.Message.from_role_and_content(h.Role.ASSISTANT, json_dumps(fn["arguments"]))
                                .with_channel("commentary").with_recipient("functions." + fn["name"]))
            else:
                rendered.append(h.Message.from_role_and_content(h.Role.ASSISTANT, msg["content"])
                                .with_channel("final"))
        ids = self.encoding.render_conversation_for_completion(
            h.Conversation.from_messages(rendered), h.Role.ASSISTANT)
        return PreparedPrompt(list(ids), list(self.encoding.stop_tokens_for_assistant_actions()))

    def parse(self, tokens, tokenizer=None):
        stop_ids = self.encoding.stop_tokens_for_assistant_actions()
        if not tokens or tokens[-1] not in stop_ids:
            raise IncompleteGeneration("Harmony generation did not finish an assistant action")
        try:
            parsed = self.encoding.parse_messages_from_completion_tokens(
                tokens, role=self.harmony.Role.ASSISTANT, strict=True)
            return self._parse_messages(parsed, self.encoding.decode_utf8([tokens[-1]]))
        except ToolCallParseError:
            raise
        except Exception as exc:
            raise ToolCallParseError(f"Invalid Harmony completion: {exc}") from exc

    def _parse_messages(self, messages, ending: str) -> AssistantTurn:
        """Structural validation after the official token parser."""
        analysis, commentary, finals, calls = [], [], [], []
        for index, msg in enumerate(messages):
            if msg.author.role != self.harmony.Role.ASSISTANT:
                raise ToolCallParseError("Model emitted a non-assistant message")
            if any(not isinstance(c, self.harmony.TextContent) for c in msg.content):
                raise ToolCallParseError("Only text Harmony messages are supported")
            body = "".join(c.text for c in msg.content)
            if msg.recipient:
                if not msg.recipient.startswith("functions."):
                    raise ToolCallParseError("Only registered functions.* recipients are supported")
                if msg.channel not in ("analysis", "commentary") or index != len(messages) - 1:
                    raise ToolCallParseError("Tool handoff must be the final message in an assistant action")
                try:
                    calls.append(ToolCall(msg.recipient[len("functions."):], arguments_dict(body)))
                except (ValueError, TypeError, RecursionError) as exc:
                    raise ToolCallParseError(f"Invalid Harmony tool arguments: {exc}") from exc
            elif msg.channel == "analysis":
                analysis.append(body)
            elif msg.channel == "commentary":
                commentary.append(body)
            elif msg.channel in ("final", None):
                finals.append(body)
            else:
                raise ToolCallParseError(f"Unsupported Harmony channel: {msg.channel}")
        if calls:
            if len(calls) != 1 or finals or ending not in ("<|ghissue|>", "<|call|>"):
                raise ToolCallParseError("Ambiguous Harmony handoff")
            return AssistantTurn("\n".join(commentary), calls, "\n".join(analysis))
        if not finals or ending not in ("<|fim_suffix|>", "<|return|>"):
            raise ToolCallParseError("Harmony action contained no final answer or executable handoff")
        return AssistantTurn(content="\n".join(finals), thinking="\n".join(analysis))

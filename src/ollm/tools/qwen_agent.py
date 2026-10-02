"""Qwen-Agent provider using oLLM's *native* model-specific tool formatting.

Importing this optional module registers `model_type='ollm'`. Qwen-Agent owns
execution in this mode; do not put an oLLM Agent loop inside this provider.
"""
from __future__ import annotations

import copy
from pathlib import Path

try:
    from qwen_agent.llm.base import BaseChatModel, register_llm
    from qwen_agent.llm.schema import Message
except ImportError as exc:
    raise ImportError("Install this integration with: pip install --no-build-isolation -e '.[qwen-agent]' (from the patched checkout)") from exc

from .backend import InferenceBackend, config_from_mapping
from .registry import _validator
from .mcp import tool_alias
from .types import (NAME_PATTERN, ToolCall, ToolCallParseError, arguments_dict,
                    json_dumps, normalize_messages)


def function_alias(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Function names must be nonempty strings")
    return name if NAME_PATTERN.fullmatch(name) else tool_alias("qa", name)


def normalize_functions(functions: list[dict] | None) -> list[dict]:
    """Accept JSON Schema or Qwen-Agent's legacy parameter-descriptor lists."""
    result, names = [], set()
    for item in functions or []:
        fn = copy.deepcopy(item.get("function", item))
        name = function_alias(fn.get("name", fn.get("name_for_model")))
        if name in names:
            raise ValueError("Duplicate function name")
        names.add(name)
        parameters = fn.get("parameters", {"type": "object", "properties": {}})
        if isinstance(parameters, list):
            properties, required = {}, []
            for parameter in parameters:
                parameter = dict(parameter)
                key = parameter.pop("name")
                if key in properties:
                    raise ValueError("Duplicate function parameter")
                if parameter.pop("required", False):
                    required.append(key)
                properties[key] = parameter
            parameters = {"type": "object", "properties": properties, "required": required,
                          "additionalProperties": False}
        _validator(parameters)
        result.append({"type": "function", "function": {"name": name,
            "description": fn.get("description", fn.get("description_for_model", "")),
            "parameters": parameters}})
    return result


def normalize_qwen_messages(messages) -> list[dict]:
    """Convert Qwen-Agent function messages, preserving parallel call pairing."""
    converted, pending = [], []
    for value in messages:
        msg = copy.deepcopy(value if isinstance(value, dict) else value.model_dump())
        role = msg["role"]
        content = msg.get("content") or ""
        if isinstance(content, list):
            pieces = []
            for block in content:
                block = block if isinstance(block, dict) else block.model_dump()
                if (not isinstance(block.get("text"), str) or block.get("type", "text") != "text"
                        or any(v for k, v in block.items() if k not in ("text", "type"))):
                    raise ValueError("oLLM's Qwen-Agent provider accepts text blocks only")
                pieces.append(block["text"])
            content = "\n".join(pieces)
        fn = msg.get("function_call")
        if role == "assistant" and fn:
            if not isinstance(fn, dict):
                fn = fn.model_dump()
            extra = msg.get("extra") or {}
            # Qwen-Agent versions differ in their function_id handling. Generate
            # our own unique IDs and pair results by supplied ID, then by name.
            call = ToolCall(function_alias(fn["name"]), arguments_dict(fn.get("arguments", {})))
            pending.append((call, extra.get("function_id")))
            if converted and converted[-1]["role"] == "assistant" and converted[-1].get("tool_calls"):
                converted[-1]["tool_calls"].append(call.to_dict())
                if content:
                    converted[-1]["content"] += "\n" + content
            else:
                converted.append({"role": "assistant", "content": content,
                                  "tool_calls": [call.to_dict()]})
            if extra.get("ollm_thinking"):
                converted[-1]["thinking"] = extra["ollm_thinking"]
        elif role == "function":
            name = function_alias(msg.get("name"))
            given_id = (msg.get("extra") or {}).get("function_id")
            candidates = [(i, call) for i, (call, source_id) in enumerate(pending)
                          if call.name == name and given_id is not None and source_id == given_id]
            if not candidates:
                candidates = [(i, call) for i, (call, _) in enumerate(pending) if call.name == name]
            if not candidates:
                raise ValueError("Qwen-Agent returned a function result without a matching call")
            index, call = candidates[0]
            pending.pop(index)
            converted.append({"role": "tool", "name": name, "tool_call_id": call.id, "content": content})
        else:
            if pending:
                raise ValueError("Missing Qwen-Agent function result before the next message")
            converted.append({**msg, "content": content})
    if pending:
        raise ValueError("Qwen-Agent history contains unanswered function calls")
    return normalize_messages(converted)


@register_llm("ollm")
class OllmChatModel(BaseChatModel):
    """Local provider for Qwen-Agent Assistant/FnCallAgent.

    `stream=True` returns a one-element iterator containing a complete assistant
    turn. This is buffered streaming, not live token streaming. Function calls
    are never emitted partially. `delta_stream`, arbitrary stop strings and
    forced tool choices are rejected, rather than silently ignored.
    """
    def __init__(self, cfg=None, *, backend: InferenceBackend | None = None):
        cfg = dict(cfg or {})
        cfg.setdefault("model_type", "ollm")
        cfg.setdefault("model", getattr(getattr(backend, "inference", None), "model_id", "qwen3-next-80B"))
        if cfg.get("cache_dir") or cfg.get("generate_cfg", {}).get("cache_dir"):
            raise ValueError("Response caching is not supported here; use kv_cache_dir for Qwen disk caching")
        super().__init__(cfg)
        if backend is None and cfg.get("bounded") is not None:
            from ollm import BudgetInference, MemoryBudget
            import torch
            options = dict(cfg["bounded"])
            if cfg.get("download") or cfg.get("force_download"):
                raise ValueError("Download bounded checkpoints explicitly with the ollm.bounded CLI first")
            if "model_dir" not in options:
                raise ValueError("bounded.model_dir is required")
            model_dir = options.pop("model_dir")
            key = options.pop("model_key", cfg["model"])
            budget = MemoryBudget(**options.pop("budget", {}))
            dtype = options.pop("dtype", "bfloat16")
            if dtype not in ("float16", "bfloat16", "float32"):
                raise ValueError("bounded.dtype must be float16, bfloat16 or float32")
            inference = BudgetInference(model_dir, model_key=key,
                device=cfg.get("device", "cuda:0"), dtype=getattr(torch, dtype),
                budget=budget, cache_dir=cfg.get("kv_cache_dir"), **options)
            backend = InferenceBackend(inference)
        if backend is None:
            models_dir = Path(cfg.get("models_dir", "./models/")).expanduser()
            download = cfg.get("download", False)
            if cfg.get("force_download", False) and not download:
                raise ValueError("force_download requires download=True")
            if not (models_dir / cfg["model"]).is_dir() and not download:
                raise FileNotFoundError("Model directory is missing; set models_dir or explicitly set download=True")
            from ollm import Inference
            inference = Inference(cfg["model"], device=cfg.get("device", "cuda:0"),
                                  logging=cfg.get("logging", False))
            inference.ini_model(models_dir=str(models_dir),
                                force_download=cfg.get("force_download", False))
            backend = InferenceBackend(inference, cache_dir=cfg.get("kv_cache_dir"))
        self.backend = backend
        self._ollm_settings = copy.deepcopy(cfg.get("generate_cfg", {}))

    def chat(self, messages, functions=None, stream=True, delta_stream=False, extra_generate_cfg=None):
        if delta_stream:
            raise ValueError("delta_stream is not supported; use buffered stream=True or stream=False")
        settings = {**self._ollm_settings, **(extra_generate_cfg or {})}
        choice = settings.pop("function_choice", "auto")
        if choice not in ("auto", "none"):
            raise ValueError("Only function_choice='auto' or 'none' is supported")
        normalized_tools = normalize_functions(functions)
        original_names = [(item.get("function", item)).get("name", (item.get("function", item)).get("name_for_model"))
                          for item in functions or []]
        aliases = {function_alias(name): name for name in original_names}
        tools = normalized_tools if choice != "none" else []
        validators = {t["function"]["name"]: _validator(t["function"]["parameters"])
                      for t in normalized_tools}
        generation = config_from_mapping(self.backend.generation, settings)
        history = normalize_qwen_messages(messages)
        return_dicts = all(isinstance(m, dict) for m in messages)

        def complete():
            turn = self.backend.generate(history, tools, generation=generation)
            if choice == "none" and turn.tool_calls:
                raise ValueError("Model requested a tool while function_choice='none'")
            output = []
            if turn.tool_calls:
                for index, call in enumerate(turn.tool_calls):
                    if call.name not in aliases:
                        raise ToolCallParseError("Model requested an unadvertised Qwen-Agent function")
                    try:
                        validators[call.name].validate(call.arguments)
                    except Exception as exc:
                        raise ToolCallParseError("Invalid arguments for Qwen-Agent function: " + str(exc)[:1500]) from exc
                    output.append(Message(role="assistant", content=turn.content if index == 0 else "",
                        function_call={"name": aliases[call.name], "arguments": json_dumps(call.arguments)},
                        extra={"function_id": call.id, "ollm_thinking": turn.thinking if index == 0 else ""}))
            else:
                output.append(Message(role="assistant", content=turn.content))
            return [m.model_dump() for m in output] if return_dicts else output

        if stream:
            def buffered():
                yield complete()
            return buffered()
        return complete()

    # Implement BaseChatModel's abstract hooks as well. The public chat override
    # intentionally bypasses Qwen-Agent's model-specific prompt rewriting.
    def _chat_no_stream(self, messages, generate_cfg):
        return self.chat(messages, stream=False, extra_generate_cfg=generate_cfg)

    def _chat_stream(self, messages, delta_stream, generate_cfg):
        return self.chat(messages, stream=True, delta_stream=delta_stream, extra_generate_cfg=generate_cfg)

    def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang="en"):
        return self.chat(messages, functions, stream, delta_stream, generate_cfg)

"""Dependency-light, model-neutral conversation and tool types."""
from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


class ToolingError(RuntimeError):
    """Base exception for errors that must stop orchestration."""


class ToolCallParseError(ToolingError):
    """Malformed or ambiguous model output; no calls from this turn may execute."""


class IncompleteGeneration(ToolingError):
    """Generation ended without the model's proper action/final stop token."""


class AgentLimitError(ToolingError):
    def __init__(self, message: str, messages: list[dict]):
        super().__init__(message)
        self.messages = copy.deepcopy(messages)


def valid_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ValueError("Tool names must match [A-Za-z_][A-Za-z0-9_]{0,63}")
    return name


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def strict_json(text: str) -> Any:
    """Reject duplicate keys and non-finite numbers, not just invalid JSON syntax."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("Non-finite JSON number")
        return parsed

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def arguments_dict(value: Any) -> dict:
    if isinstance(value, str):
        value = strict_json(value)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object")
    # Also rejects non-JSON Python values / NaN supplied through the Python API.
    return strict_json(json_dumps(value))


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid4().hex)

    def __post_init__(self):
        valid_name(self.name)
        object.__setattr__(self, "arguments", arguments_dict(self.arguments))
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Tool-call id must be a nonempty string")

    def to_dict(self) -> dict:
        return {"id": self.id, "type": "function", "function": {
            "name": self.name, "arguments": copy.deepcopy(self.arguments)}}


@dataclass
class AssistantTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str = field(default="", repr=False)

    def to_message(self) -> dict:
        message = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [call.to_dict() for call in self.tool_calls]
            if self.thinking:
                message["thinking"] = self.thinking
        return message


@dataclass
class ToolResult:
    value: Any = None
    is_error: bool = False
    error_code: str | None = None

    @classmethod
    def error(cls, code: str, message: str) -> ToolResult:
        return cls({"code": code, "message": message}, True, code)

    def to_content(self, max_chars: int = 16000) -> str:
        """Always produce valid JSON, including when clipping a large result."""
        if max_chars < 256:
            raise ValueError("max_chars must be at least 256")
        payload = {"ok": not self.is_error,
                   "error" if self.is_error else "result": self.value}
        full = json_dumps(payload)
        if len(full) <= max_chars:
            return full
        def wrapped(n):
            return json_dumps({"ok": not self.is_error, "truncated": True,
                               "original_chars": len(full), "preview": full[:n]})
        lo, hi = 0, min(len(full), max_chars)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(wrapped(mid)) <= max_chars:
                lo = mid
            else:
                hi = mid - 1
        return wrapped(lo)


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Validate text-only history and exact tool-call/result pairing.

    Function arguments use dictionaries internally (not JSON-encoded strings).
    Qwen-Agent's legacy `function` messages are translated by its own adapter.
    """
    if not messages:
        raise ValueError("At least one conversation message is required")
    result, pending, seen_ids = [], {}, set()
    instructions, started = [], False
    for original in copy.deepcopy(messages):
        role = original.get("role")
        content = original.get("content", "")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("The tool layer currently accepts text-only message content")
        if role in ("system", "developer"):
            if started:
                raise ValueError("System/developer instructions must precede conversation messages")
            instructions.append(content)
            continue
        started = True
        if role not in ("user", "assistant", "tool"):
            raise ValueError(f"Unsupported message role: {role!r}")
        if role != "tool" and pending:
            raise ValueError("Every tool call must receive a result before the next non-tool message")
        item = {"role": role, "content": content}
        if role == "assistant":
            calls = original.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ValueError("tool_calls must be a list")
            normalized = []
            for raw in calls:
                fn = raw["function"]
                call = ToolCall(fn["name"], fn.get("arguments", {}), raw["id"])
                if call.id in seen_ids:
                    raise ValueError("Duplicate tool-call id")
                seen_ids.add(call.id)
                pending[call.id] = call.name
                normalized.append(call.to_dict())
            if normalized:
                item["tool_calls"] = normalized
            thinking = original.get("thinking", original.get("reasoning_content", ""))
            if thinking:
                if not isinstance(thinking, str):
                    raise ValueError("thinking must be text")
                item["thinking"] = thinking
        elif role == "tool":
            call_id = original.get("tool_call_id")
            if call_id not in pending:
                raise ValueError("Orphan or duplicate tool result")
            name = pending.pop(call_id)
            if original.get("name", name) != name:
                raise ValueError("Tool result name does not match its call")
            item.update(name=name, tool_call_id=call_id)
        elif original.get("tool_calls"):
            raise ValueError("Only assistant messages can call tools")
        result.append(item)
    if pending:
        raise ValueError("History has tool calls without results")
    if not result or result[-1]["role"] == "assistant":
        raise ValueError("Inference requires a user message or completed tool results at the end")
    if instructions:
        result.insert(0, {"role": "system", "content": "\n\n".join(instructions)})
    return result

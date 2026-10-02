"""Explicit tool registration, schema checks, approvals and bounded execution."""
from __future__ import annotations

import math
import asyncio
import copy
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import ToolCall, ToolResult, arguments_dict, json_dumps, valid_name


def _validator(schema: dict):
    try:
        from jsonschema.validators import validator_for
    except ImportError as exc:
        raise ImportError("Install tool support with: pip install --no-build-isolation -e '.[tools]' (from the patched checkout)") from exc
    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if not isinstance(value, str) or not value.startswith("#"):
                        raise ValueError("Only local JSON Schema references are permitted")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Tool parameters must be a JSON Schema with type='object'")
    check_refs(schema)
    cls = validator_for(schema)
    cls.check_schema(schema)
    return cls(schema)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Any] = field(repr=False)
    requires_approval: bool = True
    source: str = "python"
    _schema_validator: Any = field(init=False, repr=False)

    def __post_init__(self):
        valid_name(self.name)
        if not isinstance(self.description, str) or not callable(self.handler):
            raise ValueError("A description and callable handler are required")
        self.parameters = copy.deepcopy(self.parameters)
        self._schema_validator = _validator(self.parameters)

    def to_schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": copy.deepcopy(self.parameters)}}


@dataclass(frozen=True)
class ToolPolicy:
    """Default: registered tools only; approval-required tools are denied.

    Sync handlers run in threads. A timeout cannot forcibly stop a Python thread
    or prove that a remote side effect did not happen. No calls are auto-retried.
    """
    allowed_tools: frozenset[str] | None = None
    approve: Callable[[Tool, dict], Any] | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    max_argument_chars: int = 32000
    max_result_chars: int = 16000

    def __post_init__(self):
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or self.max_argument_chars < 2 or self.max_result_chars < 256:
            raise ValueError("Invalid tool policy limits")
        if self.allowed_tools is not None:
            object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))


async def _invoke(handler, *args, **kwargs):
    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None)):
        result = handler(*args, **kwargs)
    else:
        result = await asyncio.to_thread(handler, *args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


class ToolRegistry:
    def __init__(self, policy: ToolPolicy | None = None):
        self.policy = policy or ToolPolicy()
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> Tool:
        self.add_many([tool])
        return tool

    def add_many(self, tools: list[Tool]) -> None:
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names) or any(name in self._tools for name in names):
            raise ValueError("Duplicate tool names; registration is atomic and never overwrites tools")
        self._tools.update({tool.name: tool for tool in tools})

    def remove(self, name: str, *, expected: Tool | None = None) -> None:
        if expected is not None and self._tools.get(name) is not expected:
            return
        self._tools.pop(name, None)

    def tool(self, *, parameters: dict, name: str | None = None,
             description: str | None = None, requires_approval: bool = True):
        """Decorator retaining the original Python function."""
        def decorate(handler):
            self.add(Tool(name or handler.__name__, description or inspect.getdoc(handler) or "",
                          parameters, handler, requires_approval))
            return handler
        return decorate

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._tools.values())

    def schemas(self) -> list[dict]:
        allowed = self.policy.allowed_tools
        return [tool.to_schema() for tool in self._tools.values()
                if allowed is None or tool.name in allowed]

    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult.error("unknown_tool", "This tool is not registered")
        if self.policy.allowed_tools is not None and call.name not in self.policy.allowed_tools:
            return ToolResult.error("not_allowed", "This tool is outside the host allowlist")
        try:
            arguments = arguments_dict(call.arguments)
            if len(json_dumps(arguments)) > self.policy.max_argument_chars:
                return ToolResult.error("arguments_too_large", "Tool argument limit exceeded")
            tool._schema_validator.validate(arguments)
        except Exception as exc:
            # Validation details can contain data; cap before adding to history.
            return ToolResult.error("invalid_arguments", str(exc)[:1500])

        async def execute_checked():
            if tool.requires_approval:
                if self.policy.approve is None:
                    return ToolResult.error("approval_required", "The host has not approved this tool")
                approved = await _invoke(self.policy.approve, tool, copy.deepcopy(arguments))
                if approved is not True:
                    return ToolResult.error("approval_denied", "The host denied this tool call")
            value = await _invoke(tool.handler, **arguments)
            result = value if isinstance(value, ToolResult) else ToolResult(value)
            # Verify serializability inside the error boundary.
            result.to_content(self.policy.max_result_chars)
            return result

        try:
            return await asyncio.wait_for(execute_checked(), timeout=self.policy.timeout_seconds)
        except asyncio.TimeoutError:
            return ToolResult.error("timeout", "Tool timed out; outcome may be unknown. Do not automatically retry side effects.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult.error("execution_error", f"{type(exc).__name__}: {str(exc)[:1500]}")

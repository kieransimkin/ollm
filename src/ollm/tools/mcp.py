"""Optional MCP SDK 1.x client: stdio and Streamable HTTP tool discovery.

Transport/session context managers are entered and exited in the same asyncio
Task. This matters because the SDK owns AnyIO task groups and cancel scopes.
"""
from __future__ import annotations

import math
import asyncio
import hashlib
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlsplit

from .registry import Tool, ToolRegistry
from .types import ToolResult


def tool_alias(server: str, remote_name: str) -> str:
    raw = server + "__" + remote_name
    safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if safe == raw and len(safe) <= 64:
        return safe
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return safe[:50] + "_" + digest


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = field(default=None, repr=False)
    url: str | None = None
    headers: dict[str, str] | None = field(default=None, repr=False)
    timeout_seconds: float = 30.0
    allow_tools: frozenset[str] | None = None
    auto_approve: frozenset[str] = frozenset()
    allow_insecure_http: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,23}", self.name):
            raise ValueError("MCP server names must be safe identifiers, at most 24 characters")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "args", tuple(self.args))
        object.__setattr__(self, "auto_approve", frozenset(self.auto_approve))
        if self.allow_tools is not None:
            object.__setattr__(self, "allow_tools", frozenset(self.allow_tools))
        if self.transport == "stdio":
            if not self.command or self.url or self.headers:
                raise ValueError("stdio requires command, with no URL or HTTP headers")
        elif self.transport == "streamable-http":
            if not self.url or self.command or self.args or self.env:
                raise ValueError("streamable-http requires URL and no subprocess settings")
            parsed = urlsplit(self.url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("MCP HTTP URL must have an http(s) scheme and hostname")
            if parsed.username or parsed.password or parsed.fragment:
                raise ValueError("Use headers for credentials; URL credentials/fragments are not accepted")
            if (parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1")
                    and not self.allow_insecure_http):
                raise ValueError("Non-loopback MCP endpoints require HTTPS")
        else:
            raise ValueError("transport must be stdio or streamable-http")


def mcp_result_to_tool_result(result) -> ToolResult:
    """Retain text, resource references and structured content, not base64 media."""
    raw = result.model_dump(mode="json", exclude_none=True)
    content = []
    for item in raw.get("content", []):
        item = dict(item)
        if item.get("type") in ("image", "audio"):
            item.pop("data", None)
            item["omitted"] = "Binary media is not sent to the text-only model"
        if item.get("type") == "resource" and isinstance(item.get("resource"), dict):
            resource = dict(item["resource"])
            if "blob" in resource:
                resource.pop("blob")
                resource["omitted"] = "Binary resource content omitted"
            item["resource"] = resource
        content.append(item)
    payload = {"content": content}
    if "structuredContent" in raw:
        payload["structuredContent"] = raw["structuredContent"]
    return ToolResult(payload, is_error=bool(raw.get("isError", False)),
                      error_code="mcp_tool_error" if raw.get("isError") else None)


class MCPClient:
    """Keep one or more MCP sessions alive for an agent run.

    Usage: `async with MCPClient(registry) as client: await client.connect(config)`.
    Enter/connect/exit must run in the same task; tool calls may run in child
    tasks. On exit, registered MCP tools are removed and transports are closed.
    No arbitrary endpoints, subprocess commands, or auto-approvals come from the
    model. This client implements MCP tools only, not resources or prompts.
    """
    def __init__(self, registry: ToolRegistry, *, max_tools: int = 256, max_pages: int = 100):
        if max_tools < 1 or max_pages < 1:
            raise ValueError("Discovery limits must be positive")
        self.registry, self.max_tools, self.max_pages = registry, max_tools, max_pages
        self._owner_task = None
        self._connections: list[tuple[str, AsyncExitStack, list[Tool]]] = []

    async def __aenter__(self):
        if self._owner_task is not None:
            raise RuntimeError("MCPClient is already entered")
        self._owner_task = asyncio.current_task()
        return self

    def _check_owner(self):
        if self._owner_task is None or self._owner_task is not asyncio.current_task():
            raise RuntimeError("Enter, connect and close MCPClient in the same asyncio task")

    async def __aexit__(self, exc_type, exc, tb):
        self._check_owner()
        first_error = None
        try:
            for _, stack, tools in reversed(self._connections):
                for tool in tools:
                    self.registry.remove(tool.name, expected=tool)
                try:
                    await stack.aclose()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        finally:
            self._connections.clear()
            self._owner_task = None
        if first_error is not None:
            raise first_error
        return False

    async def connect(self, config: MCPServerConfig) -> list[Tool]:
        self._check_owner()
        if any(name == config.name for name, _, _ in self._connections):
            raise ValueError("MCP server name is already connected")
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            # SDK 1.x API; deliberately bounded to <2 in optional dependencies.
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as exc:
            raise ImportError("Install MCP support with: pip install --no-build-isolation -e '.[mcp]' (from the patched checkout)") from exc
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            if config.transport == "stdio":
                parameters = StdioServerParameters(command=config.command, args=list(config.args), env=config.env)
                streams = await stack.enter_async_context(stdio_client(parameters))
            else:
                streams = await stack.enter_async_context(streamablehttp_client(
                    config.url, headers=config.headers, timeout=config.timeout_seconds,
                    sse_read_timeout=max(300.0, config.timeout_seconds)))
            session = await stack.enter_async_context(ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=config.timeout_seconds)))
            await asyncio.wait_for(session.initialize(), config.timeout_seconds)
            discovered, cursor, cursors = [], None, set()
            for _ in range(self.max_pages):
                page = await asyncio.wait_for(session.list_tools(cursor=cursor), config.timeout_seconds)
                discovered.extend(page.tools)
                if len(discovered) > self.max_tools:
                    raise ValueError("MCP server exceeds the host tool-discovery limit")
                cursor = page.nextCursor
                if not cursor:
                    break
                if cursor in cursors:
                    raise ValueError("MCP server repeated a pagination cursor")
                cursors.add(cursor)
            else:
                raise ValueError("MCP server exceeds the host discovery-page limit")
            remote_names = {tool.name for tool in discovered}
            requested = (config.allow_tools or frozenset()) | config.auto_approve
            if requested - remote_names:
                raise ValueError("Configured MCP tool names were not advertised: " + ", ".join(sorted(requested - remote_names)))
            tools = []
            for descriptor in discovered:
                if config.allow_tools is not None and descriptor.name not in config.allow_tools:
                    continue
                remote_name = descriptor.name

                def make_handler(name):
                    async def invoke(**arguments):
                        result = await asyncio.wait_for(session.call_tool(name, arguments=arguments),
                                                        config.timeout_seconds)
                        return mcp_result_to_tool_result(result)
                    return invoke

                tools.append(Tool(
                    name=tool_alias(config.name, remote_name),
                    description=descriptor.description or "",
                    parameters=descriptor.inputSchema,
                    handler=make_handler(remote_name),
                    # Never trust a server's readOnlyHint as an approval grant.
                    requires_approval=remote_name not in config.auto_approve,
                    source="mcp:" + config.name))
            self.registry.add_many(tools)
            self._connections.append((config.name, stack, tools))
            return tools
        except BaseException:
            await stack.aclose()
            raise

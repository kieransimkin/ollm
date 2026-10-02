import asyncio
from contextlib import asynccontextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ollm.tools import MCPClient, MCPServerConfig, Tool, ToolCall, ToolRegistry
from ollm.tools.mcp import mcp_result_to_tool_result, tool_alias


class Result:
    def __init__(self, **data):
        self.data = data
    def model_dump(self, **kwargs):
        return self.data


@pytest.fixture
def sdk(monkeypatch):
    state = SimpleNamespace(events=[], calls=[], pages={None: SimpleNamespace(tools=[
        SimpleNamespace(name="add", description="sum", inputSchema={"type": "object"},
                        annotations={"readOnlyHint": True})], nextCursor=None)}, fail_init=False)

    @asynccontextmanager
    async def transport(*args, **kwargs):
        owner = asyncio.current_task()
        ident = len(state.events)
        state.events.append(("open", ident, kwargs))
        try:
            yield ("read", "write", lambda: "session")
        finally:
            assert owner is asyncio.current_task(), "cross-task close"
            state.events.append(("close", ident, {}))

    class Session:
        def __init__(self, read, write, **kwargs):
            pass
        async def __aenter__(self):
            self.owner = asyncio.current_task()
            return self
        async def __aexit__(self, *args):
            assert self.owner is asyncio.current_task()
        async def initialize(self):
            if state.fail_init:
                raise RuntimeError("init failed")
        async def list_tools(self, cursor=None):
            return state.pages[cursor]
        async def call_tool(self, name, arguments):
            state.calls.append((name, arguments))
            return Result(content=[{"type": "text", "text": "42"}], isError=False,
                          structuredContent={"result": 42})

    modules = {name: ModuleType(name) for name in ("mcp", "mcp.client", "mcp.client.stdio", "mcp.client.streamable_http")}
    modules["mcp"].ClientSession = Session
    modules["mcp"].StdioServerParameters = lambda **kwargs: SimpleNamespace(**kwargs)
    modules["mcp.client.stdio"].stdio_client = transport
    modules["mcp.client.streamable_http"].streamablehttp_client = transport
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def test_local_discovery_execution_and_cleanup(sdk):
    async def run():
        registry = ToolRegistry()
        async with MCPClient(registry) as client:
            tools = await client.connect(MCPServerConfig(name="demo", command="python", auto_approve=frozenset({"add"})))
            assert tools[0].name == "demo__add"
            result = await registry.execute(ToolCall("demo__add", {"a": 17, "b": 25}))
            assert result.value["structuredContent"]["result"] == 42
        assert registry.tools == ()
    asyncio.run(run())
    assert sdk.calls == [("add", {"a": 17, "b": 25})]
    assert [event[0] for event in sdk.events] == ["open", "close"]


def test_mcp_readonly_hint_does_not_approve(sdk):
    async def run():
        registry = ToolRegistry()
        async with MCPClient(registry) as client:
            await client.connect(MCPServerConfig(name="demo", command="python"))
            result = await registry.execute(ToolCall("demo__add", {}))
            assert result.error_code == "approval_required"
    asyncio.run(run())
    assert sdk.calls == []


def test_pagination_allowlist_and_safe_alias(sdk):
    sdk.pages[None].nextCursor = "next"
    sdk.pages["next"] = SimpleNamespace(tools=[SimpleNamespace(name="read.item", description="read", inputSchema={"type": "object"})], nextCursor=None)
    async def run():
        registry = ToolRegistry()
        async with MCPClient(registry) as client:
            tools = await client.connect(MCPServerConfig(name="demo", command="python",
                allow_tools=frozenset({"read.item"}), auto_approve=frozenset({"read.item"})))
            assert len(tools) == 1
            await registry.execute(ToolCall(tools[0].name, {}))
    asyncio.run(run())
    assert sdk.calls[0][0] == "read.item"


def test_two_servers_close_lifo_and_keep_python_tools(sdk):
    async def run():
        registry = ToolRegistry()
        registry.add(Tool("local", "local", {"type": "object"}, lambda: None))
        async with MCPClient(registry) as client:
            await client.connect(MCPServerConfig(name="first", command="python"))
            await client.connect(MCPServerConfig(name="second", command="python"))
            assert len(registry.tools) == 3
        assert [t.name for t in registry.tools] == ["local"]
    asyncio.run(run())
    assert [(kind, ident) for kind, ident, _ in sdk.events] == [("open", 0), ("open", 1), ("close", 1), ("close", 0)]


def test_remote_headers_and_timeout_passed(sdk):
    async def run():
        async with MCPClient(ToolRegistry()) as client:
            await client.connect(MCPServerConfig(name="remote", transport="streamable-http",
                url="https://example.invalid/mcp", headers={"Authorization": "Bearer TEST"}, timeout_seconds=12))
    asyncio.run(run())
    assert sdk.events[0][2]["headers"] == {"Authorization": "Bearer TEST"}
    assert sdk.events[0][2]["timeout"] == 12


@pytest.mark.parametrize("failure", ["init", "unknown_allowlist", "repeated_cursor", "page_limit", "tool_limit", "duplicate"])
def test_discovery_failures_cleanup_atomically(sdk, failure):
    async def run():
        registry = ToolRegistry()
        options, config_args = {}, {}
        if failure == "init":
            sdk.fail_init = True
        if failure == "unknown_allowlist":
            config_args["allow_tools"] = frozenset({"missing"})
        if failure in ("repeated_cursor", "page_limit"):
            sdk.pages[None].nextCursor = "next"
            sdk.pages["next"] = sdk.pages[None]
            if failure == "page_limit":
                options["max_pages"] = 1
        if failure == "tool_limit":
            sdk.pages[None].tools *= 2
            options["max_tools"] = 1
        if failure == "duplicate":
            registry.add(Tool("demo__add", "original", {"type": "object"}, lambda: None))
        before = registry.tools
        async with MCPClient(registry, **options) as client:
            with pytest.raises((ValueError, RuntimeError)):
                await client.connect(MCPServerConfig(name="demo", command="python", **config_args))
            assert registry.tools == before
    asyncio.run(run())
    assert [x[0] for x in sdk.events] == ["open", "close"]


def test_cross_task_lifecycle_rejected(sdk):
    async def run():
        async with MCPClient(ToolRegistry()) as client:
            with pytest.raises(RuntimeError, match="same asyncio task"):
                await asyncio.create_task(client.connect(MCPServerConfig(name="demo", command="python")))
    asyncio.run(run())
    assert not sdk.events


@pytest.mark.parametrize("kwargs", [
    {"name": "bad-name", "command": "python"}, {"name": "x"},
    {"name": "x", "transport": "sse", "url": "https://example.org"},
    {"name": "x", "transport": "streamable-http", "url": "http://example.org/mcp"},
    {"name": "x", "transport": "streamable-http", "url": "https://user:secret@example.org"},
    {"name": "x", "transport": "streamable-http", "url": "file:///tmp/a"},
    {"name": "x", "transport": "streamable-http", "url": "https://example.org", "command": "python"},
    {"name": "x", "command": "python", "timeout_seconds": float("nan")},
])
def test_invalid_server_config(kwargs):
    with pytest.raises(ValueError):
        MCPServerConfig(**kwargs)


def test_loopback_http_and_no_secret_repr():
    config = MCPServerConfig(name="x", transport="streamable-http", url="http://127.0.0.1:8000/mcp",
                             headers={"Authorization": "SECRET"})
    assert "SECRET" not in repr(config)


def test_aliases_are_stable_distinct_and_bounded():
    assert tool_alias("demo", "add") == "demo__add"
    assert tool_alias("demo", "read.item") != tool_alias("demo", "read-item")
    assert len(tool_alias("demo", "x" * 100)) <= 64
    assert tool_alias("demo", "read.item") == tool_alias("demo", "read.item")


def test_mcp_preserves_structured_errors_and_omits_binary():
    result = mcp_result_to_tool_result(Result(isError=True, structuredContent={"why": "failed"}, content=[
        {"type": "text", "text": "failure"}, {"type": "image", "data": "BASE64", "mimeType": "image/png"},
        {"type": "resource", "resource": {"uri": "file:///report", "blob": "BASE64"}}]))
    assert result.is_error and result.error_code == "mcp_tool_error"
    assert "BASE64" not in result.to_content()
    assert result.value["structuredContent"]["why"] == "failed"

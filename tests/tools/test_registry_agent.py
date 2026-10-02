import asyncio
import json
import threading

import pytest

from ollm.tools import (Agent, AgentLimitError, AssistantTurn, Tool, ToolCall,
                        ToolCallParseError, ToolPolicy, ToolRegistry, ToolResult)


def run(value):
    return asyncio.run(value)


def test_sync_and_async_tools(pair_schema):
    registry = ToolRegistry()
    threads = []
    @registry.tool(parameters=pair_schema, requires_approval=False)
    def add(a, b):
        threads.append(threading.get_ident())
        return a + b
    @registry.tool(parameters=pair_schema, requires_approval=False)
    async def multiply(a, b):
        return a * b
    assert add(1, 2) == 3  # decorator retains function
    threads.clear()
    assert run(registry.execute(ToolCall("add", {"a": 2, "b": 3}))).value == 5
    assert threads[0] != threading.get_ident()
    assert run(registry.execute(ToolCall("multiply", {"a": 2, "b": 3}))).value == 6


@pytest.mark.parametrize("arguments", [{"a": "bad", "b": 2}, {"a": 1}, {"a": 1, "b": 2, "c": 3}])
def test_schema_validation_before_execution(arguments, pair_schema):
    calls = []
    registry = ToolRegistry()
    registry.add(Tool("add", "sum", pair_schema, lambda **args: calls.append(args), False))
    result = run(registry.execute(ToolCall("add", arguments)))
    assert result.error_code == "invalid_arguments"
    assert calls == []


def test_default_approval_denies(pair_schema):
    registry = ToolRegistry()
    registry.add(Tool("add", "sum", pair_schema, lambda **args: pytest.fail("executed")))
    assert run(registry.execute(ToolCall("add", {"a": 1, "b": 2}))).error_code == "approval_required"


@pytest.mark.parametrize("approval,expected", [(False, "approval_denied"), (1, "approval_denied"), (True, None)])
def test_explicit_approval(approval, expected, pair_schema):
    def approve(tool, args):
        args["a"] = 999  # Approval code cannot accidentally mutate the actual call.
        return approval
    registry = ToolRegistry(ToolPolicy(approve=approve))
    registry.add(Tool("add", "sum", pair_schema, lambda a, b: a + b))
    result = run(registry.execute(ToolCall("add", {"a": 1, "b": 2})))
    assert result.error_code == expected
    if expected is None:
        assert result.value == 3


def test_async_approval(pair_schema):
    async def approve(tool, args):
        return True
    registry = ToolRegistry(ToolPolicy(approve=approve))
    registry.add(Tool("add", "sum", pair_schema, lambda a, b: a + b))
    assert run(registry.execute(ToolCall("add", {"a": 1, "b": 2}))).value == 3


def test_allowlist_and_unknown(pair_schema):
    registry = ToolRegistry(ToolPolicy(allowed_tools=frozenset()))
    registry.add(Tool("add", "sum", pair_schema, lambda a, b: a + b, False))
    assert registry.schemas() == []
    assert run(registry.execute(ToolCall("add", {"a": 1, "b": 2}))).error_code == "not_allowed"
    assert run(registry.execute(ToolCall("missing", {}))).error_code == "unknown_tool"


def test_timeout_not_retried():
    calls = []
    async def slow():
        calls.append(1)
        await asyncio.sleep(1)
    registry = ToolRegistry(ToolPolicy(timeout_seconds=0.01))
    registry.add(Tool("slow", "slow", {"type": "object"}, slow, False))
    result = run(registry.execute(ToolCall("slow", {})))
    assert result.error_code == "timeout" and len(calls) == 1


def test_cancellation_propagates():
    async def cancelled():
        raise asyncio.CancelledError()
    registry = ToolRegistry()
    registry.add(Tool("cancelled", "cancel", {"type": "object"}, cancelled, False))
    with pytest.raises(asyncio.CancelledError):
        run(registry.execute(ToolCall("cancelled", {})))


@pytest.mark.parametrize("handler", [lambda: 1 / 0, lambda: object(), lambda: float("nan")])
def test_errors_are_tool_results(handler):
    registry = ToolRegistry()
    registry.add(Tool("bad", "bad", {"type": "object"}, handler, False))
    assert run(registry.execute(ToolCall("bad", {}))).error_code == "execution_error"


def test_large_arguments_not_executed():
    registry = ToolRegistry(ToolPolicy(max_argument_chars=10))
    registry.add(Tool("echo", "echo", {"type": "object"}, lambda **kwargs: pytest.fail("called"), False))
    assert run(registry.execute(ToolCall("echo", {"s": "x" * 20}))).error_code == "arguments_too_large"


def test_atomic_registration_and_schema_copy(pair_schema):
    registry = ToolRegistry()
    tool = Tool("add", "sum", pair_schema, lambda **kwargs: None)
    registry.add(tool)
    other = Tool("other", "other", pair_schema, lambda **kwargs: None)
    with pytest.raises(ValueError):
        registry.add_many([other, tool])
    assert len(registry.tools) == 1
    schema = registry.schemas()[0]
    schema["function"]["parameters"]["properties"].clear()
    assert "a" in registry.schemas()[0]["function"]["parameters"]["properties"]


@pytest.mark.parametrize("key", ["$ref", "$dynamicRef", "$recursiveRef"])
def test_schema_cannot_fetch_external_references(key):
    with pytest.raises(ValueError):
        Tool("bad", "bad", {"type": "object", "properties": {"a": {key: "https://example.invalid/schema"}}}, lambda **args: None)


def test_local_schema_reference_works():
    schema = {"type": "object", "$defs": {"positive": {"type": "number", "minimum": 0}},
              "properties": {"a": {"$ref": "#/$defs/positive"}}, "required": ["a"]}
    registry = ToolRegistry()
    registry.add(Tool("echo", "echo", schema, lambda a: a, False))
    assert run(registry.execute(ToolCall("echo", {"a": 3}))).value == 3


def test_agent_full_roundtrip(scripted_backend, pair_schema):
    call = ToolCall("add", {"a": 17, "b": 25})
    backend = scripted_backend(AssistantTurn(tool_calls=[call]), AssistantTurn(content="42"))
    registry = ToolRegistry()
    registry.add(Tool("add", "sum", pair_schema, lambda a, b: a + b, False))
    result = Agent(backend, registry).run_sync("add")
    assert result.text == "42" and result.rounds == 2 and result.tool_calls == 1
    assert backend.seen[1][0][-1]["tool_call_id"] == call.id
    assert json.loads(backend.seen[1][0][-1]["content"])["result"] == 42


def test_agent_multiple_calls_ordered(scripted_backend):
    execution = []
    registry = ToolRegistry()
    registry.add(Tool("record", "record", {"type": "object"}, lambda **a: execution.append(a["n"]), False))
    backend = scripted_backend(AssistantTurn(tool_calls=[ToolCall("record", {"n": 1}), ToolCall("record", {"n": 2})]),
                               AssistantTurn(content="done"))
    Agent(backend, registry).run_sync("record")
    assert execution == [1, 2]


def test_agent_error_returned_to_model(scripted_backend):
    backend = scripted_backend(AssistantTurn(tool_calls=[ToolCall("missing", {})]), AssistantTurn(content="unavailable"))
    Agent(backend, ToolRegistry()).run_sync("try")
    assert json.loads(backend.seen[-1][0][-1]["content"])["ok"] is False


@pytest.mark.parametrize("limits", [{"max_rounds": 1}, {"max_tool_calls": 0}, {"max_calls_per_turn": 1}])
def test_agent_budget_blocks_whole_batch(limits, scripted_backend):
    registry = ToolRegistry()
    registry.add(Tool("bad", "bad", {"type": "object"}, lambda: pytest.fail("executed"), False))
    backend = scripted_backend(AssistantTurn(tool_calls=[ToolCall("bad", {}), ToolCall("bad", {})]))
    with pytest.raises(AgentLimitError) as exc:
        Agent(backend, registry, **limits).run_sync("bad")
    assert exc.value.messages[-1]["role"] == "user"


def test_parse_errors_abort_before_execution(scripted_backend):
    backend = scripted_backend(ToolCallParseError("malformed"))
    with pytest.raises(ToolCallParseError):
        Agent(backend, ToolRegistry()).run_sync("hi")


def test_run_sync_in_event_loop_rejected(scripted_backend):
    async def attempt():
        with pytest.raises(RuntimeError, match="await"):
            Agent(scripted_backend(), ToolRegistry()).run_sync("hi")
    run(attempt())

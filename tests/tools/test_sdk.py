"""Opt-in tests against installed SDKs; no LLM weights are needed.

Run OLLM_RUN_SDK_TESTS=1 PYTHONPATH=src pytest -c pytest-tools.ini -m sdk.
These tests are skipped, not mocked, when the opt-in is absent.
"""
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from ollm.tools import MCPClient, MCPServerConfig, ToolCall, ToolRegistry

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / 'examples/tools/mcp_server.py'
pytestmark = [pytest.mark.sdk, pytest.mark.skipif(
    os.environ.get('OLLM_RUN_SDK_TESTS') != '1', reason='Set OLLM_RUN_SDK_TESTS=1 for real SDK tests')]


def require(name):
    # In the explicitly opted-in suite, a missing SDK is an error, not a green skip.
    __import__(name)


async def check_mcp(config):
    registry = ToolRegistry()
    async with MCPClient(registry) as client:
        tools = await client.connect(config)
        assert {t.name for t in tools} == {'demo__add'}
        result = await registry.execute(ToolCall('demo__add', {'a': 17, 'b': 25}))
        assert not result.is_error, result.to_content()
        assert '42' in result.to_content()
    assert registry.tools == ()


def test_real_mcp_stdio():
    require('mcp')
    asyncio.run(check_mcp(MCPServerConfig('demo', command=sys.executable, args=(str(SERVER),),
        allow_tools=frozenset({'add'}), auto_approve=frozenset({'add'}))))


def test_real_mcp_streamable_http(tmp_path):
    require('mcp')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    with (tmp_path / 'server.log').open('w+') as log:
        process = subprocess.Popen([sys.executable, str(SERVER), '--transport', 'streamable-http',
                                    '--port', str(port)], stdout=log, stderr=log, cwd=ROOT)
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    log.seek(0)
                    pytest.fail('MCP server exited: ' + log.read())
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail('MCP HTTP server did not become ready')
                    time.sleep(.1)
            asyncio.run(check_mcp(MCPServerConfig('demo', transport='streamable-http',
                url=f'http://127.0.0.1:{port}/mcp', allow_tools=frozenset({'add'}),
                auto_approve=frozenset({'add'}))))
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def harmony_completion(adapter, *messages):
    """Render actual completion tokens without hard-coding control spellings.

    The model starts inside the assistant header prepared by Harmony. Strip
    precisely that header prefix from a rendered conversation; the SDK emits
    the correct handoff/final terminal token for the installed encoding.
    """
    h = adapter.harmony
    enc = adapter.encoding
    context = [h.Message.from_role_and_content(h.Role.USER, 'SDK fixture')]
    prefix = enc.render_conversation_for_completion(
        h.Conversation.from_messages(context), h.Role.ASSISTANT)
    rendered = enc.render_conversation_for_training(
        h.Conversation.from_messages(context + list(messages)),
        config=h.RenderConversationConfig(auto_drop_analysis=False))
    assert rendered[:len(prefix)] == prefix, 'Harmony completion prefix changed'
    completion = rendered[len(prefix):]
    assert completion and completion[-1] in enc.stop_tokens_for_assistant_actions()
    return completion


def test_real_harmony_render_parse_and_tool_roundtrip(pair_schema):
    require('openai_harmony')
    from ollm.tools import GPTOSSAdapter, IncompleteGeneration, ToolCallParseError
    adapter = GPTOSSAdapter()
    tools = [{'type': 'function', 'function': {'name': 'add', 'description': 'Add', 'parameters': pair_schema}}]
    history = [{'role': 'system', 'content': 'Use tools.'}, {'role': 'user', 'content': '17 + 25?'}]
    prepared = adapter.prepare(history, tools)
    assert prepared.input_ids and prepared.stop_ids
    text = adapter.encoding.decode_utf8(prepared.input_ids)
    assert 'add' in text and '17 + 25?' in text
    h = adapter.harmony
    completion = harmony_completion(adapter,
        h.Message.from_role_and_content(h.Role.ASSISTANT, '{"a":17,"b":25}')
        .with_recipient('functions.add').with_channel('commentary'))
    turn = adapter.parse(completion)
    assert turn.tool_calls[0].arguments == {'a': 17, 'b': 25}
    history += [turn.to_message(), {'role': 'tool', 'name': 'add',
                                  'tool_call_id': turn.tool_calls[0].id, 'content': '{"result":42}'}]
    continued = adapter.prepare(history, tools)
    assert '42' in adapter.encoding.decode_utf8(continued.input_ids)
    final_ids = harmony_completion(adapter,
        h.Message.from_role_and_content(h.Role.ASSISTANT, '42').with_channel('final'))
    assert adapter.parse(final_ids).content == '42'
    with pytest.raises(IncompleteGeneration):
        adapter.parse(final_ids[:-1])
    with pytest.raises(IncompleteGeneration):
        adapter.parse(completion[:-1])

    # The fix is in the fixture, not a relaxation of execution boundaries.
    invalid_json = harmony_completion(adapter,
        h.Message.from_role_and_content(h.Role.ASSISTANT, '{"a":')
        .with_recipient('functions.add').with_channel('commentary'))
    with pytest.raises(ToolCallParseError):
        adapter.parse(invalid_json)
    mixed = harmony_completion(adapter,
        h.Message.from_role_and_content(h.Role.ASSISTANT, 'Checking the sum.').with_channel('analysis'),
        h.Message.from_role_and_content(h.Role.ASSISTANT, '42').with_channel('final'))
    turn = adapter.parse(mixed)
    assert turn.content == '42' and turn.thinking == 'Checking the sum.'


@pytest.mark.parametrize('mode', ['python', 'mcp'])
def test_real_qwen_agent_with_scripted_model(mode):
    require('qwen_agent')
    if mode == 'mcp':
        require('mcp')
    # Isolate Qwen-Agent's MCP singleton/event-loop and SDK monkey patches.
    run = subprocess.run([sys.executable, str(ROOT / 'tests/tools/sdk_qwen_runner.py'), mode],
                         cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'SDK_QWEN_AGENT_OK' in run.stdout

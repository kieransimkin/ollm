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


def test_real_harmony_render_parse_and_tool_roundtrip(pair_schema):
    require('openai_harmony')
    from ollm.tools import GPTOSSAdapter, IncompleteGeneration
    adapter = GPTOSSAdapter()
    tools = [{'type': 'function', 'function': {'name': 'add', 'description': 'Add', 'parameters': pair_schema}}]
    history = [{'role': 'system', 'content': 'Use tools.'}, {'role': 'user', 'content': '17 + 25?'}]
    prepared = adapter.prepare(history, tools)
    assert prepared.input_ids and prepared.stop_ids
    text = adapter.encoding.decode_utf8(prepared.input_ids)
    assert 'add' in text and '17 + 25?' in text
    # The renderer ends with <|start|>assistant; completion begins in the header.
    completion = adapter.encoding.encode(
        ' to=functions.add<|meta_sep|>commentary<|im_sep|>{"a":17,"b":25}<|ghissue|>',
        allowed_special='all')
    turn = adapter.parse(completion)
    assert turn.tool_calls[0].arguments == {'a': 17, 'b': 25}
    history += [turn.to_message(), {'role': 'tool', 'name': 'add',
                                  'tool_call_id': turn.tool_calls[0].id, 'content': '{"result":42}'}]
    continued = adapter.prepare(history, tools)
    assert '42' in adapter.encoding.decode_utf8(continued.input_ids)
    final_ids = adapter.encoding.encode('<|meta_sep|>final<|im_sep|>42<|fim_suffix|>', allowed_special='all')
    assert adapter.parse(final_ids).content == '42'
    with pytest.raises(IncompleteGeneration):
        adapter.parse(final_ids[:-1])


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

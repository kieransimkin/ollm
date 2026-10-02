"""Subprocess helper for real Qwen-Agent tests; model generation is scripted."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool

from ollm.tools import AssistantTurn, GenerationConfig, ToolCall
from ollm.tools.qwen_agent import OllmChatModel
from ollm.tools.types import arguments_dict

ROOT = Path(__file__).resolve().parents[2]


class Add(BaseTool):
    name = 'sdk_add'
    description = 'Add two numbers'
    parameters = {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}},
                  'required': ['a', 'b'], 'additionalProperties': False}

    def call(self, params, **kwargs):
        args = arguments_dict(params)
        return json.dumps({'result': args['a'] + args['b']})


class Backend:
    generation = GenerationConfig()
    inference = SimpleNamespace(model_id='qwen3-next-80B')

    def __init__(self):
        self.calls = 0

    def generate(self, messages, tools, **kwargs):
        self.calls += 1
        if self.calls == 1:
            assert tools
            candidates = [t['function']['name'] for t in tools
                          if 'add' in t['function']['name']]
            assert len(candidates) == 1, candidates
            return AssistantTurn(tool_calls=[ToolCall(candidates[0], {'a': 17, 'b': 25})])
        assert self.calls == 2, 'Unexpected extra generation'
        assert messages[-1]['role'] == 'tool', messages
        assert '42' in messages[-1]['content'], messages[-1]
        return AssistantTurn('42')


def main():
    backend = Backend()
    if sys.argv[1] == 'mcp':
        tools = [{'mcpServers': {'demo': {'command': sys.executable,
                   'args': [str(ROOT / 'examples/tools/mcp_server.py')]}}}]
    else:
        tools = [Add()]
    bot = Assistant(llm=OllmChatModel(backend=backend), function_list=tools,
                    system_message='Use the provided addition tool.')
    output = []
    for output in bot.run(messages=[{'role': 'user', 'content': 'What is 17 + 25?'}]):
        pass
    assert backend.calls == 2
    assert output[-1]['content'] == '42', output
    print('SDK_QWEN_AGENT_OK')


if __name__ == '__main__':
    main()

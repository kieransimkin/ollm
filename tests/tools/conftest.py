import copy
from types import SimpleNamespace

import pytest

from ollm.tools import GenerationConfig

PAIR = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"], "additionalProperties": False}


class ScriptedBackend:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen = []
        self.generation = GenerationConfig()
        self.inference = SimpleNamespace(model_id="qwen3-next-80B")

    def generate(self, messages, tools, **kwargs):
        self.seen.append((copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
        value = self.turns.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.fixture
def pair_schema():
    return copy.deepcopy(PAIR)


@pytest.fixture
def scripted_backend():
    return ScriptedBackend

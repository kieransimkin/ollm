"""Provider contract tests. These stubs are not a substitute for test_sdk.py."""
import copy
import importlib
import json
import sys
import types
from abc import ABC, abstractmethod

import pytest

from ollm.tools import AssistantTurn, ToolCall, ToolCallParseError


@pytest.fixture
def provider(monkeypatch):
    names = ['qwen_agent', 'qwen_agent.llm', 'qwen_agent.llm.base', 'qwen_agent.llm.schema']
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class Message:
        def __init__(self, **kwargs):
            self.role = kwargs.pop('role')
            self.content = kwargs.pop('content', '')
            self.name = kwargs.pop('name', None)
            self.function_call = kwargs.pop('function_call', None)
            self.extra = kwargs.pop('extra', {})
            if kwargs:
                raise ValueError(kwargs)

        def model_dump(self):
            return copy.deepcopy(vars(self))

    class BaseChatModel(ABC):
        def __init__(self, cfg=None):
            cfg = cfg or {}
            self.model = cfg.get('model', '')
            self.model_type = cfg.get('model_type', '')
            self.generate_cfg = cfg.get('generate_cfg', {})

        @abstractmethod
        def _chat_no_stream(self, messages, generate_cfg): ...
        @abstractmethod
        def _chat_stream(self, messages, delta_stream, generate_cfg): ...
        @abstractmethod
        def _chat_with_functions(self, messages, functions, stream, delta_stream, generate_cfg, lang): ...

    registry = {}
    def register_llm(name):
        def decorator(cls):
            registry[name] = cls
            return cls
        return decorator

    sys.modules['qwen_agent.llm.base'].BaseChatModel = BaseChatModel
    sys.modules['qwen_agent.llm.base'].register_llm = register_llm
    sys.modules['qwen_agent.llm.schema'].Message = Message
    module_name = 'ollm.tools.qwen_agent'
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    module._test_registry = registry
    yield module
    # Do not let these contract stubs contaminate tests against a real SDK.
    sys.modules.pop(module_name, None)
    import ollm.tools
    if getattr(ollm.tools, 'qwen_agent', None) is module:
        delattr(ollm.tools, 'qwen_agent')


def functions(schema, name='add'):
    return [{'name': name, 'description': 'Add two numbers', 'parameters': schema}]


def user():
    return [{'role': 'user', 'content': 'Calculate 17 + 25.'}]


def test_provider_registration_and_no_model_load(provider, scripted_backend):
    model = provider.OllmChatModel(backend=scripted_backend(AssistantTurn('42')))
    assert provider._test_registry['ollm'] is provider.OllmChatModel
    assert model.model_type == 'ollm'
    assert model.chat(user(), stream=False)[0]['content'] == '42'


def test_normalize_schemas_and_legacy_list(provider, pair_schema):
    original = functions(pair_schema, 'demo-add')
    snapshot = copy.deepcopy(original)
    result = provider.normalize_functions(original)
    assert result[0]['function']['name'] == provider.function_alias('demo-add')
    assert original == snapshot
    legacy = [{'name_for_model': 'add', 'description_for_model': 'Add', 'parameters': [
        {'name': 'a', 'type': 'number', 'required': True}, {'name': 'b', 'type': 'number'}]}]
    p = provider.normalize_functions(legacy)[0]['function']['parameters']
    assert p['required'] == ['a'] and p['additionalProperties'] is False
    assert 'required' not in p['properties']['a']
    assert provider.normalize_functions([{'type': 'function', 'function': original[0]}]) == result


@pytest.mark.parametrize('bad', [
    [{'name': ''}], [{'name': 'a'}, {'name': 'a'}],
    [{'name': 'a', 'parameters': {'type': 'array'}}],
    [{'name': 'a', 'parameters': [{'name': 'x'}, {'name': 'x'}]}],
    [{'name': 'a', 'parameters': {'type': 'object', '$ref': 'https://example.invalid/schema'}}],
])
def test_bad_function_schemas(provider, bad):
    with pytest.raises(ValueError):
        provider.normalize_functions(bad)


def test_parallel_pairing_original_ids_reordered(provider):
    messages = user() + [
        {'role': 'assistant', 'content': '', 'function_call': {'name': 'demo-add', 'arguments': '{"a":1,"b":2}'},
         'extra': {'function_id': 'x', 'ollm_thinking': 'need two sums'}},
        {'role': 'assistant', 'content': '', 'function_call': {'name': 'demo-add', 'arguments': '{"a":3,"b":4}'},
         'extra': {'function_id': 'y'}},
        {'role': 'function', 'name': 'demo-add', 'content': '7', 'extra': {'function_id': 'y'}},
        {'role': 'function', 'name': 'demo-add', 'content': '3', 'extra': {'function_id': 'x'}},
    ]
    snapshot = copy.deepcopy(messages)
    result = provider.normalize_qwen_messages(messages)
    calls = result[1]['tool_calls']
    assert len(calls) == 2 and calls[0]['id'] != calls[1]['id']
    assert result[2]['tool_call_id'] == calls[1]['id']
    assert result[3]['tool_call_id'] == calls[0]['id']
    assert result[1]['thinking'] == 'need two sums'
    assert messages == snapshot


def test_legacy_result_ids_fall_back_by_name(provider):
    messages = user() + [
        {'role': 'assistant', 'function_call': {'name': 'add', 'arguments': '{}'}},
        {'role': 'function', 'name': 'add', 'content': '42', 'extra': {'function_id': '1'}},
    ]
    result = provider.normalize_qwen_messages(messages)
    assert result[-1]['tool_call_id'] == result[1]['tool_calls'][0]['id']


@pytest.mark.parametrize('bad', [
    [{'role': 'user', 'content': [{'image': 'image.png'}]}],
    [{'role': 'user', 'content': [{'text': 'hello', 'image': 'image.png'}]}],
    [{'role': 'function', 'name': 'add', 'content': '42'}],
    [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'function_call': {'name': 'add', 'arguments': '{}'}}],
    [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'function_call': {'name': 'add', 'arguments': '{}'}},
     {'role': 'user', 'content': 'next'}],
])
def test_bad_qwen_histories(provider, bad):
    with pytest.raises(ValueError):
        provider.normalize_qwen_messages(bad)


def test_text_blocks_and_messages(provider):
    messages = [provider.Message(role='user', content=[{'text': 'hello'}, {'type': 'text', 'text': 'world'}])]
    assert provider.normalize_qwen_messages(messages)[0]['content'] == 'hello\nworld'


def test_native_call_converted_back_to_original_name(provider, pair_schema, scripted_backend):
    name = 'demo-add'
    alias = provider.function_alias(name)
    backend = scripted_backend(AssistantTurn('Working', [ToolCall(alias, {'a': 17, 'b': 25})], 'internal'))
    model = provider.OllmChatModel(backend=backend)
    out = list(model.chat(user(), functions(pair_schema, name)))
    assert len(out) == 1 and len(out[0]) == 1
    fn = out[0][0]['function_call']
    assert fn['name'] == name and json.loads(fn['arguments']) == {'a': 17, 'b': 25}
    assert backend.seen[0][1][0]['function']['name'] == alias
    # Full tool history can be fed back without changing the dispatcher name.
    history = user() + out[0] + [{'role': 'function', 'name': name, 'content': '42',
                                'extra': {'function_id': out[0][0]['extra']['function_id']}}]
    normalized = provider.normalize_qwen_messages(history)
    assert normalized[-1]['name'] == alias
    assert normalized[1]['thinking'] == 'internal'


def test_message_input_returns_message_not_dict(provider, scripted_backend):
    model = provider.OllmChatModel(backend=scripted_backend(AssistantTurn('42', thinking='not for display')))
    out = model.chat([provider.Message(role='user', content='sum?')], stream=False)
    assert isinstance(out[0], provider.Message) and out[0].content == '42'
    assert not out[0].extra


def test_generate_options_are_forwarded(provider, scripted_backend):
    backend = scripted_backend(AssistantTurn('42'))
    model = provider.OllmChatModel({'generate_cfg': {'max_tokens': 120, 'temperature': .3}}, backend=backend)
    model.chat(user(), stream=False, extra_generate_cfg={'max_tokens': 100, 'lang': 'en', 'seed': 9})
    cfg = backend.seen[0][2]['generation']
    assert (cfg.max_new_tokens, cfg.temperature, cfg.seed) == (100, .3, 9)


@pytest.mark.parametrize('kwargs', [
    {'delta_stream': True}, {'extra_generate_cfg': {'stop': ['STOP']}},
    {'extra_generate_cfg': {'function_choice': 'add'}},
])
def test_unsupported_options_fail_before_generation(provider, scripted_backend, kwargs):
    backend = scripted_backend(AssistantTurn('42'))
    with pytest.raises(ValueError):
        provider.OllmChatModel(backend=backend).chat(user(), stream=False, **kwargs)
    assert not backend.seen


def test_none_choice_suppresses_advertisement(provider, pair_schema, scripted_backend):
    backend = scripted_backend(AssistantTurn('42'))
    provider.OllmChatModel(backend=backend).chat(user(), functions(pair_schema), stream=False,
                                               extra_generate_cfg={'function_choice': 'none'})
    assert backend.seen[0][1] == []


@pytest.mark.parametrize('choice,error', [('auto', ToolCallParseError), ('none', ValueError)])
def test_unadvertised_calls_refused(provider, pair_schema, scripted_backend, choice, error):
    backend = scripted_backend(AssistantTurn(tool_calls=[ToolCall('unknown', {})]))
    with pytest.raises(error):
        provider.OllmChatModel(backend=backend).chat(user(), functions(pair_schema), stream=False,
            extra_generate_cfg={'function_choice': choice})


def test_response_cache_rejected(provider, scripted_backend):
    with pytest.raises(ValueError):
        provider.OllmChatModel({'cache_dir': './responses'}, backend=scripted_backend())


def test_cfg_model_loading_requires_download_optin(provider, tmp_path):
    with pytest.raises(FileNotFoundError, match='download=True'):
        provider.OllmChatModel({'models_dir': str(tmp_path)})
    with pytest.raises(ValueError, match='force_download'):
        provider.OllmChatModel({'models_dir': str(tmp_path), 'force_download': True})


def test_provider_rejects_schema_invalid_call_before_yield(provider, pair_schema, scripted_backend):
    backend = scripted_backend(AssistantTurn(tool_calls=[ToolCall('add', {'a': 'not a number', 'b': 25})]))
    stream = provider.OllmChatModel(backend=backend).chat(user(), functions(pair_schema))
    with pytest.raises(ToolCallParseError, match='Invalid arguments'):
        next(stream)

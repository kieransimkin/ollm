from types import SimpleNamespace

import pytest

from ollm.tools.adapters import GPTOSSAdapter, QwenAdapter, parse_qwen_text
from ollm.tools.types import IncompleteGeneration, ToolCallParseError


class Tokenizer:
    eos_token_id = 9
    def get_vocab(self):
        return {"<|im_end|>": 9}
    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.kwargs = messages, kwargs
        return [1, 2, 3]
    def decode(self, tokens, **kwargs):
        return self.text


def test_qwen_tools_reach_template(pair_schema):
    tokenizer = Tokenizer()
    tools = [{"type": "function", "function": {"name": "add", "parameters": pair_schema}}]
    prepared = QwenAdapter().prepare([{"role": "user", "content": "hi"}], tools, tokenizer)
    assert tokenizer.kwargs["tools"] == tools
    assert tokenizer.kwargs["enable_thinking"] is False
    assert prepared.input_ids == [1, 2, 3] and prepared.stop_ids == [9]


def test_qwen_complete_and_incomplete_tokens():
    tokenizer = Tokenizer()
    tokenizer.text = "answer"
    assert QwenAdapter().parse([1, 9], tokenizer).content == "answer"
    with pytest.raises(IncompleteGeneration):
        QwenAdapter().parse([1], tokenizer)


def test_qwen_preamble_and_multiple_calls():
    turn = parse_qwen_text('Checking.\n<tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call>\n'
                           '<tool_call>{"name":"multiply","arguments":"{\\"a\\":3,\\"b\\":4}"}</tool_call>')
    assert turn.content == "Checking."
    assert [c.name for c in turn.tool_calls] == ["add", "multiply"]
    assert turn.tool_calls[1].arguments["b"] == 4


@pytest.mark.parametrize("prefix", ["<think>", ""])
def test_calls_inside_reasoning_are_inert(prefix):
    turn = parse_qwen_text(prefix + '<tool_call>{"name":"bad","arguments":{}}</tool_call></think>All done')
    assert not turn.tool_calls and turn.content == "All done"


def test_fenced_tool_example_is_inert():
    content = 'Example:\n```xml\n<tool_call>{"name":"add","arguments":{}}</tool_call>\n```'
    assert parse_qwen_text(content).tool_calls == []


@pytest.mark.parametrize("text", [
    '<tool_call>{"name":"add","arguments":{}}',
    '</tool_call>',
    '<tool_call>{oops}</tool_call>',
    '<tool_call>{"name":"add","arguments":[],"extra":1}</tool_call>',
    '<tool_call>{"name":"add","name":"evil","arguments":{}}</tool_call>',
    '<tool_call>{"name":"add","arguments":{"x":NaN}}</tool_call>',
    '<tool_call>{"name":"a.b","arguments":{}}</tool_call>',
    '<tool_call>{"name":"add","arguments":{}}</tool_call>Now I pretend it succeeded',
    '<tool_call>{"name":"add","arguments":{}}</tool_call><tool_call>{bad}</tool_call>',
    '<think>unfinished',
    '<tool_call>{"name":"add","arguments":{}}</tool_call>text<tool_call>{"name":"add","arguments":{}}</tool_call>',
])
def test_qwen_malformed_turns_rejected(text):
    with pytest.raises(ToolCallParseError):
        parse_qwen_text(text)


class Text:
    def __init__(self, text):
        self.text = text


def message(body, channel="final", recipient=None, role="assistant"):
    return SimpleNamespace(author=SimpleNamespace(role=role), content=[Text(body)], channel=channel, recipient=recipient)


@pytest.fixture
def harmony_structural_adapter():
    # Test our structural checks independently of the official tokenizer/parser.
    adapter = object.__new__(GPTOSSAdapter)
    adapter.harmony = SimpleNamespace(Role=SimpleNamespace(ASSISTANT="assistant"), TextContent=Text)
    return adapter


def test_harmony_tool_handoff(harmony_structural_adapter):
    turn = harmony_structural_adapter._parse_messages([
        message("private plan", "analysis"), message("Checking", "commentary"),
        message('{"a":1,"b":2}', "commentary", "functions.add")], "<|ghissue|>")
    assert turn.tool_calls[0].name == "add" and turn.content == "Checking"
    assert turn.thinking == "private plan"


@pytest.mark.parametrize("ending", ["<|fim_suffix|>", "<|return|>"])
def test_harmony_final_excludes_analysis(harmony_structural_adapter, ending):
    turn = harmony_structural_adapter._parse_messages([message("thinking", "analysis"), message("42")], ending)
    assert turn.content == "42" and not turn.tool_calls


@pytest.mark.parametrize("messages,ending", [
    ([message("oops", role="tool")], "<|fim_suffix|>"),
    ([message("{}", "commentary", "python")], "<|ghissue|>"),
    ([message("not-json", "commentary", "functions.add")], "<|ghissue|>"),
    ([message("{}", "final", "functions.add")], "<|ghissue|>"),
    ([message("42")], "<|ghissue|>"),
    ([message("{}", "commentary", "functions.add")], "<|fim_suffix|>"),
    ([message("{}", "commentary", "functions.add"), message("42")], "<|fim_suffix|>"),
    ([message("42"), message("{}", "commentary", "functions.add")], "<|ghissue|>"),
    ([message("thinking only", "analysis")], "<|fim_suffix|>"),
    ([message("x", "unknown")], "<|fim_suffix|>"),
])
def test_harmony_ambiguous_output_rejected(harmony_structural_adapter, messages, ending):
    with pytest.raises(ToolCallParseError):
        harmony_structural_adapter._parse_messages(messages, ending)

import json
import subprocess
import sys

import pytest

from ollm.tools.types import (ToolCall, ToolResult, arguments_dict,
                              normalize_messages, strict_json, valid_name)


@pytest.mark.parametrize("name", ["add", "_private", "demo__add", "x" * 64])
def test_valid_names(name):
    assert valid_name(name) == name


@pytest.mark.parametrize("name", ["", "demo-add", "x.y", "1thing", "x" * 65, "a\n", None])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        valid_name(name)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{bad}'])
def test_strict_json_rejects_ambiguous_data(text):
    with pytest.raises(ValueError):
        strict_json(text)


@pytest.mark.parametrize("value", [[], None, 5, "[]", {"a": float("nan")}])
def test_arguments_must_be_json_object(value):
    with pytest.raises((ValueError, TypeError)):
        arguments_dict(value)


def test_call_copies_arguments():
    original = {"a": 1}
    call = ToolCall("add", original)
    original["a"] = 2
    assert call.arguments == {"a": 1}
    output = call.to_dict()
    output["function"]["arguments"]["a"] = 3
    assert call.arguments["a"] == 1


@pytest.mark.parametrize("value", ["a" * 10000, '\\"\n' * 5000, "你好🌟" * 2000, {"items": list(range(1000))}])
def test_truncation_is_bounded_valid_json(value):
    content = ToolResult(value).to_content(300)
    assert len(content) <= 300
    data = json.loads(content)
    assert data["truncated"] is True and data["ok"] is True


def test_error_flag_survives_truncation():
    result = ToolResult.error("failed", "x" * 10000)
    assert json.loads(result.to_content(256))["ok"] is False


def test_history_pairing_and_no_mutation():
    call = ToolCall("add", {}, "one")
    original = [{"role": "system", "content": "A"}, {"role": "developer", "content": "B"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "tool_calls": [call.to_dict()]},
                {"role": "tool", "tool_call_id": "one", "content": "result"}]
    result = normalize_messages(original)
    assert result[0]["content"] == "A\n\nB"
    assert result[-1]["name"] == "add"
    assert "name" not in original[-1]


@pytest.mark.parametrize("history", [
    [], [{"role": "function", "content": "x"}], [{"role": "tool", "content": "x"}],
    [{"role": "user", "content": []}],
    [{"role": "assistant", "content": "finished"}],
    [{"role": "user", "content": "x"}, {"role": "system", "content": "late"}],
    [{"role": "assistant", "tool_calls": [ToolCall("add", {}, "one").to_dict()]}],
    [{"role": "assistant", "tool_calls": [ToolCall("add", {}, "one").to_dict()]},
     {"role": "tool", "tool_call_id": "one", "name": "other", "content": "x"}],
    [{"role": "assistant", "tool_calls": [ToolCall("add", {}, "one").to_dict()]},
     {"role": "tool", "tool_call_id": "one", "content": "x"},
     {"role": "tool", "tool_call_id": "one", "content": "x"}],
])
def test_bad_histories_fail(history):
    with pytest.raises((ValueError, KeyError)):
        normalize_messages(history)


def test_import_does_not_load_optional_dependencies():
    code = "import sys; import ollm.tools; assert not any(x in sys.modules for x in ['torch','transformers','mcp','openai_harmony','qwen_agent'])"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=20)


def test_lazy_root_unknown_attribute():
    import ollm
    with pytest.raises(AttributeError):
        ollm.not_an_attribute
    assert "Inference" in dir(ollm)


def test_multimodal_user_content_is_local_only(tmp_path):
    from ollm.tools.types import normalize_messages
    image=tmp_path/'image.png'; image.write_bytes(b'not-decoded-here')
    messages=[{'role':'user','content':[{'type':'image','image':str(image)},
                                        {'type':'text','text':'describe this'}]}]
    normalized=normalize_messages(messages)
    assert normalized[0]['content'][0]['image']==str(image.resolve())
    with pytest.raises(ValueError,match='remote'):
        normalize_messages([{'role':'user','content':[{'type':'image','image':'https://example.com/x.png'}]}])
    with pytest.raises(ValueError,match='Video'):
        normalize_messages([{'role':'user','content':[{'type':'video','video':'x.mp4'}]}])

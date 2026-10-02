from types import SimpleNamespace
import pytest
from ollm.tools.coder_adapter import QwenCoderAdapter, TextOnlyAdapter
from ollm.tools.types import ToolCallParseError, IncompleteGeneration
from ollm.tools.backend import InferenceBackend
from ollm.bounded.api import BudgetInference

SCHEMA={'type':'object','properties':{
    'text':{'type':'string'},'count':{'type':'integer'},'flag':{'type':'boolean'},
    'items':{'type':'array','items':{'type':'string'}}},'required':['text'],'additionalProperties':False}


def adapter():
    a=QwenCoderAdapter();a.schemas={'edit':SCHEMA};return a


def wrap(params):
    return '<tool_call>\n<function=edit>\n'+params+'\n</function>\n</tool_call>'


def test_coder_schema_types_and_literal_string():
    a=adapter()
    out=a.parse_text(wrap('<parameter=text>\n001\n</parameter>\n<parameter=count>\n12\n</parameter>\n'
        '<parameter=flag>\ntrue\n</parameter>\n<parameter=items>\n["a","b"]\n</parameter>'))
    assert out.tool_calls[0].arguments==dict(text='001',count=12,flag=True,items=['a','b'])


def test_coder_preserves_code_whitespace():
    code='  def f():\n    return "&amp;"\n'
    out=adapter().parse_text(wrap('<parameter=text>\n'+code+'\n</parameter>'))
    assert out.tool_calls[0].arguments['text']==code


@pytest.mark.parametrize('payload',[
    wrap('<parameter=text>x</parameter><parameter=text>y</parameter>'),
    wrap('<parameter=missing>x</parameter>'),
    wrap('<parameter=count>1</parameter>'),
    wrap('<parameter=text>x</parameter><parameter=count>true</parameter>'),
    wrap('<parameter=text>x</parameter><parameter=count>NaN</parameter>'),
    wrap('<parameter=text>x</parameter>')+' suffix',
    '<tool_call><function=missing></function></tool_call>',
    '<tool_call><function=edit><parameter=text>x</parameter></function>',
    wrap('<parameter=text><function=bad></function></parameter>'),
])
def test_coder_refuses_ambiguous_calls(payload):
    with pytest.raises(ToolCallParseError):adapter().parse_text(payload)


def test_coder_multiple_calls_and_reasoning():
    call=wrap('<parameter=text>hello</parameter>')
    out=adapter().parse_text('<think>not an action</think>\nI will edit.\n'+call+'\n'+call)
    assert out.thinking=='not an action' and len(out.tool_calls)==2
    assert out.content=='I will edit.'


def test_fenced_tool_example_is_inert():
    text='Here is an example:\n```xml\n'+wrap('<parameter=text>x</parameter>')+'\n```'
    assert not adapter().parse_text(text).tool_calls


def test_text_only_rejects_tools():
    with pytest.raises(ValueError,match='not qualified'):
        TextOnlyAdapter().prepare([{'role':'user','content':'test'}],[{'function':{}}],None)


def test_bounded_backend_auto_selection(make_checkpoint):
    p,_,_=make_checkpoint()
    o=BudgetInference(p,device='cpu',load_tokenizer=False,allow_unqualified=True,tool_format='qwen-coder')
    b=InferenceBackend(o)
    assert isinstance(b.adapter,QwenCoderAdapter)
    assert b.generation.max_context_tokens==o.budget.max_context_tokens
    assert o.DiskCache('anywhere').root=='anywhere'


@pytest.mark.parametrize('value,expected',[('True',True),('False',False),('true',True),('false',False)])
def test_coder_accepts_official_jinja_boolean_spelling(value,expected):
    out=adapter().parse_text(wrap('<parameter=text>x</parameter><parameter=flag>'+value+'</parameter>'))
    assert out.tool_calls[0].arguments['flag'] is expected


def test_coder_rejects_remote_schema_refs_without_fetching():
    a=adapter()
    import copy
    a.schemas=copy.deepcopy(a.schemas)
    a.schemas['edit']['$ref']='https://example.invalid/untrusted-schema'
    with pytest.raises(ToolCallParseError,match='local JSON Schema'):
        a.parse_text(wrap('<parameter=text>x</parameter>'))

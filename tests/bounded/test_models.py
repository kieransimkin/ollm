from dataclasses import replace
import json
import torch
import torch.nn.functional as F
import pytest
from safetensors.torch import save_file
from ollm.bounded.budget import MemoryBudget
from ollm.bounded.model import BoundedModel
from ollm.bounded.cache import DiskState
from ollm.bounded.api import BudgetInference
from ollm.bounded.config import ModelConfig, MODELS
from ollm.bounded.kernels import rms_norm, rotary

FAMILIES = ['qwen2','qwen3','llama','qwen3_moe','qwen3_next','qwen3_5','qwen3_5_moe','deepseek_v2','deepseek_v3']


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('sharded', [False, True])
def test_full_and_chunked_logits_match(make_checkpoint, family, sharded):
    path, _, _ = make_checkpoint(family, sharded=sharded)
    m = BoundedModel(path, device='cpu', dtype=torch.float32)
    ids = [1,3,4,7,9,5,6]
    with DiskState() as state:
        full = m.logits(m.forward_tokens(ids, state, 0))
    with DiskState() as state:
        for i in range(len(ids)):
            hidden = m.forward_tokens(ids[i:i+1], state, i)
        incremental = m.logits(hidden)
        assert len(state.streams) <= 2*m.config.num_hidden_layers
    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=3e-4)


@pytest.mark.parametrize('family', ['qwen3_moe','qwen3_next','qwen3_5_moe','deepseek_v2','deepseek_v3'])
def test_packed_experts_equal_separate(make_checkpoint, family):
    first, _, _ = make_checkpoint(family)
    packed, _, _ = make_checkpoint(family, packed=True)
    one = BoundedModel(first, device='cpu', dtype=torch.float32)
    two = BoundedModel(packed, device='cpu', dtype=torch.float32)
    ids = torch.tensor([[1,8,3]])
    assert torch.equal(one.generate(ids, max_new_tokens=5), two.generate(ids, max_new_tokens=5))
    assert two.store.max_read_observed <= two.budget.weight_tile_bytes


@pytest.mark.parametrize('family', FAMILIES)
def test_generate_is_repeatable_and_cleans_cache(make_checkpoint, tmp_path, family):
    p, _, _ = make_checkpoint(family)
    cache = tmp_path/'cache'
    m = BoundedModel(p, device='cpu', dtype=torch.float32, cache_root=cache)
    ids = torch.tensor([[1,8,3]])
    a = m.generate(ids, max_new_tokens=5)
    b = m.generate(ids, max_new_tokens=5)
    assert torch.equal(a,b) and a.shape == (1,8)
    assert list(cache.iterdir()) == []
    assert not m.last_report['memory_qualified']


@pytest.mark.parametrize('family', ['qwen2','qwen3','llama'])
def test_dense_full_layer_reference(make_checkpoint, family):
    path, _, w = make_checkpoint(family)
    m = BoundedModel(path, device='cpu', dtype=torch.float32)
    c = m.config
    ids = [1,4,7,3,8]
    x = w['model.embed_tokens.weight'][ids]
    def norm(x,name):
        return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+c.rms_norm_eps)*w[name]
    for i in range(c.num_hidden_layers):
        b = f'model.layers.{i}.'
        z = norm(x,b+'input_layernorm.weight')
        def proj(z,name):
            return F.linear(z,w[b+'self_attn.'+name+'.weight'],w.get(b+'self_attn.'+name+'.bias'))
        q = proj(z,'q_proj').reshape(5,c.num_attention_heads,c.head_dim)
        k = proj(z,'k_proj').reshape(5,c.num_key_value_heads,c.head_dim)
        v = proj(z,'v_proj').reshape(5,c.num_key_value_heads,c.head_dim)
        if family == 'qwen3':
            q,k=norm(q,b+'self_attn.q_norm.weight'),norm(k,b+'self_attn.k_norm.weight')
        q,k=(rotary(y,torch.arange(5),c.head_dim,c.rope) for y in (q,k))
        k=k.repeat_interleave(c.num_attention_heads//c.num_key_value_heads,1)
        v=v.repeat_interleave(c.num_attention_heads//c.num_key_value_heads,1)
        scores=q.transpose(0,1)@k.transpose(0,1).transpose(-1,-2)/c.head_dim**.5
        scores=scores.masked_fill(torch.ones(5,5).triu(1).bool(),-torch.inf)
        out=(scores.softmax(-1)@v.transpose(0,1)).transpose(0,1).reshape(5,-1)
        x=x+proj(out,'o_proj')
        z=norm(x,b+'post_attention_layernorm.weight')
        gate=F.silu(F.linear(z,w[b+'mlp.gate_proj.weight']))
        up=F.linear(z,w[b+'mlp.up_proj.weight'])
        x=x+F.linear(gate*up,w[b+'mlp.down_proj.weight'])
    expected=F.linear(norm(x[-1:], 'model.norm.weight'),w['lm_head.weight'])[0]
    with DiskState() as state:
        actual=m.logits(m.forward_tokens(ids,state,0))
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)


def test_tiny_tiles_equal_large_tiles(make_checkpoint):
    p,_,_=make_checkpoint('qwen3')
    budget=replace(MemoryBudget(),weight_tile_bytes=256)
    small=BoundedModel(p,device='cpu',dtype=torch.float32,budget=budget)
    normal=BoundedModel(p,device='cpu',dtype=torch.float32)
    ids=torch.tensor([[1,2,3]])
    assert torch.equal(small.generate(ids,max_new_tokens=3),normal.generate(ids,max_new_tokens=3))
    assert small.store.max_read_observed<=256


@pytest.mark.parametrize('prefix', ['model.language_model.','language_model.model.'])
def test_text_only_nested_prefix(make_checkpoint,prefix):
    p,_,_=make_checkpoint('qwen3_5',prefix=prefix)
    m=BoundedModel(p,device='cpu',dtype=torch.float32)
    assert m.generate(torch.tensor([[1,3]]),max_new_tokens=2).shape==(1,4)


def test_reject_missing_and_unknown_weights(make_checkpoint):
    p,c,w=make_checkpoint()
    w.pop('model.layers.0.self_attn.q_norm.weight')
    save_file(w,str(p/'model.safetensors'))
    with pytest.raises(ValueError,match='missing'):
        BoundedModel(p,device='cpu')
    p,c,w=make_checkpoint()
    w['model.layers.0.unknown.weight']=torch.ones(1)
    save_file(w,str(p/'model.safetensors'))
    with pytest.raises(ValueError,match='Unknown tensor'):
        BoundedModel(p,device='cpu')


def test_reject_unqualified_by_default(make_checkpoint):
    p,_,_=make_checkpoint()
    with pytest.raises(ValueError,match='not been GPU-qualified'):
        BudgetInference(p,device='cpu',load_tokenizer=False)
    o=BudgetInference(p,device='cpu',load_tokenizer=False,allow_unqualified=True)
    assert o.tool_format=='text' and not o.memory_qualified


@pytest.mark.parametrize('change',[{'model_type':'qwen4_exp'}, {'rope_scaling':{'type':'dynamic'}},
    {'index_topk':2048}, {'hidden_act':'gelu'}, {'output_gate_type':'mystery'}, {'num_key_value_heads':3}])
def test_reject_unsupported_architecture(make_checkpoint,change):
    p,c,_=make_checkpoint()
    c.update(change)
    with pytest.raises(ValueError):
        ModelConfig(c)


def test_no_quantization_fallback(make_checkpoint):
    p,c,_=make_checkpoint()
    c['quantization_config']={'quant_method':'awq'}
    with pytest.raises(ValueError):
        ModelConfig(c)


def test_generation_restrictions_and_cleanup(make_checkpoint,tmp_path):
    p,_,_=make_checkpoint()
    cache=tmp_path/'cache'
    m=BoundedModel(p,device='cpu',cache_root=cache)
    for kwargs in ({'num_beams':2},{'use_cache':False},{'max_new_tokens':999999},
                   {'past_key_values':object()},{'bad_option':1}):
        with pytest.raises(ValueError):
            m.generate(torch.tensor([[1,2]]),**kwargs)
    with pytest.raises(ValueError):
        m.generate(torch.tensor([[99999]]),max_new_tokens=1)
    assert not cache.exists() or not list(cache.iterdir())


def test_session_rejects_wrong_position(make_checkpoint):
    p,_,_=make_checkpoint('qwen3_5',num_hidden_layers=1,layer_types=['linear_attention'])
    m=BoundedModel(p,device='cpu',dtype=torch.float32)
    with DiskState() as state:
        m.forward_tokens([1],state,0)
        with pytest.raises(ValueError,match='position'):m.forward_tokens([2],state,0)


def test_extra_decoder_layer_rejected(make_checkpoint):
    p,c,w=make_checkpoint()
    w['model.layers.2.self_attn.q_proj.weight']=w['model.layers.0.self_attn.q_proj.weight'].clone()
    save_file(w,str(p/'model.safetensors'))
    with pytest.raises(ValueError,match='extra decoder'):
        BoundedModel(p,device='cpu')


@pytest.mark.parametrize('family',FAMILIES)
def test_bfloat16_real_tensor_generation(make_checkpoint,family):
    p,_,_=make_checkpoint(family,dtype=torch.bfloat16)
    m=BoundedModel(p,device='cpu',dtype=torch.bfloat16)
    output=m.generate(torch.tensor([[1,4,8]]),max_new_tokens=3)
    assert output.shape==(1,6)


def test_swish_gate_is_not_silently_sigmoid(make_checkpoint):
    p,_,_=make_checkpoint('qwen3_5',output_gate_type='swish')
    m=BoundedModel(p,device='cpu',dtype=torch.float32)
    with DiskState() as state:
        a=m.logits(m.forward_tokens([1,4,8],state,0))
    m.config.data['output_gate_type']='sigmoid'
    with DiskState() as state:
        b=m.logits(m.forward_tokens([1,4,8],state,0))
    assert not torch.allclose(a,b)


def test_qwen2_window_layer_order_matches_native_configuration():
    from conftest import tiny_config
    from ollm.bounded.config import ModelConfig
    config = ModelConfig(tiny_config('qwen2', num_hidden_layers=4,
        use_sliding_window=True, sliding_window=16, max_window_layers=2))
    assert config.layer_types == ['full_attention', 'full_attention', 'sliding_attention', 'sliding_attention']


@pytest.mark.parametrize('field,value', [('num_key_value_heads',0), ('head_dim',0),
    ('moe_layer_freq',0), ('decoder_sparse_step',0), ('n_group',0)])
def test_invalid_dimensions_fail_before_loading(field,value):
    from conftest import tiny_config
    from ollm.bounded.config import ModelConfig
    with pytest.raises(ValueError):
        ModelConfig(tiny_config('deepseek_v3', **{field:value}))


def test_all_configured_eos_ids_are_retained(make_checkpoint):
    import json
    p,_,_=make_checkpoint(eos_token_id=[2,3])
    (p/'generation_config.json').write_text(json.dumps({'eos_token_id':[3,4]}))
    o=BudgetInference(p,device='cpu',load_tokenizer=False,allow_unqualified=True)
    assert o.eos_token_ids==[3,4,2]
    from ollm.tools.backend import InferenceBackend
    assert InferenceBackend(o).adapter.eos_ids==[3,4,2]

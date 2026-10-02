import math
import torch
import torch.nn.functional as F
import pytest
from ollm.bounded.budget import MemoryBudget, BudgetGuard
from ollm.bounded.kernels import (online_attention, recurrent_delta, rms_norm, rotary,
                                  rope_parameters, route_experts, StreamOps)
from ollm.bounded.model import BoundedModel
from ollm.bounded.cache import DiskState


def guard():
    return BudgetGuard(MemoryBudget(), 'cpu')


@pytest.mark.parametrize('tokens', [1, 3, 17])
@pytest.mark.parametrize('block', [1, 4, 8])
@pytest.mark.parametrize('window', [None, 3])
def test_online_attention_matches_full_matrix(tokens, block, window):
    torch.manual_seed(13)
    q, k, v = torch.randn(tokens, 4, 8), torch.randn(tokens, 2, 8), torch.randn(tokens, 2, 6)
    def blocks():
        for i in range(0, tokens, block):
            yield i, k[i:i+block], v[i:i+block]
    actual = online_attention(q, blocks(), torch.arange(tokens), scale=8**-.5,
                              value_dim=6, guard=guard(), window=window)
    keys = k.repeat_interleave(2, 1).transpose(0, 1)
    values = v.repeat_interleave(2, 1).transpose(0, 1)
    scores = q.transpose(0, 1) @ keys.transpose(-1, -2) / math.sqrt(8)
    p = torch.arange(tokens)
    mask = p[None, :] <= p[:, None]
    if window:
        mask &= p[None, :] > p[:, None] - window
    expected = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ values).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)


def test_recurrent_delta_matches_matrix_transition_and_chunking():
    torch.manual_seed(21)
    q, k, v = [torch.randn(7, 3, 4) for _ in range(3)]
    g, beta = -torch.rand(7, 3), torch.rand(7, 3)
    actual, final = recurrent_delta(q, k, v, g, beta)
    qn = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) / 2
    kn = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    s, outputs = torch.zeros(3, 4, 4), []
    eye = torch.eye(4).expand(3, 4, 4)
    for i in range(7):
        kk = kn[i, :, :, None] @ kn[i, :, None, :]
        kv = kn[i, :, :, None] @ v[i, :, None, :]
        s = (eye - beta[i, :, None, None] * kk) @ (s * g[i].exp()[:, None, None]) + beta[i, :, None, None] * kv
        outputs.append((qn[i, :, None, :] @ s)[:, 0])
    torch.testing.assert_close(actual, torch.stack(outputs), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(final, s, atol=1e-6, rtol=1e-5)
    first, state = recurrent_delta(q[:3], k[:3], v[:3], g[:3], beta[:3])
    last, state = recurrent_delta(q[3:], k[3:], v[3:], g[3:], beta[3:], state)
    torch.testing.assert_close(torch.cat((first, last)), actual)


@pytest.mark.parametrize('interleaved', [False, True])
def test_rotary_zero_and_relative_invariance(interleaved):
    x = torch.randn(2, 3, 8)
    z = rotary(x, torch.zeros(2), 4, {'rope_type': 'default'}, interleaved=interleaved)
    torch.testing.assert_close(x, z)
    y = rotary(x, torch.tensor([7, 7]), 4, {'rope_type': 'default'}, interleaved=interleaved)
    torch.testing.assert_close((x[0]*x[1]).sum(-1), (y[0]*y[1]).sum(-1), atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(x[..., 4:], y[..., 4:])


@pytest.mark.parametrize('kind', ['default', 'linear', 'yarn', 'llama3'])
def test_rope_finite_and_unsupported_refused(kind):
    p = dict(rope_type=kind, factor=8, low_freq_factor=1, high_freq_factor=4, original_max_position_embeddings=8192)
    inv, mag = rope_parameters(64, p, 'cpu')
    assert torch.isfinite(inv).all() and mag > 0
    with pytest.raises(ValueError):
        rope_parameters(64, {'rope_type': 'dynamic'}, 'cpu')


def test_noaux_routing_uses_bias_for_choice_not_weights():
    logits = torch.tensor([[2., 1., 0., -1.]])
    cfg = dict(scoring_func='sigmoid', num_experts_per_tok=2, norm_topk_prob=True,
               n_group=2, topk_group=1, topk_method='noaux_tc', routed_scaling_factor=2.5)
    ids, weights = route_experts(logits, cfg, torch.tensor([-5., -5., -2., -2.]))
    assert set(ids[0].tolist()) == {2, 3}
    expected = logits.sigmoid().gather(1, ids)
    expected = expected / expected.sum(-1, keepdim=True) * 2.5
    torch.testing.assert_close(weights, expected)


@pytest.mark.parametrize('family', ['deepseek_v2', 'deepseek_v3'])
@pytest.mark.parametrize('use_yarn', [False, True])
def test_mla_compact_cache_matches_expanded_reference(make_checkpoint, family, use_yarn):
    overrides = {'rope_scaling': dict(type='yarn', factor=40, original_max_position_embeddings=4096,
        mscale=1., mscale_all_dim=1., beta_fast=32, beta_slow=1)} if use_yarn else {}
    path, _, weights = make_checkpoint(family, **overrides)
    m = BoundedModel(path, device='cpu', dtype=torch.float32)
    c = m.config
    torch.manual_seed(4)
    x = torch.randn(9, c.hidden_size)
    b = 'model.layers.0.self_attn.'
    def linear(x, name):
        return F.linear(x, weights[b + name + '.weight'], weights.get(b + name + '.bias'))
    def norm(x, name):
        return rms_norm(x, weights[b + name + '.weight'], c.rms_norm_eps)
    if c.get('q_lora_rank'):
        q = linear(norm(linear(x, 'q_a_proj'), 'q_a_layernorm'), 'q_b_proj')
    else:
        q = linear(x, 'q_proj')
    q = q.reshape(len(x), c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim)
    qn, qp = q.split((c.qk_nope_head_dim, c.qk_rope_head_dim), -1)
    lat, kp = linear(x, 'kv_a_proj_with_mqa').split((c.kv_lora_rank, c.qk_rope_head_dim), -1)
    lat = norm(lat, 'kv_a_layernorm')
    kv = linear(lat, 'kv_b_proj').reshape(len(x), c.num_attention_heads, c.qk_nope_head_dim + c.v_head_dim)
    kn, v = kv.split((c.qk_nope_head_dim, c.v_head_dim), -1)
    qp = rotary(qp, torch.arange(len(x)), c.qk_rope_head_dim, c.rope, interleaved=True, deepseek=True)
    kp = rotary(kp[:,None], torch.arange(len(x)), c.qk_rope_head_dim, c.rope, interleaved=True, deepseek=True)
    query = torch.cat((qn, qp), -1).transpose(0,1)
    keys = torch.cat((kn, kp.expand(-1,c.num_attention_heads,-1)), -1).transpose(0,1)
    factor = c.rope.get('factor',1)
    magnitude = (1+0.1*c.rope.get('mscale_all_dim',0)*math.log(factor)) if factor > 1 else 1
    scores = query @ keys.transpose(-1,-2) / math.sqrt(c.qk_nope_head_dim+c.qk_rope_head_dim) * magnitude**2
    expected = scores.masked_fill(torch.ones(9,9).triu(1).bool(), -torch.inf).softmax(-1) @ v.transpose(0,1)
    expected = linear(expected.transpose(0,1).reshape(9,-1), 'o_proj')
    with DiskState(max_block_tokens=256) as state:
        actual = m._mla(x, 0, state, 0)
        assert set(state.streams) == {(0,'latent'),(0,'rope')}
        assert state.streams[(0,'latent')].row_shape == (c.kv_lora_rank,)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=5e-5)


@pytest.mark.parametrize('family', ['qwen3_next','qwen3_5'])
def test_hybrid_projections_and_gating_against_separate_reference(make_checkpoint,family):
    path,_,weights=make_checkpoint(family)
    m=BoundedModel(path,device='cpu',dtype=torch.float32)
    c=m.config
    x=torch.randn(5,c.hidden_size)
    base='model.layers.0.linear_attn.'
    def lin(x,name): return F.linear(x,weights[base+name+'.weight'])
    kh,vh,kd,vd=c.linear_num_key_heads,c.linear_num_value_heads,c.linear_key_head_dim,c.linear_value_head_dim
    ratio=vh//kh
    if family=='qwen3_next':
        p=lin(x,'in_proj_qkvz')
        ba=lin(x,'in_proj_ba')
        groups=p.split(2*kd+2*ratio*vd,dim=-1)
        q=torch.cat([g[:,:kd] for g in groups],-1)
        k=torch.cat([g[:,kd:2*kd] for g in groups],-1)
        v=torch.cat([g[:,2*kd:2*kd+ratio*vd] for g in groups],-1)
        z=torch.cat([g[:,2*kd+ratio*vd:] for g in groups],-1).reshape(5,vh,vd)
        bagroups=ba.split(2*ratio,dim=-1)
        b=torch.cat([g[:,:ratio] for g in bagroups],-1)
        a=torch.cat([g[:,ratio:] for g in bagroups],-1)
        mixed=torch.cat([q,k,v],-1)
    else:
        mixed=lin(x,'in_proj_qkv')
        z=lin(x,'in_proj_z').reshape(5,vh,vd)
        b,a=lin(x,'in_proj_b'),lin(x,'in_proj_a')
    convolved=F.silu(F.conv1d(mixed.T[None],weights[base+'conv1d.weight'],
        padding=c.linear_conv_kernel_dim-1,groups=mixed.shape[1])[:,:,:5])[0].T
    q,k,v=convolved.split([kh*kd,kh*kd,vh*vd],-1)
    q=q.reshape(5,kh,kd).repeat_interleave(ratio,1)
    k=k.reshape(5,kh,kd).repeat_interleave(ratio,1)
    v=v.reshape(5,vh,vd)
    q=q/torch.sqrt(q.square().sum(-1,keepdim=True)+1e-6)/math.sqrt(kd)
    k=k/torch.sqrt(k.square().sum(-1,keepdim=True)+1e-6)
    decay=-weights[base+'A_log'].exp()*F.softplus(a+weights[base+'dt_bias'])
    state=torch.zeros(vh,kd,vd)
    outputs=[]
    eye=torch.eye(kd).expand(vh,kd,kd)
    for t in range(5):
        update=k[t,:,:,None]@v[t,:,None,:]
        transition=eye-b[t].sigmoid()[:,None,None]*(k[t,:,:,None]@k[t,:,None,:])
        state=transition@(state*decay[t].exp()[:,None,None])+b[t].sigmoid()[:,None,None]*update
        outputs.append((q[t,:,None,:]@state)[:,0,:])
    output=torch.stack(outputs)
    output=output*torch.rsqrt(output.square().mean(-1,keepdim=True)+c.rms_norm_eps)
    output=output*weights[base+'norm.weight']*F.silu(z)
    expected=lin(output.flatten(1),'out_proj')
    with DiskState() as cache:
        actual=m._delta(x,0,cache)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=2e-5)

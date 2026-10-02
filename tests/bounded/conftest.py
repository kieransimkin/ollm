"""Tiny, real safetensors checkpoints; no model or SDK mocks in math tests."""
from pathlib import Path
import json
import math
import torch
import pytest
from safetensors.torch import save_file

from ollm.bounded.config import ModelConfig

# Avoid massively oversubscribing this small-matrix numerical suite.
torch.set_num_threads(1)


def tiny_config(family='qwen3', **overrides):
    if family == 'qwen3_vl':
        text_overrides = {k: v for k, v in overrides.items() if k not in {'vision_config'}}
        text = dict(model_type='qwen3_vl_text', hidden_size=16, intermediate_size=24,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=4, vocab_size=41, max_position_embeddings=256,
                    rms_norm_eps=1e-6, hidden_act='silu', tie_word_embeddings=False,
                    eos_token_id=2, bos_token_id=1, attention_bias=False, rope_theta=10000.0,
                    rope_scaling={'mrope_interleaved': True, 'mrope_section': [1, 1, 0], 'rope_type': 'default'})
        # head_dim=4 => sum(mrope_section) must be 2.
        text['rope_scaling']['mrope_section'] = [1, 1, 0]
        text.update(text_overrides)
        vision = dict(model_type='qwen3_vl', depth=3, hidden_size=8, intermediate_size=12,
                      num_heads=2, num_position_embeddings=16, out_hidden_size=16,
                      patch_size=2, temporal_patch_size=2, spatial_merge_size=2, in_channels=3,
                      hidden_act='gelu_pytorch_tanh', deepstack_visual_indexes=[0, 1])
        vision.update(overrides.get('vision_config', {}))
        return dict(model_type='qwen3_vl', image_token_id=30, video_token_id=31,
                    vision_start_token_id=32, vision_end_token_id=33,
                    tie_word_embeddings=text.get('tie_word_embeddings', False),
                    text_config=text, vision_config=vision)
    c = dict(model_type=family, hidden_size=16, intermediate_size=24,
             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
             head_dim=4, vocab_size=41, max_position_embeddings=256,
             rms_norm_eps=1e-6, hidden_act='silu', tie_word_embeddings=False,
             eos_token_id=2, bos_token_id=1, attention_bias=False,
             rope_theta=10000.0, norm_topk_prob=True)
    if family in ('qwen3_moe', 'qwen3_next', 'qwen3_5_moe'):
        c.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12)
    if family in ('qwen3_next', 'qwen3_5', 'qwen3_5_moe'):
        c.update(linear_num_key_heads=2, linear_num_value_heads=4,
                 linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=4,
                 layer_types=['linear_attention', 'full_attention'], full_attention_interval=2,
                 partial_rotary_factor=0.5, shared_expert_intermediate_size=8 if family != 'qwen3_5' else 0)
    if family.startswith('deepseek'):
        c.update(n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=12,
                 n_shared_experts=1, first_k_dense_replace=1, moe_layer_freq=1,
                 kv_lora_rank=6, q_lora_rank=None if family == 'deepseek_v2' else 8,
                 qk_nope_head_dim=4, qk_rope_head_dim=4, v_head_dim=4,
                 scoring_func='softmax' if family == 'deepseek_v2' else 'sigmoid',
                 topk_method='greedy' if family == 'deepseek_v2' else 'noaux_tc',
                 n_group=2, topk_group=1, routed_scaling_factor=1.0 if family == 'deepseek_v2' else 2.5)
    c.update(overrides)
    return c


def make_weights(raw, packed=False, prefix='model.'):
    c = ModelConfig(raw)
    h = c.hidden_size
    gen = torch.Generator().manual_seed(483)
    tensors = {}
    def weight(name, shape, norm=False, zero=False):
        if norm:
            tensors[name] = torch.randn(shape, generator=gen) * .02 + (0 if zero else 1)
        else:
            tensors[name] = torch.randn(shape, generator=gen) * .1
    def proj(b, name, o, inp, bias=False):
        weight(b + name + '.weight', (o, inp))
        if bias:
            weight(b + name + '.bias', (o,))
    def mlp(b, inter):
        for name, shape in (('gate_proj', (inter, h)), ('up_proj', (inter, h)), ('down_proj', (h, inter))):
            weight(b + name + '.weight', shape)
    weight(prefix + 'embed_tokens.weight', (c.vocab_size, h))
    if not c.tie_word_embeddings:
        weight('lm_head.weight', (c.vocab_size, h))
    weight(prefix + 'norm.weight', (h,), True, c.hybrid)
    for i, kind in enumerate(c.layer_types):
        b = prefix + f'layers.{i}.'
        weight(b + 'input_layernorm.weight', (h,), True, c.hybrid)
        weight(b + 'post_attention_layernorm.weight', (h,), True, c.hybrid)
        if kind == 'linear_attention':
            a = b + 'linear_attn.'
            kh, vh, kd, vd = c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
            conv = 2 * kh * kd + vh * vd
            weight(a + 'conv1d.weight', (conv, 1, c.linear_conv_kernel_dim))
            weight(a + 'A_log', (vh,))
            weight(a + 'dt_bias', (vh,))
            weight(a + 'norm.weight', (vd,), True)
            proj(a, 'out_proj', h, vh * vd)
            if c.family == 'qwen3_next':
                proj(a, 'in_proj_qkvz', 2 * kh * kd + 2 * vh * vd, h)
                proj(a, 'in_proj_ba', 2 * vh, h)
            else:
                for k, v in [('in_proj_qkv', conv), ('in_proj_z', vh * vd), ('in_proj_b', vh), ('in_proj_a', vh)]:
                    proj(a, k, v, h)
        elif c.deepseek:
            a = b + 'self_attn.'
            heads, r = c.num_attention_heads, c.kv_lora_rank
            qdim = c.qk_nope_head_dim + c.qk_rope_head_dim
            if c.get('q_lora_rank'):
                proj(a, 'q_a_proj', c.q_lora_rank, h, c.get('attention_bias', False))
                weight(a + 'q_a_layernorm.weight', (c.q_lora_rank,), True)
                proj(a, 'q_b_proj', heads * qdim, c.q_lora_rank)
            else:
                proj(a, 'q_proj', heads * qdim, h)
            proj(a, 'kv_a_proj_with_mqa', r + c.qk_rope_head_dim, h, c.get('attention_bias', False))
            weight(a + 'kv_a_layernorm.weight', (r,), True)
            proj(a, 'kv_b_proj', heads * (c.qk_nope_head_dim + c.v_head_dim), r)
            proj(a, 'o_proj', h, heads * c.v_head_dim, c.get('attention_bias', False))
        else:
            a = b + 'self_attn.'
            heads, kvh, dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
            gate = c.hybrid and c.get('attn_output_gate', True)
            for k, o, inp in [('q_proj', heads * dim * (2 if gate else 1), h),
                              ('k_proj', kvh * dim, h), ('v_proj', kvh * dim, h), ('o_proj', h, heads * dim)]:
                proj(a, k, o, inp, k != 'o_proj' if family_of(c) == 'qwen2' else c.get('attention_bias', False))
            if c.family not in ('qwen2', 'llama'):
                weight(a + 'q_norm.weight', (dim,), True, c.hybrid)
                weight(a + 'k_norm.weight', (dim,), True, c.hybrid)
        a = b + 'mlp.'
        if c.is_moe(i):
            proj(a, 'gate', c.num_experts, h)
            if c.get('topk_method') == 'noaux_tc':
                weight(a + 'gate.e_score_correction_bias', (c.num_experts,))
            for e in range(c.num_experts):
                mlp(a + f'experts.{e}.', c.moe_intermediate_size)
            if packed:
                gates, downs = [], []
                for e in range(c.num_experts):
                    p = a + f'experts.{e}.'
                    gates.append(torch.cat((tensors.pop(p + 'gate_proj.weight'), tensors.pop(p + 'up_proj.weight')), 0))
                    downs.append(tensors.pop(p + 'down_proj.weight'))
                tensors[a + 'experts.gate_up_proj'] = torch.stack(gates)
                tensors[a + 'experts.down_proj'] = torch.stack(downs)
            if c.deepseek:
                mlp(a + 'shared_experts.', c.n_shared_experts * c.moe_intermediate_size)
            elif c.get('shared_expert_intermediate_size', 0):
                mlp(a + 'shared_expert.', c.shared_expert_intermediate_size)
                proj(a, 'shared_expert_gate', 1, h)
        else:
            mlp(a, c.intermediate_size)
    if c.family == 'qwen3_vl':
        v = raw['vision_config']
        vp = 'model.visual.'
        vh, vi = v['hidden_size'], v['intermediate_size']
        weight(vp + 'patch_embed.proj.weight',
               (vh, v['in_channels'], v['temporal_patch_size'], v['patch_size'], v['patch_size']))
        weight(vp + 'patch_embed.proj.bias', (vh,))
        weight(vp + 'pos_embed.weight', (v['num_position_embeddings'], vh))
        for i in range(v['depth']):
            b = vp + f'blocks.{i}.'
            for norm in ('norm1', 'norm2'):
                weight(b + norm + '.weight', (vh,), True)
                weight(b + norm + '.bias', (vh,))
            proj(b + 'attn.', 'qkv', 3 * vh, vh, True)
            proj(b + 'attn.', 'proj', vh, vh, True)
            proj(b + 'mlp.', 'linear_fc1', vi, vh, True)
            proj(b + 'mlp.', 'linear_fc2', vh, vi, True)
        group = vh * v['spatial_merge_size'] ** 2
        def merger(base, post):
            norm_width = group if post else vh
            weight(base + 'norm.weight', (norm_width,), True)
            weight(base + 'norm.bias', (norm_width,))
            proj(base, 'linear_fc1', group, group, True)
            proj(base, 'linear_fc2', v['out_hidden_size'], group, True)
        merger(vp + 'merger.', False)
        for j, _ in enumerate(v['deepstack_visual_indexes']):
            merger(vp + f'deepstack_merger_list.{j}.', True)
    return tensors


def family_of(c):
    return c.family


def checkpoint(path, family='qwen3', *, packed=False, sharded=False, prefix='model.', dtype=torch.float32, **overrides):
    path.mkdir(parents=True, exist_ok=True)
    c = tiny_config(family, **overrides)
    if family == 'qwen3_vl' and prefix == 'model.':
        prefix = 'model.language_model.'
    w = {k: v.to(dtype) for k, v in make_weights(c, packed=packed, prefix=prefix).items()}
    (path / 'config.json').write_text(json.dumps(c))
    if sharded:
        names = list(w)
        halves = [names[::2], names[1::2]]
        manifest = {}
        for i, names in enumerate(halves):
            name = f'model-{i:05}.safetensors'
            save_file({k: w[k] for k in names}, str(path / name))
            manifest.update({k: name for k in names})
        (path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': manifest}))
    else:
        save_file(w, str(path / 'model.safetensors'))
    return c, w


@pytest.fixture
def make_checkpoint(tmp_path):
    def make(family='qwen3', **kwargs):
        path = tmp_path / f'{family}-{len(list(tmp_path.iterdir()))}'
        c, w = checkpoint(path, family, **kwargs)
        return path, c, w
    return make

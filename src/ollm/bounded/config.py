"""Explicit architecture selection and candidate checkpoint registry.

Registry presence means implemented candidate, NOT measured hardware support.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
import math


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    family: str
    tool_format: str = 'qwen-json'
    thinking: bool | None = False
    qualification: str = 'required'


MODELS: dict[str, ModelSpec] = {}
for size in (4, 8, 14, 32):
    MODELS[f'qwen3-{size}b'] = ModelSpec(f'Qwen/Qwen3-{size}B', 'qwen3', thinking=None)
MODELS['qwen3-4b-instruct-2507'] = ModelSpec('Qwen/Qwen3-4B-Instruct-2507', 'qwen3')
MODELS['qwen3-next-80b-thinking'] = ModelSpec('Qwen/Qwen3-Next-80B-A3B-Thinking', 'qwen3_next', thinking=True)
MODELS['qwen3-next-80b-instruct'] = ModelSpec('Qwen/Qwen3-Next-80B-A3B-Instruct', 'qwen3_next')
MODELS['qwen3-coder-next'] = ModelSpec('Qwen/Qwen3-Coder-Next', 'qwen3_next', 'qwen-coder')
for name, repo in (
    ('qwen3-30b-a3b', 'Qwen3-30B-A3B'),
    ('qwen3-30b-a3b-instruct-2507', 'Qwen3-30B-A3B-Instruct-2507'),
    ('qwen3-30b-a3b-thinking-2507', 'Qwen3-30B-A3B-Thinking-2507'),
    ('qwen3-coder-30b-a3b', 'Qwen3-Coder-30B-A3B-Instruct'),
):
    MODELS[name] = ModelSpec('Qwen/' + repo, 'qwen3_moe',
                            'qwen-coder' if 'Coder' in repo else 'qwen-json',
                            True if 'Thinking' in repo else (None if repo == 'Qwen3-30B-A3B' else False))
for size in ('1.5', '7', '14', '32'):
    for coder in (False, True):
        key = f'qwen2.5-{"coder-" if coder else ""}{size}b'
        MODELS[key] = ModelSpec(f'Qwen/Qwen2.5-{"Coder-" if coder else ""}{size}B-Instruct', 'qwen2')
    MODELS[f'deepseek-r1-distill-qwen-{size}b'] = ModelSpec(
        f'deepseek-ai/DeepSeek-R1-Distill-Qwen-{size}B', 'qwen2', 'text', True)
MODELS['deepseek-r1-distill-llama-8b'] = ModelSpec('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'llama', 'text', True)
MODELS['deepseek-r1-0528-qwen3-8b'] = ModelSpec('deepseek-ai/DeepSeek-R1-0528-Qwen3-8B', 'qwen3', 'text', True)
for size in (4, 9, 27):
    MODELS[f'qwen3.5-{size}b'] = ModelSpec(f'Qwen/Qwen3.5-{size}B', 'qwen3_5', 'qwen-coder', None)
MODELS['qwen3.5-35b-a3b'] = ModelSpec('Qwen/Qwen3.5-35B-A3B', 'qwen3_5_moe', 'qwen-coder', None)
# Later releases may use the same math, but must pass config/tensor validation;
# no blanket newer-model aliases or unverified public repository names.
for key, repo, family in (
    ('deepseek-v2-lite-chat', 'DeepSeek-V2-Lite-Chat', 'deepseek_v2'),
    ('deepseek-coder-v2-lite', 'DeepSeek-Coder-V2-Lite-Instruct', 'deepseek_v2'),
    ('deepseek-v3', 'DeepSeek-V3', 'deepseek_v3'),
    ('deepseek-r1', 'DeepSeek-R1', 'deepseek_v3'),
    ('deepseek-r1-0528', 'DeepSeek-R1-0528', 'deepseek_v3'),
):
    MODELS[key] = ModelSpec('deepseek-ai/' + repo, family, 'text', 'r1' in key)


FAMILIES = {'qwen2', 'qwen3', 'qwen3_moe', 'llama', 'qwen3_next',
            'qwen3_5', 'qwen3_5_moe', 'deepseek_v2', 'deepseek_v3'}


class ModelConfig:
    def __init__(self, raw: dict, expected_family: str | None = None):
        self.raw = copy.deepcopy(raw)
        self.data = copy.deepcopy(raw.get('text_config', raw))
        family = self.data.get('model_type', raw.get('model_type', '')).removesuffix('_text')
        if family not in FAMILIES:
            raise ValueError(f"No bounded implementation for {family!r}; no AutoModel fallback")
        if expected_family and family != expected_family:
            raise ValueError(f"Checkpoint is {family}, not registered architecture {expected_family}")
        self.family = family
        self.hybrid = family in ('qwen3_next', 'qwen3_5', 'qwen3_5_moe')
        self.deepseek = family.startswith('deepseek_')
        self.zero_centered_norm = self.hybrid
        d = self.data
        for k in ('hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size'):
            if type(d.get(k)) is not int or d[k] < 1:
                raise ValueError(f"Invalid {k}")
        d.setdefault('head_dim', d['hidden_size'] // d['num_attention_heads'])
        d.setdefault('num_key_value_heads', d['num_attention_heads'])
        d.setdefault('rms_norm_eps', 1e-6)
        d.setdefault('max_position_embeddings', 32768)
        d.setdefault('tie_word_embeddings', raw.get('tie_word_embeddings', False))
        if d.get('hidden_act', 'silu') not in ('silu', 'swish'):
            raise ValueError("Only the validated SwiGLU MLP is supported")
        if type(d['head_dim']) is not int or d['head_dim'] < 1:
            raise ValueError('Invalid head_dim')
        if type(d['num_key_value_heads']) is not int or d['num_key_value_heads'] < 1:
            raise ValueError('Invalid num_key_value_heads')
        if d['num_attention_heads'] % d['num_key_value_heads']:
            raise ValueError("Attention heads must be divisible by KV heads")
        if d.get('pretraining_tp', 1) != 1 or d.get('ep_size', 1) != 1:
            raise ValueError("Use an unpartitioned HF checkpoint, not tensor/expert-parallel shards")
        for k in ('index_topk', 'index_n_heads', 'hc_mult', 'compress_ratios'):
            if k in d:
                raise ValueError(f"{k} changes DeepSeek semantics; V3.2/V4 are not V3 aliases")
        if d.get('mlp_bias', False):
            raise ValueError("Biased MLP architecture is not implemented")
        if d.get('output_gate_type', d.get('attn_output_gate_type', 'sigmoid')) not in ('sigmoid', 'swish', 'silu'):
            raise ValueError("Unknown attention output gate")
        if d.get('linear_attn_output_gate', 'silu') not in ('silu', 'swish'):
            raise ValueError("Unsupported DeltaNet gate")
        if d.get('scoring_func', 'softmax') not in ('softmax', 'sigmoid'):
            raise ValueError('Unsupported routing scoring function')
        if d.get('rms_norm_eps', 1e-6) <= 0:
            raise ValueError('rms_norm_eps must be positive')
        q = raw.get('quantization_config') or d.get('quantization_config')
        if q and (not self.deepseek or q.get('quant_method') != 'fp8'
                  or q.get('fmt', 'e4m3') != 'e4m3'
                  or q.get('weight_block_size', [128, 128]) != [128, 128]):
            raise ValueError("Only DeepSeek block-scaled E4M3 FP8 is supported; use BF16 Qwen checkpoints")
        self.quantization = q
        rope = dict(d.get('rope_parameters') or d.get('rope_scaling') or {})
        rope.setdefault('rope_theta', d.get('rope_theta', 10000.0))
        rope.setdefault('rope_type', rope.get('type', 'default'))
        rope.setdefault('partial_rotary_factor', d.get('partial_rotary_factor', 1.0))
        if rope['rope_type'] not in ('default', 'linear', 'yarn', 'llama3'):
            raise ValueError("Dynamic/unknown RoPE requires cache re-rotation and is rejected")
        self.rope = rope
        if self.hybrid:
            for key in ('linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim',
                        'linear_value_head_dim', 'linear_conv_kernel_dim'):
                if type(d.get(key)) is not int or d[key] < 1:
                    raise ValueError(f"Missing/invalid hybrid dimension: {key}")
            if d['linear_num_value_heads'] % d['linear_num_key_heads']:
                raise ValueError("DeltaNet value/key head ratio is not integral")
        types = d.get('layer_types')
        if types is None:
            if self.hybrid:
                interval = d.get('full_attention_interval', 4)
                if type(interval) is not int or interval < 1:
                    raise ValueError("Invalid full_attention_interval")
                types = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention'
                         for i in range(d['num_hidden_layers'])]
            else:
                types = ['full_attention'] * d['num_hidden_layers']
                if d.get('use_sliding_window', False):
                    types = ['sliding_attention' if i >= d.get('max_window_layers', 28) else t
                             for i, t in enumerate(types)]
        if len(types) != d['num_hidden_layers'] or set(types) - {'linear_attention', 'full_attention', 'sliding_attention'}:
            raise ValueError("Invalid layer_types")
        if not self.hybrid and 'linear_attention' in types:
            raise ValueError("Non-hybrid architecture contains linear attention")
        self.layer_types = list(types)
        if 'sliding_attention' in types and (not isinstance(d.get('sliding_window'), int) or d['sliding_window'] < 1):
            raise ValueError("Sliding attention requires a positive window")
        self.num_experts = d.get('n_routed_experts' if self.deepseek else 'num_experts', 0) or 0
        if type(self.num_experts) is not int or self.num_experts < 0:
            raise ValueError('Invalid number of experts')
        for key in ('decoder_sparse_step', 'moe_layer_freq'):
            if type(d.get(key, 1)) is not int or d.get(key, 1) < 1:
                raise ValueError(f'Invalid {key}')
        if self.num_experts:
            topk = d.get('num_experts_per_tok', 0)
            if type(topk) is not int or not 0 < topk <= self.num_experts:
                raise ValueError("Invalid expert routing top-k")
            groups = d.get('n_group', 1)
            if (type(groups) is not int or groups < 1 or self.num_experts % groups
                    or type(d.get('topk_group', 1)) is not int
                    or not 0 < d.get('topk_group', 1) <= groups):
                raise ValueError("Invalid grouped routing dimensions")
            if d.get('topk_method') in ('group_limited_greedy', 'noaux_tc'):
                if topk > d.get('topk_group', 1) * (self.num_experts // groups):
                    raise ValueError('Routing top-k exceeds selected groups')
                if d.get('topk_method') == 'noaux_tc' and self.num_experts // groups < 2:
                    raise ValueError('noaux_tc requires at least two experts per group')
        if self.deepseek:
            for k in ('kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim'):
                if type(d.get(k)) is not int or d[k] < 1:
                    raise ValueError(f"Invalid MLA dimension {k}")
            if d.get('topk_method', 'greedy') not in ('greedy', 'group_limited_greedy', 'noaux_tc'):
                raise ValueError("Unsupported DeepSeek router")
        # No high-dimensional allocation happens here. Reject pathological dims.
        for k, v in d.items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"Non-finite config {k}")

    def get(self, name, default=None):
        return self.data.get(name, default)

    def __getattr__(self, name):
        if name in self.data:
            return self.data[name]
        raise AttributeError(name)

    def is_moe(self, i):
        if not self.num_experts:
            return False
        if self.deepseek:
            return i >= self.get('first_k_dense_replace', 0) and i % self.get('moe_layer_freq', 1) == 0
        return i not in self.get('mlp_only_layers', []) and (i + 1) % self.get('decoder_sparse_step', 1) == 0

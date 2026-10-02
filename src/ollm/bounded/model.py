"""Functional checkpoint-driven decoder with strictly tiled weight residency.

This is not AutoModel(device_map='auto'): no layer/expert module ever owns a
complete weight matrix. Only local checkpoint bytes are accepted. Batch one,
text-only and inference-only are deliberate constraints.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from .budget import BudgetGuard, MemoryBudget, MemoryBudgetError, GENERATION_LOCK
from .cache import DiskState
from .checkpoint import TensorStore, WeightRef, read_json
from .config import ModelConfig
from .kernels import StreamOps, rms_norm, rotary, online_attention, recurrent_delta, route_experts, yarn_scale


@dataclass(frozen=True)
class CacheLocation:
    root: str | Path


class BoundedModel:
    def __init__(self, root, *, config=None, budget=None, device='cuda:0', dtype=torch.bfloat16,
                 expected_family=None, cache_root=None):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.config = ModelConfig(config or read_json(self.root / 'config.json'), expected_family)
        self.budget = budget or MemoryBudget()
        self.device = torch.device(device)
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Use BF16/FP16, or FP32 for CPU reference testing")
        if self.device.type == 'cuda' and dtype == torch.float32:
            raise ValueError("CUDA profiles are BF16/FP16 only")
        self.dtype, self.cache_root = dtype, cache_root
        self.guard = BudgetGuard(self.budget, self.device)
        self.store = TensorStore(self.root, max_read_bytes=self.budget.weight_tile_bytes)
        self.ops = StreamOps(self.store, self.guard, dtype)
        self.last_report = {}
        self._resolve_layout()
        self._validate_weights()
        self.prefill_tokens = self._prefill_bound()

    def _resolve_layout(self):
        candidates = [p for p in ('model.', 'model.language_model.', 'language_model.model.', 'language_model.')
                      if self.store.has(p + 'embed_tokens.weight')]
        if len(candidates) != 1:
            raise ValueError("Checkpoint must contain exactly one recognized text embedding prefix")
        self.prefix = candidates[0]
        self.embedding_name = self.prefix + 'embed_tokens.weight'
        heads = [p for p in ('lm_head.weight', 'language_model.lm_head.weight', 'model.language_model.lm_head.weight')
                 if self.store.has(p)]
        if self.config.tie_word_embeddings:
            self.head_name = self.embedding_name
        elif len(heads) == 1:
            self.head_name = heads[0]
        else:
            raise ValueError("Missing or ambiguous untied output head")
        self.expert_refs = {}
        self.shared_refs = {}
        self.validated_names = set()

    def _expect(self, name, shape):
        ref = WeightRef(name) if isinstance(name, str) else name
        if not self.store.has(ref.name):
            raise ValueError(f"Required tensor missing: {ref.name}")
        actual = self.store.shape(ref)
        if tuple(shape) != actual:
            raise ValueError(f"Checkpoint shape mismatch: {ref.name}: expected {shape}, got {actual}")
        self.validated_names.add(ref.name)
        info = self.store.tensors[ref.name]
        if info.dtype.startswith('F8'):
            if not self.config.quantization or ref.expert is not None or len(info.shape) != 2:
                raise ValueError("FP8 requires DeepSeek block-scaled, separately named 2-D tensors")
            scale = ref.name.removesuffix('.weight') + '.weight_scale_inv'
            expected = tuple(math.ceil(x / 128) for x in info.shape)
            if not self.store.has(scale) or self.store.shape(scale) != expected:
                raise ValueError(f"Missing or invalid scales for {ref.name}")
            self.validated_names.add(scale)

    def _projection(self, base, name, out_dim, in_dim, *, bias=False):
        self._expect(base + name + '.weight', (out_dim, in_dim))
        key = base + name + '.bias'
        if bias:
            self._expect(key, (out_dim,))
        elif self.store.has(key):
            raise ValueError(f"Unexpected projection bias: {key}")

    def _triplet(self, base, intermediate, *, expert=None):
        h = self.config.hidden_size
        if expert is not None:
            separate = f'{base}experts.{expert}.'
            if self.store.has(separate + 'gate_proj.weight'):
                return self._triplet(separate, intermediate)
            # HF packed exports: slice the expert axis before reading bytes.
            packed = base + 'experts.'
            gate_up = next((packed + k for k in ('gate_up_proj', 'gate_up_proj.weight') if self.store.has(packed + k)), None)
            down = next((packed + k for k in ('down_proj', 'down_proj.weight') if self.store.has(packed + k)), None)
            if gate_up and down:
                self._expect(gate_up, (self.config.num_experts, 2 * intermediate, h))
                self._expect(down, (self.config.num_experts, h, intermediate))
                return (WeightRef(gate_up, expert, 0, intermediate),
                        WeightRef(gate_up, expert, intermediate, 2 * intermediate), WeightRef(down, expert))
            raise ValueError(f"Unrecognized expert layout at {separate}")
        refs = (WeightRef(base + 'gate_proj.weight'), WeightRef(base + 'up_proj.weight'), WeightRef(base + 'down_proj.weight'))
        for ref, shape in zip(refs, ((intermediate, h), (intermediate, h), (h, intermediate))):
            self._expect(ref, shape)
        return refs

    def _validate_weights(self):
        c, h = self.config, self.config.hidden_size
        self._expect(self.embedding_name, (c.vocab_size, h))
        self._expect(self.head_name, (c.vocab_size, h))
        self._expect(self.prefix + 'norm.weight', (h,))
        for i, kind in enumerate(c.layer_types):
            base = self.prefix + f'layers.{i}.'
            for norm in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
                self._expect(base + norm, (h,))
            if kind == 'linear_attention':
                b = base + 'linear_attn.'
                kh, vh = c.linear_num_key_heads, c.linear_num_value_heads
                kd, vd = c.linear_key_head_dim, c.linear_value_head_dim
                conv_dim = 2 * kh * kd + vh * vd
                self._expect(b + 'conv1d.weight', (conv_dim, 1, c.linear_conv_kernel_dim))
                self._expect(b + 'A_log', (vh,))
                self._expect(b + 'dt_bias', (vh,))
                self._expect(b + 'norm.weight', (vd,))
                self._projection(b, 'out_proj', h, vh * vd)
                if c.family == 'qwen3_next':
                    self._projection(b, 'in_proj_qkvz', 2 * kh * kd + 2 * vh * vd, h)
                    self._projection(b, 'in_proj_ba', 2 * vh, h)
                else:
                    for name, dim in (('in_proj_qkv', conv_dim), ('in_proj_z', vh * vd), ('in_proj_b', vh), ('in_proj_a', vh)):
                        self._projection(b, name, dim, h)
            elif c.deepseek:
                b = base + 'self_attn.'
                heads, rank = c.num_attention_heads, c.kv_lora_rank
                qdim = c.qk_nope_head_dim + c.qk_rope_head_dim
                if c.get('q_lora_rank'):
                    self._projection(b, 'q_a_proj', c.q_lora_rank, h, bias=c.get('attention_bias', False))
                    self._expect(b + 'q_a_layernorm.weight', (c.q_lora_rank,))
                    self._projection(b, 'q_b_proj', heads * qdim, c.q_lora_rank)
                else:
                    self._projection(b, 'q_proj', heads * qdim, h)
                self._projection(b, 'kv_a_proj_with_mqa', rank + c.qk_rope_head_dim, h, bias=c.get('attention_bias', False))
                self._expect(b + 'kv_a_layernorm.weight', (rank,))
                self._projection(b, 'kv_b_proj', heads * (c.qk_nope_head_dim + c.v_head_dim), rank)
                self._projection(b, 'o_proj', h, heads * c.v_head_dim, bias=c.get('attention_bias', False))
            else:
                b = base + 'self_attn.'
                heads, kvh, dim = c.num_attention_heads, c.num_key_value_heads, c.head_dim
                gated = c.hybrid and c.get('attn_output_gate', True)
                for name, out_dim, in_dim in (('q_proj', heads * dim * (2 if gated else 1), h),
                                              ('k_proj', kvh * dim, h), ('v_proj', kvh * dim, h), ('o_proj', h, heads * dim)):
                    # Qwen2 has biases on Q/K/V only; Qwen3 uses attention_bias.
                    bias = name != 'o_proj' if c.family == 'qwen2' else c.get('attention_bias', False)
                    self._projection(b, name, out_dim, in_dim, bias=bias)
                if c.family not in ('qwen2', 'llama'):
                    for name in ('q_norm.weight', 'k_norm.weight'):
                        self._expect(b + name, (dim,))
            b = base + 'mlp.'
            if c.is_moe(i):
                self._projection(b, 'gate', c.num_experts, h)
                if c.get('topk_method') == 'noaux_tc':
                    self._expect(b + 'gate.e_score_correction_bias', (c.num_experts,))
                self.expert_refs[i] = [self._triplet(b, c.moe_intermediate_size, expert=e) for e in range(c.num_experts)]
                if c.deepseek and c.get('n_shared_experts'):
                    self.shared_refs[i] = self._triplet(b + 'shared_experts.', c.moe_intermediate_size * c.n_shared_experts)
                elif c.get('shared_expert_intermediate_size', 0):
                    self.shared_refs[i] = self._triplet(b + 'shared_expert.', c.shared_expert_intermediate_size)
                    self._projection(b, 'shared_expert_gate', 1, h)
            else:
                self.expert_refs[i] = self._triplet(b, c.intermediate_size)
        # Do not silently ignore unknown parameters in executed text layers.
        layer_prefix = self.prefix + 'layers.'
        for name in self.store.tensors:
            if name.startswith(layer_prefix):
                tail = name[len(layer_prefix):]
                idx = tail.split('.', 1)[0]
                # Some DeepSeek checkpoints append MTP layers. They are unused
                # for ordinary autoregressive decoding, explicitly not enabled.
                if not idx.isdigit():
                    raise ValueError(f"Invalid layer index in checkpoint: {name}")
                if int(idx) < c.num_hidden_layers:
                    if name not in self.validated_names:
                        raise ValueError(f"Unknown tensor in executed layer: {name}")
                elif not (c.deepseek and int(idx) < c.num_hidden_layers + c.get('num_nextn_predict_layers', 0)):
                    raise ValueError(f"Unexpected extra decoder layer: {name}")

    def _prefill_bound(self):
        c = self.config
        # Conservative per-token intermediate bound, including residuals,
        # projections, router and FP32 recurrent/attention arithmetic.
        width = max(c.hidden_size * 8, c.get('intermediate_size', 0) * 6,
                    c.num_attention_heads * c.head_dim * 8,
                    c.get('moe_intermediate_size', 0) * 6,
                    c.num_experts * 8)
        if c.hybrid:
            width = max(width, (2 * c.linear_num_key_heads * c.linear_key_head_dim +
                               2 * c.linear_num_value_heads * c.linear_value_head_dim) * 12)
            state_bytes = c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4
            self.guard.workspace(6 * state_bytes)
        if c.deepseek:
            width = max(width, c.num_attention_heads * (c.qk_nope_head_dim + c.qk_rope_head_dim + c.v_head_dim) * 8)
        max_tokens = self.budget.workspace_bytes // (width * 4)
        if max_tokens < 1:
            raise MemoryBudgetError("A single-token activation exceeds the workspace budget")
        return min(self.budget.prefill_tokens, self.budget.attention_block_tokens, max_tokens)

    def _linear(self, x, base, name, *, compute_dtype=None):
        bias = base + name + '.bias'
        return self.ops.linear(x, base + name + '.weight', bias if self.store.has(bias) else None, compute_dtype=compute_dtype)

    def _norm(self, x, name, *, zero=None):
        return rms_norm(x, self.ops.small(name), self.config.rms_norm_eps,
                        zero_centered=self.config.zero_centered_norm if zero is None else zero)

    def _kv_blocks(self, state, i, length, *, start=0):
        # KV chunks are read from disk and discarded between blocks. There is no
        # full-layer cache reconstruction or GPU tail after long generation.
        step = self.budget.attention_block_tokens
        for lo in range(start, length, step):
            hi = min(length, lo + step)
            self.guard.check((hi - lo) * self.config.num_key_value_heads * self.config.head_dim * 16)
            k = state.read((i, 'k'), lo, hi, device=self.device)
            v = state.read((i, 'v'), lo, hi, device=self.device)
            yield lo, k, v
            del k, v

    def _attention(self, x, i, state, position):
        c = self.config
        base = self.prefix + f'layers.{i}.self_attn.'
        n, heads, kvh, dim = len(x), c.num_attention_heads, c.num_key_value_heads, c.head_dim
        q = self._linear(x, base, 'q_proj')
        gate = None
        if c.hybrid and c.get('attn_output_gate', True):
            q, gate = q.reshape(n, heads, 2 * dim).chunk(2, dim=-1)
        else:
            q = q.reshape(n, heads, dim)
        k = self._linear(x, base, 'k_proj').reshape(n, kvh, dim)
        v = self._linear(x, base, 'v_proj').reshape(n, kvh, dim)
        if c.family not in ('qwen2', 'llama'):
            q, k = self._norm(q, base + 'q_norm.weight'), self._norm(k, base + 'k_norm.weight')
        positions = torch.arange(position, position + n, device=x.device)
        rotary_dim = int(dim * c.rope['partial_rotary_factor'])
        q = rotary(q, positions, rotary_dim, c.rope)
        k = rotary(k, positions, rotary_dim, c.rope)
        state.append((i, 'k'), k, position=position)
        state.append((i, 'v'), v, position=position)
        del k, v
        window = c.get('sliding_window') if c.layer_types[i] == 'sliding_attention' else None
        start = max(0, position - window + 1) if window else 0
        out = online_attention(q, self._kv_blocks(state, i, position + n, start=start), positions,
                               scale=dim**-0.5, value_dim=dim, guard=self.guard, window=window)
        if gate is not None:
            gate_type = c.get('output_gate_type', c.get('attn_output_gate_type', 'sigmoid'))
            out = out * (F.silu(gate) if gate_type in ('silu', 'swish') else gate.sigmoid())
        return self._linear(out.reshape(n, heads * dim), base, 'o_proj')

    def _delta(self, x, i, state):
        c = self.config
        base = self.prefix + f'layers.{i}.linear_attn.'
        n, kh, vh, kd, vd = len(x), c.linear_num_key_heads, c.linear_num_value_heads, c.linear_key_head_dim, c.linear_value_head_dim
        ratio = vh // kh
        if c.family == 'qwen3_next':
            packed = self._linear(x, base, 'in_proj_qkvz').reshape(n, kh, 2 * kd + 2 * ratio * vd)
            q, k, v, z = packed.split((kd, kd, ratio * vd, ratio * vd), dim=-1)
            ba = self._linear(x, base, 'in_proj_ba').reshape(n, kh, 2 * ratio)
            b, a = (y.reshape(n, vh) for y in ba.chunk(2, dim=-1))
            mixed = torch.cat((q.reshape(n, -1), k.reshape(n, -1), v.reshape(n, -1)), -1)
            z = z.reshape(n, vh, vd)
        else:
            mixed = self._linear(x, base, 'in_proj_qkv')
            z = self._linear(x, base, 'in_proj_z').reshape(n, vh, vd)
            b, a = self._linear(x, base, 'in_proj_b'), self._linear(x, base, 'in_proj_a')
        kernel = self.ops.small(base + 'conv1d.weight')
        history = state.state((i, 'conv'), device=x.device)
        if history is None:
            history = torch.zeros((mixed.shape[1], c.linear_conv_kernel_dim - 1), dtype=x.dtype, device=x.device)
        # History contains pre-convolution inputs, not activated values.
        inputs = torch.cat((history, mixed.T), -1)
        mixed = F.silu(F.conv1d(inputs.unsqueeze(0), kernel, groups=mixed.shape[1]))[0].T
        state.save_state((i, 'conv'), inputs[:, -(c.linear_conv_kernel_dim - 1):] if c.linear_conv_kernel_dim > 1 else inputs[:, :0])
        q, k, v = mixed.split((kh * kd, kh * kd, vh * vd), -1)
        q = q.reshape(n, kh, kd).repeat_interleave(ratio, dim=1)
        k = k.reshape(n, kh, kd).repeat_interleave(ratio, dim=1)
        v = v.reshape(n, vh, vd)
        log_decay = -self.ops.small(base + 'A_log', dtype=torch.float32).exp() * F.softplus(a.float() + self.ops.small(base + 'dt_bias', dtype=torch.float32))
        previous = state.state((i, 'recurrent'), device=x.device)
        out, recurrent = recurrent_delta(q, k, v, log_decay, b.sigmoid(), previous)
        state.save_state((i, 'recurrent'), recurrent)
        # Gated RMSNorm is standard (not zero-centered), evaluated in FP32.
        out32 = out.float() * torch.rsqrt(out.float().square().mean(-1, keepdim=True) + c.rms_norm_eps)
        out32 = out32 * self.ops.small(base + 'norm.weight', dtype=torch.float32) * F.silu(z.float())
        return self._linear(out32.to(x.dtype).reshape(n, vh * vd), base, 'out_proj')

    def _mla(self, x, i, state, position):
        c = self.config
        base = self.prefix + f'layers.{i}.self_attn.'
        n, heads, rank = len(x), c.num_attention_heads, c.kv_lora_rank
        nope, rope_dim, vd = c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim
        if c.get('q_lora_rank'):
            q = self._linear(self._norm(self._linear(x, base, 'q_a_proj'), base + 'q_a_layernorm.weight', zero=False), base, 'q_b_proj')
        else:
            q = self._linear(x, base, 'q_proj')
        q = q.reshape(n, heads, nope + rope_dim)
        q_nope, q_rope = q.split((nope, rope_dim), -1)
        compressed = self._linear(x, base, 'kv_a_proj_with_mqa')
        latent, k_rope = compressed.split((rank, rope_dim), -1)
        latent = self._norm(latent, base + 'kv_a_layernorm.weight', zero=False)
        positions = torch.arange(position, position + n, device=x.device)
        q_rope = rotary(q_rope, positions, rope_dim, c.rope, interleaved=True, deepseek=True)
        k_rope = rotary(k_rope[:, None], positions, rope_dim, c.rope, interleaved=True, deepseek=True)[:, 0]
        state.append((i, 'latent'), latent, position=position)
        state.append((i, 'rope'), k_rope, position=position)
        scale = (nope + rope_dim)**-0.5 * yarn_scale(c.rope.get('factor', 1), c.rope.get('mscale_all_dim', 0))**2
        result = torch.empty((n, heads, vd), dtype=x.dtype, device=x.device)
        # Process heads independently: never expand the cache to head-count K/V.
        for h in range(heads):
            self.guard.check((nope + vd) * rank * 16)
            w = self.store.rows(base + 'kv_b_proj.weight', h * (nope + vd), (h + 1) * (nope + vd), dtype=x.dtype, device=x.device)
            query = torch.cat((q_nope[:, h] @ w[:nope], q_rope[:, h]), -1)[:, None]
            def blocks():
                for lo in range(0, position + n, self.budget.attention_block_tokens):
                    hi = min(position + n, lo + self.budget.attention_block_tokens)
                    self.guard.check((hi - lo) * (rank * 2 + rope_dim) * 16)
                    lat = state.read((i, 'latent'), lo, hi, device=x.device)
                    pe = state.read((i, 'rope'), lo, hi, device=x.device)
                    yield lo, torch.cat((lat, pe), -1)[:, None], lat[:, None]
            attended = online_attention(query, blocks(), positions, scale=scale, value_dim=rank, guard=self.guard)
            result[:, h] = attended[:, 0] @ w[nope:].T
            del w, attended
        return self._linear(result.reshape(n, heads * vd), base, 'o_proj')

    def _mlp(self, x, refs):
        intermediate = self.store.shape(refs[0])[0]
        chunk = max(1, min(len(x), self.budget.workspace_bytes // (max(intermediate, x.shape[-1]) * 32)))
        self.guard.workspace(x.numel() * x.element_size())
        result = torch.empty_like(x)
        for lo in range(0, len(x), chunk):
            z = x[lo:lo + chunk]
            gate = self.ops.linear(z, refs[0])
            up = self.ops.linear(z, refs[1])
            gate = F.silu(gate) * up
            del up
            result[lo:lo + chunk] = self.ops.linear(gate, refs[2])
        return result

    def _moe(self, x, i):
        c = self.config
        base = self.prefix + f'layers.{i}.mlp.'
        # Config-bounded prefill means router logits are only [chunk,E].
        logits = self._linear(x, base, 'gate', compute_dtype=torch.float32 if c.deepseek else None)
        correction_name = base + 'gate.e_score_correction_bias'
        correction = self.ops.small(correction_name, dtype=torch.float32) if self.store.has(correction_name) else None
        routing = dict(c.data)
        routing.setdefault('norm_topk_prob', True)
        indices, weights = route_experts(logits, routing, correction)
        # DeepSeek combines expert outputs in FP32; Qwen uses input dtype.
        accumulation_dtype = torch.float32 if c.deepseek else x.dtype
        result = torch.zeros(x.shape, dtype=accumulation_dtype, device=x.device)
        for e in torch.unique(indices).tolist():
            token, slot = torch.where(indices == e)
            out = self._mlp(x[token], self.expert_refs[i][e])
            contribution = out.to(accumulation_dtype) * weights[token, slot, None].to(accumulation_dtype)
            result.index_add_(0, token, contribution)
            del out, contribution
        result = result.to(x.dtype)
        if i in self.shared_refs:
            shared = self._mlp(x, self.shared_refs[i])
            if not c.deepseek:
                shared = shared * self._linear(x, base, 'shared_expert_gate').sigmoid()
            result = result + shared
        return result

    @torch.inference_mode()
    def forward_tokens(self, tokens, state, position):
        if position != state.position:
            raise ValueError("Cache session position does not match the input prefix")
        if not tokens or len(tokens) > self.prefill_tokens:
            raise ValueError("forward_tokens requires a nonempty bounded chunk")
        if position < 0 or position + len(tokens) > min(self.config.max_position_embeddings, self.budget.max_context_tokens):
            raise ValueError("Context limit exceeded")
        x = self.ops.embedding(tokens, self.embedding_name)
        for i, kind in enumerate(self.config.layer_types):
            self.guard.check()
            base = self.prefix + f'layers.{i}.'
            normalized = self._norm(x, base + 'input_layernorm.weight')
            if kind == 'linear_attention':
                update = self._delta(normalized, i, state)
            elif self.config.deepseek:
                update = self._mla(normalized, i, state, position)
            else:
                update = self._attention(normalized, i, state, position)
            x = x + update
            normalized = self._norm(x, base + 'post_attention_layernorm.weight')
            update = self._moe(normalized, i) if self.config.is_moe(i) else self._mlp(normalized, self.expert_refs[i])
            x = x + update
        state.position = position + len(tokens)
        return self._norm(x[-1:], self.prefix + 'norm.weight')

    def logits(self, hidden):
        # Only the final required token is projected, in bounded vocabulary rows.
        if hidden.shape != (1, self.config.hidden_size):
            raise ValueError("Only last-token logits are supported")
        return self.ops.linear(hidden, self.head_name).float()[0]

    @torch.inference_mode()
    def generate(self, input_ids, *, attention_mask=None, max_new_tokens=128, eos_token_id=None,
                 pad_token_id=None, do_sample=False, temperature=1.0, top_p=1.0, top_k=None,
                 repetition_penalty=1.0, use_cache=True, num_beams=1,
                 return_dict_in_generate=False, past_key_values=None, **unsupported):
        if unsupported:
            raise ValueError('Unsupported generation options: ' + ', '.join(sorted(unsupported)))
        if num_beams != 1 or not use_cache or return_dict_in_generate:
            raise ValueError("Only batch-one cached autoregressive generation is supported")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a [1,T] int64 tensor")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not torch.all(attention_mask == 1)):
            raise ValueError("Padding, packed sequences and custom masks are not supported")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.budget.max_output_tokens:
            raise ValueError("Requested output exceeds profile max_output_tokens")
        if not all(math.isfinite(float(v)) for v in (temperature, top_p, repetition_penalty)):
            raise ValueError("Non-finite sampling options")
        if temperature <= 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("Invalid sampling options")
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("top_k must be nonnegative")
        tokens = input_ids[0].detach().cpu().tolist()
        media = {self.config.raw.get(k) for k in ('image_token_id', 'video_token_id', 'vision_start_token_id', 'vision_end_token_id')}
        media.discard(None)
        if media.intersection(tokens):
            raise ValueError("Multimodal placeholder tokens are not valid in the text-only bounded backend")
        if not tokens or len(tokens) + max_new_tokens > min(self.config.max_position_embeddings, self.budget.max_context_tokens):
            raise ValueError("Prompt plus output exceeds the configured context profile")
        stops = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
        if any(type(v) is not int or not 0 <= v < self.config.vocab_size for v in stops):
            raise ValueError("Invalid EOS token")
        root = self.cache_root
        if past_key_values is not None:
            if not isinstance(past_key_values, CacheLocation):
                raise ValueError("Cross-turn cache reuse is deliberately not implemented")
            root = past_key_values.root
        with GENERATION_LOCK, DiskState(root, max_bytes=self.budget.max_cache_bytes,
                                       max_block_tokens=self.budget.attention_block_tokens) as state:
            start = time.monotonic()
            initial = len(tokens)
            starting_io = self.store.bytes_read
            self.guard.check()
            try:
                for position in range(0, initial, self.prefill_tokens):
                    hidden = self.forward_tokens(tokens[position:position + self.prefill_tokens], state, position)
                prefill_seconds = time.monotonic() - start
                for step in range(max_new_tokens):
                    logits = self.logits(hidden)
                    if not torch.isfinite(logits).all():
                        raise RuntimeError("Non-finite logits; aborting rather than producing corrupt output")
                    if repetition_penalty != 1:
                        seen = torch.tensor(sorted(set(tokens)), device=logits.device)
                        scores = logits[seen]
                        logits[seen] = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                    if do_sample:
                        logits /= temperature
                        if top_k:
                            threshold = logits.topk(min(top_k, logits.numel())).values[-1]
                            logits[logits < threshold] = -torch.inf
                        if top_p < 1:
                            ordered, order = logits.sort(descending=True)
                            cumulative = ordered.softmax(-1).cumsum(-1)
                            remove = cumulative > top_p
                            remove[1:] = remove[:-1].clone()
                            remove[0] = False
                            logits[order[remove]] = -torch.inf
                        token = torch.multinomial(logits.softmax(-1), 1).item()
                    else:
                        token = logits.argmax().item()
                    tokens.append(token)
                    if token in stops or step + 1 == max_new_tokens:
                        break
                    hidden = self.forward_tokens([token], state, len(tokens) - 1)
                self.guard.check()
                self.last_report = dict(self.guard.report(), prompt_tokens=initial,
                                        generated_tokens=len(tokens) - initial, prefill_seconds=prefill_seconds,
                                        cache_peak_bytes=state.peak_bytes, cache_bytes_read=state.bytes_read,
                                        checkpoint_bytes_read=self.store.bytes_read - starting_io,
                                        max_checkpoint_read_bytes=self.store.max_read_observed,
                                        prefill_chunk_tokens=self.prefill_tokens, memory_qualified=False)
                return torch.tensor([tokens], dtype=torch.long, device=input_ids.device)
            except torch.cuda.OutOfMemoryError as exc:
                raise MemoryBudgetError("CUDA OOM inside bounded execution; this profile is NOT qualified") from exc

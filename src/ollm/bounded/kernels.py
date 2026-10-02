"""Bounded reference PyTorch kernels; no T-by-T attention matrix.

Equations follow the primary implementations listed in docs/bounded_sources.md.
No optional Flash/DeltaNet kernel is silently substituted. Prefill is deliberately
conservative and sequential; this is a low-memory path, not a throughput engine.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from .checkpoint import WeightRef
from .budget import MemoryBudgetError


def rms_norm(x, weight, eps, *, zero_centered=False):
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    if zero_centered:
        return (y * (1 + weight.float())).to(x.dtype)
    return y.to(x.dtype) * weight.to(x.dtype)


def yarn_scale(factor, multiplier=1.0):
    return 1.0 if factor <= 1 else 1 + 0.1 * multiplier * math.log(factor)


def rope_parameters(dim, params, device, *, deepseek=False):
    if dim < 2 or dim % 2:
        raise ValueError("Rotary dimension must be positive and even")
    base = float(params.get('rope_theta', 10000))
    if not math.isfinite(base) or base <= 1:
        raise ValueError("Invalid RoPE theta")
    inv = base ** (-torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
    kind = params.get('rope_type', 'default')
    factor = float(params.get('factor', 1.0))
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Invalid RoPE factor")
    magnitude = 1.0
    if kind == 'linear':
        inv = inv / factor
    elif kind == 'yarn':
        original = params.get('original_max_position_embeddings', 32768)
        def correction(rotations):
            return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))
        low = max(0, math.floor(correction(params.get('beta_fast', 32))))
        high = min(dim - 1, math.ceil(correction(params.get('beta_slow', 1))))
        if high == low:
            high += 0.001
        ramp = ((torch.arange(dim // 2, device=device) - low) / (high - low)).clamp(0, 1)
        inv = inv * (1 - ramp) + inv / factor * ramp
        if deepseek:
            magnitude = yarn_scale(factor, params.get('mscale', 1)) / yarn_scale(factor, params.get('mscale_all_dim', 0))
        else:
            magnitude = params.get('attention_factor', yarn_scale(factor))
    elif kind == 'llama3':
        low, high = params['low_freq_factor'], params['high_freq_factor']
        original = params['original_max_position_embeddings']
        wave = 2 * math.pi / inv
        smooth = (original / wave - low) / (high - low)
        interpolated = (1 - smooth) * inv / factor + smooth * inv
        inv = torch.where(wave > original / low, inv / factor,
                          torch.where(wave < original / high, inv, interpolated))
    elif kind != 'default':
        raise ValueError(f"Unknown RoPE {kind}")
    return inv, magnitude


def rotary(x, positions, dim, params, *, interleaved=False, deepseek=False):
    # x: [tokens, heads, dim]; all three multimodal coordinates are equal for
    # text-only Qwen3.5, so its interleaved MRoPE reduces to ordinary text RoPE.
    inv, magnitude = rope_parameters(dim, params, x.device, deepseek=deepseek)
    freq = positions.float()[:, None] * inv[None, :]
    cos, sin = (freq.cos() * magnitude).to(x.dtype)[:, None], (freq.sin() * magnitude).to(x.dtype)[:, None]
    a = x[..., :dim]
    if interleaved:
        even, odd = a[..., 0::2], a[..., 1::2]
        rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    else:
        first, second = a.chunk(2, dim=-1)
        rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, x[..., dim:]), dim=-1)


class StreamOps:
    def __init__(self, store, guard, dtype):
        self.store, self.guard, self.dtype = store, guard, dtype
        self.device = guard.device

    def small(self, name, *, dtype=None):
        size = self.store.tensors[name].nbytes
        self.guard.workspace(size * 4)
        return self.store.small(name, dtype=dtype or self.dtype, device=self.device)

    def linear(self, x, weight, bias=None, *, compute_dtype=None):
        weight = WeightRef(weight) if isinstance(weight, str) else weight
        shape = self.store.shape(weight)
        if len(shape) != 2 or shape[1] != x.shape[-1]:
            raise ValueError(f"Linear shape mismatch for {weight.name}: {shape} vs {tuple(x.shape)}")
        dtype = compute_dtype or x.dtype
        tokens = math.prod(x.shape[:-1])
        element = torch.empty((), dtype=dtype).element_size()
        out_bytes = tokens * shape[0] * element
        self.guard.workspace(out_bytes + x.numel() * element)
        # Conversion may use fp32 even for fp8 weights. Bound it, not just the
        # compressed bytes on disk. At least one complete input row must fit.
        row_bytes = shape[1] * max(4, element)
        if row_bytes > self.guard.budget.weight_tile_bytes:
            raise MemoryBudgetError("One weight row exceeds the configured tile size")
        rows = max(1, self.guard.budget.weight_tile_bytes // row_bytes)
        rows = min(rows, max(1, self.store.max_read_bytes // row_bytes))
        self.guard.check(out_bytes + 4 * rows * row_bytes)
        out = torch.empty((*x.shape[:-1], shape[0]), dtype=dtype, device=x.device)
        source = x.to(dtype)
        for lo in range(0, shape[0], rows):
            hi = min(lo + rows, shape[0])
            self.guard.check(4 * (hi - lo) * row_bytes)
            w = self.store.rows(weight, lo, hi, dtype=dtype, device=x.device)
            b = None if bias is None else self.store.rows(bias, lo, hi, dtype=dtype, device=x.device)
            out[..., lo:hi] = F.linear(source, w, b)
            del w, b
        return out

    def embedding(self, ids, name):
        shape = self.store.shape(name)
        if len(shape) != 2 or any(not 0 <= n < shape[0] for n in ids):
            raise ValueError("Invalid token ID or embedding shape")
        self.guard.workspace(len(ids) * shape[1] * 4)
        result = torch.empty((len(ids), shape[1]), device=self.device, dtype=self.dtype)
        for i, token in enumerate(ids):
            self.guard.check(shape[1] * 16)
            result[i] = self.store.rows(name, token, token + 1, dtype=self.dtype, device=self.device)[0]
        return result


def online_attention(query, blocks, positions, *, scale, value_dim, guard, window=None):
    """Stable exact causal attention with bounded KV tiles and grouped heads.

    query [T,H,D]; blocks yields (absolute_start, K[S,KVH,D], V[S,KVH,V]).
    Accumulation is FP32; no masked softmax row can create NaNs.
    """
    t, heads, _ = query.shape
    guard.workspace(heads * t * (value_dim + 2) * 4 + query.numel() * 4)
    maxima = torch.full((heads, t), -torch.inf, dtype=torch.float32, device=query.device)
    denom = torch.zeros((heads, t), dtype=torch.float32, device=query.device)
    accum = torch.zeros((heads, t, value_dim), dtype=torch.float32, device=query.device)
    for offset, keys, values in blocks:
        if keys.shape[0] != values.shape[0] or heads % keys.shape[1]:
            raise ValueError("Invalid grouped attention cache")
        end = offset + keys.shape[0]
        key_pos = torch.arange(offset, end, device=query.device)
        mask = key_pos[None, :] <= positions[:, None]
        if window is not None:
            mask &= key_pos[None, :] > positions[:, None] - window
        guard.workspace((keys.numel() + values.numel()) * 4 + t * keys.shape[0] * 16)
        ratio = heads // keys.shape[1]
        for h in range(heads):
            logits = (query[:, h].float() @ keys[:, h // ratio].float().T) * scale
            logits.masked_fill_(~mask, -torch.inf)
            top = torch.maximum(maxima[h], logits.max(-1).values)
            safe_top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
            old = torch.exp(maxima[h] - safe_top)
            probs = torch.exp(logits - safe_top[:, None])
            accum[h].mul_(old[:, None]).add_(probs @ values[:, h // ratio].float())
            denom[h].mul_(old).add_(probs.sum(-1))
            maxima[h] = top
    if not torch.all(denom > 0):
        raise ValueError("Attention has a query without any visible keys")
    return (accum / denom[..., None]).transpose(0, 1).to(query.dtype)


def recurrent_delta(query, key, value, log_decay, beta, state=None):
    """Gated DeltaNet scan, including FP32 recurrent state and Q/K L2 norm."""
    q, k, v = query.float(), key.float(), value.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q *= q.shape[-1] ** -0.5
    if state is None:
        state = torch.zeros((q.shape[1], q.shape[2], v.shape[2]), dtype=torch.float32, device=q.device)
    else:
        state = state.float()
    out = torch.empty_like(v)
    for i in range(q.shape[0]):
        state = state * log_decay[i].float().exp()[:, None, None]
        correction = (v[i] - (state * k[i, :, :, None]).sum(-2)) * beta[i].float()[:, None]
        state = state + k[i, :, :, None] * correction[:, None, :]
        out[i] = (state * q[i, :, :, None]).sum(-2)
    return out.to(value.dtype), state


def route_experts(logits, config, correction_bias=None):
    """Bounded [chunk,experts] routing, never [tokens,topk,experts] one-hot."""
    scores = logits.float().sigmoid() if config.get('scoring_func', 'softmax') == 'sigmoid' else logits.float().softmax(-1)
    choice = scores if correction_bias is None else scores + correction_bias.float()
    method = config.get('topk_method', 'greedy')
    if method in ('group_limited_greedy', 'noaux_tc'):
        groups = config.get('n_group', 1)
        grouped = choice.reshape(choice.shape[0], groups, -1)
        group_score = (grouped.topk(2, dim=-1).values.sum(-1) if method == 'noaux_tc'
                       else grouped.max(-1).values)
        selected = group_score.topk(config.get('topk_group', 1), dim=-1, sorted=False).indices
        active = torch.zeros_like(group_score, dtype=torch.bool).scatter_(1, selected, True)
        mask = active[:, :, None].expand_as(grouped).reshape_as(choice)
        # V2 group-limited routing masks to zero; V3 noaux_tc uses -inf.
        choice = choice.masked_fill(~mask, -torch.inf if method == "noaux_tc" else 0)
    indices = choice.topk(config['num_experts_per_tok'], dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    if config.get('norm_topk_prob', True) and (config['num_experts_per_tok'] > 1 or not config.get('model_type', '').startswith('deepseek')):
        weights = weights / weights.sum(-1, keepdim=True)
    weights *= config.get('routed_scaling_factor', 1.0)
    return indices, weights

"""Memory-bounded Qwen3-VL image inference.

The implementation follows the Qwen3-VL 4.57 architecture while retaining
only bounded weight tiles and explicitly-sized activation workspaces on CUDA.
Images are supported; video is deliberately rejected until it has a separate
hardware qualification profile.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .budget import GENERATION_LOCK, MemoryBudgetError
from .cache import DiskState
from .model import BoundedModel
from .kernels import online_attention, rope_parameters


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def mrope(x, position_ids, params):
    """Qwen3-VL interleaved 3-axis MRoPE for x[T,H,D]."""
    if position_ids.shape != (3, x.shape[0]):
        raise ValueError("MRoPE positions must have shape [3,tokens]")
    dim = x.shape[-1]
    inv, magnitude = rope_parameters(dim, params, x.device)
    freqs = position_ids.to(device=x.device, dtype=torch.float32)[:, :, None] * inv[None, None, :]
    sections = params.get('mrope_section')
    if not isinstance(sections, list) or len(sections) != 3 or sum(sections) * 2 != dim:
        raise ValueError("Invalid Qwen3-VL MRoPE sections")
    mixed = freqs[0].clone()
    for axis, offset in ((1, 1), (2, 2)):
        mixed[:, offset:sections[axis] * 3:3] = freqs[axis, :, offset:sections[axis] * 3:3]
    emb = torch.cat((mixed, mixed), dim=-1)
    cos = (emb.cos() * magnitude).to(x.dtype)[:, None]
    sin = (emb.sin() * magnitude).to(x.dtype)[:, None]
    return x * cos + _rotate_half(x) * sin


def _layer_norm(x, weight, bias, eps=1e-6):
    return F.layer_norm(x, (x.shape[-1],), weight.to(x.dtype), bias.to(x.dtype), eps)


class Qwen3VLBoundedModel(BoundedModel):
    """Qwen3-VL text decoder + streamed ViT/DeepStack image path."""

    def __init__(self, *args, **kwargs):
        self._active_mrope = None
        self._rope_delta = None
        super().__init__(*args, **kwargs)
        unit = self.vision['spatial_merge_size'] ** 2
        if self.budget.max_visual_tokens % unit:
            raise ValueError("max_visual_tokens must be divisible by the Qwen3-VL spatial merge unit")
        if self.budget.max_images * unit > self.budget.max_visual_tokens:
            raise ValueError("max_visual_tokens is too small to exercise max_images")

    @property
    def vision(self):
        return self.config.raw['vision_config']

    def _validate_weights(self):
        super()._validate_weights()
        c, v = self.config, self.config.raw['vision_config']
        if c.family != 'qwen3_vl':
            raise ValueError("Qwen3VLBoundedModel requires a qwen3_vl checkpoint")
        candidates = [p for p in ('model.visual.', 'visual.') if self.store.has(p + 'patch_embed.proj.weight')]
        if len(candidates) != 1:
            raise ValueError("Checkpoint must contain exactly one recognized Qwen3-VL vision prefix")
        self.vision_prefix = candidates[0]
        p, h, inter = self.vision_prefix, v['hidden_size'], v['intermediate_size']
        patch_shape = (h, v['in_channels'], v['temporal_patch_size'], v['patch_size'], v['patch_size'])
        self._expect(p + 'patch_embed.proj.weight', patch_shape)
        self._expect(p + 'patch_embed.proj.bias', (h,))
        self._expect(p + 'pos_embed.weight', (v['num_position_embeddings'], h))
        for i in range(v['depth']):
            b = p + f'blocks.{i}.'
            for norm in ('norm1', 'norm2'):
                self._expect(b + norm + '.weight', (h,))
                self._expect(b + norm + '.bias', (h,))
            self._expect(b + 'attn.qkv.weight', (3 * h, h))
            self._expect(b + 'attn.qkv.bias', (3 * h,))
            self._expect(b + 'attn.proj.weight', (h, h))
            self._expect(b + 'attn.proj.bias', (h,))
            self._expect(b + 'mlp.linear_fc1.weight', (inter, h))
            self._expect(b + 'mlp.linear_fc1.bias', (inter,))
            self._expect(b + 'mlp.linear_fc2.weight', (h, inter))
            self._expect(b + 'mlp.linear_fc2.bias', (h,))
        group = h * v['spatial_merge_size'] ** 2
        self._validate_merger(p + 'merger.', group, postshuffle=False)
        for j, _ in enumerate(v['deepstack_visual_indexes']):
            self._validate_merger(p + f'deepstack_merger_list.{j}.', group, postshuffle=True)
        # Vision is executed, so do not silently tolerate unknown vision weights.
        for name in self.store.tensors:
            if name.startswith(p) and name not in self.validated_names:
                raise ValueError(f"Unknown tensor in executed vision tower: {name}")

    def _validate_merger(self, base, group, *, postshuffle):
        v = self.vision
        norm_width = group if postshuffle else v['hidden_size']
        self._expect(base + 'norm.weight', (norm_width,))
        self._expect(base + 'norm.bias', (norm_width,))
        self._expect(base + 'linear_fc1.weight', (group, group))
        self._expect(base + 'linear_fc1.bias', (group,))
        self._expect(base + 'linear_fc2.weight', (v['out_hidden_size'], group))
        self._expect(base + 'linear_fc2.bias', (v['out_hidden_size'],))

    def _rotary_qk(self, q, k, position, n):
        causal = torch.arange(position, position + n, device=q.device)
        if self._active_mrope is None:
            positions = causal[None].expand(3, -1)
        else:
            positions = self._active_mrope.to(q.device)
            if positions.shape != (3, n):
                raise ValueError("Active multimodal positions do not match decoder chunk")
        return mrope(q, positions, self.config.rope), mrope(k, positions, self.config.rope), causal

    def forward_tokens(self, tokens, state, position):
        if self._rope_delta is None:
            return super().forward_tokens(tokens, state, position)
        p = torch.arange(position, position + len(tokens), device=self.device) + int(self._rope_delta)
        self._active_mrope = p[None].expand(3, -1)
        try:
            return super().forward_tokens(tokens, state, position)
        finally:
            self._active_mrope = None

    # ------------------------------- vision -------------------------------
    def _vision_linear_flat(self, x, weight_name, bias_name):
        """Linear over a Conv3d kernel flattened per output channel, row tiled."""
        shape = self.store.shape(weight_name)
        out_dim, in_dim = shape[0], math.prod(shape[1:])
        if x.ndim != 2 or x.shape[1] != in_dim:
            raise ValueError("Vision patch tensor width does not match Conv3d kernel")
        element = torch.empty((), dtype=self.dtype).element_size()
        resident = (x.numel() + x.shape[0] * out_dim) * element
        self.guard.workspace(resident + min(self.budget.weight_tile_bytes, out_dim * in_dim * element))
        source = x.to(device=self.device, dtype=self.dtype)
        out = torch.empty((len(x), out_dim), dtype=self.dtype, device=self.device)
        row_bytes = in_dim * max(4, element)
        rows = max(1, min(out_dim, self.budget.weight_tile_bytes // row_bytes))
        for lo in range(0, out_dim, rows):
            hi = min(out_dim, lo + rows)
            w = self.store.rows(weight_name, lo, hi, dtype=self.dtype, device=self.device).reshape(hi - lo, in_dim)
            b = self.store.rows(bias_name, lo, hi, dtype=self.dtype, device=self.device)
            out[:, lo:hi] = F.linear(source, w, b)
            del w, b
        return out

    def _vision_position_embeddings(self, grid):
        v = self.vision
        side = int(math.isqrt(v['num_position_embeddings']))
        table = self.ops.small(self.vision_prefix + 'pos_embed.weight')
        merge = v['spatial_merge_size']
        pieces = []
        for t, h, w in grid.tolist():
            h_idx = torch.linspace(0, side - 1, h, device=self.device)
            w_idx = torch.linspace(0, side - 1, w, device=self.device)
            hf, wf = h_idx.long(), w_idx.long()
            hc, wc = (hf + 1).clamp(max=side - 1), (wf + 1).clamp(max=side - 1)
            dh, dw = h_idx - hf, w_idx - wf
            result = torch.zeros((h * w, v['hidden_size']), dtype=self.dtype, device=self.device)
            terms = (
                (hf[:, None] * side + wf[None, :], (1 - dh)[:, None] * (1 - dw)[None, :]),
                (hf[:, None] * side + wc[None, :], (1 - dh)[:, None] * dw[None, :]),
                (hc[:, None] * side + wf[None, :], dh[:, None] * (1 - dw)[None, :]),
                (hc[:, None] * side + wc[None, :], dh[:, None] * dw[None, :]),
            )
            for idx, weight in terms:
                result.add_(table[idx.flatten()] * weight.flatten()[:, None].to(self.dtype))
            result = result.repeat(t, 1)
            result = (result.view(t, h // merge, merge, w // merge, merge, -1)
                      .permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            pieces.append(result)
        return torch.cat(pieces, dim=0)

    def _vision_position_cos_sin(self, grid):
        v = self.vision
        merge = v['spatial_merge_size']
        head_dim = v['hidden_size'] // v['num_heads']
        rope_dim = head_dim // 2
        inv = 10000.0 ** (-torch.arange(0, rope_dim, 2, device=self.device, dtype=torch.float32) / rope_dim)
        pieces = []
        for t, h, w in grid.tolist():
            mh, mw = h // merge, w // merge
            br = torch.arange(mh, device=self.device)[:, None, None, None] * merge
            bc = torch.arange(mw, device=self.device)[None, :, None, None] * merge
            ir = torch.arange(merge, device=self.device)[None, None, :, None]
            ic = torch.arange(merge, device=self.device)[None, None, None, :]
            row = (br + ir).expand(mh, mw, merge, merge).reshape(-1)
            col = (bc + ic).expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack((row, col), -1).repeat(t, 1)
            freq = coords.float()[:, :, None] * inv[None, None, :]
            freq = freq.flatten(1)
            emb = torch.cat((freq, freq), -1)
            pieces.append(emb)
        emb = torch.cat(pieces, 0)
        return emb.cos(), emb.sin()

    def _vision_segments(self, grid):
        out, start = [], 0
        for t, h, w in grid.tolist():
            for _ in range(t):
                end = start + h * w
                out.append((start, end))
                start = end
        return out

    def _vision_norm(self, x, base):
        weight = self.ops.small(base + '.weight')
        bias = self.ops.small(base + '.bias')
        return _layer_norm(x, weight, bias)

    def _vision_attention(self, x, layer, cos, sin, segments):
        v = self.vision
        base = self.vision_prefix + f'blocks.{layer}.attn.'
        hidden, heads = v['hidden_size'], v['num_heads']
        dim = hidden // heads
        qkv = self.ops.linear(x, base + 'qkv.weight', base + 'qkv.bias')
        q, k, val = qkv.reshape(len(x), 3, heads, dim).unbind(1)
        c, s = cos[:, None].float(), sin[:, None].float()
        qf, kf = q.float(), k.float()
        q = (qf * c + _rotate_half(qf) * s).to(x.dtype)
        k = (kf * c + _rotate_half(kf) * s).to(x.dtype)
        del qkv, qf, kf
        base_bytes = (q.numel() + k.numel() + val.numel() + x.numel()) * x.element_size()
        block = self.budget.attention_block_tokens
        self.guard.workspace(base_bytes + heads * block * (dim + block + 4) * 4)
        attended = torch.empty_like(q)
        for start, end in segments:
            for qlo in range(start, end, block):
                qhi = min(end, qlo + block)
                query = q[qlo:qhi]
                def blocks():
                    for klo in range(start, end, block):
                        khi = min(end, klo + block)
                        yield klo, k[klo:khi], val[klo:khi]
                dummy = torch.arange(qlo, qhi, device=x.device)
                attended[qlo:qhi] = online_attention(
                    query, blocks(), dummy, scale=dim ** -0.5, value_dim=dim,
                    guard=self.guard, causal=False)
        return self.ops.linear(attended.reshape(len(x), hidden), base + 'proj.weight', base + 'proj.bias')

    def _vision_mlp(self, x, layer):
        v = self.vision
        base = self.vision_prefix + f'blocks.{layer}.mlp.'
        inter = v['intermediate_size']
        per = max(inter, v['hidden_size']) * 20
        chunk = max(1, min(len(x), self.budget.workspace_bytes // per))
        result = torch.empty_like(x)
        for lo in range(0, len(x), chunk):
            z = self.ops.linear(x[lo:lo + chunk], base + 'linear_fc1.weight', base + 'linear_fc1.bias')
            z = F.gelu(z, approximate='tanh')
            result[lo:lo + chunk] = self.ops.linear(z, base + 'linear_fc2.weight', base + 'linear_fc2.bias')
        return result

    def _vision_merger(self, x, base, *, postshuffle):
        v = self.vision
        merge = v['spatial_merge_size']
        group = v['hidden_size'] * merge * merge
        if len(x) % (merge * merge):
            raise ValueError("Vision patch count is not divisible by spatial merge unit")
        if postshuffle:
            grouped = x.reshape(-1, group)
            grouped = _layer_norm(grouped, self.ops.small(base + 'norm.weight'), self.ops.small(base + 'norm.bias'))
        else:
            normalized = _layer_norm(x, self.ops.small(base + 'norm.weight'), self.ops.small(base + 'norm.bias'))
            grouped = normalized.reshape(-1, group)
        per = max(group, v['out_hidden_size']) * 24
        chunk = max(1, min(len(grouped), self.budget.workspace_bytes // per))
        output = torch.empty((len(grouped), v['out_hidden_size']), dtype=x.dtype, device=x.device)
        for lo in range(0, len(grouped), chunk):
            z = self.ops.linear(grouped[lo:lo + chunk], base + 'linear_fc1.weight', base + 'linear_fc1.bias')
            z = F.gelu(z)
            output[lo:lo + chunk] = self.ops.linear(z, base + 'linear_fc2.weight', base + 'linear_fc2.bias')
        return output

    def _validate_vision_inputs(self, pixel_values, image_grid_thw):
        v = self.vision
        if image_grid_thw is None or not torch.is_tensor(image_grid_thw) or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
            raise ValueError("image_grid_thw must be an [images,3] integer tensor")
        grid = image_grid_thw.detach().to('cpu', dtype=torch.long)
        if not len(grid) or len(grid) > self.budget.max_images or torch.any(grid <= 0):
            raise ValueError("Image count/grid exceeds the bounded image profile")
        merge = v['spatial_merge_size']
        if torch.any(grid[:, 1:] % merge):
            raise ValueError("Image patch grid must be divisible by spatial_merge_size")
        raw_tokens = int(grid.prod(-1).sum().item())
        if raw_tokens > self.budget.max_visual_tokens:
            raise MemoryBudgetError(
                f"Image needs {raw_tokens} pre-merge patches; profile allows {self.budget.max_visual_tokens}")
        patch_width = v['in_channels'] * v['temporal_patch_size'] * v['patch_size'] ** 2
        if not torch.is_tensor(pixel_values):
            raise ValueError("pixel_values must be a tensor produced by the local processor")
        pixels = pixel_values.detach()
        if pixels.numel() != raw_tokens * patch_width:
            raise ValueError("Pixel patch tensor does not match image_grid_thw")
        pixels = pixels.reshape(raw_tokens, patch_width)
        resident = raw_tokens * (patch_width + 4 * v['hidden_size']) * torch.empty((), dtype=self.dtype).element_size()
        self.guard.workspace(resident)
        return pixels, grid, raw_tokens

    @torch.inference_mode()
    def encode_images(self, pixel_values, image_grid_thw):
        pixels, grid, raw_tokens = self._validate_vision_inputs(pixel_values, image_grid_thw)
        p = self.vision_prefix
        x = self._vision_linear_flat(pixels, p + 'patch_embed.proj.weight', p + 'patch_embed.proj.bias')
        x = x + self._vision_position_embeddings(grid.to(self.device))
        cos, sin = self._vision_position_cos_sin(grid.to(self.device))
        segments = self._vision_segments(grid)
        deep = []
        indexes = self.vision['deepstack_visual_indexes']
        for layer in range(self.vision['depth']):
            b = p + f'blocks.{layer}.'
            z = self._vision_norm(x, b + 'norm1')
            x = x + self._vision_attention(z, layer, cos, sin, segments)
            z = self._vision_norm(x, b + 'norm2')
            x = x + self._vision_mlp(z, layer)
            if layer in indexes:
                j = indexes.index(layer)
                feature = self._vision_merger(x, p + f'deepstack_merger_list.{j}.', postshuffle=True)
                deep.append(feature.detach().to('cpu'))
        final = self._vision_merger(x, p + 'merger.', postshuffle=False).detach().to('cpu')
        if len(deep) != len(indexes):
            raise RuntimeError("DeepStack extraction did not produce every configured feature")
        return final, deep, raw_tokens

    # -------------------------- multimodal text ---------------------------
    def _mrope_positions(self, tokens, grid):
        c = self.config.raw
        image_token, video_token = c['image_token_id'], c['video_token_id']
        vision_start = c['vision_start_token_id']
        if video_token in tokens:
            raise ValueError("Video is not enabled in the bounded Qwen3-VL image profile")
        starts = [i for i, token in enumerate(tokens[:-1]) if token == vision_start and tokens[i + 1] == image_token]
        if len(starts) != len(grid):
            raise ValueError("Image placeholders do not match image_grid_thw")
        merge = self.vision['spatial_merge_size']
        chunks, st = [], 0
        for image_index, start_marker in enumerate(starts):
            ed = start_marker + 1
            t, h, w = grid[image_index].tolist()
            mt, mh, mw = t, h // merge, w // merge
            count = mt * mh * mw
            if tokens[ed:ed + count] != [image_token] * count:
                raise ValueError("Image token run length does not match the processed grid")
            base = int(chunks[-1].max().item()) + 1 if chunks else 0
            text_len = ed - st
            if text_len:
                chunks.append(torch.arange(text_len, dtype=torch.long).view(1, -1).expand(3, -1) + base)
                base += text_len
            ti = torch.arange(mt).view(-1, 1).expand(-1, mh * mw).flatten()
            hi = torch.arange(mh).view(1, -1, 1).expand(mt, -1, mw).flatten()
            wi = torch.arange(mw).view(1, 1, -1).expand(mt, mh, -1).flatten()
            chunks.append(torch.stack((ti, hi, wi)) + base)
            st = ed + count
        if st < len(tokens):
            base = int(chunks[-1].max().item()) + 1 if chunks else 0
            chunks.append(torch.arange(len(tokens) - st, dtype=torch.long).view(1, -1).expand(3, -1) + base)
        positions = torch.cat(chunks, dim=1) if chunks else torch.arange(len(tokens)).view(1, -1).expand(3, -1)
        if positions.shape[1] != len(tokens):
            raise ValueError("MRoPE position construction did not cover the prompt")
        return positions, int(positions.max().item()) + 1 - len(tokens)

    def _multimodal_prefill_chunk(self, tokens, state, position, mrope_positions,
                                  image_features, deepstack, image_token):
        n = len(tokens)
        x = self.ops.embedding(tokens, self.embedding_name)
        local_mask = torch.tensor([t == image_token for t in tokens], dtype=torch.bool, device=self.device)
        before = self._image_tokens_before
        count = int(local_mask.sum().item())
        if count:
            x = x.clone()
            x[local_mask] = image_features[before:before + count].to(device=self.device, dtype=self.dtype)
            ds = [feat[before:before + count].to(device=self.device, dtype=self.dtype) for feat in deepstack]
        else:
            ds = [torch.empty((0, self.config.hidden_size), device=self.device, dtype=self.dtype) for _ in deepstack]
        self._image_tokens_before += count
        self._active_mrope = mrope_positions[:, position:position + n].to(self.device)
        try:
            return self._forward_embeds(x, state, position, visual_mask=local_mask, deepstack=ds)
        finally:
            self._active_mrope = None

    @torch.inference_mode()
    def generate(self, input_ids, *, pixel_values=None, image_grid_thw=None,
                 pixel_values_videos=None, video_grid_thw=None, **kwargs):
        if pixel_values_videos is not None or video_grid_thw is not None:
            raise ValueError("Video requires a separately qualified bounded profile and is not enabled")
        if pixel_values is None and image_grid_thw is None:
            return super().generate(input_ids, **kwargs)
        if pixel_values is None or image_grid_thw is None:
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        return self._generate_images(input_ids, pixel_values, image_grid_thw, **kwargs)

    def _generate_images(self, input_ids, pixel_values, image_grid_thw, *, attention_mask=None,
                         max_new_tokens=128, eos_token_id=None, pad_token_id=None, do_sample=False,
                         temperature=1.0, top_p=1.0, top_k=None, repetition_penalty=1.0,
                         use_cache=True, num_beams=1, return_dict_in_generate=False,
                         past_key_values=None, **unsupported):
        if unsupported:
            raise ValueError('Unsupported generation options: ' + ', '.join(sorted(unsupported)))
        if num_beams != 1 or not use_cache or return_dict_in_generate or past_key_values is not None:
            raise ValueError("Multimodal generation supports batch-one fresh cached decoding only")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a [1,T] int64 tensor")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape or not torch.all(attention_mask == 1)):
            raise ValueError("Padding/packed multimodal prompts are not supported")
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.budget.max_output_tokens:
            raise ValueError("Requested output exceeds profile max_output_tokens")
        if not all(math.isfinite(float(v)) for v in (temperature, top_p, repetition_penalty)):
            raise ValueError("Non-finite sampling options")
        if temperature <= 0 or not 0 < top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("Invalid sampling options")
        if top_k is not None and (type(top_k) is not int or top_k < 0):
            raise ValueError("top_k must be nonnegative")
        tokens = input_ids[0].detach().cpu().tolist()
        if not tokens or len(tokens) + max_new_tokens > min(self.config.max_position_embeddings, self.budget.max_context_tokens):
            raise ValueError("Prompt plus output exceeds the configured context profile")
        if self.config.raw['video_token_id'] in tokens:
            raise ValueError("Video placeholder found in image-only bounded profile")
        stops = set([eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or []))
        if any(type(v) is not int or not 0 <= v < self.config.vocab_size for v in stops):
            raise ValueError("Invalid EOS token")
        with GENERATION_LOCK:
            start = time.monotonic()
            image_features, deepstack, raw_visual = self.encode_images(pixel_values, image_grid_thw)
            grid = image_grid_thw.detach().to('cpu', dtype=torch.long)
            image_token = self.config.raw['image_token_id']
            llm_visual = int((grid.prod(-1) // self.vision['spatial_merge_size'] ** 2).sum().item())
            if tokens.count(image_token) != llm_visual or len(image_features) != llm_visual:
                raise ValueError("Image features and image placeholder token count differ")
            positions, self._rope_delta = self._mrope_positions(tokens, grid)
            self._image_tokens_before = 0
            with DiskState(self.cache_root, max_bytes=self.budget.max_cache_bytes,
                           max_block_tokens=self.budget.attention_block_tokens) as state:
                starting_io = self.store.bytes_read
                self.guard.check()
                try:
                    initial = len(tokens)
                    for position in range(0, initial, self.prefill_tokens):
                        part = tokens[position:position + self.prefill_tokens]
                        hidden = self._multimodal_prefill_chunk(part, state, position, positions,
                                                               image_features, deepstack, image_token)
                    if self._image_tokens_before != llm_visual:
                        raise RuntimeError("Not every image feature was consumed by the prompt")
                    prefill_seconds = time.monotonic() - start
                    for step in range(max_new_tokens):
                        logits = self.logits(hidden)
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("Non-finite logits")
                        if repetition_penalty != 1:
                            seen = torch.tensor(sorted(set(tokens)), device=logits.device)
                            scores = logits[seen]
                            logits[seen] = torch.where(scores < 0, scores * repetition_penalty, scores / repetition_penalty)
                        if do_sample:
                            logits = logits / temperature
                            if top_k:
                                threshold = logits.topk(min(top_k, logits.numel())).values[-1]
                                logits[logits < threshold] = -torch.inf
                            if top_p < 1:
                                ordered, order = logits.sort(descending=True)
                                cumulative = ordered.softmax(-1).cumsum(-1)
                                remove = cumulative > top_p
                                remove[1:] = remove[:-1].clone(); remove[0] = False
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
                        visual_input_tokens=raw_visual, visual_llm_tokens=llm_visual,
                        images=len(grid), cache_peak_bytes=state.peak_bytes,
                        cache_bytes_read=state.bytes_read,
                        checkpoint_bytes_read=self.store.bytes_read - starting_io,
                        max_checkpoint_read_bytes=self.store.max_read_observed,
                        prefill_chunk_tokens=self.prefill_tokens, memory_qualified=False)
                    return torch.tensor([tokens], dtype=torch.long, device=input_ids.device)
                except torch.cuda.OutOfMemoryError as exc:
                    raise MemoryBudgetError("CUDA OOM inside bounded Qwen3-VL execution; profile is NOT qualified") from exc
                finally:
                    self._active_mrope = None
                    self._rope_delta = None
                    self._image_tokens_before = 0

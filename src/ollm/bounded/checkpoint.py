"""Validated safetensors byte-range reader; no model materialization or pickle.

Only a requested row range is read. Packed expert banks are sliced on the expert
axis BEFORE a tensor is created. FP8 weights are dequantized per bounded row tile.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import torch

from .budget import MemoryBudgetError


def _pairs(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f"Duplicate JSON key: {k}")
        result[k] = v
    return result


def read_json(path, max_bytes=64 * 1024**2):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError(f"JSON file too large: {path.name}")
    return json.loads(path.read_text('utf-8'), object_pairs_hook=_pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(f"Invalid JSON number {v}")))


DTYPES = {'F32': torch.float32, 'F16': torch.float16, 'BF16': torch.bfloat16,
          'F64': torch.float64, 'I64': torch.int64, 'I32': torch.int32,
          'F8_E4M3': torch.float8_e4m3fn, 'F8_E4M3FN': torch.float8_e4m3fn}


@dataclass(frozen=True)
class TensorInfo:
    path: Path
    shape: tuple[int, ...]
    dtype: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class WeightRef:
    name: str
    expert: int | None = None
    row_start: int = 0
    row_stop: int | None = None


class TensorStore:
    def __init__(self, root, *, max_read_bytes=64 * 1024**2, fp8_block=(128, 128)):
        self.root = Path(root).expanduser().resolve(strict=True)
        self.max_read_bytes = int(max_read_bytes)
        self.fp8_block = tuple(fp8_block)
        self.tensors: dict[str, TensorInfo] = {}
        self.bytes_read = 0
        self.max_read_observed = 0
        self.fingerprint_parts = []
        index = self.root / 'model.safetensors.index.json'
        if index.exists():
            manifest = read_json(index).get('weight_map')
            if not isinstance(manifest, dict) or not manifest:
                raise ValueError("Empty or invalid safetensors index")
            filenames = sorted(set(manifest.values()))
        else:
            manifest = None
            filenames = ['model.safetensors']
        for filename in filenames:
            if not isinstance(filename, str):
                raise ValueError("Invalid checkpoint filename")
            path = (self.root / filename).resolve(strict=True)
            if not path.is_relative_to(self.root) or path.suffix != '.safetensors':
                raise ValueError("Checkpoint shard escapes the model directory or is not safetensors")
            self._header(path)
        if manifest is not None:
            for name, filename in manifest.items():
                if name not in self.tensors or self.tensors[name].path != (self.root / filename).resolve():
                    raise ValueError(f"Index does not match shard tensor: {name}")
            self.tensors = {k: v for k, v in self.tensors.items() if k in manifest}
        self.layout_fingerprint = hashlib.sha256('\n'.join(self.fingerprint_parts).encode()).hexdigest()

    def _header(self, path):
        size = path.stat().st_size
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("Truncated safetensors header")
            n, = struct.unpack('<Q', raw)
            if n < 2 or n > 64 * 1024**2 or n + 8 > size:
                raise ValueError("Invalid safetensors header size")
            header_bytes = f.read(n)
            header = json.loads(header_bytes, object_pairs_hook=_pairs)
        self.fingerprint_parts.append(f"{path.relative_to(self.root)}:{size}:" + hashlib.sha256(header_bytes).hexdigest())
        spans = []
        for name, info in header.items():
            if name == '__metadata__':
                continue
            if name in self.tensors:
                raise ValueError(f"Duplicate tensor in shards: {name}")
            if info['dtype'] not in DTYPES:
                raise ValueError(f"Unsupported dtype {info['dtype']} for {name}; no silent quantized fallback")
            shape = info['shape']
            offsets = info['data_offsets']
            if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                or len(offsets) != 2 or any(type(x) is not int for x in offsets)):
                raise ValueError(f"Invalid shape/offsets for {name}")
            lo, hi = offsets
            itemsize = torch.empty((), dtype=DTYPES[info['dtype']]).element_size()
            if lo < 0 or hi - lo != math.prod(shape) * itemsize or hi > size - n - 8:
                raise ValueError(f"Tensor range is outside shard: {name}")
            spans.append((lo, hi))
            self.tensors[name] = TensorInfo(path, tuple(shape), info['dtype'], n + 8 + lo, hi - lo)
        end = 0
        for lo, hi in sorted(spans):
            if lo < end:
                raise ValueError("Overlapping safetensors tensor ranges")
            end = hi

    def has(self, name):
        return name in self.tensors

    def shape(self, ref: str | WeightRef):
        ref = WeightRef(ref) if isinstance(ref, str) else ref
        shape = self.tensors[ref.name].shape
        if ref.expert is not None:
            if not shape or not 0 <= ref.expert < shape[0]:
                raise ValueError(f"Expert index out of range: {ref}")
            shape = shape[1:]
        if ref.row_start or ref.row_stop is not None:
            if not shape:
                raise ValueError("Cannot slice a scalar")
            stop = shape[0] if ref.row_stop is None else ref.row_stop
            if not 0 <= ref.row_start <= stop <= shape[0]:
                raise ValueError("Invalid weight row slice")
            shape = (stop - ref.row_start,) + shape[1:]
        return shape

    def raw_rows(self, name, start=0, stop=None, *, expert=None):
        info = self.tensors[name]
        shape = info.shape
        offset = info.offset
        itemsize = torch.empty((), dtype=DTYPES[info.dtype]).element_size()
        if expert is not None:
            if len(shape) < 2 or not 0 <= expert < shape[0]:
                raise ValueError("Invalid expert slice")
            offset += expert * math.prod(shape[1:]) * itemsize
            shape = shape[1:]
        if not shape:  # scalar
            shape = (1,)
        stop = shape[0] if stop is None else stop
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError("Weight slice out of range")
        row_bytes = math.prod(shape[1:]) * itemsize
        n = (stop - start) * row_bytes
        if n > self.max_read_bytes:
            raise MemoryBudgetError(f"Unbounded tensor read {name}: {n:,} bytes")
        with info.path.open('rb') as f:
            f.seek(offset + start * row_bytes)
            data = bytearray(f.read(n))
        if len(data) != n:
            raise ValueError(f"Short checkpoint read for {name}")
        self.bytes_read += n
        self.max_read_observed = max(self.max_read_observed, n)
        if not n:
            return torch.empty((stop - start,) + shape[1:], dtype=DTYPES[info.dtype])
        return torch.frombuffer(data, dtype=DTYPES[info.dtype]).reshape((stop - start,) + shape[1:])

    def rows(self, ref: str | WeightRef, start=0, stop=None, *, dtype=torch.bfloat16, device='cpu'):
        ref = WeightRef(ref) if isinstance(ref, str) else ref
        shape = self.shape(ref)
        stop = shape[0] if stop is None else stop
        if not 0 <= start <= stop <= shape[0]:
            raise ValueError("Weight slice out of range")
        lo, hi = ref.row_start + start, ref.row_start + stop
        raw = self.raw_rows(ref.name, lo, hi, expert=ref.expert)
        if raw.dtype == torch.float8_e4m3fn:
            if raw.ndim != 2 or ref.expert is not None:
                raise ValueError("FP8 support requires ordinary 2-D per-expert weight tensors")
            scale_name = ref.name.removesuffix('.weight') + '.weight_scale_inv'
            if scale_name not in self.tensors:
                raise ValueError(f"Missing block scales: {scale_name}")
            br, bc = self.fp8_block
            full = self.tensors[ref.name].shape
            expected = (math.ceil(full[0] / br), math.ceil(full[1] / bc))
            if self.tensors[scale_name].shape != expected:
                raise ValueError(f"Invalid FP8 block-scale shape for {ref.name}")
            scales = self.raw_rows(scale_name, lo // br, math.ceil(hi / br)).float()
            # CPU expansion is limited to one tile, never the whole checkpoint.
            row_index = torch.arange(lo, hi) // br - lo // br
            col_index = torch.arange(raw.shape[1]) // bc
            converted = raw.float()
            converted.mul_(scales[row_index[:, None], col_index[None, :]])
            raw = converted
        return raw.to(device=device, dtype=dtype)

    def small(self, name, *, dtype=torch.float32, device='cpu'):
        info = self.tensors[name]
        if info.nbytes > self.max_read_bytes:
            raise MemoryBudgetError(f"{name} must be accessed by tiles")
        out = (self.raw_rows(name).to(device=device, dtype=dtype) if not info.shape
               else self.rows(name, dtype=dtype, device=device))
        return out.reshape(info.shape)

    def content_hash(self):
        """Slow but bounded-memory checksum used by hardware qualification."""
        digest = hashlib.sha256()
        for path in sorted({x.path for x in self.tensors.values()}):
            digest.update(str(path.relative_to(self.root)).encode())
            with path.open('rb') as f:
                while data := f.read(8 * 1024**2):
                    digest.update(data)
        return digest.hexdigest()

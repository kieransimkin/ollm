"""Session-isolated, append-only disk states with bounded block reads.

No generated-token GPU tail. Sparse hybrid layer IDs are dictionary keys, not
positions in an append-only list. Files contain raw tensors, never pickle.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile
import torch


@dataclass
class Stream:
    path: Path
    row_shape: tuple[int, ...]
    dtype: torch.dtype
    length: int = 0


class DiskState:
    def __init__(self, root=None, *, max_bytes=128 * 1024**3, max_block_tokens=256):
        if root is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix='ollm-bounded-', dir=root))
        self.max_bytes = max_bytes
        self.max_block_tokens = max_block_tokens
        self.streams: dict[tuple[int, str], Stream] = {}
        self.states: dict[tuple[int, str], Stream] = {}
        self.bytes_stored = 0
        self.bytes_read = 0
        self.peak_bytes = 0
        self.closed = False
        self.position = 0

    def _check(self, key):
        if self.closed:
            raise RuntimeError("Cache session is closed")
        if (not isinstance(key, tuple) or len(key) != 2 or type(key[0]) is not int
            or key[0] < 0 or key[1] not in ('k', 'v', 'latent', 'rope', 'conv', 'recurrent')):
            raise ValueError("Invalid cache key")

    def _reserve(self, delta):
        if self.bytes_stored + delta > self.max_bytes:
            raise RuntimeError("Disk-cache quota exceeded")
        free = shutil.disk_usage(self.path).free
        if delta > max(0, free - 1024**2):
            raise RuntimeError("Insufficient free disk space for cache")

    @staticmethod
    def _bytes(tensor):
        t = tensor.detach().to('cpu').contiguous()
        return t.view(torch.uint8).numpy().tobytes()

    def append(self, key, values, *, position):
        self._check(key)
        if values.ndim < 1 or values.shape[0] > self.max_block_tokens:
            raise ValueError("Append exceeds cache block bound")
        stream = self.streams.get(key)
        if stream is None:
            stream = Stream(self.path / f'{key[0]}-{key[1]}.bin', tuple(values.shape[1:]), values.dtype)
            self.streams[key] = stream
        if position != stream.length or tuple(values.shape[1:]) != stream.row_shape or values.dtype != stream.dtype:
            raise ValueError("Non-contiguous or incompatible cache append")
        n = values.numel() * values.element_size()
        self._reserve(n)
        with stream.path.open('ab') as f:
            f.write(self._bytes(values))
        stream.length += values.shape[0]
        self.bytes_stored += n
        self.peak_bytes = max(self.peak_bytes, self.bytes_stored)

    def length(self, key):
        self._check(key)
        return self.streams[key].length if key in self.streams else 0

    def read(self, key, start, stop, *, device='cpu'):
        self._check(key)
        s = self.streams[key]
        if not 0 <= start <= stop <= s.length or stop - start > self.max_block_tokens:
            raise ValueError("Invalid or oversized cache read")
        return self._read(s, start, stop, device)

    def _read(self, s, start, stop, device):
        import math
        itemsize = torch.empty((), dtype=s.dtype).element_size()
        row_bytes = math.prod(s.row_shape) * itemsize
        with s.path.open('rb') as f:
            f.seek(start * row_bytes)
            data = bytearray(f.read((stop - start) * row_bytes))
        if len(data) != (stop - start) * row_bytes:
            raise RuntimeError("Truncated cache state")
        self.bytes_read += len(data)
        if stop == start or row_bytes == 0:
            return torch.empty((stop - start,) + s.row_shape, dtype=s.dtype, device=device)
        return torch.frombuffer(data, dtype=s.dtype).reshape((stop - start,) + s.row_shape).to(device)

    def save_state(self, key, tensor):
        self._check(key)
        if key[1] not in ('conv', 'recurrent'):
            raise ValueError("Only recurrent and convolution states may be replaced")
        old = self.states.get(key)
        n = tensor.numel() * tensor.element_size()
        old_n = old.path.stat().st_size if old else 0
        self._reserve(n)  # atomic replacement temporarily needs both files
        self.peak_bytes = max(self.peak_bytes, self.bytes_stored + n)
        path = self.path / f'{key[0]}-{key[1]}.state'
        temporary = path.with_suffix('.tmp')
        try:
            temporary.write_bytes(self._bytes(tensor))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self.states[key] = Stream(path, tuple(tensor.shape), tensor.dtype, 1)
        self.bytes_stored += n - old_n
        self.peak_bytes = max(self.peak_bytes, self.bytes_stored)

    def state(self, key, *, device='cpu'):
        self._check(key)
        if key not in self.states:
            return None
        return self._read(self.states[key], 0, 1, device)[0]

    def close(self):
        if not self.closed:
            shutil.rmtree(self.path)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

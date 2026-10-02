"""Conservative, byte-based inference budgets (never a hardware certification)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
import torch


class MemoryBudgetError(RuntimeError):
    """The requested working set does not fit the configured budget."""


@dataclass(frozen=True)
class MemoryBudget:
    # Decimal bytes: deliberately below BOTH 8 GB and 8 GiB.
    vram_bytes: int = 7_000_000_000
    workspace_bytes: int = 128 * 1024**2
    weight_tile_bytes: int = 32 * 1024**2
    safety_bytes: int = 256 * 1024**2
    prefill_tokens: int = 64
    attention_block_tokens: int = 256
    max_context_tokens: int = 32768
    max_output_tokens: int = 512
    max_cache_bytes: int = 128 * 1024**3
    # Vision tokens are pre-merge ViT patches (not LLM image tokens).
    max_visual_tokens: int = 4096
    max_images: int = 4

    def __post_init__(self):
        for k, v in asdict(self).items():
            if type(v) is not int or v < 1:
                raise ValueError(f"{k} must be a positive integer")
        if self.vram_bytes >= 8_000_000_000:
            raise ValueError("The bounded profile must be strictly below 8,000,000,000 bytes")
        if self.workspace_bytes + 3 * self.weight_tile_bytes + self.safety_bytes >= self.vram_bytes:
            raise ValueError("Workspace, conversion tiles and safety reserve exceed the VRAM budget")

    def as_dict(self):
        return asdict(self)


class BudgetGuard:
    """Checks device-wide usage at operation boundaries; no opaque CUDA kernels.

    A competing process may allocate between checks. This guard cannot guarantee
    a device-wide hard limit; qualification records sampling and allocator peaks.
    All kernels in this package explicitly bound their individual workspaces.
    """
    def __init__(self, budget: MemoryBudget, device):
        self.budget = budget
        self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError("The bounded runtime supports CPU reference tests and CUDA only")
        self.peak_device_bytes = 0
        self.peak_requested_bytes = 0
        self.checks = 0
        self.started = time.monotonic()
        self.check(0)

    def check(self, extra: int = 0):
        if extra < 0:
            raise ValueError("Negative memory reservation")
        self.checks += 1
        self.peak_requested_bytes = max(self.peak_requested_bytes, extra)
        if extra + self.budget.safety_bytes >= self.budget.vram_bytes:
            raise MemoryBudgetError(f"Requested working set {extra:,} bytes exceeds the memory profile")
        if self.device.type == 'cuda':
            # mem_get_info sees allocations outside PyTorch too. memory_reserved
            # is not added again: it is already included in device-wide usage.
            with torch.cuda.device(self.device):
                free, total = torch.cuda.mem_get_info()
                used = total - free
                self.peak_device_bytes = max(self.peak_device_bytes, used)
                reusable = max(0, torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device))
                if used + max(0, extra - reusable) + self.budget.safety_bytes > min(total, self.budget.vram_bytes):
                    torch.cuda.empty_cache()
                    free, total = torch.cuda.mem_get_info()
                    used = total - free
                    if used + extra + self.budget.safety_bytes > min(total, self.budget.vram_bytes):
                        raise MemoryBudgetError(
                            f"Device uses {used:,} bytes; operation needs up to {extra:,}; "
                            f"budget is {self.budget.vram_bytes:,}. Reduce chunks or free the GPU.")

    def workspace(self, size: int):
        if size > self.budget.workspace_bytes:
            raise MemoryBudgetError(f"Workspace {size:,} exceeds {self.budget.workspace_bytes:,} bytes")
        self.check(size)

    def report(self):
        d = dict(peak_device_bytes_at_checks=self.peak_device_bytes,
                 peak_requested_bytes=self.peak_requested_bytes, checks=self.checks,
                 elapsed_seconds=time.monotonic() - self.started, device=str(self.device))
        if self.device.type == 'cuda':
            d.update(torch_peak_allocated=torch.cuda.max_memory_allocated(self.device),
                     torch_peak_reserved=torch.cuda.max_memory_reserved(self.device))
        return d


# Generation is single-stream. This also avoids multiple bounded models fighting
# over an otherwise individually sensible per-device memory budget.
GENERATION_LOCK = threading.RLock()

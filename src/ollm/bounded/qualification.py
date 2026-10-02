"""Local CUDA memory-profile measurements, not fabricated model benchmarks.

Memory qualification and mathematical/model-quality qualification are separate.
A sampled driver trace cannot prove an absolute global peak between samples.
Reports explicitly retain this limitation and the hardware/software identity.
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import threading
import time
from pathlib import Path
import torch
from .checkpoint import read_json


def implementation_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def identity(model):
    config_hash = hashlib.sha256(json.dumps(model.config.raw, sort_keys=True).encode()).hexdigest()
    return dict(layout_digest=model.store.layout_fingerprint, config_digest=config_hash,
                implementation_digest=implementation_digest(), dtype=str(model.dtype),
                torch_version=torch.__version__, family=model.config.family)


class DeviceMonitor:
    def __init__(self, device, interval=0.01):
        self.device = torch.device(device)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self.error = None

    def sample(self):
        if self.device.type == 'cuda':
            with torch.cuda.device(self.device):
                free, total = torch.cuda.mem_get_info()
                self.samples.append({'seconds': time.monotonic() - self.start,
                                     'device_used_bytes': total - free})

    def __enter__(self):
        self.start = time.monotonic()
        if self.device.type == 'cuda':
            self.sample()
            def worker():
                while not self._stop.wait(self.interval):
                    try:
                        self.sample()
                    except Exception as exc:
                        self.error = str(exc)
                        return
            self._thread = threading.Thread(target=worker, name='ollm-vram-trace', daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
            self.sample()


def validate_profile(path, model):
    report = read_json(path)
    if model.device.type != 'cuda' or report.get('memory_pass') is not True:
        raise ValueError("A successful real CUDA memory profile is required")
    if report.get('identity') != identity(model):
        raise ValueError("Profile does not match checkpoint layout/config, dtype, PyTorch or implementation")
    if report.get('device_name') != torch.cuda.get_device_name(model.device):
        raise ValueError("Profile was measured on a different GPU model")
    if report.get('device_total_bytes') != torch.cuda.get_device_properties(model.device).total_memory:
        raise ValueError("Profile GPU memory capacity differs")
    # Exact budgets/chunks. A larger context/output or changed tile plan must
    # never inherit a passing smaller-profile result.
    if report.get('budget') != model.budget.as_dict():
        raise ValueError("Memory profile budget/context/output/chunk settings differ")
    if getattr(model.config, 'multimodal', False):
        if report.get('multimodal_qualification') is not True:
            raise ValueError("Qwen3-VL requires a multimodal CUDA qualification profile")
        if report.get('max_observed_visual_tokens', 0) < model.budget.max_visual_tokens:
            raise ValueError("The profile did not exercise max_visual_tokens")
        if report.get('max_observed_images', 0) < model.budget.max_images:
            raise ValueError("The profile did not exercise max_images")
    if report.get('max_observed_prompt_tokens', 0) + report.get('forced_output_tokens', 0) < model.budget.max_context_tokens:
        raise ValueError("The measured run did not exercise the advertised context bound")
    if report.get('forced_output_tokens', 0) < model.budget.max_output_tokens:
        raise ValueError("The measured run did not exercise the advertised output bound")
    if report.get('checkpoint_content_sha256') is not None:
        if report['checkpoint_content_sha256'] != model.store.content_hash():
            raise ValueError('Profile checkpoint payload hash differs')
    return report


def host_peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except (ImportError, AttributeError):
        return None


def benchmark(model, *, prompt_tokens=1024, rounds=3, report_path=None, hash_weights=False):
    """Exercise cold generation, sustained decode and rebuilt-history rounds.

    Synthetic valid token IDs make lengths exact; this is NOT an agent-quality
    test. EOS stopping is disabled to exercise every requested decode step.
    CPU runs are useful smoke tests but can NEVER produce memory_pass=True.
    """
    if getattr(model.config, 'multimodal', False):
        raise ValueError("Use benchmark_multimodal/qualify-vl for Qwen3-VL; text-only runs cannot qualify image support")
    if type(rounds) is not int or rounds < 2 or type(prompt_tokens) is not int or prompt_tokens < 1:
        raise ValueError("Qualification requires >=2 rounds and a positive prompt size")
    budget = model.budget
    largest_prompt = budget.max_context_tokens - budget.max_output_tokens
    if prompt_tokens > largest_prompt:
        raise ValueError("Initial prompt plus output exceeds profile context")
    lengths = [round(prompt_tokens + (largest_prompt - prompt_tokens) * i / (rounds - 1)) for i in range(rounds)]
    report = dict(format_version=1, memory_pass=False, numerical_qualification=False,
                  tool_quality_qualification=False, workload='synthetic text; forced output; rebuilt histories',
                  identity=identity(model), budget=budget.as_dict(), device=str(model.device),
                  platform=platform.platform(), cases=[], forced_output_tokens=budget.max_output_tokens,
                  max_observed_prompt_tokens=0,
                  peak_measurement='sampled device-wide memory plus PyTorch allocator high-water marks',
                  limitation='Device-wide samples can miss between-sample peaks; external GPU activity is not controlled.')
    if hash_weights:
        report['checkpoint_content_sha256'] = model.store.content_hash()
    try:
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(model.device)
            report.update(device_name=torch.cuda.get_device_name(model.device),
                          device_total_bytes=torch.cuda.get_device_properties(model.device).total_memory)
            free, total = torch.cuda.mem_get_info(model.device)
            baseline_non_torch = max(0, total - free - torch.cuda.memory_reserved(model.device))
        else:
            baseline_non_torch = 0
        with DeviceMonitor(model.device) as monitor:
            for length in lengths:
                # All token IDs are within the vocabulary. Do not insert image/
                # video placeholders into the explicitly text-only path.
                sequence = [3 + (i % max(1, min(model.config.vocab_size - 3, 97))) for i in range(length)]
                sequence = [min(x, model.config.vocab_size - 1) for x in sequence]
                ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
                start = time.monotonic()
                result = model.generate(ids, max_new_tokens=budget.max_output_tokens, eos_token_id=[])
                if model.device.type == 'cuda':
                    torch.cuda.synchronize(model.device)
                generated = result.shape[1] - length
                if generated != budget.max_output_tokens:
                    raise RuntimeError("Benchmark did not exercise the requested output length")
                report['cases'].append(dict(model.last_report, elapsed_seconds=time.monotonic() - start))
                report['max_observed_prompt_tokens'] = max(report['max_observed_prompt_tokens'], length)
                del ids, result
        report['driver_trace'] = monitor.samples
        report['monitor_error'] = monitor.error
        if model.device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(model.device)
            end_non_torch = max(0, total - free - torch.cuda.memory_reserved(model.device))
            allocator_peak = torch.cuda.max_memory_reserved(model.device)
            sampled_peak = max(s['device_used_bytes'] for s in monitor.samples)
            conservative_peak = max(sampled_peak, max(baseline_non_torch, end_non_torch) + allocator_peak,
                                    model.guard.peak_device_bytes)
            report.update(torch_peak_allocated=torch.cuda.max_memory_allocated(model.device),
                          torch_peak_reserved=allocator_peak, sampled_device_peak_bytes=sampled_peak,
                          conservative_peak_bytes=conservative_peak)
            report['memory_pass'] = bool(monitor.error is None and conservative_peak < budget.vram_bytes < 8_000_000_000)
            report['status'] = 'memory_pass' if report['memory_pass'] else 'memory_failed'
        else:
            report['status'] = 'cpu_reference_only'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['host_process_peak_rss_bytes'] = host_peak_rss_bytes()
        report['checkpoint_tensor_bytes'] = sum(info.nbytes for info in model.store.tensors.values())
        if report_path:
            path = Path(report_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + '\n')
    return report



def _visual_grids(total_patches, images, merge):
    unit = merge * merge
    if total_patches % unit or images < 1 or images * unit > total_patches:
        raise ValueError("Visual qualification budget cannot be represented by valid merged image grids")
    units = total_patches // unit
    base, remainder = divmod(units, images)
    result = []
    for i in range(images):
        n = base + (1 if i < remainder else 0)
        # h=merge and w=merge*n are both divisible by merge and use n merge units.
        result.append((1, merge, merge * n))
    return torch.tensor(result, dtype=torch.long)


def benchmark_multimodal(model, *, rounds=2, report_path=None, hash_weights=False):
    """Exercise Qwen3-VL at the exact visual/context/output bounds.

    This uses synthetic processor-equivalent patch tensors. It measures the
    complete bounded ViT, DeepStack and language path, not image decoding or
    preprocessing libraries (which remain on CPU in the public API).
    """
    if model.config.family != 'qwen3_vl':
        raise ValueError("Multimodal qualification currently targets qwen3_vl only")
    if type(rounds) is not int or rounds < 2:
        raise ValueError("Multimodal qualification requires at least two rounds")
    budget, v = model.budget, model.config.raw['vision_config']
    merge = v['spatial_merge_size']
    raw_visual = budget.max_visual_tokens
    llm_visual = raw_visual // (merge * merge)
    max_prompt = budget.max_context_tokens - budget.max_output_tokens
    # Two wrappers per image plus at least one text token.
    if max_prompt < llm_visual + 2 * budget.max_images + 1:
        raise ValueError("Context budget is too small for max_visual_tokens/max_images plus output")
    patch_width = v['in_channels'] * v['temporal_patch_size'] * v['patch_size'] ** 2
    report = dict(format_version=1, memory_pass=False, numerical_qualification=False,
                  tool_quality_qualification=False, multimodal_qualification=True,
                  workload='synthetic Qwen3-VL image+text; max vision/context; forced output',
                  identity=identity(model), budget=budget.as_dict(), device=str(model.device),
                  platform=platform.platform(), cases=[], forced_output_tokens=budget.max_output_tokens,
                  max_observed_prompt_tokens=0, max_observed_visual_tokens=0, max_observed_images=0,
                  peak_measurement='sampled device-wide memory plus PyTorch allocator high-water marks',
                  limitation='Synthetic normalized patches exercise model memory, not PIL/image preprocessing; '
                             'device-wide samples can miss between-sample peaks.')
    if hash_weights:
        report['checkpoint_content_sha256'] = model.store.content_hash()
    media = model.config.raw
    filler = next(i for i in range(model.config.vocab_size)
                  if i not in {media['image_token_id'], media['video_token_id'], media['vision_start_token_id'], media['vision_end_token_id']})
    image_counts = [1, budget.max_images]
    try:
        if model.device.type == 'cuda':
            torch.cuda.synchronize(model.device); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(model.device)
            report.update(device_name=torch.cuda.get_device_name(model.device),
                          device_total_bytes=torch.cuda.get_device_properties(model.device).total_memory)
            free, total = torch.cuda.mem_get_info(model.device)
            baseline_non_torch = max(0, total - free - torch.cuda.memory_reserved(model.device))
        else:
            baseline_non_torch = 0
        with DeviceMonitor(model.device) as monitor:
            for case_idx in range(rounds):
                images = image_counts[min(case_idx, len(image_counts)-1)]
                grid = _visual_grids(raw_visual, images, merge)
                sequence = [filler]
                for t, h, w in grid.tolist():
                    count = t * (h // merge) * (w // merge)
                    sequence += [media['vision_start_token_id']] + [media['image_token_id']] * count + [media['vision_end_token_id']]
                if len(sequence) > max_prompt:
                    raise ValueError("Synthetic multimodal prompt exceeds context before text fill")
                sequence += [filler] * (max_prompt - len(sequence))
                ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
                pixels = torch.zeros((raw_visual, patch_width), dtype=torch.float32)
                start = time.monotonic()
                result = model.generate(ids, pixel_values=pixels, image_grid_thw=grid,
                                        max_new_tokens=budget.max_output_tokens, eos_token_id=[])
                if model.device.type == 'cuda':
                    torch.cuda.synchronize(model.device)
                generated = result.shape[1] - len(sequence)
                if generated != budget.max_output_tokens:
                    raise RuntimeError("Multimodal benchmark did not force the requested output length")
                report['cases'].append(dict(model.last_report, elapsed_seconds=time.monotonic()-start,
                                            qualification_images=images))
                report['max_observed_prompt_tokens'] = max(report['max_observed_prompt_tokens'], len(sequence))
                report['max_observed_visual_tokens'] = max(report['max_observed_visual_tokens'], raw_visual)
                report['max_observed_images'] = max(report['max_observed_images'], images)
                del ids, pixels, result
        report['driver_trace'] = monitor.samples; report['monitor_error'] = monitor.error
        if model.device.type == 'cuda':
            free, total = torch.cuda.mem_get_info(model.device)
            end_non_torch = max(0, total - free - torch.cuda.memory_reserved(model.device))
            allocator_peak = torch.cuda.max_memory_reserved(model.device)
            sampled_peak = max(s['device_used_bytes'] for s in monitor.samples)
            conservative_peak = max(sampled_peak, max(baseline_non_torch, end_non_torch)+allocator_peak,
                                    model.guard.peak_device_bytes)
            report.update(torch_peak_allocated=torch.cuda.max_memory_allocated(model.device),
                          torch_peak_reserved=allocator_peak, sampled_device_peak_bytes=sampled_peak,
                          conservative_peak_bytes=conservative_peak)
            report['memory_pass'] = bool(monitor.error is None and conservative_peak < budget.vram_bytes < 8_000_000_000)
            report['status'] = 'memory_pass' if report['memory_pass'] else 'memory_failed'
        else:
            report['status'] = 'cpu_reference_only'
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['host_process_peak_rss_bytes'] = host_peak_rss_bytes()
        report['checkpoint_tensor_bytes'] = sum(info.nbytes for info in model.store.tensors.values())
        if report_path:
            path = Path(report_path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + '\n')
    return report

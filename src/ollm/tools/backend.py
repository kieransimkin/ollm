"""Serialized oLLM generation with fresh, isolated per-request caches."""
from __future__ import annotations

import math
import threading
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from .adapters import GPTOSSAdapter, QwenAdapter

# oLLM's model implementations have module-level loaders. Sharing models across
# overlapping generation calls is unsafe; serialize this integration's calls.
_GENERATION_LOCK = threading.RLock()


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int | None = None
    repetition_penalty: float = 1.0
    max_context_tokens: int | None = None
    reasoning_effort: str = "low"
    enable_thinking: bool = False
    seed: int | None = None

    def __post_init__(self):
        if any(not math.isfinite(x) for x in (self.temperature, self.top_p, self.repetition_penalty)):
            raise ValueError("Sampling settings must be finite numbers")
        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool):
            raise ValueError("max_new_tokens must be an integer")
        if self.max_new_tokens < 1 or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid token or sampling limits")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.max_context_tokens is not None and (not isinstance(self.max_context_tokens, int)
                                                   or isinstance(self.max_context_tokens, bool)
                                                   or self.max_context_tokens < 1):
            raise ValueError("max_context_tokens must be positive")
        if self.reasoning_effort not in ("low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium or high")


class InferenceBackend:
    """Wrap an already-loaded oLLM Inference object; never downloads on import.

    Set cache_dir for Qwen disk caching. Each model turn gets a fresh subdirectory
    that is removed afterward. There is intentionally no cross-turn KV reuse.
    """
    def __init__(self, inference, *, adapter=None, generation: GenerationConfig | None = None,
                 cache_dir: str | Path | None = None):
        self.inference = inference
        self.generation = generation or GenerationConfig()
        model_id = getattr(inference, "model_id", "")
        if adapter is None:
            if model_id == "qwen3-next-80B":
                adapter = QwenAdapter()
            elif model_id == "gpt-oss-20B":
                adapter = GPTOSSAdapter()
            else:
                raise ValueError("Select qwen3-next-80B or gpt-oss-20B, or explicitly supply a compatible adapter")
        self.adapter = adapter
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
        if self.cache_dir is not None and adapter.family == "gpt-oss":
            raise ValueError("Upstream oLLM does not support gpt-oss DiskCache; omit cache_dir")

    def generate(self, messages: list[dict], tools: list[dict], *, generation=None):
        import torch
        cfg = generation or self.generation
        with _GENERATION_LOCK:
            prepared = self.adapter.prepare(messages, tools, self.inference.tokenizer,
                reasoning_effort=cfg.reasoning_effort, enable_thinking=cfg.enable_thinking)
            if not prepared.input_ids or not prepared.stop_ids:
                raise ValueError("Adapter returned empty prompt or stop-token list")
            model_config = self.inference.model.config
            limits = [x for x in (cfg.max_context_tokens, getattr(model_config, "max_position_embeddings", None))
                      if isinstance(x, int) and x > 0]
            if limits and len(prepared.input_ids) + cfg.max_new_tokens > min(limits):
                raise ValueError("Prompt plus requested output exceeds the context limit; shorten history or tool results")
            vocab_size = getattr(model_config, "vocab_size", None)
            if vocab_size is not None and max(prepared.input_ids + prepared.stop_ids) >= vocab_size:
                raise ValueError("Adapter token IDs exceed the model vocabulary")
            device = torch.device(self.inference.device)
            ids = torch.tensor([prepared.input_ids], dtype=torch.long, device=device)
            kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=cfg.max_new_tokens, eos_token_id=prepared.stop_ids,
                pad_token_id=prepared.stop_ids[0], do_sample=cfg.temperature > 0,
                repetition_penalty=cfg.repetition_penalty, use_cache=True,
                return_dict_in_generate=False, num_beams=1)
            if cfg.temperature > 0:
                kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
                if cfg.top_k is not None:
                    kwargs["top_k"] = cfg.top_k
            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_context = (TemporaryDirectory(prefix="ollm-agent-", dir=self.cache_dir)
                             if self.cache_dir is not None else nullcontext(None))
            devices = ([device.index if device.index is not None else torch.cuda.current_device()]
                       if device.type == "cuda" else [])
            rng = torch.random.fork_rng(devices=devices) if cfg.seed is not None else nullcontext()
            with cache_context as cache_path, rng, torch.inference_mode():
                if cfg.seed is not None:
                    torch.manual_seed(cfg.seed)
                if cache_path is not None:
                    cache = self.inference.DiskCache(cache_dir=cache_path)
                    if cache is None:
                        raise ValueError("Selected model did not provide the requested disk cache")
                    kwargs["past_key_values"] = cache
                output = self.inference.model.generate(**kwargs)
                completion = output[0, ids.shape[-1]:].detach().cpu().tolist()
                del output
            return self.adapter.parse(completion, self.inference.tokenizer)


def config_from_mapping(base: GenerationConfig, settings: dict) -> GenerationConfig:
    """Translate a deliberately small Qwen-Agent generation configuration."""
    settings = dict(settings)
    # These only control Qwen-Agent behavior, not oLLM generation.
    for key in ("lang", "parallel_function_calls", "thought_in_content"):
        settings.pop(key, None)
    for old, new in (("max_tokens", "max_new_tokens"), ("max_input_tokens", "max_context_tokens")):
        if old in settings:
            if new in settings:
                raise ValueError(f"Specify {old} or {new}, not both")
            settings[new] = settings.pop(old)
    unknown = set(settings) - set(GenerationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError("Unsupported generation settings: " + ", ".join(sorted(unknown)))
    return replace(base, **settings)

import concurrent.futures
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from ollm.tools import AssistantTurn, GenerationConfig, InferenceBackend
from ollm.tools.adapters import PreparedPrompt
from ollm.tools.backend import config_from_mapping


class Adapter:
    family = "qwen"
    def prepare(self, *args, **kwargs):
        return PreparedPrompt([1, 2, 3], [9])
    def parse(self, completion, tokenizer):
        assert completion == [4, 9]
        return AssistantTurn("done")


@pytest.fixture
def inference():
    torch = pytest.importorskip("torch")
    class Model:
        config = SimpleNamespace(max_position_embeddings=4096, vocab_size=100)
        def __init__(self):
            self.kwargs = []
            self.active = self.max_active = 0
            self.lock = threading.Lock()
            self.fail = False
        def generate(self, **kwargs):
            self.kwargs.append(kwargs)
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.005)
                if self.fail:
                    raise RuntimeError("generation failed")
                return torch.cat([kwargs["input_ids"], torch.tensor([[4, 9]])], dim=1)
            finally:
                with self.lock:
                    self.active -= 1
    class Inference:
        model_id = "qwen3-next-80B"
        device = "cpu"
        tokenizer = None
        def __init__(self):
            self.model = Model()
            self.paths = []
        def DiskCache(self, cache_dir):
            path = Path(cache_dir)
            self.paths.append(path)
            (path / "marker").write_text("cache", encoding="utf-8")
            return object()
    return Inference()


def test_backend_uses_generation_and_slices_prompt(inference):
    backend = InferenceBackend(inference, adapter=Adapter())
    assert backend.generate([], []).content == "done"
    args = inference.model.kwargs[0]
    assert args["eos_token_id"] == [9]
    assert args["do_sample"] is False
    assert "temperature" not in args and "past_key_values" not in args
    assert args["attention_mask"].tolist() == [[1, 1, 1]]


def test_sample_settings(inference):
    generation = GenerationConfig(temperature=0.7, top_p=0.8, top_k=10, seed=7)
    InferenceBackend(inference, adapter=Adapter(), generation=generation).generate([], [])
    kwargs = inference.model.kwargs[0]
    assert kwargs["do_sample"] and kwargs["temperature"] == 0.7 and kwargs["top_k"] == 10


def test_fresh_disk_cache_isolated_and_cleaned(inference, tmp_path):
    sentinel = tmp_path / "unrelated"
    sentinel.write_text("keep", encoding="utf-8")
    backend = InferenceBackend(inference, adapter=Adapter(), cache_dir=tmp_path)
    backend.generate([], [])
    backend.generate([], [])
    assert len(set(inference.paths)) == 2
    assert all(not path.exists() for path in inference.paths)
    assert sentinel.read_text() == "keep"
    assert all("past_key_values" in x for x in inference.model.kwargs)


def test_disk_cache_cleaned_on_failure(inference, tmp_path):
    inference.model.fail = True
    with pytest.raises(RuntimeError):
        InferenceBackend(inference, adapter=Adapter(), cache_dir=tmp_path).generate([], [])
    assert list(tmp_path.iterdir()) == []


def test_gpt_disk_cache_rejected_without_importing_harmony(inference):
    adapter = Adapter()
    adapter.family = "gpt-oss"
    with pytest.raises(ValueError, match="DiskCache"):
        InferenceBackend(inference, adapter=adapter, cache_dir="somewhere")


def test_context_limit_prevents_generation(inference):
    config = GenerationConfig(max_new_tokens=10, max_context_tokens=12)
    with pytest.raises(ValueError, match="context"):
        InferenceBackend(inference, adapter=Adapter(), generation=config).generate([], [])
    assert inference.model.kwargs == []


def test_vocabulary_mismatch_prevents_generation(inference):
    inference.model.config.vocab_size = 8
    with pytest.raises(ValueError, match="vocabulary"):
        InferenceBackend(inference, adapter=Adapter()).generate([], [])


def test_generation_is_serialized(inference):
    backend = InferenceBackend(inference, adapter=Adapter())
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(backend.generate, [], []) for _ in range(2)]
        assert all(f.result().content == "done" for f in futures)
    assert inference.model.max_active == 1


@pytest.mark.parametrize("kwargs", [
    {"temperature": float("nan")}, {"temperature": float("inf")}, {"temperature": -1},
    {"top_p": 0}, {"max_new_tokens": 0}, {"max_new_tokens": 1.5},
    {"max_new_tokens": True}, {"repetition_penalty": 0}, {"max_context_tokens": -1},
    {"reasoning_effort": "minimal"}, {"top_k": -1},
])
def test_invalid_generation_settings(kwargs):
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


def test_mapping_translation_and_strict_unknowns():
    config = config_from_mapping(GenerationConfig(), {"max_tokens": 100, "lang": "en", "max_input_tokens": 2000})
    assert config.max_new_tokens == 100 and config.max_context_tokens == 2000
    with pytest.raises(ValueError):
        config_from_mapping(config, {"stop": ["stop"]})
    with pytest.raises(ValueError):
        config_from_mapping(config, {"max_tokens": 1, "max_new_tokens": 2})

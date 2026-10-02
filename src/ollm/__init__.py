"""oLLM public API; heavy inference dependencies are loaded on first use."""
from importlib import import_module

__all__ = ["Inference", "AutoInference", "file_get_contents", "TextStreamer"]


def __getattr__(name):
    modules = {
        "Inference": ".inference", "AutoInference": ".inference",
        "file_get_contents": ".utils", "TextStreamer": "transformers",
    }
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

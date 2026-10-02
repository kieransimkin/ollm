"""Shared model-loading options; examples require an explicit opt-in to download."""
import argparse
import json
from pathlib import Path


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", choices=["qwen3-next-80B", "gpt-oss-20B"], default="qwen3-next-80B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--download", action="store_true", help="Allow downloading missing model weights (can be very large)")
    p.add_argument("--kv-cache-dir", help="Qwen only: parent for temporary per-turn disk caches")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-context-tokens", type=int)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="low")
    p.add_argument("--thinking", action="store_true", help="Qwen: enable thinking if supported by the model/template")
    p.add_argument("--prompt", default="Use the tools to calculate (17 + 25) * 3, then give the answer.")
    p.add_argument("--transcript", type=Path, help="Save the full transcript; may contain sensitive tool data and model reasoning")
    return p


def backend(args):
    from ollm import Inference
    from ollm.tools import GenerationConfig, InferenceBackend
    if args.kv_cache_dir and args.model == "gpt-oss-20B":
        raise ValueError("gpt-oss DiskCache is not supported upstream; omit --kv-cache-dir")
    if not (Path(args.models_dir) / args.model).is_dir() and not args.download:
        raise FileNotFoundError("Model directory is missing. Supply --models-dir or explicitly add --download.")
    inference = Inference(args.model, device=args.device, logging=False)
    inference.ini_model(models_dir=args.models_dir, force_download=False)
    return InferenceBackend(inference, cache_dir=args.kv_cache_dir, generation=GenerationConfig(
        max_new_tokens=args.max_new_tokens, max_context_tokens=args.max_context_tokens,
        temperature=args.temperature, reasoning_effort=args.reasoning_effort, enable_thinking=args.thinking))


def print_result(result, args):
    print(result.text)
    print(f"[rounds={result.rounds}, tool_calls={result.tool_calls}]")
    if args.transcript:
        args.transcript.write_text(json.dumps(result.messages, indent=2, ensure_ascii=False), encoding="utf-8")


def print_qwen_result(response, args):
    for message in reversed(response):
        if message.get("role") == "assistant" and not message.get("function_call"):
            print(message.get("content", ""))
            break
    if args.transcript:
        args.transcript.write_text(json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")


NUMBER_PAIR = {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
               "required": ["a", "b"], "additionalProperties": False}

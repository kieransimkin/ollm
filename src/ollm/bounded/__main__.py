"""List, inspect, run and measure local bounded model candidates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
from .api import BudgetInference
from .budget import MemoryBudget
from .config import MODELS
from .model import BoundedModel, load_bounded_model
from .qualification import benchmark, benchmark_multimodal


def main(argv=None):
    parser = argparse.ArgumentParser(description='oLLM bounded text inference; no model is pre-certified below 8 GB')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='List implemented candidate checkpoints and their qualification status')
    for command in ('inspect', 'generate', 'qualify', 'qualify-vl'):
        p = sub.add_parser(command)
        p.add_argument('model_dir', type=Path)
        p.add_argument('--model-key', choices=sorted(MODELS))
        p.add_argument('--device', default='cuda:0')
        p.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
        p.add_argument('--context', type=int, default=4096)
        p.add_argument('--max-new-tokens', type=int, default=128)
        p.add_argument('--prefill-chunk', type=int, default=64)
        p.add_argument('--attention-block', type=int, default=256)
        p.add_argument('--cache-dir', type=Path)
        p.add_argument('--budget-file', type=Path, help='Exact MemoryBudget JSON; overrides the four size/chunk flags')
        p.add_argument('--download', action='store_true', help='Explicitly download the registered checkpoint into model_dir')
        p.add_argument('--revision', help='Optional immutable HF revision for a requested download')
        if command == 'generate':
            p.add_argument('--prompt', default='Explain why expert streaming reduces GPU memory requirements.')
            p.add_argument('--image', type=Path, action='append', help='Local image; repeat for multiple images')
            p.add_argument('--allow-unqualified', action='store_true')
            p.add_argument('--memory-profile', type=Path)
            p.add_argument('--temperature', type=float, default=0.)
        elif command in ('qualify', 'qualify-vl'):
            if command == 'qualify':
                p.add_argument('--prompt-tokens', type=int, default=1024)
            p.add_argument('--rounds', type=int, default=3)
            p.add_argument('--report', type=Path, required=True)
            p.add_argument('--hash-weights', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'list':
        for key, spec in MODELS.items():
            print(f'{key:40s} {spec.family:15s} {spec.tool_format:12s} qualification={spec.qualification}')
        return 0
    if args.download:
        if not args.model_key:
            parser.error('--download requires --model-key')
        from huggingface_hub import snapshot_download
        snapshot_download(MODELS[args.model_key].repo_id, revision=args.revision,
                          local_dir=str(args.model_dir), allow_patterns=['*.json', '*.jinja', '*.safetensors'])
    from .checkpoint import read_json
    budget = (MemoryBudget(**read_json(args.budget_file)) if args.budget_file else MemoryBudget(
        max_context_tokens=args.context, max_output_tokens=args.max_new_tokens,
        prefill_tokens=args.prefill_chunk, attention_block_tokens=args.attention_block))
    options = dict(device=args.device, dtype=getattr(torch, args.dtype), budget=budget)
    if args.command == 'generate':
        o = BudgetInference(args.model_dir, model_key=args.model_key, cache_dir=args.cache_dir,
                            allow_unqualified=args.allow_unqualified, memory_profile=args.memory_profile, **options)
        content = args.prompt
        if args.image:
            content = ([{'type': 'image', 'image': str(path)} for path in args.image] +
                       [{'type': 'text', 'text': args.prompt}])
        print(o.generate([{'role': 'user', 'content': content}], max_new_tokens=budget.max_output_tokens,
                         temperature=args.temperature))
        print(json.dumps(o.model.last_report, indent=2), file=sys.stderr)
    else:
        if args.command == 'inspect':
            options['device'] = 'cpu'
        model = load_bounded_model(args.model_dir, expected_family=MODELS[args.model_key].family if args.model_key else None,
                                   cache_root=args.cache_dir, **options)
        if args.command == 'inspect':
            print(json.dumps(dict(family=model.config.family, text_prefix=model.prefix,
                 checkpoint_tensors=len(model.store.tensors), validated_tensors=len(model.validated_names),
                 prefill_tokens=model.prefill_tokens, budget=budget.as_dict(), memory_qualified=False,
                 layout_digest=model.store.layout_fingerprint), indent=2))
        else:
            if args.command == 'qualify-vl':
                report = benchmark_multimodal(model, rounds=args.rounds, report_path=args.report,
                                              hash_weights=args.hash_weights)
            else:
                report = benchmark(model, prompt_tokens=args.prompt_tokens, rounds=args.rounds,
                                   report_path=args.report, hash_weights=args.hash_weights)
            print(json.dumps({k:v for k,v in report.items() if k not in ('driver_trace','cases')}, indent=2))
            return 0 if report['memory_pass'] else 2
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'ollm bounded: {exc}', file=sys.stderr)
        raise SystemExit(2)

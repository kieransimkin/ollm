"""Shared explicit local-checkpoint arguments; --help does not load any SDK/model."""
import argparse
import json
from pathlib import Path


def parser(description):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('model_dir',type=Path)
    p.add_argument('--model-key',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dtype',choices=['bfloat16','float16','float32'],default='bfloat16')
    p.add_argument('--budget-file',type=Path)
    p.add_argument('--context',type=int,default=4096)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--memory-profile',type=Path)
    p.add_argument('--allow-unqualified',action='store_true')
    p.add_argument('--prompt',default='Use the tools to add 17 and 25, then multiply the result by 3.')
    return p


def inference(args):
    import torch
    from ollm import BudgetInference,MemoryBudget
    from ollm.bounded.checkpoint import read_json
    budget=(MemoryBudget(**read_json(args.budget_file)) if args.budget_file else
            MemoryBudget(max_context_tokens=args.context,max_output_tokens=args.max_new_tokens))
    return BudgetInference(args.model_dir,model_key=args.model_key,device=args.device,
        dtype=getattr(torch,args.dtype),budget=budget,cache_dir=args.cache_dir,
        memory_profile=args.memory_profile,allow_unqualified=args.allow_unqualified)


def backend(args):
    from ollm.tools import InferenceBackend
    return InferenceBackend(inference(args))


NUMBER_PAIR={'type':'object','properties':{'a':{'type':'number'},'b':{'type':'number'}},
             'required':['a','b'],'additionalProperties':False}

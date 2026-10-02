"""Public bounded inference facade compatible with ollm.tools.InferenceBackend."""
from __future__ import annotations
from pathlib import Path
import torch

from .budget import MemoryBudget
from .checkpoint import read_json
from .config import MODELS
from .model import BoundedModel, CacheLocation


def local_tokenizer(root):
    """Load the fast tokenizer file, not an architecture-specific model class.

    This keeps the legacy Transformers model pin intact. No remote code,
    AutoConfig, AutoModel, or network lookup is used for newer architectures.
    """
    try:
        from transformers import PreTrainedTokenizerFast
    except ImportError as exc:
        raise ImportError("Tokenization requires Transformers; numeric-token CPU tests do not") from exc
    root = Path(root)
    if not (root / 'tokenizer.json').is_file():
        raise ValueError("A local tokenizer.json is required; remote tokenizer code is not executed")
    settings = read_json(root / 'tokenizer_config.json') if (root / 'tokenizer_config.json').exists() else {}
    special = {}
    for key in ('bos_token', 'eos_token', 'pad_token', 'unk_token'):
        value = settings.get(key)
        if isinstance(value, dict):
            value = value.get('content')
        if isinstance(value, str):
            special[key] = value
    tok = PreTrainedTokenizerFast(tokenizer_file=str(root / 'tokenizer.json'), **special)
    template_file = root / 'chat_template.jinja'
    template = template_file.read_text('utf-8') if template_file.exists() else settings.get('chat_template')
    if isinstance(template, list):
        template = {x['name']: x['template'] for x in template}
    tok.chat_template = template
    config = read_json(root / 'config.json')
    text = config.get('text_config', config)
    for name in ('eos_token', 'bos_token'):
        value = text.get(name + '_id', config.get(name + '_id'))
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, int):
            token = tok.convert_ids_to_tokens(value)
            if token is None:
                raise ValueError(f"Tokenizer is missing configured {name}_id")
            setattr(tok, name, token)
    if not tok.chat_template:
        raise ValueError("No local chat template; provide explicit input token IDs or a tokenizer")
    return tok


class BudgetInference:
    """New memory-bounded path; the old Inference implementation is unchanged.

    Candidate checkpoints require allow_unqualified=True until a matching real
    CUDA memory profile has been produced. CPU executions are reference tests,
    never evidence of fitting in GPU memory.
    """
    bounded_runtime = True

    def __init__(self, model_dir, *, model_key=None, device='cuda:0', dtype=torch.bfloat16,
                 budget: MemoryBudget | None = None, cache_dir=None, tokenizer=None,
                 load_tokenizer=True, allow_unqualified=False, memory_profile=None,
                 tool_format=None):
        spec = MODELS.get(model_key) if model_key else None
        if model_key and spec is None:
            raise ValueError(f"Unknown model key {model_key!r}; list candidates with python -m ollm.bounded list")
        self.model_id = model_key or str(Path(model_dir).name)
        self.device = torch.device(device)
        self.budget = budget or MemoryBudget()
        self.spec = spec
        self.model = BoundedModel(model_dir, expected_family=spec.family if spec else None,
                                  budget=self.budget, device=self.device, dtype=dtype, cache_root=cache_dir)
        if memory_profile is not None:
            from .qualification import validate_profile
            validate_profile(memory_profile, self.model)
        elif not allow_unqualified:
            raise ValueError("This checkpoint/profile has not been GPU-qualified. Run the qualify command, "
                             "or explicitly set allow_unqualified=True for testing. No under-8-GB claim is implied.")
        self.memory_qualified = memory_profile is not None
        # Do not guess a tool protocol from a generic family. Local anonymous
        # checkpoints default to text-only unless the caller chooses explicitly.
        self.tool_format = tool_format or (spec.tool_format if spec else 'text')
        if self.tool_format not in ('text', 'qwen-json', 'qwen-coder'):
            raise ValueError("Unknown bounded tool protocol")
        self.default_thinking = spec.thinking if spec else None
        self.tokenizer = tokenizer if tokenizer is not None else (local_tokenizer(model_dir) if load_tokenizer else None)
        generation_path = Path(model_dir) / 'generation_config.json'
        generation = read_json(generation_path) if generation_path.exists() else {}
        stop_values = [generation.get('eos_token_id'), self.model.config.get('eos_token_id'),
                       self.model.config.raw.get('eos_token_id'), getattr(self.tokenizer, 'eos_token_id', None)]
        self.eos_token_ids = []
        for value in stop_values:
            for token_id in value if isinstance(value, list) else ([] if value is None else [value]):
                if type(token_id) is not int or not 0 <= token_id < self.model.config.vocab_size:
                    raise ValueError('Invalid checkpoint/tokenizer EOS token ID')
                if token_id not in self.eos_token_ids:
                    self.eos_token_ids.append(token_id)

    def DiskCache(self, cache_dir='./kv_cache'):
        # A descriptor: the runtime owns its isolated session and closes it even
        # on exceptions/cancellation. Does not delete a caller's existing cache.
        return CacheLocation(cache_dir)

    def generate(self, messages, *, max_new_tokens=128, temperature=0.0):
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for text generation")
        kwargs = {}
        if self.default_thinking is not None:
            kwargs['enable_thinking'] = self.default_thinking
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
        ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                  eos_token_id=self.eos_token_ids,
                                  do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0)
        return self.tokenizer.decode(out[0, ids.shape[1]:].cpu().tolist(), skip_special_tokens=False)

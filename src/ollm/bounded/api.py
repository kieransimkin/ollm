"""Public bounded inference facade compatible with ollm.tools.InferenceBackend."""
from __future__ import annotations
from pathlib import Path
import torch

from .budget import MemoryBudget
from .checkpoint import read_json
from .config import MODELS
from .model import BoundedModel, CacheLocation, load_bounded_model


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


def local_processor(root):
    """Load the installed Qwen3-VL processor from local metadata only."""
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise ImportError("Qwen3-VL processing requires Transformers >=4.57") from exc
    processor = AutoProcessor.from_pretrained(
        str(Path(root)), local_files_only=True, trust_remote_code=False, use_fast=False)
    if not hasattr(processor, 'tokenizer'):
        raise ValueError("Qwen3-VL processor did not expose a tokenizer")
    return processor


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
        self.model = load_bounded_model(model_dir, expected_family=spec.family if spec else None,
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
        self.processor = None
        if self.model.config.family == 'qwen3_vl' and load_tokenizer and tokenizer is None:
            self.processor = local_processor(model_dir)
            self.tokenizer = self.processor.tokenizer
        else:
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

    def _stop_ids(self):
        ids = []
        if self.tokenizer is not None:
            vocab = self.tokenizer.get_vocab()
            if '<|im_end|>' in vocab:
                ids.append(vocab['<|im_end|>'])
        ids.extend(self.eos_token_ids)
        return list(dict.fromkeys(ids))

    def prepare_model_inputs(self, messages, tools=None, *, enable_thinking=False):
        """Prepare structured Qwen3-VL image messages without remote fetches.

        Returns CPU processor tensors where possible; the bounded model stages
        only bounded image/weight chunks onto CUDA. Video blocks are rejected.
        """
        if self.model.config.family != 'qwen3_vl':
            raise ValueError("prepare_model_inputs is only used by the bounded Qwen3-VL path")
        if self.processor is None:
            raise ValueError("Qwen3-VL requires its local AutoProcessor")
        from ollm.tools.types import normalize_messages
        history = normalize_messages(messages)
        kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt')
        if tools:
            kwargs['tools'] = tools
        prepared = self.processor.apply_chat_template(history, **kwargs)
        if not isinstance(prepared, dict) and not hasattr(prepared, 'items'):
            raise ValueError("Qwen3-VL processor must return a tensor mapping")
        prepared = dict(prepared)
        if 'pixel_values_videos' in prepared or 'video_grid_thw' in prepared:
            raise ValueError("Video is not enabled in the bounded Qwen3-VL image profile")
        if 'input_ids' not in prepared:
            raise ValueError("Processor returned no input_ids")
        ids = prepared.pop('input_ids')
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("Only batch-one multimodal prompts are supported")
        attention = prepared.pop('attention_mask', None)
        if attention is not None and (attention.shape != ids.shape or not torch.all(attention == 1)):
            raise ValueError("Padding/packed multimodal prompts are not supported")
        allowed = {'pixel_values', 'image_grid_thw'}
        unknown = set(prepared) - allowed
        if unknown:
            raise ValueError("Unsupported Qwen3-VL processor outputs: " + ', '.join(sorted(unknown)))
        # Validate image bounds before any CUDA transfer. The model repeats this
        # validation at the execution boundary.
        if ('pixel_values' in prepared) != ('image_grid_thw' in prepared):
            raise ValueError("Processor must return pixel_values and image_grid_thw together")
        if 'image_grid_thw' in prepared:
            grid = prepared['image_grid_thw']
            if grid.ndim != 2 or grid.shape[1] != 3 or len(grid) > self.budget.max_images:
                raise ValueError("Processor image grid exceeds the image-count profile")
            visual = int(grid.to('cpu', dtype=torch.long).prod(-1).sum().item())
            if visual > self.budget.max_visual_tokens:
                raise ValueError(f"Processed image has {visual} vision patches; profile allows {self.budget.max_visual_tokens}")
        return ids, self._stop_ids(), prepared

    def generate(self, messages, *, max_new_tokens=128, temperature=0.0):
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required for text generation")
        if self.model.config.family == 'qwen3_vl':
            ids, stops, extra = self.prepare_model_inputs(messages)
            ids = ids.to(self.device)
            out = self.model.generate(ids, max_new_tokens=max_new_tokens, eos_token_id=stops,
                                      do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0,
                                      **extra)
        else:
            kwargs = {}
            if self.default_thinking is not None:
                kwargs['enable_thinking'] = self.default_thinking
            raw = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
            ids = torch.tensor([raw], dtype=torch.long, device=self.device)
            out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                      eos_token_id=self.eos_token_ids,
                                      do_sample=temperature > 0, temperature=temperature if temperature > 0 else 1.0)
        return self.tokenizer.decode(out[0, ids.shape[1]:].cpu().tolist(), skip_special_tokens=False)

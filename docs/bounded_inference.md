# Bounded runtime: architecture and API

[Overview](../BOUNDED.md) · [Models](bounded_models.md) · [Qualification](bounded_qualification.md)

## Contract and non-goals

The hardened path is explicit: `ollm.BudgetInference`. It supports one unpadded text
sequence, inference only, CPU reference runs or a single CUDA device, and BF16/FP16
CUDA arithmetic. FP32 is restricted to CPU reference work. Checkpoint weights stay
on disk; only bounded pieces and activations are resident. All checkpoints remain
unqualified until the user measures a matching profile.

This is a new functional decoder, not a thin wrapper around Transformers generation.
There is no ordinary-model fallback, beam search, batching, training, PEFT, speculative
/MTP decoding, cross-turn prefix reuse, distributed inference, custom attention masks,
or automatic checkpoint-code execution. Qwen3-VL-2B has a separately implemented
image-only multimodal path; video and other multimodal architectures remain unsupported.
Unsupported options fail explicitly. Legacy `Inference` remains available separately.

## Shared memory infrastructure

### Checkpoint reading and linear operations

`TensorStore` reads safetensor metadata and bounded byte ranges, accepting one
`model.safetensors` or an indexed set of shards. It checks duplicate JSON keys,
paths, file lengths, tensor shapes, byte ranges, overlapping storage and index
consistency. Shards escaping the model directory, including symlink targets outside
it, are rejected. Use an actual local-directory download, not symlinked HF cache
snapshots, or materialize those symlinks first.

`StreamOps.linear` preallocates the bounded activation output, then reads contiguous
output rows and performs each partial linear projection. Conversion size, rather
than only compressed disk bytes, limits the row count. There is no GPU module
holding all layer or expert weights. Embeddings read only requested token rows.
The output head projects only the final required hidden token, tiled over vocabulary
rows; it does not create a prompt-length by vocabulary logits tensor.

Separately named experts and supported packed `[experts, output, input]` gate/up/down
layouts are validated explicitly. A packed expert axis is sliced before reading;
loading a whole expert bank is not necessary. Unsupported layouts fail rather than
using `ignore_mismatched_sizes` or silently dropping parameters.

DeepSeek FP8 support accepts block-scaled E4M3 two-dimensional tensors plus matching
`weight_scale_inv` tensors with 128 by 128 blocks. It reads/converts a row tile on CPU
and transfers the requested BF16/FP16 result. Scales crossing row/block boundaries are
handled. This is **weight-dequantized BF16/FP16 inference**, not native FP8 activation
kernels or recovery of unavailable original precision. AWQ, GPTQ, GGUF, MXFP4 and
other weight formats are outside this runtime. GPT-OSS stays on the legacy path.

### Cache behavior

`DiskState` creates one unpredictable private session subdirectory inside the chosen
cache root. Attention keys/values or MLA latent/position vectors are append-only raw
binary streams. Both prompt tokens and generated tokens go to disk, with bounded
block reads: there is no indefinitely growing generated-token GPU tail and no whole
layer-cache reconstruction. Numeric layer identifiers are dictionary keys, so sparse
hybrid attention layers do not depend on contiguous cache-array positions.

Convolution history and FP32 recurrent state are stored separately, atomically
replaced, and loaded only for the layer being processed. The history stores raw
pre-convolution inputs. Disk quota checks account for temporary replacement space.
On success or an exception the session removes only its own directory, never an
existing user's cache folder. Crashed processes can leave session directories;
remove those only after confirming their owning process has stopped. Disk-cache
files contain sensitive input-derived data and are not encrypted by this package.

Every generation creates fresh state. Each agent tool round rebuilds its history;
this costs prefill time but avoids claiming unsafe prefix reuse. Low-level
`forward_tokens` requires the exact next session position, including pure-linear
hybrid configurations. Sessions are not persistent resume files.

### Attention and activations

Ordinary causal/GQA attention uses bounded key/value tiles and numerically stable
online softmax, processing heads without repeating the full KV cache. Sliding-window
visibility preserves absolute positions. The implementation does not materialize an
unbounded square attention matrix or silently select an eager whole-context fallback.

Prefill chunks adapt downward to the configured workspace and architecture widths.
MLP token chunks are bounded separately; residuals and projection outputs are
included in conservative planning. The portable implementation uses PyTorch tensor
operations, not the legacy kvikio/FlashAttention fast path. Sequential expert/head
loops favor predictable residency over performance.

### Hybrid Qwen

Next uses its native packed, key-head-grouped QKVZ/BA ordering; Qwen3.5 uses separate
QKV, Z, beta and decay projections. Both use causal depthwise convolution, L2-normalized
Q/K, FP32 gated-DeltaNet recurrence, gated RMS normalization and the appropriate
output projection. The three ordinary text positions used by Qwen3.5 MRoPE are equal,
so the text-only computation reduces to text RoPE. Zero-centered normalization and
partial RoPE are retained. Supported sigmoid and Swish attention output gates are
explicitly distinguished.

Layer types/configuration and tensors, not the marketing name, choose execution.
Qwen3-VL has an explicit vision backend described in [qwen3_vl.md](qwen3_vl.md);
other vision placeholders and separate MTP tensors remain rejected. Unknown tensors
in executed decoder/vision layers are rejected. Configured DeepSeek appended MTP
layers are recognized as unused, not silently treated as regular layers.

### DeepSeek

V2/V3 use native MLA. Persistent state contains the normalized compressed latent and
rotary positional component only. Per-head weight absorption moves the non-positional
key and value projections into query/output operations. Tiled online attention consumes
compact cache blocks; it does not retain expanded head-count K/V tensors.

Dense initial layers, routed experts, shared experts, grouped routing, correction bias,
normalization, routing scale and DeepSeek YaRN scaling are preserved. Routing stores
`[prefill_chunk, experts]` scores and `[prefill_chunk, selected_experts]` indices rather
than an enormous token/expert one-hot mask. Experts are executed sequentially and
weighted outputs accumulated. There is no expert prediction, expert dropping or
approximate pruning.

MLA comparisons exercise equivalence against explicitly expanded FP32 attention on
tiny checkpoints, including YaRN. Different floating-point operation order means
BF16/FP16 results are not promised bitwise identical to an expanded or native FP8
implementation. Full-model quality still needs pretrained/reference validation.

## Public API

```python
import torch
from ollm import BudgetInference, MemoryBudget

budget = MemoryBudget(max_context_tokens=4096, max_output_tokens=128)
o = BudgetInference(
    './models/qwen3-8b', model_key='qwen3-8b',
    device='cuda:0', dtype=torch.bfloat16, budget=budget,
    cache_dir='./runtime-cache',
    allow_unqualified=True,  # explicit development opt-in, not certification
)
print(o.generate([{'role': 'user', 'content': 'Explain grouped attention.'}],
                 max_new_tokens=128))
print(o.model.last_report)
```

`BudgetInference` arguments:

| Argument | Meaning |
|---|---|
| `model_dir` | Local config, tokenizer and safetensors directory. No download. |
| `model_key` | Optional registry entry; enforces architecture family and selects a tool protocol/thinking default. The actual local tensor layout is always checked. A registry name is not proof of repository provenance. |
| `device`, `dtype` | `cuda:N` with BF16/FP16, or CPU reference arithmetic. |
| `budget` | A `MemoryBudget`; see defaults below. |
| `cache_dir` | Parent for automatically isolated disk sessions. |
| `allow_unqualified` | Default false. Explicit testing bypass, with no performance assertion. |
| `memory_profile` | Successful matching local CUDA profile instead of the bypass. |
| `tokenizer` | Optional already loaded compatible tokenizer. |
| `load_tokenizer=False` | Numeric-token/low-level testing without Transformers. |
| `tool_format` | Explicit `text`, `qwen-json`, or `qwen-coder` override. Anonymous checkpoints default to text. |

Local tokenization uses `tokenizer.json`, its special-token configuration and the
checkpoint chat template with `PreTrainedTokenizerFast`. It does not call AutoConfig,
load model classes, run `trust_remote_code`, or fetch anything. The framework's legacy
Transformers pin is retained. Actual newer tokenizer compatibility needs on-machine
verification; numeric CPU tests do not substitute for that check.

`o.model.generate(input_ids=...)` accepts a `[1,T]` int64 tensor, `max_new_tokens`,
EOS ID/list, an all-ones mask, greedy or temperature/top-p/top-k sampling and repetition
penalty. It returns the complete prompt-plus-output token tensor. Only requested
positions are projected to logits. Other generation features fail explicitly.
`o.DiskCache(path)` is a location descriptor for existing tool-layer compatibility,
not a reusable live KV state object.

## Budget fields

| Field | Default |
|---|---:|
| `vram_bytes` | 7,000,000,000 decimal bytes |
| `workspace_bytes` | 134,217,728 bytes (128 MiB) |
| `weight_tile_bytes` | 33,554,432 bytes (32 MiB) |
| `safety_bytes` | 268,435,456 bytes (256 MiB) |
| `prefill_tokens` | 64, further reduced by architecture planning |
| `attention_block_tokens` | 256 |
| `max_context_tokens` | 32,768 in the API; 4,096 in CLI/examples |
| `max_output_tokens` | 512 in the API; 128 in CLI/examples |
| `max_cache_bytes` | 137,438,953,472 bytes (128 GiB) |

Every field is a positive integer. Budgets at or above 8,000,000,000 bytes are rejected.
A budget file is an exact set of constructor fields; CLI `--budget-file` overrides its
context/output/chunk flags. Larger context limits never inherit a smaller-profile
result. The model's own configured context limit is also enforced; unsupported dynamic
RoPE modes are rejected rather than changing an already cached prefix.

`BudgetGuard` considers device-wide used memory, PyTorch reusable reserved blocks and
an explicit safety allowance before bounded operations. It raises `MemoryBudgetError`
when the planned operation cannot fit and translates CUDA OOM into a failed profile.
It is **not an OS-level reservation or a proof of a device-wide hard cap**. Other
processes, allocator fragmentation and library workspaces can invalidate a plan.
Do not remove safeguards or mark a CPU run as a CUDA qualification to work around this.

## Operational and performance limitations

Disk capacity and traffic can dominate. Full V3/R1 retain their large checkpoint and
may reread selected experts and compact cache blocks many times. This initial MLA path
also rereads cache blocks per head. No throughput figures for real checkpoints were
measured here. Use reported bytes read, prefill time, generated-token count, host peak
RSS (where available) and elapsed time to judge suitability, not parameter counts alone.

Generation is serialized inside a process. Different processes are not coordinated.
There is no hard cancellation/sandbox boundary, no transactional tool rollback and no
automatic retry after a tool side effect. Existing MCP security and approval guidance
still applies. Qwen-Agent owns execution when that route is selected; native registry
limits are not automatically inherited by it.

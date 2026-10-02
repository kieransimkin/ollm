# Implemented model candidates and qualification status

[Overview](../BOUNDED.md) · [Runtime](bounded_inference.md) · [Qualification](bounded_qualification.md)

**All entries below are implemented but pretrained/GPU-unqualified in this patch.**
No model is advertised as measured below 8 GB. `python -m ollm.bounded list` prints
exact keys. Support means there is a native memory-bounded execution path, not a
fallback into an ordinary full-model loader.

## Implementation sequence

| Stage | Backend / checkpoints | Memory implementation | Tool adapter |
|---|---|---|---|
| 1 | Qwen3 dense 4B, 4B-Instruct-2507, 8B, 14B, 32B | Row-streamed weights, Q/K normalization, tiled GQA, disk KV, bounded MLP/prefill | JSON tool calls |
| 1 | Next-80B-A3B Instruct/Thinking; Qwen3-Coder-Next | Native hybrid attention/DeltaNet, routed/shared expert streaming, sparse-layer disk states | JSON for Next; parameter tags for Coder-Next |
| 2 | Qwen3-30B-A3B; Instruct/Thinking-2507; Coder-30B-A3B-Instruct | Native Qwen3 MoE with bounded routing and individual experts | JSON or Coder tags, checkpoint-specific |
| 2 | Qwen2.5 and Qwen2.5-Coder 1.5B/7B/14B/32B Instruct | Native biased Q/K/V, rotary/GQA, tiled dense execution | JSON |
| 2 | R1-Distill-Qwen 1.5B/7B/14B/32B, R1-Distill-Llama-8B, R1-0528-Qwen3-8B | Respective Qwen2/Llama/Qwen3 mathematical backend | Text only |
| 3 | Qwen3.5 4B/9B/27B text path, 35B-A3B text MoE path | Native separate hybrid projections, text MRoPE, zero-centered norm, tiled weights and state | Parameter tags |
| 4 | DeepSeek-V2-Lite-Chat; Coder-V2-Lite-Instruct | Compact MLA, dense/routed/shared MLPs, grouped routing, disk cache | Text only |
| 4 | DeepSeek-V3; R1; R1-0528 | Compact MLA plus V3 noaux routing and bounded official FP8 conversion | Text only; experimental large-checkpoint workloads |

The 14B/32B Qwen qualification requirement is implemented as a CUDA report workflow,
not fulfilled by invented figures. Run each model separately with explicit budgets.
The same is true of 27B/35B and full DeepSeek: implementation is not a hardware result.

## Larger or newer compatible variants

A local checkpoint can use `BudgetInference(model_dir, ...)` without a registry key.
The explicit `model_type`, text config and required tensors select one of the allowlisted
backends; unknown families are refused. This admits compatible same-architecture
variants for testing without incorrectly calling every new Qwen a Next alias.

For Qwen3.5-family text checkpoints, both sigmoid and Swish output gates are
implemented, including the documented Swish gate used by Qwen3.8-27B. That model still
requires exact local config/tensor/tokenizer checks and its own reference/GPU runs.
For anonymous checkpoints specify `tool_format='qwen-coder'` only after verifying the
actual chat template uses the parameter-tag format. New aliases are not evidence of
qualification. Config fields or tensor layouts outside the implemented semantics
must be ported rather than bypassing validation.

Other multimodal Qwen variants, Qwen4-exp/Flash-Next, DeepSeek V3.2's sparse indexer, DeepSeek
V4-family changes and GPT-OSS are **not** added to this new decoder. Existing GPT-OSS
inference/tools remain unchanged. Unsupported architectures, quantization modes,
unknown executed-layer weights and distributed-shard assumptions fail explicitly.

## What to verify for every real checkpoint

1. Match immutable revision and local files; inspect shapes without executing model code.
2. Run reference/numerical tests, including prompt chunks versus incremental decoding.
3. Check the real tokenizer and assistant/tool handoff framing for that revision.
4. Measure cold execution, sustained output, and rebuilt long histories on the actual GPU.
5. Record storage reads, peak host RSS, elapsed time and disk-cache space along with VRAM.

The registry checks the family; it does not claim a 4B alias proves the local weights
came from that particular repository. For reproducibility retain the source revision
and use `qualify --hash-weights` to bind a report to full checkpoint bytes as well as
configuration and safetensor layout.

Distillations are not the native DeepSeek flagship architecture. They remain explicitly
identified as Qwen/Llama backbones in the registry. Native V2/V3 are separately implemented.


## Qwen3-VL

| Key | Architecture | Scope | Qualification |
| --- | --- | --- | --- |
| `qwen3-vl-2b-instruct` | Qwen3-VL dense text + 24-layer ViT | text + local images; video rejected | required (`qualify-vl`) |

The image backend validates and executes the vision checkpoint rather than
ignoring the tower. Patch embedding, learned position interpolation, visual
RoPE, ViT blocks, mergers, DeepStack injection and interleaved MRoPE all use
the bounded runtime. See [qwen3_vl.md](qwen3_vl.md).

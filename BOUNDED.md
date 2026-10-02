# Bounded-memory Qwen and DeepSeek inference

This extension adds **real tiled text-inference implementations**, not model aliases
that fall back to `AutoModel` or `device_map="auto"`. It targets the fork at
`b395639e7397573728c7c4a1f82565ef1cb542c4`.

**Important status:** the backends have CPU numerical/contract tests. No pretrained
checkpoint has been run or certified below 8 GB on a GPU during development of this
patch. All registry entries are **unqualified candidates**. The normal public API
refuses to run them without either an explicit experimental opt-in or a matching
locally measured CUDA memory profile. Memory qualification is not model-quality or
native-tool-quality qualification.

## What changed

The new `BudgetInference` path has a shared byte-based memory budget, row-tiled
safetensor reading, disk-backed attention state, chunked prefill/MLPs, online
softmax attention, streamed expert execution, and last-token-only output projection.
It implements Qwen2, Qwen3 dense/MoE, Qwen3-Next, Qwen3.5 text dense/MoE, Llama
for compatible distillations, and native DeepSeek V2/V3. Native DeepSeek keeps the
compressed MLA cache; official block-scaled FP8 weights are converted a bounded
tile at a time. Full V3/R1 are experimental, potentially extremely slow targets.

Coder-Next, Coder-30B and Qwen3.5 have a schema-aware parameter-tag tool parser.
The existing native Python/MCP loop and Qwen-Agent provider accept this new runtime.
DeepSeek and R1 distillations are text-only at the tool-adapter boundary: offering
native functions to those checkpoints is explicitly rejected, not guessed.

Legacy `Inference`, its accelerated loaders, and its original cache classes remain
unchanged. Their older memory limitations are not retroactively fixed by this
extension. Use `BudgetInference` or the commands below for the hardened path.

## Start from a working oLLM environment

```bash
python -m pip install --no-build-isolation -e '.[agents,test-bounded]'
python -m ollm.bounded list

# No CUDA/model execution: inspect a local checkpoint's config and tensor headers.
python -m ollm.bounded inspect ./models/qwen3-4b --model-key qwen3-4b

# Explicit experimental execution. This does NOT assert measured 8-GB support.
python -m ollm.bounded generate ./models/qwen3-4b --model-key qwen3-4b \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified \
  --prompt 'Explain how a model can be larger than GPU memory.'
```

Checkpoints are never downloaded implicitly. For an intentional download, add
`--download --model-key KEY`, optionally `--revision COMMIT`, to `inspect`, `generate`
or `qualify`. That can fetch very large files; confirm the chosen checkpoint and
available SSD capacity first. Downloads are into the specified local directory.

## Measure before claiming support

```bash
python -m ollm.bounded qualify ./models/qwen3-4b --model-key qwen3-4b \
  --budget-file examples/bounded/budget-7gb.json \
  --prompt-tokens 1024 --rounds 3 --report ./profiles/qwen3-4b.json

# Accepts only a successful matching CUDA profile. No experimental bypass.
python -m ollm.bounded generate ./models/qwen3-4b --model-key qwen3-4b \
  --budget-file examples/bounded/budget-7gb.json \
  --memory-profile ./profiles/qwen3-4b.json --prompt 'What is expert streaming?'
```

The sample budget is **7,000,000,000 decimal bytes**, below both 8 GB and 8 GiB.
It is a requested ceiling, not a fabricated measurement. The profiler also checks
sampled device-wide usage and PyTorch allocator peaks. Another process or an
unobserved driver allocation can affect the device; read the measurement limits.

## Documentation

- [Runtime, memory policy, API and limitations](docs/bounded_inference.md)
- [Model matrix and implementation order](docs/bounded_models.md)
- [Numerical tests and CUDA qualification](docs/bounded_qualification.md)
- [Runnable Python, MCP and Qwen-Agent examples](examples/bounded/README.md)
- [Primary architecture references](docs/bounded_sources.md)

The new execution path is conservative portable PyTorch, not a throughput claim.
It deliberately does not depend on new Transformers model classes or change the
legacy Transformers pin. Long contexts and hundreds-of-billion-parameter models
can be dominated by repeated SSD reads even when their VRAM working set is small.

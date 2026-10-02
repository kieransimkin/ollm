# Testing and real CUDA memory qualification

[Overview](../BOUNDED.md) · [Models](bounded_models.md) · [Examples](../examples/bounded/README.md)

## Three independent questions

**Does the implementation perform the intended mathematics?** CPU tests compare
bounded attention with ordinary attention matrices, compact MLA with expanded K/V,
hybrid recurrence with independent matrix transitions, dense decoders with an eager
reference, and chunked/incremental executions. Real tiny safetensor checkpoints are
used for all nine implemented architectures, including BF16 generation and FP8 tile
conversion. These tests are not pretrained-quality results.

**Does the real checkpoint fit the GPU profile?** The CUDA measurement runner exercises
explicit prompt/output lengths and checks device-wide and allocator memory. CPU runs
always have `memory_pass=false`. No passing GPU report ships with this patch.

**Does the model use tools reliably?** Grammar tests reject incomplete/ambiguous calls;
they do not measure whether a pretrained model chooses the right tool. Run the native
and Qwen-Agent examples with actual checkpoints and assess tool behavior independently.
DeepSeek native tool use is intentionally not advertised by this patch.

## CPU tests

From a checkout with the required CPU dependencies (does not download checkpoints):

```bash
python -m pip install 'torch>2.6' --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'pytest>=8,<10' 'safetensors>=0.5,<1' 'numpy>=1.26' 'jsonschema>=4.23,<5'
PYTHONPATH=src python -m pytest -c pytest-bounded.ini -q
PYTHONPATH=src python -m pytest -c pytest-tools.ini -m 'not sdk and not model' -q
```

On PowerShell, set `$env:PYTHONPATH='src'` before the two pytest commands. The bounded
pytest config also sets its source path, but the legacy suite needs that environment
variable when the package is not installed.

Optional real Transformers-class comparisons (still tiny local weights):

```bash
OLLM_RUN_REFERENCE=1 PYTHONPATH=src \
  python -m pytest -c pytest-bounded.ini -m reference -q -rs
```

These nine tests require a Transformers version containing each native class. Missing
classes are explicit skips. The project retains its legacy `transformers==4.57.0` pin;
newer hybrid class comparisons should run in a **separate** reference environment rather
than silently upgrading the production legacy loaders. Optional SDK suites from the
previous tools patch remain in `tests/tools`; their status must be reported separately.

`.github/workflows/bounded-tests.yml` runs CPU tests without full checkpoint weights.
An explicit workflow-dispatch option runs available native-class comparisons. CI has
not been executed remotely during patch creation. Do not run untrusted PR code on a
self-hosted GPU or automatically download hundreds of gigabytes in CI.

## Checkpoint preparation and inspection

```bash
# Explicit download example. Add --revision with the resolved commit for reproducibility.
python -m ollm.bounded inspect ./models/qwen3-4b --model-key qwen3-4b --download

# Existing local checkpoint; metadata inspection does not require CUDA.
python -m ollm.bounded inspect ./models/deepseek-v2-lite \
  --model-key deepseek-v2-lite-chat
```

Inspection validates all required executed-layer tensors but does not prove their
values are trained/correct. It reports `memory_qualified=false`. Full model downloads,
particularly V3/R1, require appropriate storage and may not be practical for every user.

## CUDA measurement

```bash
python -m ollm.bounded qualify ./models/qwen3-8b --model-key qwen3-8b \
  --device cuda:0 --dtype bfloat16 --budget-file examples/bounded/budget-7gb.json \
  --prompt-tokens 1024 --rounds 3 --report ./profiles/qwen3-8b.json
```

The budget file fixes every operating limit. The last rebuilt-history round reaches
`max_context_tokens - max_output_tokens`; EOS is disabled so all output steps are
exercised. There must be at least two rounds. Prompt lengths are synthetic valid text
token IDs, not claims of actual semantic agent conversations. Larger profiles take
substantially more computation and I/O; no completion-time estimate is implied.

Run the same command with separate paths/keys for Qwen3 4B, 14B, 32B, Next Thinking,
Coder-Next, 30B-A3B, Qwen2.5, Qwen3.5, and native DeepSeek. Begin the latter with
`deepseek-v2-lite-chat` or `deepseek-coder-v2-lite`. Full V3/R1 remain experimental.
No result for one size, architecture or quantization scheme transfers to another.

`--hash-weights` additionally streams all checkpoint bytes to compute a payload hash.
This adds I/O; using such a profile checks the payload again. Without it, identity binds
config and safetensor layout/size, not arbitrary same-shaped modifications to weights.
Retain the download revision alongside the report either way.

## Report contents and limitations

A report records family/config/layout hashes, runtime implementation digest, PyTorch
version, dtype, device name/capacity, exact budget/chunks, prompt/output counts, per-run
cache and weight reads, prefill/elapsed time, host-process peak RSS where supported,
allocator peaks, and a sampled device-wide trace. CUDA success requires the conservative
observed peak to be strictly below the configured budget and below 8,000,000,000 bytes.
Failures during measured execution write `memory_pass=false`, an error and completed
cases. Invalid arguments/checkpoint validation can stop before measured execution.

**This is an empirical profile, not a proof or secure certificate.** Device memory is
sampled approximately every 10 ms; samples can miss between-sample allocations. PyTorch
high-water marks add coverage for PyTorch allocations but not every possible driver
or third-party allocation. The conservative result includes baseline/end non-PyTorch
usage. Another process can change memory use at any time. Reports are local editable
JSON and must not be accepted from untrusted parties as an attestation.

`numerical_qualification` and `tool_quality_qualification` remain false in a memory
report because the benchmark does not run those evaluations. A passing memory report
is not a blanket release-quality certificate, an accurate-answer guarantee or an
unlimited-context claim. Repeat after relevant hardware/driver/software changes.

## Use a matching report

```bash
python -m ollm.bounded generate ./models/qwen3-8b --model-key qwen3-8b \
  --budget-file examples/bounded/budget-7gb.json \
  --memory-profile ./profiles/qwen3-8b.json --prompt 'Explain latent attention.'
```

Profile loading checks identity, GPU name/capacity, exact budgets and exercised context
/output bounds. Changed source code invalidates the runtime digest. A profile measured
with smaller chunks/lengths cannot be reused with larger ones. During development only,
`--allow-unqualified` bypasses the profile requirement while retaining runtime guards;
it does not set `memory_pass` or imply successful measurement.

## Acceptance gate for publishing a supported profile

Publish no sub-8-GB support claim until there are real checkpoint reference/quality
results, successful repeated GPU measurements at advertised bounds, and documented
storage/RAM/throughput requirements. Include the GPU, dtype, context, output budget,
checkpoint revision and software identity. For tools, include multi-round real-model
handoff tests and approval/security checks. CPU tests and metadata estimates alone do
not meet that gate.


## Multimodal qualification

Qwen3-VL refuses the text-only `qualify` workload because it would not measure
the vision tower. Use `qualify-vl`. The report must exercise the exact
`max_visual_tokens`, `max_images`, context and output settings from the budget.
Successful Qwen3-VL profiles carry `multimodal_qualification: true`; a text-only
profile is rejected by `validate_profile`. Synthetic normalized patches are
used so model GPU memory is measured independently of CPU image decoding.

# Testing and verification boundaries

[Overview](../TOOLS.md) · [Native guide](tool_usage.md) ·
[Qwen-Agent guide](qwen_agent.md) · [Examples](../examples/tools/README.md)

## What was verified during patch development

The local suite has **176 passing unit/contract tests** and **six explicitly
skipped opt-in tests**. It ran with Python 3.13.5, pytest 9.0.2, jsonschema
4.26.0 and PyTorch 2.10.0+cpu. No model weights, Transformers, MCP SDK,
`openai-harmony` or Qwen-Agent were installed in that environment.

| Area | Executed locally | Boundary |
| --- | --- | --- |
| Strict JSON, schema checks and tool/result types | Yes | Real Python and jsonschema |
| Python sync/async dispatch, approvals, cancellation, limits and tool loop | Yes | Real handlers; scripted model decisions |
| Native Qwen parsing | Yes | Synthetic completed native output; tokenizer contract stub |
| GPT-OSS structural action validation | Yes | Harmony-shaped message fixtures, not the real renderer/parser |
| oLLM backend tensor handling and temporary-cache isolation | Yes | Real CPU tensors; fake inference model/cache |
| MCP discovery, pagination, namespacing, cleanup and failures | Yes | SDK contract stubs, not real subprocess/HTTP transport |
| Qwen-Agent provider/schema/message conversion | Yes | Framework interface stubs, not actual Assistant execution |
| All seven example CLIs with `--help` | Yes | No models or optional SDKs imported |
| Real MCP stdio and Streamable HTTP | Included, not run locally | Opt-in real SDK tests |
| Official Harmony token renderer/parser | Included, not run locally | Opt-in SDK test; tokenizer assets may need provisioning |
| Real Qwen-Agent Python-tool and MCP loops | Included, not run locally | Actual framework + scripted generation in subprocesses |
| Real Qwen/GPT-OSS model choosing and using a tool | Included, not run locally | Requires local weights and compatible hardware |
| GitHub Actions workflow | Added, not run remotely here | Runs when the patch is committed/pushed to a suitable repository |

Passing mocked contracts does not establish real SDK compatibility, checkpoint
quality, GPU fit, latency or tool-calling accuracy. Run the SDK suite and then
the model smoke test in the target environment before deployment.

The patch targets upstream commit
`12e16463473188af27f7e6d0449b30ea95365735`. The two modified existing files were
retrieved through GitHub and their original contents verified by Git blob SHA:

| File | Original blob SHA |
| --- | --- |
| `pyproject.toml` | `02306cf3545d91f2169d8a91e184f8d7c08334f7` |
| `src/ollm/__init__.py` | `3188cce5c91fc2d1e17a8ca5fc20b55a56e41cea` |

Patch application is checked against those exact preimages and the new-file
layout. A full upstream inference build, editable package installation and GPU
run were not performed in the authoring environment. The inference engine files
are not modified by this patch.

## Unit and contract tests without an inference installation

Run from the patched repository root:

```bash
python -m pip install 'pytest>=8,<10' 'jsonschema>=4.23,<5'
# Needed to include actual CPU tensor/backend tests:
python -m pip install 'torch>2.6' --index-url https://download.pytorch.org/whl/cpu

PYTHONPATH=src python -m pytest -c pytest-tools.ini -m 'not sdk and not model' -q
```

The source-tree path avoids installation of the base package's GPU-oriented
build dependencies. `pytest-tools.ini` limits discovery to the new test suite;
it does not run the upstream ad hoc model scripts under `scripts/`.

PowerShell equivalent:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -c pytest-tools.ini -m "not sdk and not model" -q
```

Without torch, tests that explicitly require its tensors skip instead of
claiming a backend pass. The local verification above did have CPU torch.

## Real SDK integration tests, no LLM weights

Use a separate clean environment when practical:

```bash
python -m pip install -r tests/tools/requirements-sdk.txt
OLLM_RUN_SDK_TESTS=1 PYTHONPATH=src python -m pytest -c pytest-tools.ini -m sdk -q
```

The compatibility target file pins **MCP 1.30.0**, **openai-harmony 0.0.8** and
**Qwen-Agent 0.0.34**; it is not a full transitive-dependency or model-runtime
lockfile. MCP 2.x is intentionally outside this patch's dependency range.

The five SDK cases cover real stdio discovery/calling/cleanup, a real local
Streamable HTTP server, native Harmony render/parse/continuation, and real
Qwen-Agent tool loops for Python and MCP. Model generation in the two Qwen-Agent
SDK cases is scripted so no model weights are required. Those cases run in
subprocesses to isolate Qwen-Agent's MCP singleton and SDK monkey patches.
The HTTP test starts a loopback server on a temporary port and terminates it
in `finally`.

When `OLLM_RUN_SDK_TESTS=1` is explicitly set, a missing SDK is an error, not a
successful skipped compatibility check. Without the opt-in all five cases are
skipped. Harmony may need to fetch encoding data on its first initialization;
pre-provision it for an offline test environment.

## Real model smoke tests

First verify upstream model loading works. The model test requires a directory
that already exists and never intentionally downloads missing weights:

```bash
OLLM_LIVE_MODEL=qwen3-next-80B OLLM_MODELS_DIR=./models OLLM_DEVICE=cuda:0 \
  PYTHONPATH=src python -m pytest -c pytest-tools.ini -m model -q

OLLM_LIVE_MODEL=gpt-oss-20B OLLM_MODELS_DIR=./models OLLM_DEVICE=cuda:0 \
  PYTHONPATH=src python -m pytest -c pytest-tools.ini -m model -q
```

Optional `OLLM_KV_CACHE_DIR=./agent-kv` applies only to Qwen. The smoke test
requests `add(17, 25)`, asserts that a real Python handler was called with those
arguments, and checks the final response includes 42. A model that answers 42
without calling the tool does **not** pass. This is a smoke check, not an
accuracy benchmark or proof of safe autonomous operation.

After those pass, exercise both models with the full examples, particularly
`mcp_tools.py` and `qwen_agent_mcp.py`. Audit their actual transcripts and tool
permissions before substituting credentialed services for the read-only demo.

## CI

`.github/workflows/tool-tests.yml` adds read-only-permission GitHub Actions jobs:
unit/contract tests on Python 3.10 and 3.12 with CPU torch, and SDK integration
tests on Python 3.12. It does not download LLM weights, publish packages or modify
repository contents. It installs SDK packages only in the SDK job; it does not
pretend an optional test ran merely because an import was skipped.

No CI run or publishing action was triggered during patch preparation.

## Additional checks

```bash
python -m compileall -q src/ollm/tools examples/tools tests/tools
# From an unmodified target checkout, before applying the downloaded patch:
git apply --check /path/to/ollm-tools-mcp-qwen-agent.patch
# Once applied, verify it could be reversed (this does not reverse it):
git apply --reverse --check /path/to/ollm-tools-mcp-qwen-agent.patch
```

Preserve a clean working tree or a separate feature branch for patching. Review
the diff and test output, including skips, rather than using a pass count as a
substitute for SDK and model validation.

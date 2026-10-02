# Runnable tool examples

[Overview](../../TOOLS.md) · [Native guide](../../docs/tool_usage.md) ·
[Qwen-Agent guide](../../docs/qwen_agent.md) · [Tests](../../docs/tool_testing.md)

Run these commands from the patched repository root in a working oLLM
environment. Use `python -m pip install --no-build-isolation -e '.[agents]'`
for every example's optional dependencies. Smaller extras are listed in the
native guide. Ordinary help does not load models or optional SDKs:

```bash
python examples/tools/python_tools.py --help
python examples/tools/mcp_tools.py --help
python examples/tools/qwen_agent_mcp.py --help
```

## Example map

| File | Demonstrates | Model needed |
| --- | --- | --- |
| `python_tools.py` | Synchronous addition and asynchronous multiplication, native registry and agent | Yes |
| `mcp_server.py` | Read-only arithmetic server over stdio or Streamable HTTP | No |
| `mcp_probe.py` | Actual MCP discovery and arithmetic call without inference | No |
| `mcp_tools.py` | Native agent, MCP stdio or remote HTTP, allowlists, approvals, bearer-header environment variable | Yes |
| `mixed_tools.py` | Python beats-to-seconds conversion plus MCP arithmetic in one registry | Yes |
| `qwen_agent_tools.py` | Qwen-Agent Assistant and custom BaseTool classes using the native oLLM provider | Yes |
| `qwen_agent_mcp.py` | Qwen-Agent's own MCP manager with the local arithmetic server | Yes |
| `common.py` | Shared model-loading options and final/transcript display helpers | Helper only |

## Start with a model-free MCP check

```bash
python examples/tools/mcp_probe.py
```

This starts the included stdio server with the current Python interpreter,
registers only its `add` tool, executes `17 + 25`, prints a structured result
containing 42, and closes the session. No arbitrary program or downloaded MCP
server is launched.

For HTTP, use two terminals. Terminal 1 runs a foreground server:

```bash
python examples/tools/mcp_server.py --transport streamable-http --port 8765
```

Terminal 2:

```bash
python examples/tools/mcp_probe.py --url http://127.0.0.1:8765/mcp
```

Stop the foreground server normally after testing. It is bound to loopback;
this example does not deploy a public service or add authentication.

Without a full oLLM installation, the no-model examples can run from source:

```bash
python -m pip install 'jsonschema>=4.23,<5' 'mcp>=1.30,<2'
PYTHONPATH=src python examples/tools/mcp_probe.py
```

In PowerShell set `$env:PYTHONPATH="src"` first instead of the shell prefix.

## Native Python and MCP tools with both models

```bash
python examples/tools/python_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/python_tools.py --model gpt-oss-20B --models-dir ./models

python examples/tools/mcp_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/mcp_tools.py --model gpt-oss-20B --models-dir ./models

python examples/tools/mixed_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/mixed_tools.py --model gpt-oss-20B --models-dir ./models
```

The arithmetic default requests `(17 + 25) * 3`; a successful actual tool run
should produce 126. The mixed example converts 24 beats at 120 BPM to 12 seconds
with Python, then adds five seconds through MCP for 17 seconds. These describe
expected outcomes, not prerecorded successful model runs.

To permit first-time weight download explicitly, add `--download`. The expected
directories are `./models/qwen3-next-80B` and `./models/gpt-oss-20B`; upstream
oLLM handles the checkpoint formats. `--download` permits missing weights; it
does not force replacement of an existing model.

Qwen disk-cache example:

```bash
python examples/tools/mcp_tools.py --model qwen3-next-80B --models-dir ./models \
  --kv-cache-dir ./agent-kv --max-new-tokens 1024
```

Do not give GPT-OSS a cache directory; upstream does not support that mode.
Use `--reasoning-effort low`, `medium` or `high` for Harmony. The `--thinking`
flag passes Qwen's thinking-template option when supported by the checkpoint.

## Remote MCP with explicit approvals

For the already running local HTTP server, explicitly allow and exempt the
reviewed arithmetic tools:

```bash
python examples/tools/mcp_tools.py --model qwen3-next-80B --models-dir ./models \
  --url http://127.0.0.1:8765/mcp \
  --tool add --tool multiply --approve-tool add --approve-tool multiply
```

Omitting `--approve-tool` makes the example ask for approval for each call.
Omitting `--tool` exposes all discovered remote definitions, still subject to
approval. Arguments to these flags use original remote names. Native examples
name the URL-based server `remote`; prompted names are normally `remote__add`
and `remote__multiply`.

For a trusted authenticated service, set `OLLM_MCP_TOKEN` securely and give its
HTTPS endpoint through `--url`. `--token-env NAME` selects a different variable.
The example sends it as an Authorization Bearer header and does not print it.
Do not place real secrets in command lines, screenshots or committed files.
Remote servers may have different tool names and schemas: change `--tool`,
`--approve-tool` and `--prompt` accordingly after reviewing their capabilities.

## Qwen-Agent routes

```bash
python examples/tools/qwen_agent_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/qwen_agent_tools.py --model gpt-oss-20B --models-dir ./models
python examples/tools/qwen_agent_mcp.py --model qwen3-next-80B --models-dir ./models
python examples/tools/qwen_agent_mcp.py --model gpt-oss-20B --models-dir ./models
```

These use `OllmChatModel` directly; no HTTP model service or cloud key is needed.
Qwen-Agent owns tool execution and its MCP manager's policy/lifecycle. Native
registry approval and result limits do not apply automatically. The examples
expose only bundled arithmetic tools. See the provider guide before connecting
services with read/write access to real data.

## Shared options and transcripts

All model examples accept `--model`, `--device`, `--models-dir`, `--download`,
`--kv-cache-dir`, `--max-new-tokens`, `--max-context-tokens`, `--temperature`,
`--reasoning-effort`, `--thinking`, `--prompt` and `--transcript`.

```bash
python examples/tools/python_tools.py --model gpt-oss-20B --models-dir ./models \
  --max-new-tokens 1024 --reasoning-effort low \
  --prompt 'Use add to calculate 203 + 419.' --transcript ./tool-transcript.json
```

Transcripts are opt-in and may contain sensitive tool results, arguments and
model reasoning needed for continuation. Protect them and avoid committing
them. Ensure the target parent directory exists. The display prints the final
answer, not the model's private reasoning channels. Output is buffered per model
turn; these examples do not claim live token streaming.

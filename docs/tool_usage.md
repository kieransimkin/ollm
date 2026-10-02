# Native tool and MCP integration

[Overview](../TOOLS.md) · [Examples](../examples/tools/README.md) ·
[Qwen-Agent](qwen_agent.md) · [Testing](tool_testing.md)

## 1. Architecture

The model requests tools; the host executes them. MCP is a transport and tool
discovery layer, not a replacement for model-specific function-call formatting.

```text
Agent.run(history)
  -> InferenceBackend.generate(history, registry.schemas())
     -> QwenAdapter or GPTOSSAdapter prepares native model tokens
     -> oLLM model.generate(...)
     -> adapter validates a complete assistant action
  -> ToolRegistry validates, authorizes and executes each requested tool
     -> ordinary Python callable OR an existing MCP ClientSession
  -> append correlated, bounded JSON tool results
  -> generate again until a final answer or a budget/error stops the run
```

The registry and agent are dependency-light. `import ollm.tools` does not load
PyTorch, Transformers, MCP, Harmony or Qwen-Agent. The package's existing public
exports (`Inference`, `AutoInference`, `file_get_contents`, `TextStreamer`)
remain available through lazy imports.

Use one loaded oLLM model per process. Generation calls through this backend
share a process-wide lock because upstream model loaders use module-global
state. This does not make multiple simultaneously loaded model instances safe,
nor coordinate code that calls the raw model outside this backend.

## 2. Dependencies and installation

Python 3.10 or later is required by the base package. Start with the original
oLLM installation and verify its existing `example.py` works on your hardware.
The patch does not change its inference architecture, checkpoint provisioning,
CUDA installation or Transformers pin.

| Extra | Additional functionality |
| --- | --- |
| `tools` | JSON Schema validation, native Qwen tools and the shared agent |
| `mcp` | Native agent plus MCP SDK 1.x transports |
| `gpt-oss-tools` | Native tools plus the official Harmony runtime |
| `qwen-agent` | Qwen-Agent provider with native Qwen model formatting |
| `agents` | All of the above, including Qwen-Agent's MCP extra |
| `test-tools` | Pytest and JSON Schema validation |

Examples of editable installation from the patched checkout:

```bash
python -m pip install --no-build-isolation -e '.[tools]'
python -m pip install --no-build-isolation -e '.[mcp,gpt-oss-tools]'
python -m pip install --no-build-isolation -e '.[agents]'
```

Upstream `pyproject.toml` currently lists `flash-attn` and
`flash-linear-attention` as base dependencies. A tool-only extra does not bypass
them during normal package installation. To run tests or the no-model MCP probe
without an inference installation, use the **source-tree commands** in the
[testing guide](tool_testing.md), rather than attempting a full oLLM install.

The MCP dependency is `mcp>=1.30,<2`. SDK 2.x has a different API; this patch
uses the maintained 1.x client API also used by the targeted Qwen-Agent
integration. Compatibility target versions are recorded in
`tests/tools/requirements-sdk.txt`. They are not claims of completed runtime
verification in the authoring environment.

For offline operation, provision weights, dependencies and tokenizer assets
before disconnecting. Harmony's encoding initialization can need tokenizer data
on first use. Initialize `GPTOSSAdapter()` once while online and preserve its
runtime cache in the deployed environment. Do not assume a cached Hugging Face
checkpoint automatically provides every Harmony asset.

## 3. A complete Python-tool example

Run this from a working patched oLLM installation. The explicit `ini_model`
call below follows upstream behavior and may download a missing checkpoint.
The bundled command-line examples instead require `--download` for that case.

```python
from ollm import Inference
from ollm.tools import Agent, GenerationConfig, InferenceBackend, ToolRegistry

inference = Inference("qwen3-next-80B", device="cuda:0", logging=False)
inference.ini_model(models_dir="./models", force_download=False)
backend = InferenceBackend(
    inference,
    generation=GenerationConfig(max_new_tokens=512, temperature=0.0),
    # cache_dir="./agent-kv",  # Optional, Qwen only; see caching below.
)
registry = ToolRegistry()

@registry.tool(
    parameters={
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    requires_approval=False,  # Deliberate host approval of this harmless function.
)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

agent = Agent(backend, registry, max_rounds=8, max_tool_calls=16)
result = agent.run_sync("Call add with 17 and 25, then report the result.")
print(result.text)
print(result.rounds, result.tool_calls)
```

Changing the model ID to `gpt-oss-20B` selects Harmony automatically; install
`gpt-oss-tools` and omit `cache_dir`. Model selection alone does not download or
instantiate a different model inside an existing backend.

Synchronous handlers run in a worker thread, so they do not block the MCP event
loop. `async def` handlers are awaited directly. Handlers receive keyword
arguments and may return any JSON-serializable value or a `ToolResult`.
The decorator retains the original callable, so ordinary direct Python calls
remain possible outside the registry. Direct calls do not apply its policy.

For explicit registration:

```python
from ollm.tools import Tool

registry.add(Tool(
    name="add_explicit",
    description="Add two numbers.",
    parameters=registry.tools[0].parameters,
    handler=add,
    requires_approval=False,
))
```

Names must match `[A-Za-z_][A-Za-z0-9_]{0,63}`. Duplicate registration fails;
`add_many()` is atomic and never silently overwrites an existing tool. Schemas
are copied and checked at registration. Only local JSON Schema references are
accepted; network schema resolution is not enabled. The default JSON Schema
validator does not enforce `format` as an application security rule: validate
paths, URLs, identifiers and business rules in the actual handler too.

## 4. Permissions, errors and limits

Tools default to `requires_approval=True`. With the default policy those calls
are denied; merely registering a tool does not authorize its execution.

```python
from ollm.tools import ToolPolicy, ToolRegistry

def approve(tool, arguments):
    # Replace with a trusted UI or a host-side policy. Never delegate approval
    # to the same model requesting the action.
    return tool.name == "read_status" and arguments.get("project") == "demo"

registry = ToolRegistry(ToolPolicy(
    allowed_tools=frozenset({"read_status"}),
    approve=approve,
    timeout_seconds=30,
    max_argument_chars=32000,
    max_result_chars=16000,
))
```

An approval callback can be synchronous or asynchronous and must return the
literal boolean `True`. An allowlist limits both advertisement and execution;
it does not replace approvals. `requires_approval=False` is an explicit
host-side exemption. Validate arguments before showing an approval request.
The callback receives a copy of the arguments.

Limits and defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `Agent.max_rounds` | 8 | Maximum model generation turns, including final answer |
| `Agent.max_tool_calls` | 16 | Total attempted calls, including returned tool errors |
| `Agent.max_calls_per_turn` | 8 | Maximum calls in one Qwen-generated batch |
| `ToolPolicy.timeout_seconds` | 30 | Wait budget for approval plus handler execution |
| `ToolPolicy.max_argument_chars` | 32,000 | Compact JSON character count before execution |
| `ToolPolicy.max_result_chars` | 16,000 | Maximum JSON characters fed back per result |
| `MCPClient.max_tools` | 256 | Maximum advertised tools discovered per server |
| `MCPClient.max_pages` | 100 | Maximum discovery pages per server |

A whole over-budget batch is refused before any of its tools execute. The last
available model round must produce a final answer: tools requested in that
round are not executed. `AgentLimitError.messages` contains the completed
history, not an invented final answer. Batches execute sequentially even when
Qwen requests several calls; there is no parallel tool execution in this agent.

`ToolResult` serializes as `{"ok":true,"result":...}` or
`{"ok":false,"error":...}`. Oversized results become a valid JSON wrapper
with `truncated`, `original_chars` and a bounded `preview`; they are not sliced
into invalid JSON. Limits are character counts, not byte or token counts.
Result bounding limits model context consumption, not the amount of memory a
handler or remote SDK can allocate before returning.

Error codes include `unknown_tool`, `not_allowed`, `invalid_arguments`,
`arguments_too_large`, `approval_required`, `approval_denied`, `timeout` and
`execution_error`. They are sent back as tool results so the model can respond
appropriately. `ToolResult.error(code, message)` lets a handler return an
application error. MCP `isError` is preserved; this is distinct from a failed
transport. No failed tool call is automatically retried by the dispatcher.

A timeout is **not a hard cancellation guarantee**. Python cannot forcibly kill
a thread safely, a coroutine may mishandle cancellation, and a remote service
may already have performed an action. `asyncio.run()`/`run_sync()` can also wait
for worker-thread shutdown. Use a supervised subprocess or external service
for hard runtime/memory limits. Do not blindly retry timed-out writes. Model
inference itself has a token budget, not a wall-clock watchdog.

## 5. MCP sessions and discovery

Use the official SDK client for stdio or Streamable HTTP. `MCPClient` discovers
`tools/list`, validates schemas, registers namespaced wrappers, and invokes
`tools/call`. Initialization and discovery have request timeouts; pagination
has tool/page limits and repeated-cursor detection. A failed connection leaves
no partial registrations.

```python
import asyncio
import sys
from pathlib import Path
from ollm.tools import Agent, MCPClient, MCPServerConfig, ToolRegistry

# backend is an already initialized InferenceBackend from section 3.
async def run_mcp(backend):
    registry = ToolRegistry()
    server = Path("examples/tools/mcp_server.py").resolve()
    config = MCPServerConfig(
        name="math",
        command=sys.executable,
        args=(str(server),),
        allow_tools=frozenset({"add", "multiply"}),
        auto_approve=frozenset({"add", "multiply"}),
    )
    async with MCPClient(registry) as client:
        discovered = await client.connect(config)
        print([tool.name for tool in discovered])  # math__add, math__multiply
        return await Agent(backend, registry).run("Use tools to calculate (17 + 25) * 3.")

# result = asyncio.run(run_mcp(backend))
```

`allow_tools` and `auto_approve` contain **original remote names**, not aliases.
The registry policy's `allowed_tools` contains **model-facing aliases**. Omitted
`allow_tools` exposes all discovered tools, but they still require approval
unless specifically exempted. Empty `allow_tools=frozenset()` exposes none.
Unknown configured names fail connection instead of silently being ignored.
Server `readOnlyHint` annotations do not grant approval.

Names are normally `server__tool`. Remote names with punctuation or excessive
length are sanitized and receive a stable SHA-256-derived suffix. The wrapper
always dispatches the original remote name; the model sees only the alias.
Server names are safe identifiers up to 24 characters. Duplicate server names
within one client are rejected. Multiple servers and Python tools can share a
registry; all tool-name collisions are checked.

Enter, connect and exit a client **in the same asyncio task**. The SDK's AnyIO
context managers own task groups and cancellation scopes. Do not open sessions
with one `asyncio.run()` and use/close them with another. Do not open each
connection in a different task with `gather()`. Calls themselves can be awaited
by child tasks. Exit removes that client's registrations, closes all sessions
in reverse order and propagates cleanup errors after attempting the remaining
closes. Use `async with`; there is no finalizer-based cleanup guarantee.

### Remote HTTP and authentication

```python
import os
from ollm.tools import MCPServerConfig

remote = MCPServerConfig(
    name="catalog",
    transport="streamable-http",
    url="https://your-trusted-service.example/mcp",
    headers={"Authorization": "Bearer " + os.environ["CATALOG_MCP_TOKEN"]},
    timeout_seconds=45,
    allow_tools=frozenset({"search"}),
    # No auto_approve here: connect this to an approval-enabled registry.
)
```

Headers and subprocess environment fields are excluded from the config's
`repr`; this is not a comprehensive logging redaction system. Do not put
secrets in URLs or commit them in example files. URL credentials/fragments are
rejected. Non-loopback HTTP requires HTTPS unless the host explicitly sets
`allow_insecure_http=True`. That opt-out is for controlled development, not
production. A loopback exception exists for `localhost`, `127.0.0.1` and `::1`.
The URL and launch command are host configuration, never generated by the model.

Subprocess arguments are passed as a list, not through a shell. `env` is passed
to the SDK's subprocess launcher; manage inherited environment and operating
system permissions deliberately. Install trusted server executables ahead of
time instead of automatically running unreviewed package downloads.

This version has static host-provided HTTP headers, not automatic OAuth
registration/discovery/refresh. It does not implement legacy HTTP+SSE transport,
MCP resources/prompts/sampling/elicitation, subscriptions, reconnection or
list-changed refresh. Discovery occurs once per successful connection. Use a
new controlled session to refresh definitions. These are explicit scope limits,
not claims that every feature of the MCP specification is supported.

### MCP result content

Text blocks, structured JSON content, resource references and textual embedded
resources are preserved. Image/audio base64 data and embedded binary resource
blobs are omitted with a marker. Referenced URLs and resource URIs are not
followed automatically. This integration is text-only; it does not pass media
to oLLM's vision/audio models.

## 6. Model formats and strict stopping

**Qwen:** the loaded tokenizer's native `apply_chat_template(..., tools=...)`
renders tools and history. The parser accepts framed `<tool_call>` JSON objects
with `name` and `arguments`, including multiple calls at the tail of an
assistant turn. Reasoning and fenced code examples do not become tool calls.
Extra keys, malformed delimiters, trailing ambiguous text, duplicate JSON keys,
non-finite numbers and truncated turns fail closed. No `eval()` or Python code
execution is involved in parsing. This is not a general parser for every Qwen
family: automatic selection targets upstream `qwen3-next-80B`.

**GPT-OSS:** `openai-harmony` renders and parses actual native token IDs, using
structured system/developer/function-tool definitions and assistant channels.
The adapter distinguishes action handoff from final completion, validates
`functions.<tool>` recipients, and preserves current-chain analysis separately
from the user-visible answer. It supports one tool handoff per assistant
action, followed by a tool result and another model action. Reasoning is not
searched for JSON that merely resembles a call.

Both adapters require proper model action/turn stop tokens. Reaching
`max_new_tokens` without one raises `IncompleteGeneration`; no tool from that
unfinished generation executes. Increasing the output budget can help, but
there is no automatic regeneration or prompt-repair loop. A malformed turn
raises `ToolCallParseError`; calls parsed earlier in that same turn are not
partially dispatched. A schema-invalid but syntactically valid call returns a
tool error, allowing a later model turn to correct its arguments within budget.

Model behavior is not guaranteed by transport correctness. Test your actual
checkpoint on representative tasks and adversarial tool output before enabling
sensitive tools. No tool-calling accuracy benchmark is claimed by this patch.

## 7. Generation, caching and conversation API

`InferenceBackend(inference, adapter=None, generation=None, cache_dir=None)`
wraps an existing loaded model. Automatic adapter selection recognizes only the
two supported upstream IDs. An explicitly supplied compatible adapter can
implement `prepare(...) -> PreparedPrompt` and `parse(...) -> AssistantTurn`.

`GenerationConfig` defaults:

| Field | Default | Notes |
| --- | --- | --- |
| `max_new_tokens` | 512 | Includes reasoning and tool-call tokens |
| `temperature` | 0.0 | Greedy; positive values enable sampling |
| `top_p` | 0.9 | Used only with sampling |
| `top_k` | None | Does not override the model's setting unless explicitly set while sampling |
| `repetition_penalty` | 1.0 | Passed to generation |
| `max_context_tokens` | None | Optional total prompt plus reserved output cap |
| `reasoning_effort` | `low` | GPT-OSS: `low`, `medium`, `high` |
| `enable_thinking` | False | Qwen: passed to the native template when supported |
| `seed` | None | Optional seed with RNG-state restoration around generation |

The minimum of the configured context cap and model `max_position_embeddings`
is enforced when available. The backend does not silently truncate history.
Shorten history/tool results or change the explicit output budget instead.
A context cap does not promise that the workload will fit in available VRAM.

Every model call recomputes the full prompt. Qwen disk caching, when requested,
uses a new temporary subdirectory per generation and cleans it on success or
failure. This avoids sharing/deleting another session's cache, but **does not
implement prefix-cache reuse between tool calls**. Large-context multi-step
agents can therefore be slow. Provision sufficient scratch storage and avoid
large unnecessary tool outputs. Abrupt process termination may leave temporary
files; clean stale directories only after verifying no process owns them.

The upstream GPT-OSS `DiskCache` method is unsupported and returns `None`.
This backend rejects a requested GPT-OSS cache directory instead of pretending
it offloaded the cache. Its ordinary generation uses the model's default cache.

`await Agent.run(text_or_messages)` returns `AgentResult` with final `text`,
full `messages`, `rounds` and `tool_calls`. `run_sync` wraps it for ordinary
synchronous programs; use `await` in notebooks/async applications. String input
adds the default instruction about untrusted tool results. Supplied histories
are used as supplied after validation; include your own appropriate system
instructions. Histories are copied, and caller-owned messages are not mutated.

Canonical histories use text `system`/`user`/`assistant`/`tool` messages,
assistant `tool_calls`, and matching `tool_call_id` values. Tool arguments are
objects internally, not double-encoded JSON strings. Leading developer/system
instructions are combined. Every call must have exactly one matching result
before another user/assistant turn. Completed histories end in an assistant
answer: append the next user message before calling `run` again.

```python
messages = result.messages + [{"role": "user", "content": "Now add 10 to that using the tool."}]
next_result = agent.run_sync(messages)
```

The native agent returns completed answers; there is no live token-streaming
API in this patch. Full transcripts can contain model reasoning from tool
turns, sensitive arguments and tool data. Store them only with suitable access
controls; do not log them by default.

## 8. Operational security and troubleshooting

Tool metadata, descriptions and results can contain prompt injection. Treat
all of them as untrusted data and keep authority in host-side allowlists,
approvals and handler validation. The default instruction is helpful context,
not a security boundary. There is no filesystem, network or process sandbox.
Do not expose unrestricted shell/Python execution or broad write access merely
because a model supports function calling.

| Symptom | Action |
| --- | --- |
| Missing optional dependency | Install the relevant extra, or source-tree test dependencies |
| Missing checkpoint directory | Set `--models-dir`, provision weights, or explicitly add `--download` |
| Unsupported model ID | Use an upstream supported ID; this patch does not add new checkpoint architectures |
| IncompleteGeneration | Inspect the task/template; increase `--max-new-tokens` for genuine longer generations |
| ToolCallParseError | No calls in that turn executed; inspect the selected model and framing, not a broad JSON regex fallback |
| approval_required/denied | Supply a host callback or explicitly exempt reviewed harmless tools |
| invalid_arguments | Align the schema and Python signature; return actionable validation messages |
| Same-task MCP error | Keep enter/connect/exit in a single `async with` scope in one task |
| Context limit or out-of-memory | Reduce prompt/results/output budget; Qwen disk cache is optional, GPT-OSS disk cache is unavailable |
| Slow multi-round response | Full-prefill and model offloading costs recur; reduce the number and size of model turns |
| MCP 2.x import/signature failure | Install the declared `<2` dependency range, do not silently fall back to another API |

## References

API decisions were checked against these primary sources; runtime verification
is described separately in the testing guide.

- [oLLM source at the patch baseline](https://github.com/Mega4alik/ollm/tree/12e16463473188af27f7e6d0449b30ea95365735)
- [Official Harmony Python API](https://github.com/openai/harmony/blob/main/docs/python.md)
- [Official Harmony runtime](https://pypi.org/project/openai-harmony/)
- [MCP SDK 1.x documentation](https://py.sdk.modelcontextprotocol.io/v1/)
- [MCP SDK release lines and migration notice](https://pypi.org/project/mcp/)
- [Qwen3-Next model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)

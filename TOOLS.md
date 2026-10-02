# Tools, MCP and Qwen-Agent for oLLM

This extension adds native tool calling for **qwen3-next-80B** and
**gpt-oss-20B** without replacing oLLM's model loaders or offloading code.
It was prepared against oLLM commit
`12e16463473188af27f7e6d0449b30ea95365735` (version 1.0.4).

## Choose an execution layer

| Route | Model formatting and generation | Tool execution |
| --- | --- | --- |
| `ollm.tools.Agent` | Qwen native chat template or official GPT-OSS Harmony | oLLM registry, validation, approvals and limits; Python + MCP tools can be mixed |
| Qwen-Agent `Assistant` + `OllmChatModel` | The same native adapters and oLLM inference | Qwen-Agent's own tools/MCP manager and execution policy |

Do not nest the two agent loops. The Qwen-Agent provider performs **one model
turn**, not tool execution. Both routes accept either supported model.

## Install and try

In a checkout with a working oLLM inference environment:

```bash
python -m pip install --no-build-isolation -e '.[agents]'

# No model or GPU needed: discover and call the bundled read-only MCP server.
python examples/tools/mcp_probe.py

# Existing local model weights are used unless --download is explicitly added.
python examples/tools/python_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/mcp_tools.py --model gpt-oss-20B --models-dir ./models
python examples/tools/qwen_agent_tools.py --model qwen3-next-80B --models-dir ./models
python examples/tools/qwen_agent_mcp.py --model gpt-oss-20B --models-dir ./models
```

The original inference dependencies, including the upstream Transformers pin
and Flash Attention requirement, are unchanged. Installing an extra does not
remove those requirements. MCP is deliberately constrained to the maintained
1.x API (`>=1.30,<2`); this is not an SDK 2.x integration.

## Documentation

- [Native tool/MCP guide and API](docs/tool_usage.md)
- [Qwen-Agent provider guide](docs/qwen_agent.md)
- [All runnable examples and commands](examples/tools/README.md)
- [Tests, compatibility targets and verification limits](docs/tool_testing.md)

## Scope and verification

The implementation includes strict model-specific parsing, structured tool
results, namespaced MCP discovery, stdio/Streamable HTTP clients, explicit
permissions, argument validation, execution budgets, and isolated temporary
Qwen disk caches. GPT-OSS uses the official `openai-harmony` package.

Generation is buffered. Qwen-Agent's streaming interface yields a **completed
model turn**, not live tokens. There is no cross-turn KV-prefix reuse, hard
sandbox, automatic OAuth flow, MCP resources/prompts implementation, or
OpenAI-compatible HTTP server in this patch. Upstream GPT-OSS disk caching
remains unsupported.

Unit/contract tests and CPU tensor tests were executed during development;
real MCP/Harmony/Qwen-Agent SDK tests and model-weight tests are included as
separate opt-in suites. See the testing guide for the distinction before
using this with sensitive tools.


## Additional Qwen and DeepSeek architectures

The [bounded-memory extension](BOUNDED.md) adds an explicit `BudgetInference`
backend. The native Python/MCP loop and this Qwen-Agent provider can use it,
including the new Qwen Coder parameter-tag parser. Read the
[bounded examples](examples/bounded/README.md) for invocation. Legacy inference
remains unchanged. New checkpoints are unqualified candidates until measured
on the actual GPU; native DeepSeek function calling is not claimed.


### Qwen3-VL images

The bounded runtime exposes `qwen3-vl-2b-instruct` to the same native tool/MCP
loop. User messages may contain local `image` and `text` content blocks. Remote
image URLs and video blocks are rejected. See [docs/qwen3_vl.md](docs/qwen3_vl.md).

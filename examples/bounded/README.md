# Runnable bounded examples

[Overview](../../BOUNDED.md) · [Memory/API guide](../../docs/bounded_inference.md)

Run from the fork's root in the working oLLM environment. The example `--help` commands
do not load a model or optional MCP/Qwen-Agent SDK. Local weights/tokenizer are required
for actual execution. These examples use unqualified opt-in explicitly; replace it with
`--memory-profile PATH` once the exact budget has been measured on the GPU.

```bash
# Standard text; 4B, 8B, 14B and 32B use the same actual dense backend.
python examples/bounded/chat.py ./models/qwen3-8b --model-key qwen3-8b \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Native DeepSeek compact MLA and streamed experts. No native function-call claim.
python examples/bounded/chat.py ./models/deepseek-v2-lite --model-key deepseek-v2-lite-chat \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Coder-Next parameter-tag parser plus explicit read-only Python tools.
python examples/bounded/native_tools.py ./models/coder-next --model-key qwen3-coder-next \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Native MCP manager and execution policy; launches only the bundled arithmetic server.
python examples/bounded/mcp_tools.py ./models/coder-30b --model-key qwen3-coder-30b-a3b \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Qwen3.5 text-only hybrid plus native Python tools.
python examples/bounded/native_tools.py ./models/qwen3.5-4b --model-key qwen3.5-4b \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Qwen-Agent owns orchestration and tool execution.
python examples/bounded/qwen_agent.py ./models/qwen3-8b --model-key qwen3-8b \
  --budget-file examples/bounded/budget-7gb.json --allow-unqualified

# Same provider, using Qwen-Agent's own MCP manager.
python examples/bounded/qwen_agent.py ./models/coder-next --model-key qwen3-coder-next \
  --mcp --budget-file examples/bounded/budget-7gb.json --allow-unqualified
```

The native MCP example explicitly allowlists/auto-approves only the bundled `add`
and `multiply` tools. Do not copy auto-approval to untrusted/write-capable tools.
Qwen-Agent's execution route does not inherit the native registry's policy. The
examples never execute arbitrary generated Python or shell code.

## Reuse the backend

```python
from ollm import BudgetInference, MemoryBudget
from ollm.tools import InferenceBackend
from ollm.tools.qwen_agent import OllmChatModel

budget = MemoryBudget(max_context_tokens=4096, max_output_tokens=128)
o = BudgetInference('./models/coder-next', model_key='qwen3-coder-next',
                    budget=budget, allow_unqualified=True)
provider = OllmChatModel(backend=InferenceBackend(o))
```

Or configure the registered Qwen-Agent provider directly:

```python
# Import registers model_type='ollm' with Qwen-Agent.
import ollm.tools.qwen_agent
from qwen_agent.agents import Assistant

agent = Assistant(llm={
    'model_type': 'ollm',
    'model': 'qwen3-coder-next',
    'device': 'cuda:0',
    'bounded': {
        'model_dir': './models/coder-next',
        'dtype': 'bfloat16',
        'budget': {'max_context_tokens': 4096, 'max_output_tokens': 128},
        'allow_unqualified': True,
    },
    'generate_cfg': {'max_new_tokens': 128},
}, function_list=[])
```

In a measured deployment, replace `allow_unqualified` with `memory_profile`; the
profile's exact budget must match. Download beforehand with the explicit CLI: the
provider rejects download flags on this path. No inference HTTP server or cloud API
is involved. Both native and Qwen-Agent outputs remain buffered by complete model
turn; no partial function calls are executed.

## Coder schema rules and errors

Coder-Next/30B and Qwen3.5 parameter values are interpreted using the advertised
JSON Schema. Numeric-looking strings remain strings; code whitespace is retained
apart from one protocol framing newline. Numbers, arrays and objects must be valid JSON. Boolean/null scalars also accept
`True`, `False` and `None` when their schema allows those types, matching the official
Jinja template’s scalar stringification. Required fields and additional-property constraints are checked.
Parameter names must be identifier-style; each needs an explicit `type`. Nested
`$ref`-only/`anyOf`-only type inference is not guessed. Ambiguous delimiter-like content,
unadvertised names, duplicate parameters, incomplete frames and text after calls
cause a fail-closed parse error. Larger output budgets may be necessary for thinking
models, but changing the budget requires a new measurement profile.

## CPU toy tests are not these pretrained examples

The automated suite runs each script's `--help` without optional SDKs and tests real
small synthetic safetensor decoders. It does not pretend to have run these commands
with large pretrained weights, real SDK transports, or CUDA in the development
container. See the delivered validation report and the qualification guide.

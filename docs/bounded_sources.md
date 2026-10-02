# Primary implementation references

These are architecture/template references, not evidence of this fork's GPU speed or
memory consumption. Configurations and templates can change; retain immutable model
revisions for real runs. The decoder is a functional implementation of these operations;
no remote checkpoint Python is executed.

## Fork baseline

- https://github.com/kieransimkin/ollm/commit/b395639e7397573728c7c4a1f82565ef1cb542c4
- Existing `src/ollm/gds_loader.py`, `kvcache.py`, `qwen3_next.py`, `llama.py` inspected
  for layer/expert offloading and cache constraints. Legacy files are not modified by
  this patch; the bounded path is separate.

## Qwen dense, MoE and hybrid

- https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3/modeling_qwen3.py
- https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen2/configuration_qwen2.py
  (blob `4d75e25092f4c04ef119682f61b9b32241ffb398`; sliding layers start at `max_window_layers`)
- https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_next/modeling_qwen3_next.py
  (blob `e15e3435f732076de2426422742744e5bbfaa85c`; grouped packed projections, DeltaNet, gated attention)
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
  (inspected blob `d7afe5803811e6ee82ddf43b00f85471a3ec7a36`; separate projections and text hybrid semantics)
- https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3-Coder-Next/blob/main/chat_template.jinja
- https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja
- https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
  (same family with explicit Swish output gate; not a shipped GPU qualification)

## DeepSeek

- https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
  (compact MLA/weight absorption, routed/shared expert definitions)
- https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
  (HF checkpoint tensor names, YaRN, grouped noaux routing and correction bias)
- https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json
  (block-scaled FP8, expert/attention dimensions, appended MTP count)
- https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/blob/main/config.json
- https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
- https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v2/modeling_deepseek_v2.py
  (inspected blob `0d1286bbdf81e6c8fc253d05e6713ad55fb2d962`)

Model licenses and acceptable-use terms belong to their respective checkpoints and
are not replaced by oLLM's source license. This patch does not redistribute weights.
The CPU tests create small random fixtures locally. Reference-library tests use
installed upstream packages rather than bundling their source implementations.


### Qwen3-VL

- Qwen3-VL-2B-Instruct configuration: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
- Transformers 4.57 implementation: https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- Model card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

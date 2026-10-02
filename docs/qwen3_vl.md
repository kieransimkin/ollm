# Qwen3-VL-2B-Instruct in the bounded runtime

`qwen3-vl-2b-instruct` is the first bounded multimodal checkpoint in this fork.
The implementation is native to `ollm.bounded`: it does not load the complete
Hugging Face model onto CUDA and it does not use `device_map="auto"`.

## Implemented scope

- `Qwen/Qwen3-VL-2B-Instruct`, BF16 checkpoint layout.
- Text-only requests and one or more **images** in a batch-one conversation.
- The 24-layer ViT is executed with row-tiled checkpoint reads.
- Vision attention is exact non-causal attention, query/key tiled per image/frame
  segment so no full visual attention matrix is materialized.
- Vision MLPs and the patch mergers are chunked against the configured workspace.
- DeepStack features from vision layers 5, 11 and 17 are merged, moved to CPU,
  and staged back only when injected after the first text decoder layers.
- Qwen3-VL three-axis interleaved MRoPE is implemented for the language decoder.
- Native Python tools and MCP can be used with structured image messages.
- Images must be local files (or PIL images when using the Python API). HTTP,
  HTTPS and data URLs are rejected before the processor runs.

**Video is not enabled.** The checkpoint supports video, but temporal workloads
need a separate implementation/qualification profile. Supplying a video block
or video processor tensors raises instead of silently falling back.

## Install

The base project still pins the Transformers version used by this fork. Pillow
is opt-in for direct multimodal use:

```bash
python -m pip install --no-build-isolation -e '.[multimodal]'
```

For image + MCP/Qwen-Agent workflows:

```bash
python -m pip install --no-build-isolation -e '.[agents]'
```

No checkpoint is downloaded on import. An explicit CLI download is available:

```bash
python -m ollm.bounded inspect ./models/qwen3-vl-2b \
  --model-key qwen3-vl-2b-instruct --download
```

## Image chat

```bash
python examples/bounded/qwen3_vl.py ./models/qwen3-vl-2b \
  --model-key qwen3-vl-2b-instruct \
  --image ./photo.jpg \
  --budget-file examples/bounded/budget-7gb.json \
  --allow-unqualified
```

The explicit `--allow-unqualified` is required until the checkpoint has a
matching CUDA profile. It is not an under-8-GB certification.

Python uses structured content:

```python
from ollm import BudgetInference

model = BudgetInference(
    './models/qwen3-vl-2b',
    model_key='qwen3-vl-2b-instruct',
    allow_unqualified=True,
)
answer = model.generate([{
    'role': 'user',
    'content': [
        {'type': 'image', 'image': './photo.jpg'},
        {'type': 'text', 'text': 'What is happening in this image?'},
    ],
}])
```

Multiple local image blocks are accepted up to `MemoryBudget.max_images` and
the total pre-merge patch count must not exceed
`MemoryBudget.max_visual_tokens`. The default bounded profile uses 4096 visual
patches and four images. The post-merge image token count is lower by
`spatial_merge_size ** 2` (four for this checkpoint).

## MCP with an image

The existing agent loop preserves image blocks across model/tool/model rounds:

```bash
python examples/bounded/qwen3_vl_mcp.py ./models/qwen3-vl-2b \
  --model-key qwen3-vl-2b-instruct \
  --image ./numbers.png \
  --allow-unqualified
```

This example exposes only the bundled read-only arithmetic MCP server.
Reprocessing the image on each agent round is deliberate; cross-turn visual/KV
cache reuse is not yet implemented.

## Memory qualification

Text-only qualification is deliberately refused for a multimodal model. Run:

```bash
python -m ollm.bounded qualify-vl ./models/qwen3-vl-2b \
  --model-key qwen3-vl-2b-instruct \
  --budget-file examples/bounded/budget-7gb.json \
  --rounds 2 \
  --report ./profiles/qwen3-vl-2b.json
```

The workload uses synthetic normalized patch tensors but executes the real
checkpoint's complete vision tower, DeepStack path, MRoPE decoder, maximum
configured context and forced output. It exercises one-image and maximum-image
cases. CPU runs are reference-only and can never produce `memory_pass=true`.

A matching successful profile is then required in place of the experimental
opt-in:

```bash
python examples/bounded/qwen3_vl.py ./models/qwen3-vl-2b \
  --model-key qwen3-vl-2b-instruct \
  --image ./photo.jpg \
  --memory-profile ./profiles/qwen3-vl-2b.json
```

A memory profile is hardware/checkpoint/runtime specific. It does not certify
vision quality, OCR accuracy, tool quality or arbitrary video/image sizes.

## Architecture references

The implementation follows the Qwen3-VL configuration and the Transformers
4.57 Qwen3-VL architecture: Conv3D patch embedding, interpolated learned visual
position embeddings, axial visual RoPE, packed non-causal ViT attention,
DeepStack mergers, and interleaved 3-axis text MRoPE.

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
- https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py

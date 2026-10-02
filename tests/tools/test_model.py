"""Opt-in local-weight smoke test. Never downloads a model automatically."""
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.model, pytest.mark.skipif(
    not os.environ.get('OLLM_LIVE_MODEL'), reason='Set OLLM_LIVE_MODEL for real inference')]


def test_live_model_executes_python_tool(pair_schema):
    from ollm import Inference
    from ollm.tools import Agent, GenerationConfig, InferenceBackend, Tool, ToolRegistry
    model = os.environ['OLLM_LIVE_MODEL']
    assert model in ('qwen3-next-80B', 'gpt-oss-20B')
    models_dir = Path(os.environ.get('OLLM_MODELS_DIR', './models'))
    assert (models_dir / model).is_dir(), 'Provision model weights first; tests never download them'
    inference = Inference(model, device=os.environ.get('OLLM_DEVICE', 'cuda:0'), logging=False)
    inference.ini_model(models_dir=str(models_dir), force_download=False)
    observed = []

    def add(a, b):
        observed.append((a, b))
        return a + b

    registry = ToolRegistry()
    registry.add(Tool('add', 'Add two numbers', pair_schema, add, requires_approval=False))
    backend = InferenceBackend(inference, generation=GenerationConfig(max_new_tokens=1024),
                               cache_dir=os.environ.get('OLLM_KV_CACHE_DIR'))
    result = Agent(backend, registry).run_sync(
        'You must call add with a=17 and b=25. Use its result to answer, do not calculate without the tool.')
    assert observed and observed[0] == (17, 25), observed
    assert '42' in result.text, result.text

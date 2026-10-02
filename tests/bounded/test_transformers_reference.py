"""Optional REAL Transformers comparisons, distinct from local tensor tests.

Enable OLLM_RUN_REFERENCE=1 in an environment with the desired Transformers
version. No pretrained weights/downloads: native classes consume tiny local
checkpoints. Missing family classes are explicit skips, not passing tests.
"""
import os
import pytest
import torch
from ollm.bounded.model import BoundedModel
from ollm.bounded.cache import DiskState

pytestmark = pytest.mark.reference


@pytest.mark.parametrize('family,class_name',[
    ('qwen2','Qwen2ForCausalLM'),('qwen3','Qwen3ForCausalLM'),
    ('qwen3_moe','Qwen3MoeForCausalLM'),('qwen3_next','Qwen3NextForCausalLM'),
    ('llama','LlamaForCausalLM'),
    ('qwen3_5','Qwen3_5ForCausalLM'),('qwen3_5_moe','Qwen3_5MoeForCausalLM'),
    ('deepseek_v2','DeepseekV2ForCausalLM'),('deepseek_v3','DeepseekV3ForCausalLM'),
])
def test_real_transformers_logits(make_checkpoint,family,class_name):
    if os.environ.get('OLLM_RUN_REFERENCE')!='1':
        pytest.skip('Set OLLM_RUN_REFERENCE=1 to run actual Transformers comparisons')
    transformers=pytest.importorskip('transformers')
    cls=getattr(transformers,class_name,None)
    if cls is None:
        pytest.skip(f'{class_name} not available in Transformers {transformers.__version__}')
    path,_,_=make_checkpoint(family, **({'norm_topk_prob':False} if family=='deepseek_v2' else {}))
    ref=cls.from_pretrained(str(path),local_files_only=True,torch_dtype=torch.float32,attn_implementation='eager')
    ref.eval()
    actual=BoundedModel(path,device='cpu',dtype=torch.float32)
    ids=torch.tensor([[1,5,7,3,9]])
    with torch.inference_mode():
        expected=ref(ids,use_cache=False).logits[0,-1].float()
        with DiskState() as state:
            logits=actual.logits(actual.forward_tokens(ids[0].tolist(),state,0))
    torch.testing.assert_close(logits,expected,atol=3e-5,rtol=5e-4)

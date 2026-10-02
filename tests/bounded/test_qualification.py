from dataclasses import replace
import json
import subprocess
import os
from pathlib import Path
import sys
import pytest
import torch
from ollm.bounded.budget import MemoryBudget
from ollm.bounded.model import BoundedModel
from ollm.bounded.qualification import benchmark, validate_profile


def test_cpu_report_can_never_certify_vram(make_checkpoint,tmp_path):
    p,_,_=make_checkpoint()
    budget=replace(MemoryBudget(),max_context_tokens=12,max_output_tokens=2)
    m=BoundedModel(p,device='cpu',dtype=torch.float32,budget=budget)
    report_path=tmp_path/'report.json'
    report=benchmark(m,prompt_tokens=3,rounds=2,report_path=report_path,hash_weights=True)
    assert report['status']=='cpu_reference_only'
    assert not report['memory_pass'] and not report['numerical_qualification']
    assert report['max_observed_prompt_tokens']==10
    assert len(report['cases'])==2
    assert len(report['checkpoint_content_sha256'])==64
    with pytest.raises(ValueError,match='real CUDA'):validate_profile(report_path,m)


def test_failed_measurement_writes_failure(make_checkpoint,tmp_path):
    p,_,_=make_checkpoint()
    budget=replace(MemoryBudget(),max_context_tokens=12,max_output_tokens=2,max_cache_bytes=1)
    m=BoundedModel(p,device='cpu',dtype=torch.float32,budget=budget)
    path=tmp_path/'failed.json'
    report=benchmark(m,prompt_tokens=3,rounds=2,report_path=path)
    assert report['status']=='failed' and not report['memory_pass']
    assert 'quota' in report['error']
    assert json.loads(path.read_text())['status']=='failed'


@pytest.mark.parametrize('cmd',[['--help'],['list'],['qualify','--help'],['generate','--help'],['inspect','--help']])
def test_cli_no_download_help(cmd):
    r=subprocess.run([sys.executable,'-m','ollm.bounded']+cmd,capture_output=True,text=True,
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')})
    assert r.returncode==0,r.stderr


def test_qwen3vl_requires_multimodal_qualification(make_checkpoint, tmp_path):
    import torch
    from ollm.bounded.budget import MemoryBudget
    from ollm.bounded.model import load_bounded_model
    from ollm.bounded.qualification import benchmark, benchmark_multimodal
    p,_,_=make_checkpoint('qwen3_vl')
    budget=MemoryBudget(max_visual_tokens=16,max_images=2,max_context_tokens=32,
                        max_output_tokens=2,prefill_tokens=8,attention_block_tokens=4)
    model=load_bounded_model(p,device='cpu',dtype=torch.float32,budget=budget)
    with pytest.raises(ValueError,match='qualify-vl'):
        benchmark(model,prompt_tokens=8,rounds=2)
    report=benchmark_multimodal(model,rounds=2,report_path=tmp_path/'vl.json')
    assert report['status']=='cpu_reference_only'
    assert report['multimodal_qualification'] is True
    assert report['max_observed_visual_tokens']==16
    assert report['max_observed_images']==2

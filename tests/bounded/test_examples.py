import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('name',['chat','native_tools','mcp_tools','qwen_agent'])
def test_example_help_needs_no_model_or_sdk(name):
    root=Path(__file__).resolve().parents[2]
    result=subprocess.run([sys.executable,str(root/'examples'/'bounded'/(name+'.py')),'--help'],
        capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(root/'src')})
    assert result.returncode==0,result.stderr

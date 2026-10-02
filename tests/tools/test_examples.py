"""Ensure every CLI can describe itself without optional SDKs or model loading."""
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['python_tools', 'mcp_server', 'mcp_probe', 'mcp_tools',
                                  'mixed_tools', 'qwen_agent_tools', 'qwen_agent_mcp'])
def test_example_help(name):
    result = subprocess.run([sys.executable, str(ROOT / f'examples/tools/{name}.py'), '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'usage:' in result.stdout


def test_sdk_helper_paths():
    # Loading test_sdk only defines functions; no SDK import/network call occurs.
    values = runpy.run_path(str(ROOT / 'tests/tools/test_sdk.py'))
    assert values['ROOT'] == ROOT
    assert values['SERVER'].is_file()

import json, tempfile, os, shutil
from pathlib import Path
from PIL import Image
import pytest

from eyetools.core.env_manager import EnvManager

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def _make_image() -> str:
    td = tempfile.mkdtemp()
    p = Path(td) / "img.jpg"
    Image.new("RGB", (64, 64), color=(120, 45, 210)).save(p)
    return str(p)


def _run_tool(task: str):
    if shutil.which("uv") is None:  # safety
        pytest.skip("uv not installed")
    img = _make_image()
    # choose cuda if available inside env
    code = (
        "import sys,json,torch;"
        "sys.path.insert(0,'tools');"
        f"from classification.tool_impl import load_tool;"
        f"device='cuda' if torch.cuda.is_available() else 'cpu';"
        f"out=load_tool('{task}').predict('{img}');"
        "print(json.dumps(out))"
    )
    mgr = EnvManager(WORKSPACE_ROOT)
    meta = {"environment_ref": "py312", "runtime": {}}
    cp = mgr.run_in_env(meta, ["python", "-c", code])
    assert cp.returncode == 0, cp.stderr
    line = cp.stdout.strip().splitlines()[-1]
    return json.loads(line)


def test_modality_in_env():
    out = _run_tool("modality")
    assert out["task"] == "modality"
    assert len(out["predictions"]) >= 1


def test_cfp_age_in_env():
    out = _run_tool("cfp_age")
    assert out["task"] == "cfp_age"
    assert 40 <= out["prediction"] <= 70


def test_cfp_quality_in_env():
    out = _run_tool("cfp_quality")
    assert out["task"] == "cfp_quality"
    assert len(out["predictions"]) >= 1


def test_laterality_in_env():
    out = _run_tool("laterality")
    assert out["task"] == "laterality"
    assert len(out["predictions"]) == 1


def test_multidis_in_env():
    out = _run_tool("multidis")
    assert out["task"] == "multidis"
    # probabilities dict present or predictions list
    assert "probabilities" in out or "predictions" in out

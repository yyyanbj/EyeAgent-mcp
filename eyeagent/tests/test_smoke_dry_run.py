import os
import json
from pathlib import Path

# Ensure DRY-RUN mode
os.environ.setdefault("EYEAGENT_DRY_RUN", "1")

# Prefer local package path
import sys
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "eyeagent"))

from eyeagent.diagnostic_workflow import run_diagnosis


def test_smoke_dry_run():
    patient = {"id": "test-001", "age": 55, "instruction": "Screening."}
    images = [
        {"image_id": "img-1", "path": "/tmp/od.jpg"},
        {"image_id": "img-2", "path": "/tmp/os.jpg"},
    ]
    report = run_diagnosis(patient, images)
    assert isinstance(report, dict)
    assert report.get("final_report") is not None
    # Validate minimal keys
    fr = report.get("final_report") or {}
    assert "diagnoses" in fr
    assert "lesions" in fr
    assert "management" in fr
    assert "reasoning" in fr

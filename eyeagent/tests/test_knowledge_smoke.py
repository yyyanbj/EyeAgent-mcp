import os
from pathlib import Path
import sys

os.environ.setdefault("EYEAGENT_DRY_RUN", "1")

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "eyeagent"))

from eyeagent.diagnostic_workflow import run_diagnosis


def test_knowledge_integration_smoke():
    patient = {"id": "kn-001", "age": 60, "instruction": "What management for suspected AMD?"}
    images = [{"image_id": "OD", "path": "/tmp/od.jpg"}]
    report = run_diagnosis(patient, images)
    assert isinstance(report, dict)
    fr = report.get("final_report") or {}
    assert "reasoning" in fr
    # knowledge block is optional, but ensure pipeline doesn't crash
    # If knowledge agent executed, it is included back into the final fragment
    # Skip strict assertion due to conditional routing

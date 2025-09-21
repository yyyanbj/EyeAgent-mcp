import os
from typing import Dict, Any, List

os.environ.setdefault("EYEAGENT_DRY_RUN", "1")

from eyeagent.diagnostic_workflow import run_diagnosis


def main():
    patient: Dict[str, Any] = {
        "id": "demo-oph-001",
        "age": 58,
        "instruction": "Comprehensive retinal screening with knowledge references."
    }
    images: List[Dict[str, Any]] = [
        {"image_id": "OD", "path": "/tmp/od.jpg"},
        {"image_id": "OS", "path": "/tmp/os.jpg"},
    ]
    # Optionally request a profile-driven pipeline with knowledge step if configured in config/pipelines.yml
    # os.environ.setdefault("EYEAGENT_PIPELINE_PROFILE", "default")
    report = run_diagnosis(patient, images)
    from pprint import pprint
    pprint(report.get("final_report"))


if __name__ == "__main__":
    main()

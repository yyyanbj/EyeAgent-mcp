from __future__ import annotations
import json
import os
from pathlib import Path

import types
import asyncio

from eyeagent.cli import main as eyeagent_main


async def _stub_run(patient, images, **kwargs):
    # In incremental mode, CLI passes merged images and prior
    assert isinstance(images, list)
    assert any(i.get("image_id") == "IMG1" for i in images), "previous image missing"
    assert any(i.get("image_id") == "IMG2" for i in images), "new image missing"
    # prior only passed for supported backends; CLI defaults to profile in incremental mode
    prior = kwargs.get("prior")
    assert prior is not None and isinstance(prior, dict)
    return {"ok": True, "images": images, "patient": patient, "case_id": kwargs.get("case_id")}


def test_cli_incremental_merges_and_calls_prior(monkeypatch, tmp_path):
    # Isolate cases directory
    cases_dir = tmp_path / "cases"
    os.environ["EYEAGENT_CASES_DIR"] = str(cases_dir)
    case_id = "testcase123"
    case_dir = cases_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal prior trace.json
    prior_doc = {
        "case_id": case_id,
        "patient": {"patient_id": "P1"},
        "images": [
            {"image_id": "IMG1", "path": "/tmp/old.jpg"}
        ],
        "events": []
    }
    (case_dir / "trace.json").write_text(json.dumps(prior_doc), encoding="utf-8")

    # Monkeypatch dispatcher to our stub
    import eyeagent.diagnostic_workflow as dw
    monkeypatch.setattr(dw, "run_diagnosis_async", _stub_run)

    # Run CLI incremental with a new image
    rc = eyeagent_main([
        "run",
        "--case-id", case_id,
        "--incremental",
        "--images", json.dumps([{"image_id": "IMG2", "path": "/tmp/new.jpg"}])
    ])

    assert rc == 0

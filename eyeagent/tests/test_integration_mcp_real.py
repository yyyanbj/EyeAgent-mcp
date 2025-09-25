import os
import sys
from pathlib import Path
import pytest

# This test uses a real MCP server. It is skipped by default unless explicitly enabled.
# Enable by setting EYEAGENT_RUN_REAL_MCP=1 (and ensure the server is running at MCP_SERVER_URL).

RUN_REAL = os.getenv("EYEAGENT_RUN_REAL_MCP", "0").lower() in ("1", "true", "yes")
MCP_URL = os.getenv("MCP_SERVER_URL") or os.getenv("EYEAGENT_MCP_URL") or "http://localhost:5789/mcp"

@pytest.mark.skipif(not RUN_REAL, reason="Set EYEAGENT_RUN_REAL_MCP=1 to run real MCP integration test.")
def test_with_real_mcp_server():
    # Make sure DRY-RUN is disabled and MCP URL is set BEFORE importing workflow
    os.environ.pop("EYEAGENT_DRY_RUN", None)
    os.environ["MCP_SERVER_URL"] = MCP_URL

    # Resolve a real image file from the repository (eyetools examples)
    repo_root = Path(__file__).resolve().parents[2]
    # Prefer a stable test image
    candidates = [
        repo_root / "eyetools" / "examples" / "test_images" / "DR lesion.jpg",
        repo_root / "eyetools" / "examples" / "test_images" / "DR lesion1.jpg",
        repo_root / "eyetools" / "examples" / "test_images" / "retinal vessel.jpg",
    ]
    image_path = None
    for p in candidates:
        if p.exists():
            image_path = p
            break
    if not image_path:
        pytest.skip("No example test image found in eyetools/examples/test_images")

    # Import after env setup
    sys.path.insert(0, str(repo_root / "eyeagent"))
    from eyeagent.diagnostic_workflow import run_diagnosis

    patient = {"id": "it-mcp-001", "age": 55, "instruction": "Comprehensive screening."}
    images = [{"image_id": image_path.stem, "path": str(image_path)}]

    # Execute full workflow using real MCP tools
    report = run_diagnosis(patient, images)
    assert isinstance(report, dict)
    fr = report.get("final_report") or {}
    # Minimal sanity checks
    assert "reasoning" in fr
    assert "diagnoses" in fr

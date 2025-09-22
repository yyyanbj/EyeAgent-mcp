import time
from fastapi.testclient import TestClient

from eyetools.mcp_server import create_app
from eyetools.core.tool_manager import ToolManager


def get_test_app():
    # Create app with eager mode so tools are registered; do not preload segmentation models automatically
    app = create_app(
        lifecycle_mode="eager",
        parallel_subprocess=True,
    )
    return app


def test_single_tool_warmup(monkeypatch):
    app = get_test_app()
    client = TestClient(app)

    # Discover a tool id that supports warmup (segmentation or classification). We'll introspect resource snapshot
    # Access registry through app state (FastAPI TestClient mounts app with state preserved)
    tool_ids = [m.id for m in app.state.registry.list()]
    assert tool_ids, "No tools discovered for warmup test"
    target_tool = tool_ids[0]

    # Call warmup
    resp = client.post(f"/admin/warmup?tool_id={target_tool}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data.get("tool_id") == target_tool
    assert data.get("status") in ("ok", "skipped", "error")

    # After warmup the resource snapshot should have last_warmup timestamp for that tool (subprocess tools only)
    snapshot2 = client.get("/admin/resources").json()
    last_warmup = snapshot2.get("subprocess", {}).get("last_warmup", {})
    # Not all tools are subprocess; if missing we just ensure no error path
    # If present, timestamp should be recent
    if target_tool in last_warmup:
        now = time.time()
        assert now - last_warmup[target_tool] < 60, "last_warmup timestamp not updated recently"


def test_batch_warmup():
    app = get_test_app()
    client = TestClient(app)
    # If there are many subprocess tools (segmentation variants), limit runtime filter test only
    subproc_tool_ids = [m.id for m in app.state.registry.list() if m.runtime.get("load_mode", "auto") == "subprocess"]
    if len(subproc_tool_ids) > 5:
        # Just test subprocess filter path (faster) instead of warming everything
        resp2 = client.post("/admin/warmup_all?runtime=subprocess")
        assert resp2.status_code == 200, resp2.text
        data2 = resp2.json()
        assert "count" in data2
        for r in data2["results"]:
            assert r.get("runtime") == "subprocess"
    else:
        resp = client.post("/admin/warmup_all")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert "count" in data
        assert isinstance(data["results"], list)
        for r in data["results"]:
            assert "tool_id" in r and "status" in r

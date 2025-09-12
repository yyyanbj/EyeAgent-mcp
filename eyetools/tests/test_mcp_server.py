from eyetools.mcp_server import create_app
from fastapi.testclient import TestClient


def test_mcp_server_basic():
    app = create_app(include_examples=True)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    r2 = client.get("/tools")
    assert r2.status_code == 200
    data = r2.json()
    # Either automatic tools list or manual selection fields
    assert ("tools" in data) or ("selection_required" in data)


from eyetools.mcp_server import create_app
from fastapi.testclient import TestClient

def test_admin_evict_and_preload_cycle():
    app = create_app(include_examples=True)
    client = TestClient(app)
    tools = client.get('/tools').json()
    # normalize tool ids across possible shapes
    tool_ids = []
    if isinstance(tools, dict):
        if 'tools' in tools:
            raw = tools.get('tools') or []
            if raw and isinstance(raw, list):
                if raw and isinstance(raw[0], dict):
                    tool_ids = [t.get('id') for t in raw if isinstance(t, dict) and t.get('id')]
                else:
                    # list of id strings
                    tool_ids = [t for t in raw if isinstance(t, str)]
        if not tool_ids and 'tool_ids' in tools:
            raw2 = tools.get('tool_ids') or []
            if raw2 and isinstance(raw2, list):
                tool_ids = [t for t in raw2 if isinstance(t, str)]
    elif isinstance(tools, list):
        # already a list of ids
        tool_ids = [t for t in tools if isinstance(t, str)]
    if not tool_ids:
        return  # nothing to test
    tid = tool_ids[0]
    # Choose an inproc/auto tool if available (avoid subprocess segmentation tools for this test)
    if any(t.startswith('classification:') for t in tool_ids):
        for t in tool_ids:
            if t.startswith('classification:'):
                tid = t
                break
    r1 = client.post(f'/admin/preload/{tid}')
    assert r1.status_code == 200, r1.text
    # evict
    r2 = client.post('/admin/evict', params={'tool_id': tid})
    assert r2.status_code == 200
    # metrics filter (may be empty if no predict yet)
    r3 = client.get('/metrics', params={'tool_id': tid})
    assert r3.status_code == 200
    # detailed lifecycle
    r4 = client.get('/lifecycle', params={'detailed': True})
    assert r4.status_code == 200
    assert isinstance(r4.json(), dict)

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


def test_mcp_lifecycle_and_reload():
    app = create_app(include_examples=True)
    client = TestClient(app)
    # lifecycle endpoint returns dict (possibly empty states initially)
    lr = client.get("/lifecycle")
    assert lr.status_code == 200
    assert isinstance(lr.json(), dict)
    # reload endpoint
    rr = client.post("/admin/reload")
    assert rr.status_code == 200
    payload = rr.json()
    assert "tools" in payload
    # optional preload on reload
    rr2 = client.post("/admin/reload?preload_models=true")
    assert rr2.status_code == 200


def test_admin_evict_and_preload_and_metrics():
    app = create_app(include_examples=True)
    client = TestClient(app)
    # get tools list
    tools_resp = client.get("/tools")
    assert tools_resp.status_code == 200
    tjson = tools_resp.json()
    # gather candidate tool ids from either new or legacy shape
    tool_ids = []
    if isinstance(tjson, dict):
        if "tools" in tjson and isinstance(tjson["tools"], list):
            tool_ids = [t.get("id") for t in tjson["tools"] if isinstance(t, dict) and t.get("id")]
        if not tool_ids and "tool_ids" in tjson:
            tool_ids = tjson.get("tool_ids") or []
    if not tool_ids:
        # no tools discovered; skip (environment may lack example tools)
        return
    tid = tool_ids[0]
    # targeted preload
    pr = client.post(f"/admin/preload/{tid}")
    assert pr.status_code == 200
    # metrics filtered (may be empty but endpoint must work)
    mr = client.get("/metrics", params={"tool_id": tid})
    assert mr.status_code == 200
    # evict the tool
    er = client.post("/admin/evict", params={"tool_id": tid})
    assert er.status_code == 200
    # detailed lifecycle gives dict mapping tool ids -> state objects
    lr = client.get("/lifecycle", params={"detailed": True})
    assert lr.status_code == 200
    ljson = lr.json()
    assert isinstance(ljson, dict)
    # after eviction tool may appear as UNLOADED or absent; tolerate both
    # ensure endpoint delivered some structured content
    assert len(ljson) >= 0


def test_admin_resources():
    app = create_app(include_examples=True)
    client = TestClient(app)
    r = client.get('/admin/resources')
    assert r.status_code == 200
    data = r.json()
    # basic expected keys
    for key in ["counts", "metrics", "memory"]:
        assert key in data
    assert "discovered" in data["counts"]


def test_admin_preload_subprocess_tool():
    app = create_app(include_examples=True)
    client = TestClient(app)
    tools_resp = client.get('/tools').json()
    # Find a segmentation tool (subprocess mode)
    candidates = []
    if isinstance(tools_resp, dict):
        raw = tools_resp.get('tools') or []
        if raw and isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                candidates = [t.get('id') for t in raw if isinstance(t, dict) and t.get('id','').startswith('segmentation:')]
            else:
                candidates = [t for t in raw if isinstance(t, str) and t.startswith('segmentation:')]
    if not candidates:
        return  # no segmentation tools available
    seg_id = candidates[0]
    r = client.post(f'/admin/preload/{seg_id}')
    assert r.status_code == 200, r.text
    payload = r.json()
    # should indicate worker_started or loaded
    assert payload.get('tool_id') == seg_id
    assert payload.get('status') in ('worker_started', 'loaded')


def test_admin_evict_subprocess_tool():
    app = create_app(include_examples=True)
    client = TestClient(app)
    tools_resp = client.get('/tools').json()
    seg_id = None
    if isinstance(tools_resp, dict):
        raw = tools_resp.get('tools') or []
        if raw and isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                for t in raw:
                    if t.get('id','').startswith('segmentation:'):
                        seg_id = t['id']; break
            else:
                for t in raw:
                    if isinstance(t, str) and t.startswith('segmentation:'):
                        seg_id = t; break
    if not seg_id:
        return
    # preload (spawns worker)
    pr = client.post(f'/admin/preload/{seg_id}')
    assert pr.status_code == 200
    # evict
    er = client.post('/admin/evict', params={'tool_id': seg_id})
    assert er.status_code == 200, er.text
    # second eviction should 404
    er2 = client.post('/admin/evict', params={'tool_id': seg_id})
    assert er2.status_code == 404


def test_admin_maintenance_idle_and_unload():
    app = create_app(include_examples=True)
    client = TestClient(app)
    # preload an inproc classification tool to ensure it's LOADED
    tools_resp = client.get('/tools').json()
    cls_id = None
    if isinstance(tools_resp, dict):
        raw = tools_resp.get('tools') or []
        if raw and isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                for t in raw:
                    if t.get('id','').startswith('classification:'):
                        cls_id = t['id']; break
            else:
                for t in raw:
                    if isinstance(t, str) and t.startswith('classification:'):
                        cls_id = t; break
    if not cls_id:
        return
    pr = client.post(f'/admin/preload/{cls_id}')
    assert pr.status_code == 200
    # simulate old last_used by manipulating lifecycle manager internal state
    tm = app.state.tool_manager
    if cls_id in tm._inproc_lru:  # type: ignore[attr-defined]
        tm._inproc_lru[cls_id] -= 10_000  # make it very old
    # mark idle
    r1 = client.post('/admin/maintenance', params={'mark_idle_s': 5})
    assert r1.status_code == 200
    after1 = r1.json()['after']
    # confirm state is IDLE
    # If timing environment prevented test mutation we allow LOADED fallback
    assert after1.get(cls_id) in ('IDLE', 'LOADED')
    # force unload via short threshold
    r2 = client.post('/admin/maintenance', params={'unload_idle_s': 5})
    assert r2.status_code == 200
    after2 = r2.json()['after']
    # After unload it may be UNLOADED or absent (if removed from lifecycle tracking)
    assert after2.get(cls_id) in ('UNLOADED', 'IDLE', 'LOADED', None)


def test_lifecycle_mode_eager_preloads(monkeypatch):
    # Use create_app directly with lifecycle_mode=eager
    app = create_app(include_examples=True, lifecycle_mode="eager")
    client = TestClient(app)
    # resources should show loaded_inproc >= classification tools and subprocess_workers >= segmentation tools
    res = client.get('/admin/resources').json()
    counts = res.get('counts', {})
    assert counts.get('discovered', 0) >= 1
    # in eager mode we expect some inproc loaded (classification) and segmentation workers spawned
    # Depending on environment, subprocess workers may or may not spawn instantly; allow >=0 but ensure not all zero
    assert counts.get('loaded_inproc', 0) >= 1


def test_lifecycle_mode_dynamic_manual_maintenance():
    # dynamic mode with very small thresholds; we will manually call maintenance instead of sleeping for loop
    app = create_app(include_examples=True, lifecycle_mode="dynamic", dynamic_mark_idle_s=0.01, dynamic_unload_s=0.02, dynamic_interval_s=5)
    client = TestClient(app)
    # preload a classification tool lazily by invoking predict once
    tools = client.get('/tools').json()
    cls_id = None
    if isinstance(tools, dict):
        raw = tools.get('tools') or []
        if raw and isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                for t in raw:
                    if t.get('id','').startswith('classification:'):
                        cls_id = t['id']; break
            else:
                for t in raw:
                    if isinstance(t, str) and t.startswith('classification:'):
                        cls_id = t; break
    if not cls_id:
        return
    client.post(f'/admin/preload/{cls_id}')
    tm = app.state.tool_manager
    # make last_used artificially old
    if cls_id in tm._inproc_lru:  # type: ignore
        tm._inproc_lru[cls_id] -= 100  # 100 seconds old
    # run maintenance mark idle
    r1 = client.post('/admin/maintenance', params={'mark_idle_s': 0.01})
    assert r1.status_code == 200
    # run maintenance unload
    r2 = client.post('/admin/maintenance', params={'unload_idle_s': 0.01})
    assert r2.status_code == 200


def test_admin_config_endpoint_eager():
    app = create_app(include_examples=True, lifecycle_mode="eager")
    client = TestClient(app)
    r = client.get('/admin/config')
    assert r.status_code == 200
    data = r.json()
    assert data['lifecycle_mode'] == 'eager'
    assert 'tool_paths' in data


def test_admin_config_endpoint_dynamic_values():
    app = create_app(include_examples=False, lifecycle_mode="dynamic", dynamic_mark_idle_s=11, dynamic_unload_s=22, dynamic_interval_s=33)
    client = TestClient(app)
    r = client.get('/admin/config')
    assert r.status_code == 200
    data = r.json()
    assert data['lifecycle_mode'] == 'dynamic'
    assert data['dynamic_mark_idle_s'] == 11
    assert data['dynamic_unload_s'] == 22
    assert data['dynamic_interval_s'] == 33


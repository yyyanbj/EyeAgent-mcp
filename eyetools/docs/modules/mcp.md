# MCP Integration

This module integrates the EyeTools server with FastMCP (Model Context Protocol) so MCP-compatible clients can discover and invoke tools.

## Key Capabilities
- Optional activation (disable via `enable_mcp=False` or omit `fastmcp` dependency)
- Automatic registration of all discovered tools after registry population (before any eager preload)
- Multiple parameter mapping strategies (in order):
  1. explicit (generated async function with explicit parameters inferred from `input_schema` or `meta.io`)
  2. kwargs (single `**kwargs` entry point)
  3. payload (single dictionary argument) – fallback for older FastMCP builds that reject `**kwargs`
- Input schema best-effort inference from `ToolMeta.io` (`io.input` if present, else treat remaining keys as string properties)
- Reload-safe re‑registration (`/admin/reload` triggers `_mcp_reregister`)
- Introspection endpoint `/mcp/tools` with: name, description, input_schema, param_style, lifecycle_state
- Manual refresh without re-discovery: `POST /mcp/refresh`

## Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `EYETOOLS_MCP_MOUNT_PATH` | Change mount path for MCP app (`/` or `/mcp` etc.) | `/` |
| `EYETOOLS_MCP_COMPAT_MOUNT` | Also mount at `/mcp` when using custom path (set to `1`) | disabled |
| `EYETOOLS_MCP_PREFER_EXPLICIT` | Disable explicit param generation when set to `0` | `1` |

## Lifecycle Strategy
MCP registration occurs after tools are DISCOVERED + REGISTERED but before any eager preload. UNLOADED / IDLE states do not remove MCP routes: calls lazily re-load tools. `/admin/reload` rebuilds registry, then `_mcp_reregister` reconstructs MCP registration.

## Minimal Server Test (Fast Path)
Run only the MCP server test without loading all heavy tool packages.

### 1. Create a temporary minimal tool config
If you want to avoid scanning the full `tools/` tree, point to an empty temp directory.
```bash
mkdir -p /tmp/eyetools-empty
export EYETOOLS_TOOL_PATHS=/tmp/eyetools-empty
```
This ensures discovery yields zero tools but server endpoints still initialize.

### 2. Run the single test file
```bash
pytest -q tests/test_mcp_server.py::test_mcp_server_basic \
  -k mcp_server_basic \
  --maxfail=1 --disable-warnings
```
(You can remove the `-k` if you want the whole file.)

### 3. Run with MCP enabled but no tools (validation only)
```bash
python - <<'PY'
from eyetools.mcp_server import create_app
from fastapi.testclient import TestClient
app = create_app(include_examples=False, tool_paths=[])  # No discovery
c = TestClient(app)
print('health:', c.get('/health').json())
print('mcp tools:', c.get('/mcp/tools').json())
PY
```
Expected output should show `tools: 0` and `enabled: true` if `fastmcp` is installed.

### 4. Overriding Mount Path
```bash
EYETOOLS_MCP_MOUNT_PATH=/mcp uvicorn eyetools.mcp_server:create_app --factory --port 8001
```
Then visit: `http://127.0.0.1:8001/mcp/tools`.

### 5. Refresh MCP Registration
After manual changes (e.g., env flips):
```bash
curl -X POST http://127.0.0.1:8001/mcp/refresh
```

## Adding a Lightweight Test Case (Optional)
You can add a focused test that asserts payload shape when no tools are discovered:
```python
# tests/test_mcp_tools_introspection.py
from eyetools.mcp_server import create_app
from fastapi.testclient import TestClient

def test_mcp_tools_empty():
    app = create_app(include_examples=False, tool_paths=[])
    client = TestClient(app)
    data = client.get('/mcp/tools').json()
    assert data['enabled'] is True
    assert data['count'] == 0
```

## Troubleshooting
| Symptom | Cause | Action |
|---------|-------|--------|
| `enabled: false` in `/mcp/tools` | `fastmcp` not installed | `pip install fastmcp` |
| All tools show `param_style=payload` | Old FastMCP rejects `**kwargs` & explicit build failed | Upgrade `fastmcp` or inspect logs for explicit build errors |
| Missing new tool after reload | Not re-registered | Call `POST /mcp/refresh` or use `/admin/reload` |
| 404 invoking tool via MCP | Tool id mismatch | Verify `tool_id` from `/mcp/tools` exactly |

## Future Extensions
- Output schema adaptation if model returns structured artifacts
- Auth / role filtering on MCP layer
- Tool disable/unregister endpoint
- Streaming predict wrapper

---
This document complements `docs/testing.md` and focuses specifically on MCP behavior.

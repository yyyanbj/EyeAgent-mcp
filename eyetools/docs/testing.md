# Testing & Quality

Current coverage:
- Discovery: test_discovery
- Classification placeholder functions: test_classification
- Metrics & errors: test_metrics_and_errors
- Role routing: test_role_router
- ToolManager inproc/subprocess: test_tool_manager_inproc / test_tool_manager_subprocess
- Added config/env/process/cli/server coverage.

## Minimal MCP Server Test

For fast CI signal we maintain a minimal MCP smoke test that:
1. Starts the FastAPI app with MCP enabled (automatic if `fastmcp` installed)
2. Supplies an empty `tool_paths` list (or points `EYETOOLS_TOOL_PATHS` to an empty directory)
3. Asserts `/mcp/tools` returns an empty tool list with `enabled: true`

Example:
```python
from eyetools.mcp_server import create_app
from fastapi.testclient import TestClient

def test_mcp_server_basic():
	app = create_app(include_examples=False, tool_paths=[])
	c = TestClient(app)
	data = c.get('/mcp/tools').json()
	assert data['enabled'] is True
	assert data['count'] == 0
```

Run just this test:
```bash
pytest -q tests/test_mcp_server.py::test_mcp_server_basic --maxfail=1
```

More advanced MCP behavior (param styles, refresh) is documented in `docs/modules/mcp.md` and can be covered by incremental tests later.

Potential future additions:
- Lifecycle eviction tests (mark_idle / unload_idle)
- Worker error propagation & timeout path
- Coverage & mutation testing (pytest-cov, mutmut)
 - Expanded MCP tests (explicit vs kwargs vs payload fallback)
 - `/mcp/refresh` behavior & post-reload re-registration


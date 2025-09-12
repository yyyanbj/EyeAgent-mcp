# CLI / MCP Server / Logging

## CLI
Command: `eyetools-mcp serve` supports host/port/tool-path/include-examples/role-config/reload.

## MCP Server Routes
- GET /health: health status
- GET /tools?role=xxx: role-filtered tool list
- POST /predict: {tool_id, request, role?}
- GET /metrics: metrics snapshot

## Logging
- core_logger: unified debug output for core modules.


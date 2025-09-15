# Quick Start

## Environment
The project uses `uv` for dependency and multi-Python management. Install `uv`:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

See the official docs for alternative installation methods.

## Install Dependencies
```
uv sync
```

## Run Tests
```
uv run pytest -q
```

## Launch MCP Server
Basic run (will auto-load local `./tools` directory if it exists):
```
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000
```
Add examples and an explicit tools directory (repeat `--tools-dir` to add multiple roots):
```
uv run eyetools-mcp serve --host 0.0.0.0 --port 8000 --include-examples --tools-dir tools
```
`--tools-dir` is an alias of `--tool-path`; both are accepted.

### Lifecycle Modes
Select how tools are loaded / unloaded:

* `eager` (default): preload every tool (inproc + subprocess) at startup.
* `lazy`: load on first request only.
* `dynamic`: lazy plus a background loop that marks tools idle and unloads them after inactivity thresholds.

Flags:
```
--lifecycle-mode {eager|lazy|dynamic}
--dynamic-mark-idle-s <seconds>
--dynamic-unload-s <seconds>
--dynamic-interval-s <seconds>
```

Examples:
```
# Lazy mode (on-demand)
uv run eyetools-mcp serve --lifecycle-mode lazy

# Dynamic: mark idle after 5m, unload after 15m, sweep every minute
uv run eyetools-mcp serve --lifecycle-mode dynamic \
	--dynamic-mark-idle-s 300 --dynamic-unload-s 900 --dynamic-interval-s 60
```

Legacy flags kept for backward compatibility (ignored if lifecycle-mode provided):
```
--preload            # preload all inproc tools
--preload-subprocess # also preload subprocess tools
```
`eager` mode supersedes these (always preloads everything). Use them only if you keep older scripts without updating to `--lifecycle-mode`.

Open http://localhost:8000/docs to view the FastAPI interactive docs.

## Tool Discovery
Use one (or many) `--tools-dir` / `--tool-path` flags to point at directories containing tool packages. You can also set environment variables:

* `EYETOOLS_TOOL_PATHS` (colon separated)
* `EYETOOLS_EXTRA_TOOL_PATHS` (colon separated)

If neither flags nor env vars are provided and a local `./tools` directory exists, it will be used automatically.

See `examples/tool_package_template` for a reference structure.

## Management Endpoints
Additional operational endpoints are exposed:

* `GET /lifecycle` – current lifecycle states for instantiated tools.
* `POST /admin/reload` – rediscover tools (query param `preload_models=true` to eagerly load again).
* `GET /metrics` – aggregated latency & count metrics. Add `?tool_id=<id>` for one tool.
* `POST /admin/preload/{tool_id}` – for inproc: instantiate & load; for subprocess: start worker.
* `POST /admin/evict?tool_id=<id>` – unload an inproc tool or terminate a subprocess worker.
* `GET /lifecycle?detailed=true` – include `last_used` timestamp & loaded flag per tool.
* `GET /admin/resources` – snapshot of counts, RSS memory, aggregate metrics (GPU if available).
* `POST /admin/maintenance` – run idle marking/unloading. Params: `mark_idle_s`, `unload_idle_s`.

These can be useful after adding new tool directories at runtime.

### Quick Examples
Preload a single tool:
```
curl -X POST http://localhost:8000/admin/preload/classification:modality
```
Evict it again:
```
curl -X POST 'http://localhost:8000/admin/evict?tool_id=classification:modality'
```
Inspect detailed lifecycle:
```
curl 'http://localhost:8000/lifecycle?detailed=true'
```
Tool-specific metrics:
```
curl 'http://localhost:8000/metrics?tool_id=classification:modality'
```
Resource snapshot:
```
curl http://localhost:8000/admin/resources
```
Run maintenance (mark idle >300s & unload idle >900s, sequential calls):
```
curl -X POST 'http://localhost:8000/admin/maintenance?mark_idle_s=300'
curl -X POST 'http://localhost:8000/admin/maintenance?unload_idle_s=900'
```


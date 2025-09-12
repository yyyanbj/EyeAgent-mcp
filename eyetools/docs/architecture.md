# Architecture Overview

EyeTools has two layers:
1. Legacy/compat layer `eyetools/`
2. New core under `eyetools/core/` (discovery, registration, scheduling, runtime)

Core flow:
```
discover (loader.discover_tools) -> parse config (config_loader) -> register (registry.ToolRegistry) ->
ToolManager chooses inproc | subprocess based on load_mode -> lazy model load -> predict -> metrics & lifecycle management
```

## Key Modules
- config_loader: parse config (single / variants / tools), normalize + defaults.
- registry: holds ToolMeta (thread-safe).
- loader: walks paths, calls config_loader, registers ToolMeta.
- tool_manager: predict dispatch, inproc instance cache, subprocess workers, metrics & lifecycle (LOADED/IDLE/UNLOADED).
- process_manager: spawn workers & JSON line protocol IPC.
- env_manager: compose `uv run` invocation with python tag + extra dependencies.
- role_router: role-based include/exclude + tag filtering.
- logging: core_logger utilities.
- worker_entry / worker_protocol: worker process protocol implementation.
- mcp_server: FastAPI app exposing health/tools/predict/metrics with role filtering.
- cli: command entry launching uvicorn.

## Lifecycle & Lazy Loading
ToolManager states:
- REGISTERED -> first inproc predict creates instance (prepare) -> LOADED
- mark_idle transitions LOADED -> IDLE (time-based)
- unload_idle transitions IDLE -> UNLOADED (eviction)
- next predict reloads -> LOADED

## Execution Modes
- inproc: direct import + instantiate.
- subprocess: ProcessManager spawns worker.

## Metrics
- predict_count, total_latency_ms, last_latency_ms, avg_latency_ms
- aggregate summary under `__aggregate__`.


"""MCP Server (FastAPI) wrapping ToolManager + RoleRouter.

Environment variables:
  EYETOOLS_TOOL_PATHS=path1:path2   (additional discovery roots)
  EYETOOLS_ROLE_CONFIG_JSON='{"roles":{...}}'

CLI will import this module and call create_app().
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import os, json
from pathlib import Path

from .core.registry import ToolRegistry
from .core.loader import discover_tools
from .core.tool_manager import ToolManager
from .core.role_router import RoleRouter
from .core.logging import core_logger


class PredictRequest(BaseModel):
    tool_id: str
    request: Dict[str, Any] = {}
    role: Optional[str] = None


def _load_role_config(role_config_path: Optional[str]) -> Dict[str, Any]:
    if role_config_path and Path(role_config_path).exists():
        import yaml
        with open(role_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    env_json = os.getenv("EYETOOLS_ROLE_CONFIG_JSON")
    if env_json:
        try:
            return json.loads(env_json)
        except json.JSONDecodeError as e:  # noqa
            core_logger.error(f"Invalid EYETOOLS_ROLE_CONFIG_JSON: {e}")
    return {}


def _compute_tool_paths(include_examples: bool, extra_paths: List[str]) -> List[Path]:
    paths: List[Path] = []
    # Extra paths (CLI or env)
    for p in extra_paths:
        if p:
            paths.append(Path(p).resolve())
    # Examples only if explicitly requested
    if include_examples:
        examples_dir = Path(__file__).resolve().parent / "examples"
        if examples_dir.exists():
            paths.append(examples_dir)
    # Deduplicate
    dedup = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup.append(p)
            seen.add(p)
    return dedup


def create_app(
    include_examples: bool = False,
    tool_paths: Optional[List[str]] = None,
    role_config_path: Optional[str] = None,
    preload: bool = False,
    preload_subprocess: bool = False,
    lifecycle_mode: Optional[str] = None,
    dynamic_mark_idle_s: float = 300.0,
    dynamic_unload_s: float = 900.0,
    dynamic_interval_s: float = 60.0,
    parallel_subprocess: bool = False,
    parallel_subprocess_workers: int = 4,
):
    registry = ToolRegistry()
    # Two env vars supported for flexibility: legacy EYETOOLS_TOOL_PATHS and new EYETOOLS_EXTRA_TOOL_PATHS
    extra_paths_env = os.getenv("EYETOOLS_TOOL_PATHS", "")
    extra_paths_env2 = os.getenv("EYETOOLS_EXTRA_TOOL_PATHS", "")
    env_paths = [p for p in (extra_paths_env.split(":") + extra_paths_env2.split(":")) if p]
    combined_paths = (tool_paths or []) + env_paths
    # If user did not specify any tool path flags or env vars, but a local ./tools directory exists, include it by default.
    if not combined_paths:
        default_dir = Path.cwd() / "tools"
        if default_dir.exists():
            combined_paths.append(str(default_dir))
    path_objs = _compute_tool_paths(include_examples, combined_paths)
    errors = discover_tools(path_objs, registry)
    if errors:
        for e in errors:
            core_logger.warning(f"discover error: {e}")
    # Log discovered tools summary
    if registry.list():
        core_logger.info("Discovered tools (count=%d):", len(registry.list()))
        for meta in registry.list():
            core_logger.info(
                " - id=%s package=%s variant=%s runtime=%s lazy=%s root=%s",
                meta.id,
                meta.package,
                meta.variant,
                meta.runtime.get("load_mode", "auto"),
                meta.model.get("lazy", True),
                meta.root_dir,
            )
    else:
        core_logger.warning("No tools discovered. Check --tools-dir paths or environment variables.")
    tm = ToolManager(registry)
    role_router = RoleRouter(_load_role_config(role_config_path))

    app = FastAPI(title="eyetools-mcp", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "tools": len(registry.list())}

    @app.get("/tools")
    def list_tools(role: Optional[str] = Query(None)):
        result = role_router.filter_tools(role, registry.list())
        return result.to_payload()

    @app.post("/predict")
    def predict(req: PredictRequest):
        # Optional role checking: ensure tool in allowed set if role provided and auto mode
        if req.role:
            fr = role_router.filter_tools(req.role, registry.list())
            if not fr.selection_required and req.tool_id not in fr.tool_ids:
                raise HTTPException(status_code=403, detail="Tool not permitted for role")
        try:
            data = tm.predict(req.tool_id, req.request)
        except KeyError:
            raise HTTPException(status_code=404, detail="Tool not found")
        except Exception as e:  # noqa
            raise HTTPException(status_code=500, detail=str(e))
        return {"tool_id": req.tool_id, "data": data}

    @app.get("/metrics")
    def metrics(tool_id: Optional[str] = Query(None)):
        return tm.get_metrics(tool_id=tool_id)

    @app.get("/lifecycle")
    def lifecycle(detailed: bool = Query(False)):
        return tm.lifecycle_states_detailed() if detailed else tm.lifecycle_states()

    @app.post("/admin/reload")
    def admin_reload(preload_models: bool = False):
        """Rediscover tools using original discovery context.
        Optionally preload models again.
        """
        ctx = app.state.discovery_ctx
        registry.clear()
        errors = discover_tools(_compute_tool_paths(ctx["include_examples"], (ctx["tool_paths"] or [])), registry)
        if errors:
            for e in errors:
                core_logger.warning(f"reload discover error: {e}")
        if preload_models:
            tm.preload_all()
        return {"tools": [m.id for m in registry.list()], "errors": [str(e) for e in errors]}

    @app.post("/admin/evict")
    def admin_evict(tool_id: str = Query(...)):
        # unified unload supporting inproc or subprocess workers
        ok = tm.unload_any(tool_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Tool not loaded or not found")
        return {"tool_id": tool_id, "status": "unloaded"}

    @app.post("/admin/preload/{tool_id}")
    def admin_preload_one(tool_id: str):
        meta = registry.get(tool_id)
        if not meta:
            raise HTTPException(status_code=404, detail="Tool not found")
        try:
            result = tm.ensure_loaded(tool_id)
        except Exception as e:  # noqa
            raise HTTPException(status_code=500, detail=str(e))
        return result

    @app.get("/admin/resources")
    def admin_resources():
        return tm.resource_snapshot()

    @app.post("/admin/maintenance")
    def admin_maintenance(mark_idle_s: float | None = Query(None), unload_idle_s: float | None = Query(None)):
        return tm.maintenance(mark_idle_s=mark_idle_s, unload_idle_s=unload_idle_s)

    @app.post("/admin/warmup")
    def admin_warmup(tool_id: str = Query(...)):
        try:
            return tm.warmup_tool(tool_id)
        except Exception as e:  # noqa
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/admin/warmup_all")
    def admin_warmup_all(runtime: str | None = Query(None, description="Filter: subprocess|inproc")):
        if runtime not in (None, "subprocess", "inproc"):
            raise HTTPException(status_code=400, detail="runtime must be subprocess or inproc")
        try:
            return tm.warmup_all(runtime=runtime)
        except Exception as e:  # noqa
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/admin/config")
    def admin_config():
        ctx = app.state.discovery_ctx
        return {
            "lifecycle_mode": ctx.get("lifecycle_mode"),
            "dynamic_mark_idle_s": ctx.get("dynamic_mark_idle_s"),
            "dynamic_unload_s": ctx.get("dynamic_unload_s"),
            "dynamic_interval_s": ctx.get("dynamic_interval_s"),
            "tool_paths": ctx.get("tool_paths"),
            "include_examples": ctx.get("include_examples"),
        }

    # Lifecycle mode handling precedence:
    # 1. lifecycle_mode if provided (eager|lazy|dynamic)
    # 2. fallback to legacy --preload flags
    if lifecycle_mode:
        app.state.lifecycle_mode = lifecycle_mode
        if lifecycle_mode == "eager":
            # In eager mode we always include subprocess tools regardless of legacy preload_subprocess flag
            core_logger.info(
                "[lifecycle] eager mode: preloading all tools (include_subprocess=True parallel_subprocess=%s)",
                parallel_subprocess,
            )
            tm.preload_all(
                include_subprocess=True,
                parallel_subprocess=parallel_subprocess,
                max_workers=parallel_subprocess_workers,
            )  # eager always includes subprocess
        elif lifecycle_mode == "lazy":
            core_logger.info("[lifecycle] lazy mode: models load on first request")
            # nothing extra
        elif lifecycle_mode == "dynamic":
            core_logger.info(
                "[lifecycle] dynamic mode: lazy load + background maintenance idle=%ss unload=%ss interval=%ss",
                dynamic_mark_idle_s,
                dynamic_unload_s,
                dynamic_interval_s,
            )
            # Start background maintenance task
            import asyncio

            async def _maintenance_loop():  # pragma: no cover (timing loop)
                await asyncio.sleep(dynamic_interval_s)
                while True:
                    try:
                        tm.maintenance(mark_idle_s=dynamic_mark_idle_s, unload_idle_s=dynamic_unload_s)
                    except Exception as e:  # noqa
                        core_logger.exception(f"maintenance loop error: {e}")
                    await asyncio.sleep(dynamic_interval_s)

            @app.on_event("startup")
            async def _start_maintenance():  # pragma: no cover
                if getattr(app.state, "_maintenance_started", False):
                    return
                loop = asyncio.get_event_loop()
                loop.create_task(_maintenance_loop())
                app.state._maintenance_started = True
        else:
            core_logger.warning("Unknown lifecycle_mode=%s (ignored)", lifecycle_mode)
    else:
        # Legacy preload behavior
        if preload:
            core_logger.info("Preloading all tools (subprocess=%s)...", preload_subprocess)
            tm.preload_all(include_subprocess=preload_subprocess)

    # Expose references for instrumentation/introspection
    app.state.tool_manager = tm
    app.state.registry = registry
    app.state.role_router = role_router
    app.state.discovery_ctx = {
        "include_examples": include_examples,
        "tool_paths": tool_paths,
        "role_config_path": role_config_path,
        "lifecycle_mode": lifecycle_mode,
        "dynamic_mark_idle_s": dynamic_mark_idle_s,
        "dynamic_unload_s": dynamic_unload_s,
        "dynamic_interval_s": dynamic_interval_s,
    }

    return app

__all__ = ["create_app"]

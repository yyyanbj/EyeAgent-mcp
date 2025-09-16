"""MCP Server (FastAPI) wrapping ToolManager + RoleRouter.

Environment variables:
  EYETOOLS_TOOL_PATHS=path1:path2   (additional discovery roots)
  EYETOOLS_ROLE_CONFIG_JSON='{"roles":{...}}'

CLI will import this module and call create_app().
"""
from __future__ import annotations
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Callable
import os, json
from pathlib import Path

from .core.registry import ToolRegistry
from .core.loader import discover_tools
from .core.tool_manager import ToolManager
from .core.role_router import RoleRouter
from .core.logging import core_logger

# Optional fastmcp import. We keep it optional so existing users without MCP needs are not blocked.
try:  # pragma: no cover - import side effect
    from fastmcp import FastMCP  # type: ignore
except Exception:  # noqa
    FastMCP = None  # type: ignore


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
    enable_mcp: bool = True,
    # Revert default back to root for backward compatibility; can override via param or EYETOOLS_MCP_MOUNT_PATH env.
    mcp_mount_path: str = "/",
):
    # Allow environment variable to override mount path regardless of default/argument
    mount_env = os.getenv("EYETOOLS_MCP_MOUNT_PATH")
    if mount_env:
        if not mount_env.startswith("/"):
            mount_env = "/" + mount_env
        mcp_mount_path = mount_env
    # Normalize trailing slash (except keep root as '/')
    if mcp_mount_path != "/" and mcp_mount_path.endswith("/"):
        mcp_mount_path = mcp_mount_path[:-1]
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

    # ---- MCP Integration (refactored) ----------------------------------------
    def _setup_mcp():
        if not enable_mcp:
            return None
        if FastMCP is None:  # pragma: no cover - optional path
            core_logger.warning("[MCP] fastmcp not installed; skipping MCP endpoints")
            return None
        try:
            mcp_instance = FastMCP()
        except Exception as e:  # noqa
            core_logger.exception(f"[MCP] init failed: {e}")
            return None

        prefer_explicit = os.getenv("EYETOOLS_MCP_PREFER_EXPLICIT", "1") != "0"
        param_styles: Dict[str, str] = {}

        def build_input_schema(meta) -> Dict[str, Any]:
            if isinstance(meta.io, dict):
                if isinstance(meta.io.get("input"), dict):
                    return meta.io["input"]
                props = {k: {"type": "string"} for k in meta.io.keys() if k not in {"batchable", "input"}}
                if props:
                    return {"type": "object", "properties": props, "additionalProperties": True}
            return {"type": "object", "properties": {}, "additionalProperties": True}

        def extract_param_names(input_schema: Dict[str, Any], meta) -> List[str]:
            names: List[str] = []
            if input_schema.get("type") == "object":
                props = input_schema.get("properties", {})
                if isinstance(props, dict):
                    names.extend(props.keys())
            if not names and isinstance(meta.io, dict):
                for k in meta.io.keys():
                    if k not in {"batchable", "input"}:
                        names.append(k)
            return list(dict.fromkeys(names))  # de-dup preserve order

        def register_single(meta):  # noqa: C901 complexity acceptable here
            tool_id = meta.id
            input_schema = build_input_schema(meta)
            # Attempt explicit signature
            def make_explicit():
                params = extract_param_names(input_schema, meta)
                if not params:
                    raise ValueError("no_params")
                import re
                mapping = {}
                used = set()
                san_list = []
                for p in params:
                    san = re.sub(r"[^0-9a-zA-Z_]+", "_", p) or "p"
                    if san[0].isdigit():
                        san = "p_" + san
                    base = san; i = 1
                    while san in used:
                        san = f"{base}_{i}"; i += 1
                    used.add(san)
                    mapping[san] = p
                    san_list.append(san)
                sig_params = ", ".join(f"{s}: Any = None" for s in san_list)
                body = [
                    "    try:",
                    "        _locals = {k: v for k, v in locals().items() if k not in {'tid','tm','_map'}}",
                    "        inputs = {}",
                    "        for _san,_orig in _map.items():",
                    "            val = _locals.get(_san)",
                    "            if val is not None: inputs[_orig] = val",
                    "        return tm.predict(tid, {'inputs': inputs} if inputs else {})",
                    "    except Exception as e:",
                    "        return {'error': str(e)}",
                ]
                src = f"async def _f({sig_params}):\n" + "\n".join(body)
                glb = {"tm": tm, "tid": tool_id, "Any": Any, "_map": mapping}
                loc: Dict[str, Any] = {}
                exec(src, glb, loc)  # noqa
                fn = loc["_f"]
                fn.__name__ = f"tool_{tool_id.replace(':','_')}_explicit"
                fn.__doc__ = f"MCP wrapper explicit params for '{tool_id}' params={params}"
                return fn

            async def func_kwargs(**kwargs):  # type: ignore
                try:
                    return tm.predict(tool_id, {"inputs": kwargs} if kwargs else {})
                except Exception as e:  # noqa
                    return {"error": str(e)}
            func_kwargs.__name__ = f"tool_{tool_id.replace(':','_')}"
            func_kwargs.__doc__ = f"MCP wrapper for eyetools tool '{tool_id}' (kwargs)"

            async def func_payload(payload: Dict[str, Any] | None = None):  # type: ignore
                try:
                    return tm.predict(tool_id, {"inputs": (payload or {})} if payload else {})
                except Exception as e:  # noqa
                    return {"error": str(e)}
            func_payload.__name__ = f"tool_{tool_id.replace(':','_')}_payload"
            func_payload.__doc__ = f"MCP wrapper for eyetools tool '{tool_id}' (payload)"

            # Decorator acquisition (handles input_schema support variance)
            try:
                try:
                    deco = mcp_instance.tool(name=tool_id, description=tool_id, input_schema=input_schema)  # type: ignore
                except TypeError as te:
                    if 'input_schema' in str(te):
                        deco = mcp_instance.tool(name=tool_id, description=tool_id)  # type: ignore
                        core_logger.debug("[MCP] no input_schema support (tool=%s)", tool_id)
                    else:
                        raise
                # explicit -> kwargs -> payload fallback
                if prefer_explicit:
                    try:
                        explicit_fn = make_explicit()
                        deco(explicit_fn)
                        param_styles[tool_id] = "explicit"
                        return
                    except Exception as e:  # noqa
                        core_logger.debug("[MCP] explicit build failed tool=%s reason=%s", tool_id, e)
                try:
                    deco(func_kwargs)
                    param_styles[tool_id] = "kwargs"
                except Exception as e:  # noqa
                    if "**kwargs" in str(e) or "kwargs" in str(e):
                        deco2 = mcp_instance.tool(name=tool_id, description=tool_id)  # type: ignore
                        deco2(func_payload)
                        param_styles[tool_id] = "payload"
                        core_logger.info("[MCP] payload fallback tool=%s", tool_id)
                    else:
                        raise
            except Exception as e:  # noqa
                core_logger.warning("[MCP] failed to register tool id=%s error=%s", tool_id, e)

        for meta in registry.list():
            register_single(meta)

        # Mount HTTP app
        try:
            mcp_app = mcp_instance.http_app(transport="streamable-http")  # type: ignore
            if mcp_mount_path == "/":
                for r in mcp_app.routes:
                    if not any(getattr(er, 'path', None) == r.path for er in app.routes):
                        app.router.routes.append(r)
                app.router.lifespan_context = mcp_app.router.lifespan_context  # type: ignore
                core_logger.info("[MCP] routes merged at root")
            else:
                mount_path = mcp_mount_path.rstrip("/") or "/mcp"
                app.mount(mount_path, mcp_app)
                core_logger.info("[MCP] mounted at %s", mount_path)
                if os.getenv("EYETOOLS_MCP_COMPAT_MOUNT") == "1" and mount_path not in {"/mcp", "/"}:
                    try:
                        app.mount("/mcp", mcp_app)
                        core_logger.info("[MCP] compatibility mount /mcp active")
                    except Exception as ce:  # noqa
                        core_logger.warning(f"[MCP] compat mount failed: {ce}")
        except Exception as e:  # noqa
            core_logger.exception(f"[MCP] attach routes failed: {e}")

        # Final summary log
        names = sorted(param_styles.keys())
        core_logger.info("[MCP] available tools (count=%d): %s", len(names), ", ".join(names))
        app.state.fastmcp = mcp_instance
        app.state._mcp_param_styles = param_styles  # type: ignore

        def re_register_all():
            # clear and re-run registration (simple strategy: new instance)
            core_logger.info("[MCP] re-registering tools after reload...")
            app.state.fastmcp = None
            return _setup_mcp()

        app.state._mcp_reregister = re_register_all  # type: ignore
        return mcp_instance

    _setup_mcp()

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
        # Re-register MCP tools if integration enabled
        if enable_mcp and getattr(app.state, "_mcp_reregister", None):  # pragma: no cover
            try:
                app.state._mcp_reregister()  # type: ignore
            except Exception as e:  # noqa
                core_logger.warning(f"[MCP] MCP re-registration failed: {e}")
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

    # Debug / introspection endpoint for MCP (lists tool IDs FastMCP has registered)
    if enable_mcp:
        @app.get("/mcp/tools")  # pragma: no cover - simple passthrough
        def mcp_tools():
            inst = getattr(app.state, "fastmcp", None)
            if not inst:
                return {"enabled": False, "tools": [], "tool_names": []}
            names: list[str] = []
            detailed: List[Dict[str, Any]] = []
            lifecycle_map = tm.lifecycle_states()
            try:
                maybe = getattr(inst, "_tools", None) or getattr(inst, "tools", None)
                # fastmcp stores tools as name->FastMCPTool (with attributes .name, .description, .input_schema)
                if isinstance(maybe, dict):
                    for k, v in maybe.items():
                        names.append(k)
                        try:
                            desc = getattr(v, "description", "")
                            schema = getattr(v, "input_schema", {})
                        except Exception:  # noqa
                            desc, schema = "", {}
                        style_map: Dict[str, str] = getattr(app.state, "_mcp_param_styles", {})
                        detailed.append({
                            "name": k,
                            "description": desc,
                            "input_schema": schema,
                            "param_style": style_map.get(k, "unknown"),
                            "lifecycle_state": lifecycle_map.get(k),
                        })
                names = sorted(names)
                detailed = sorted(detailed, key=lambda x: x["name"])
            except Exception:  # noqa
                pass
            return {
                "enabled": True,
                "mount_path": mcp_mount_path,
                "tool_names": names,
                "tools": detailed,
                "count": len(names),
            }

        @app.post("/mcp/refresh")  # pragma: no cover - light endpoint
        def mcp_refresh():
            rer = getattr(app.state, "_mcp_reregister", None)
            if not rer:
                raise HTTPException(status_code=400, detail="MCP not enabled")
            rer()
            tools_after = getattr(app.state, "_mcp_param_styles", {})
            return {"status": "ok", "count": len(tools_after), "tools": sorted(tools_after.keys())}

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

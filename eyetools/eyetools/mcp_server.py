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


def create_app(include_examples: bool = False, tool_paths: Optional[List[str]] = None, role_config_path: Optional[str] = None):
    registry = ToolRegistry()
    extra_paths_env = os.getenv("EYETOOLS_TOOL_PATHS", "")
    env_paths = [p for p in extra_paths_env.split(":") if p]
    combined_paths = (tool_paths or []) + env_paths
    path_objs = _compute_tool_paths(include_examples, combined_paths)
    errors = discover_tools(path_objs, registry)
    if errors:
        for e in errors:
            core_logger.warning(f"discover error: {e}")
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
    def metrics():
        return tm.get_metrics()

    return app

__all__ = ["create_app"]

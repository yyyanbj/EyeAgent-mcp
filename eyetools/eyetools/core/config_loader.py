"""Tool configuration loading and normalization.

This module centralizes parsing of tool package configs (YAML or Python) into
normalized ToolDefinition dicts suitable for registry ingestion.

MVP Scope:
- Detect mode: single / variants / tools.
- Merge shared fields for variants.
- Generate stable tool IDs.
- Basic validation of required fields.
- Provide load_configs_in_dir(path) -> List[ToolDefinition].

Later:
- Pydantic schemas
- Rich error classes
- Requirements conflict analysis
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import runpy
import yaml
from .errors import ConfigError

REQUIRED_SINGLE_FIELDS = ["entry"]
VALID_LOAD_MODES = {"auto", "inproc", "subprocess"}
VALID_PRECISIONS = {"auto", "fp32", "fp16", "bf16", "int8"}


 # ConfigError now imported from .errors


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_py(path: Path) -> Dict[str, Any]:
    ns = runpy.run_path(str(path))
    if "get_config" in ns:
        cfg = ns["get_config"]()
    elif "CONFIGS" in ns:
        cfg = {"tools": ns["CONFIGS"]}
    else:
        raise ConfigError("Python config must expose get_config() or CONFIGS")
    if not isinstance(cfg, dict):
        raise ConfigError("Python config entry must return dict")
    return cfg


def _detect_mode(data: Dict[str, Any]) -> str:
    if "variants" in data:
        return "variants"
    if "tools" in data:
        return "tools"
    return "single"


def _ensure_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def _gen_id(base: Dict[str, Any]) -> str:
    if base.get("id"):
        return base["id"]
    pkg = base.get("package")
    variant = base.get("variant")
    name = base.get("name")
    category = base.get("category")
    if pkg and variant:
        core = f"{pkg}:{variant}"
        return f"{category}.{core}" if category else core
    if pkg:
        return f"{category}.{pkg}" if category else pkg
    if name:
        return f"{category}.{name}" if category else name
    # fallback to entry class
    entry = base.get("entry", "unknown:Unknown")
    core = entry.split(":", 1)[-1]
    return f"{category}.{core}" if category else core


def _merge_defaults(td: Dict[str, Any]) -> Dict[str, Any]:
    runtime = td.get("runtime", {})
    model = td.get("model", {})
    warmup = td.get("warmup", {})
    io = td.get("io", {})

    runtime.setdefault("load_mode", "auto")
    runtime.setdefault("max_workers", 1)
    runtime.setdefault("queue_size", 16)
    runtime.setdefault("idle_timeout_s", 600)

    model.setdefault("precision", "auto")
    model.setdefault("device", "auto")
    model.setdefault("lazy", True)

    warmup.setdefault("on_first_call", True)
    warmup.setdefault("on_start", False)

    io.setdefault("batchable", False)

    td.setdefault("version", "0.1.0")

    td["runtime"] = runtime
    td["model"] = model
    td["warmup"] = warmup
    td["io"] = io
    td["id"] = _gen_id(td)
    return td


def _validate(td: Dict[str, Any]):
    if td.get("runtime", {}).get("load_mode") not in VALID_LOAD_MODES:
        raise ConfigError(f"Invalid load_mode: {td.get('runtime', {}).get('load_mode')}")
    prec = td.get("model", {}).get("precision")
    if prec not in VALID_PRECISIONS:
        raise ConfigError(f"Invalid precision: {prec}")
    if "entry" not in td:
        raise ConfigError("Missing entry field")
    if ":" not in td["entry"]:
        raise ConfigError("entry must be 'module:Class'")


def _expand_variants(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    shared = data.get("shared", {})
    pkg = data.get("package")
    entry = data.get("entry")
    res = []
    for v in data.get("variants", []):
        merged = {}
        # shallow merge precedence: shared < variant
        for src in (shared, v):
            for k, val in src.items():
                if k in {"runtime", "model", "warmup", "io"}:
                    # nested dict merge
                    base_sub = merged.get(k, {})
                    new_sub = base_sub.copy()
                    if isinstance(val, dict):
                        new_sub.update(val)
                    else:
                        new_sub = val
                    merged[k] = new_sub
                else:
                    merged[k] = val
        merged["package"] = pkg
        merged.setdefault("entry", entry)
        merged["variant"] = v.get("variant")
        res.append(merged)
    return res


def _expand_tools_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(data.get("tools", []))


def parse_config_file(path: Path) -> List[Dict[str, Any]]:
    data = _read_yaml(path) if path.suffix in (".yml", ".yaml") else _read_py(path)
    mode = _detect_mode(data)
    if mode == "single":
        items = [data]
    elif mode == "variants":
        items = _expand_variants(data)
    else:
        items = _expand_tools_list(data)
    normalized: List[Dict[str, Any]] = []
    for td in items:
        td = _merge_defaults(td)
        _validate(td)
        normalized.append(td)
    return normalized


def load_configs_in_dir(dir_path: Path) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for fname in ("config.yaml", "config.yml", "config.py"):
        p = dir_path / fname
        if p.exists():
            configs.extend(parse_config_file(p))
    return configs

__all__ = [
    "parse_config_file",
    "load_configs_in_dir",
    "ConfigError",
]

"""Filesystem discovery of tool packages."""
from __future__ import annotations
from pathlib import Path
from typing import List
from .config_loader import load_configs_in_dir, ConfigError
from .errors import RegistrationError
from .registry import ToolRegistry, ToolMeta


def discover_tools(root_dirs: List[Path], registry: ToolRegistry, errors: List[str] | None = None):
    errors = errors if errors is not None else []
    for root in root_dirs:
        if not root.exists():
            continue
        for path in root.rglob("config.yaml"):
            tool_dir = path.parent
            try:
                configs = load_configs_in_dir(tool_dir)
                for c in configs:
                    # Infer category if not set: expecting .../tools/<category>/<pkg>/config.*
                    if not c.get("category"):
                        parts = list(tool_dir.parts)
                        if "tools" in parts:
                            idx = parts.index("tools")
                            if idx + 1 < len(parts):
                                c["category"] = parts[idx + 1]
                    meta = ToolMeta(
                        id=c["id"],
                        entry=c["entry"],
                        version=c.get("version", "0.1.0"),
                        package=c.get("package"),
                        variant=c.get("variant"),
                        runtime=c.get("runtime", {}),
                        model=c.get("model", {}),
                        params=c.get("params", {}),
                        warmup=c.get("warmup", {}),
                        io=c.get("io", {}),
                        tags=c.get("tags", []),
                        root_dir=str(tool_dir),
                        extra_requires=c.get("extra_requires", []),
                        python=c.get("python") or c.get("shared", {}).get("python"),
                        category=c.get("category"),
                        environment_ref=(c.get("environment_ref") or c.get("shared", {}).get("environment_ref")),
                    )
                    try:
                        registry.register(meta)
                    except Exception as re:  # noqa
                        raise RegistrationError(f"Failed to register {c.get('id')}: {re}") from re
            except ConfigError as e:
                errors.append(f"{tool_dir}: {e}")
        for path in root.rglob("config.py"):
            # Avoid double-processing if YAML already handled same folder
            tool_dir = path.parent
            if (tool_dir / "config.yaml").exists() or (tool_dir / "config.yml").exists():
                continue
            try:
                configs = load_configs_in_dir(tool_dir)
                for c in configs:
                    if not c.get("category"):
                        parts = list(tool_dir.parts)
                        if "tools" in parts:
                            idx = parts.index("tools")
                            if idx + 1 < len(parts):
                                c["category"] = parts[idx + 1]
                    meta = ToolMeta(
                        id=c["id"],
                        entry=c["entry"],
                        version=c.get("version", "0.1.0"),
                        package=c.get("package"),
                        variant=c.get("variant"),
                        runtime=c.get("runtime", {}),
                        model=c.get("model", {}),
                        params=c.get("params", {}),
                        warmup=c.get("warmup", {}),
                        io=c.get("io", {}),
                        tags=c.get("tags", []),
                        root_dir=str(tool_dir),
                        extra_requires=c.get("extra_requires", []),
                        python=c.get("python") or c.get("shared", {}).get("python"),
                        category=c.get("category"),
                        environment_ref=(c.get("environment_ref") or c.get("shared", {}).get("environment_ref")),
                    )
                    try:
                        registry.register(meta)
                    except Exception as re:  # noqa
                        raise RegistrationError(f"Failed to register {c.get('id')}: {re}") from re
            except ConfigError as e:
                errors.append(f"{tool_dir}: {e}")
    return errors

__all__ = ["discover_tools"]

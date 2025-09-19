from __future__ import annotations
from typing import Dict, Type, Optional, List
from loguru import logger
import importlib
from ..config.settings import get_configured_agents


_REGISTRY: Dict[str, Type] = {}


def register_agent(cls: Type) -> Type:
    """Class decorator to register an agent class by both role and class name.

    The class is expected to define attributes: `role` and `name`.
    """
    role = getattr(cls, "role", None)
    name = getattr(cls, "name", None) or cls.__name__
    if role:
        _REGISTRY[role] = cls
    _REGISTRY[name] = cls
    logger.debug(f"[registry] registered agent role={role} name={name}")
    return cls


def get_agent_class(key: str) -> Optional[Type]:
    return _REGISTRY.get(key)


def list_agents() -> List[str]:
    return sorted(list(_REGISTRY.keys()))


def register_builtins() -> None:
    """Register agents from config. Each item: role -> {class: dotted.path, enabled: bool}.

    Falls back to UnifiedAgent if nothing configured or import fails."""
    agents_cfg = get_configured_agents()
    registered = []
    for role, meta in (agents_cfg or {}).items():
        if not isinstance(meta, dict) or not meta.get("enabled", False):
            continue
        dotted = meta.get("class")
        if not dotted or not isinstance(dotted, str):
            continue
        try:
            mod_name, cls_name = dotted.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            r = getattr(cls, "role", None) or role
            n = getattr(cls, "name", None) or cls.__name__
            _REGISTRY.setdefault(r, cls)
            _REGISTRY.setdefault(n, cls)
            registered.append((r, n))
        except Exception as e:
            logger.warning(f"[registry] failed to import {dotted}: {e}")
            continue
    # Ensure at least UnifiedAgent is present
    if not _REGISTRY:
        try:
            from .unified_agent import UnifiedAgent  # type: ignore
            _REGISTRY.setdefault(getattr(UnifiedAgent, "role", "unified"), UnifiedAgent)
            _REGISTRY.setdefault(getattr(UnifiedAgent, "name", "UnifiedAgent"), UnifiedAgent)
            registered.append((getattr(UnifiedAgent, "role", "unified"), getattr(UnifiedAgent, "name", "UnifiedAgent")))
        except Exception as e:
            logger.error(f"[registry] fallback import UnifiedAgent failed: {e}")
    logger.debug(f"[registry] built-ins registered keys={list(_REGISTRY.keys())} items={registered}")

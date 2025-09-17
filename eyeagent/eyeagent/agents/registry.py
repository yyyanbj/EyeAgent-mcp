from __future__ import annotations
from typing import Dict, Type, Optional, List
from loguru import logger


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
    """Register built-in agents. Import side effects will trigger decorator if used.

    We support both decorator-based registration and explicit adds here to be robust.
    """
    # Local imports to avoid circulars
    from .orchestrator_agent import OrchestratorAgent
    from .image_analysis_agent import ImageAnalysisAgent
    from .specialist_agent import SpecialistAgent
    from .followup_agent import FollowUpAgent
    from .report_agent import ReportAgent

    for cls in [OrchestratorAgent, ImageAnalysisAgent, SpecialistAgent, FollowUpAgent, ReportAgent]:
        role = getattr(cls, "role", None)
        name = getattr(cls, "name", None) or cls.__name__
        if role:
            _REGISTRY.setdefault(role, cls)
        _REGISTRY.setdefault(name, cls)
    logger.debug(f"[registry] built-ins registered keys={list(_REGISTRY.keys())}")

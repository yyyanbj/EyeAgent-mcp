from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
import re
from pathlib import Path
from .settings import Settings
from ..tools.tool_registry import role_tool_ids as _role_tool_ids  # type: ignore
from ..tools.tool_registry import TOOL_REGISTRY  # type: ignore

# Config structure expected (in eyeagent.yml):
# tools_filter:
#   OrchestratorAgent:
#     include: ["classification:.*"]
#     exclude: []
#   ImageAnalysisAgent:
#     include: ["classification:cfp_quality", "segmentation:cfp_.*", "segmentation:oct_.*", "segmentation:ffa_.*"]
#   SpecialistAgent:
#     include: ["disease_specific_cls:.*"]
#
# Any item in include/exclude is treated as a regex pattern (fullmatch by default).


def _compile_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for p in patterns or []:
        try:
            out.append(re.compile(p))
        except Exception:
            # fallback: escape literal
            out.append(re.compile(re.escape(str(p))))
    return out


def _matches_any(name: str, pats: List[re.Pattern]) -> bool:
    for r in pats:
        # Use search to be convenient for users; can be changed to fullmatch if needed
        if r.search(name):
            return True
    return False


def get_agent_filter(agent_name: str) -> Dict[str, Any]:
    cfg = Settings().load() or {}
    tf = cfg.get("tools_filter") or {}
    return tf.get(agent_name) or {}


def filter_tool_ids(agent_name: str, tool_ids: List[str]) -> List[str]:
    """Backward-compatible filtering against a provided base list.

    Note: Prefer select_tool_ids for config-first selection. This function only
    prunes the given list based on include/exclude patterns.
    """
    f = get_agent_filter(agent_name)
    inc = _compile_patterns(f.get("include")) if isinstance(f.get("include"), list) else []
    exc = _compile_patterns(f.get("exclude")) if isinstance(f.get("exclude"), list) else []
    ids = list(tool_ids or [])
    if inc:
        ids = [t for t in ids if _matches_any(t, inc)]
    if exc:
        ids = [t for t in ids if not _matches_any(t, exc)]
    return ids


def _all_tool_ids() -> List[str]:
    try:
        return list(TOOL_REGISTRY.keys())
    except Exception:
        return []


def select_tool_ids(agent_name: str, base_tool_ids: Optional[List[str]] = None, role: Optional[str] = None) -> List[str]:
    """Config-first selection of allowed tool ids for an agent.

    Priority:
    1) Start set = role tools if role provided; else base_tool_ids if provided; else all tools known to registry.
    2) Apply include patterns from config (if any) to keep matches.
    3) Apply exclude patterns from config to remove matches.

    This allows the configuration (eyeagent.yml -> tools_filter) to be the source
    of truth without being constrained by hardcoded lists inside agents.
    """
    start: List[str]
    if role:
        try:
            start = list(_role_tool_ids(role))
        except Exception:
            start = []
        # If role yields nothing (e.g., role labels differ), fallback to base/all
        if not start:
            start = list(base_tool_ids or _all_tool_ids())
    elif base_tool_ids is not None:
        start = list(base_tool_ids)
    else:
        start = _all_tool_ids()
    return filter_tool_ids(agent_name, start)

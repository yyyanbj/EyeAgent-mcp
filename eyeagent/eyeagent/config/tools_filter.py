from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
import re
from pathlib import Path
from .settings import Settings

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
    f = get_agent_filter(agent_name)
    inc = _compile_patterns(f.get("include")) if isinstance(f.get("include"), list) else []
    exc = _compile_patterns(f.get("exclude")) if isinstance(f.get("exclude"), list) else []
    ids = list(tool_ids or [])
    # If include provided, keep only matches; else keep all
    if inc:
        ids = [t for t in ids if _matches_any(t, inc)]
    # Apply excludes
    if exc:
        ids = [t for t in ids if not _matches_any(t, exc)]
    return ids

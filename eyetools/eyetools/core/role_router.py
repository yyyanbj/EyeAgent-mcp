"""Role-based tool exposure router.

Config schema (YAML/JSON -> dict):
{
  "roles": {
     "radiologist": {
        "include": ["classification.*", "report.generate"],
        "exclude": ["*.experimental"],
        "tags_any": ["vision"],
        "tags_all": ["stable"],
        "select_mode": "auto"   # or "manual"
     },
     ...
  },
  "defaults": { "include": ["*"], "exclude": [] }
}

Rules:
 1. Determine role block (fallback to defaults if missing) -> start set = include patterns.
 2. If include empty -> treat as defaults.include or all tools if defaults absent.
 3. Exclude patterns remove matches.
 4. Tag filters (tags_any OR; tags_all AND) applied after pattern filtering.
 5. Manual mode returns structure {"selection_required": True, "candidates": [...ids...]}

Pattern matching uses fnmatch on tool id and also id segments to allow simple globs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable
import fnmatch


def _as_list(v):
    if not v:
        return []
    if isinstance(v, list):
        return v
    return [v]


@dataclass
class RoleMatchResult:
    selection_required: bool
    tool_ids: List[str]

    def to_payload(self):
        if self.selection_required:
            return {"selection_required": True, "candidates": self.tool_ids}
        return {"tools": self.tool_ids}


class RoleRouter:
    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.roles_cfg = self.config.get("roles", {})
        self.defaults = self.config.get("defaults", {"include": ["*"], "exclude": []})

    @staticmethod
    def _match_patterns(tool_id: str, patterns: Iterable[str]) -> bool:
        for p in patterns:
            if fnmatch.fnmatch(tool_id, p):
                return True
        return False

    def filter_tools(self, role: str | None, tools: List[Any]) -> RoleMatchResult:
        role_cfg = self.roles_cfg.get(role or "", {}) if role else {}
        include = _as_list(role_cfg.get("include")) or _as_list(self.defaults.get("include", ["*"]))
        exclude = _as_list(role_cfg.get("exclude")) or []
        tags_any = set(_as_list(role_cfg.get("tags_any")))
        tags_all = set(_as_list(role_cfg.get("tags_all")))
        select_mode = role_cfg.get("select_mode", "auto")

        # Phase 1 include
        candidates = []
        for m in tools:
            tid = m.id
            if include and not self._match_patterns(tid, include):
                continue
            candidates.append(m)
        # Phase 2 exclude
        if exclude:
            filtered = []
            for m in candidates:
                if self._match_patterns(m.id, exclude):
                    continue
                filtered.append(m)
            candidates = filtered
        # Phase 3 tags
        if tags_any:
            candidates = [m for m in candidates if tags_any.intersection(set(getattr(m, 'tags', []) or []))]
        if tags_all:
            candidates = [m for m in candidates if tags_all.issubset(set(getattr(m, 'tags', []) or []))]

        tool_ids = [m.id for m in candidates]
        if select_mode == "manual":
            return RoleMatchResult(selection_required=True, tool_ids=tool_ids)
        return RoleMatchResult(selection_required=False, tool_ids=tool_ids)

__all__ = ["RoleRouter", "RoleMatchResult"]

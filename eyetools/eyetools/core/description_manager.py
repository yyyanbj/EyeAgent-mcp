"""Centralized description manager for tools.

Loads a YAML/JSON config that maps tool IDs or glob patterns to descriptions.
Supports hot reload via .reload().

Config formats supported (choose one):

1) Simple mapping (YAML):
   classification: "Classify general attributes"
   classification:modality: "Detect image modality"
   rag:query: "Retrieve and answer ophthalmology questions"

2) Explicit keys:
   descriptions:
     classification: "..."
     segmentation: "..."
   patterns:
     - match: "classification:*"
       description: "Image classification"
     - match: "segmentation:*"
       description: "Segmentation tools"

Environment override: EYETOOLS_DESCRIPTION_CONFIG can specify the config file path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import fnmatch


@dataclass
class _CompiledDescriptions:
    exact: Dict[str, str]
    patterns: List[Tuple[str, str]]
    default: Optional[str] = None


class DescriptionManager:
    def __init__(self, config_path: Optional[str] = None):
        self._path: Optional[Path] = Path(config_path).resolve() if config_path else None
        self._compiled: _CompiledDescriptions = _CompiledDescriptions(exact={}, patterns=[], default=None)
        self._last_mtime: Optional[float] = None
        # Attempt initial load (best-effort)
        self._load_if_available()

    @staticmethod
    def _normalize(obj: Any) -> _CompiledDescriptions:
        exact: Dict[str, str] = {}
        patterns: List[Tuple[str, str]] = []
        default: Optional[str] = None
        if not obj:
            return _CompiledDescriptions(exact, patterns, default)
        data: Dict[str, Any]
        if isinstance(obj, dict):
            data = obj
        else:
            # Unknown type -> empty
            return _CompiledDescriptions(exact, patterns, default)
        # format 2 explicit
        if isinstance(data.get("descriptions"), dict):
            for k, v in data["descriptions"].items():
                if isinstance(v, str):
                    exact[str(k)] = v
        if isinstance(data.get("patterns"), list):
            for item in data["patterns"]:
                if isinstance(item, dict):
                    m = item.get("match")
                    d = item.get("description")
                    if isinstance(m, str) and isinstance(d, str):
                        patterns.append((m, d))
        if isinstance(data.get("default"), str):
            default = data.get("default")
        # format 1 simple mapping (top-level strings)
        # Only include string values not already captured in explicit descriptions
        for k, v in data.items():
            if k in {"descriptions", "patterns", "default"}:
                continue
            if isinstance(v, str) and k not in exact:
                exact[str(k)] = v
        return _CompiledDescriptions(exact, patterns, default)

    def _read_config(self) -> Optional[_CompiledDescriptions]:
        if not self._path:
            return None
        if not self._path.exists():
            return _CompiledDescriptions({}, [], None)
        raw: Any
        text = self._path.read_text(encoding="utf-8")
        # Try YAML then JSON
        try:
            import yaml  # type: ignore
            raw = yaml.safe_load(text)
        except Exception:
            try:
                raw = json.loads(text)
            except Exception:
                raw = None
        return self._normalize(raw)

    def _load_if_available(self):
        compiled = self._read_config()
        if compiled is not None:
            self._compiled = compiled
            try:
                self._last_mtime = self._path.stat().st_mtime if self._path and self._path.exists() else None
            except Exception:
                self._last_mtime = None

    def set_path(self, config_path: Optional[str]):
        self._path = Path(config_path).resolve() if config_path else None
        self.reload()

    def reload(self) -> Dict[str, Any]:
        before = {"count_exact": len(self._compiled.exact), "count_patterns": len(self._compiled.patterns)}
        self._load_if_available()
        after = {"count_exact": len(self._compiled.exact), "count_patterns": len(self._compiled.patterns)}
        return {"before": before, "after": after, "path": str(self._path) if self._path else None}

    def get(self, tool_id: str, meta: Optional[Any] = None) -> str:
        # 1. exact match
        if tool_id in self._compiled.exact:
            return self._compiled.exact[tool_id]
        # 2. glob patterns (first match wins)
        for pat, desc in self._compiled.patterns:
            try:
                if fnmatch.fnmatch(tool_id, pat):
                    return desc
            except Exception:
                continue
        # 3. fallback to meta-provided description if present
        try:
            if meta and getattr(meta, "io", None):
                io = getattr(meta, "io") or {}
                if isinstance(io, dict) and isinstance(io.get("description"), str):
                    return io["description"]
        except Exception:
            pass
        # 4. optional default
        if self._compiled.default:
            return self._compiled.default  # type: ignore[return-value]
        # 5. last resort: use tool_id as description
        return tool_id

    def summary(self) -> Dict[str, Any]:
        return {
            "path": str(self._path) if self._path else None,
            "count_exact": len(self._compiled.exact),
            "count_patterns": len(self._compiled.patterns),
            "default": self._compiled.default,
            "examples": list(self._compiled.exact.items())[:5],
        }

__all__ = ["DescriptionManager"]

"""In-memory tool registry."""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import threading


@dataclass
class ToolMeta:
    id: str
    entry: str
    version: str
    package: Optional[str] = None
    variant: Optional[str] = None
    runtime: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    warmup: Dict[str, Any] = field(default_factory=dict)
    io: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    root_dir: Optional[str] = None
    extra_requires: List[str] = field(default_factory=list)
    python: Optional[str] = None
    category: Optional[str] = None


class ToolRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._tools: Dict[str, ToolMeta] = {}

    def register(self, meta: ToolMeta):
        with self._lock:
            if meta.id in self._tools:
                raise ValueError(f"Duplicate tool id: {meta.id}")
            self._tools[meta.id] = meta

    def get(self, tool_id: str) -> Optional[ToolMeta]:
        return self._tools.get(tool_id)

    def list(self) -> List[ToolMeta]:
        return list(self._tools.values())

    def remove(self, tool_id: str):
        with self._lock:
            self._tools.pop(tool_id, None)

    def clear(self):
        with self._lock:
            self._tools.clear()

__all__ = ["ToolRegistry", "ToolMeta"]

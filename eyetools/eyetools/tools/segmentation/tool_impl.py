"""Shim module delegating to legacy ``tools.segmentation.tool_impl``."""

from __future__ import annotations

from importlib import import_module as _im

_legacy = _im("tools.segmentation.tool_impl")
for _k, _v in list(_legacy.__dict__.items()):
    if not _k.startswith("_"):
        globals()[_k] = _v

__all__ = [n for n in globals() if not n.startswith("_")]

del _im, _legacy

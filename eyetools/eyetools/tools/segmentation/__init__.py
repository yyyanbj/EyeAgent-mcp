"""Shim subpackage for legacy ``tools.segmentation`` package."""

from __future__ import annotations

from importlib import import_module as _im

try:  # pragma: no cover
    _mod = _im("tools.segmentation")
    for _name, _value in list(_mod.__dict__.items()):
        if not _name.startswith("_"):
            globals()[_name] = _value
    __all__ = getattr(_mod, "__all__", [n for n in _mod.__dict__ if not n.startswith("_")])  # type: ignore
except Exception:  # pragma: no cover
    __all__ = []  # type: ignore

del _im, _mod  # type: ignore

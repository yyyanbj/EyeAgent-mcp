"""Shim module delegating to legacy ``tools.disease_specific_cls.tool_impl``.

All public names are copied into this module's namespace so that imports via
``eyetools.tools.disease_specific_cls.tool_impl`` behave identically to the legacy
location. Keep this lightweight to avoid import-time side effects duplication.
"""

from __future__ import annotations

from importlib import import_module as _im

_legacy = _im("tools.disease_specific_cls.tool_impl")
for _k, _v in list(_legacy.__dict__.items()):
    if not _k.startswith("_"):
        globals()[_k] = _v

__all__ = [n for n in globals() if not n.startswith("_")]

del _im, _legacy

"""Shim subpackage delegating to legacy ``tools.disease_specific_cls``.

Re-exports public symbols so that ``eyetools.tools.disease_specific_cls`` mirrors
``tools.disease_specific_cls`` until a filesystem relocation occurs.
"""

from __future__ import annotations

from importlib import import_module as _im

try:  # pragma: no cover - simple pass-through
    _mod = _im("tools.disease_specific_cls")
    # Replicate its public (non-underscore) attributes.
    for _name, _value in list(_mod.__dict__.items()):
        if not _name.startswith("_"):
            globals()[_name] = _value
    # Provide an __all__ if the source defines it; else synthesize.
    __all__ = getattr(_mod, "__all__", [n for n in _mod.__dict__ if not n.startswith("_")])  # type: ignore
except Exception:  # pragma: no cover - absence tolerated
    __all__ = []  # type: ignore

del _im, _mod  # type: ignore

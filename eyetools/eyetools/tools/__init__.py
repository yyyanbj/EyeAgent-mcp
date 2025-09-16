"""Shim package for backward/forward compatibility.

This directory provides a transitional bridge so that code may import either:

    import eyetools.tools.disease_specific_cls.tool_impl

while the canonical implementation still physically lives in the legacy top-level
``tools`` package at the repository root. Once the project is reorganized so the
``tools`` package is located inside ``eyetools/`` this shim can be removed.

Resolution strategy:
  * Attribute access (``eyetools.tools.X``) dynamically proxies to ``tools.X``.
  * Explicit subpackages we need for tests (``disease_specific_cls`` and
    ``segmentation``) contain small re-export wrappers so nested imports like
    ``eyetools.tools.disease_specific_cls.tool_impl`` succeed.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

try:  # Attempt to import the legacy top-level tools package.
    _legacy_tools = importlib.import_module("tools")  # type: ignore
except Exception:  # pragma: no cover - absence is tolerated
    _legacy_tools = None  # type: ignore


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dynamic proxy
    """Dynamically resolve attributes to the legacy ``tools`` package.

    This allows patterns like ``from eyetools.tools import DiseaseSpecificClassificationTool``
    provided the symbol exists in the original ``tools`` package.
    """
    if _legacy_tools is None:
        raise AttributeError(name)
    try:
        return getattr(_legacy_tools, name)
    except AttributeError:
        # Fallback: try importing a submodule from the legacy namespace.
        try:
            return importlib.import_module(f"tools.{name}")
        except ModuleNotFoundError as e:  # pragma: no cover
            raise AttributeError(name) from e


def __dir__():  # pragma: no cover - convenience
    base = []
    if _legacy_tools is not None:
        base.extend(getattr(_legacy_tools, "__all__", []))
    return sorted(set(base))


__all__ = []  # Intentionally empty; dynamic resolution performed in __getattr__.

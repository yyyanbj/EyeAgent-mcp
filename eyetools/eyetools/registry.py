"""Deprecated legacy registry module.

This module has been replaced by eyetools.core.registry. It remains as a thin
shim to preserve import compatibility (e.g., `from eyetools.registry import ToolRegistry`).
New code should import from `eyetools.core.registry`.
"""
from __future__ import annotations

import warnings

from .core.registry import ToolRegistry, ToolMeta  # re-export

warnings.warn(
    "eyetools.registry is deprecated; import from eyetools.core.registry instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ToolRegistry", "ToolMeta"]

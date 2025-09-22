"""Deprecated legacy tool_manager module.

This module has been replaced by eyetools.core.tool_manager. It remains as a thin
shim to preserve import compatibility (e.g., `from eyetools.tool_manager import ToolManager`).
New code should import from `eyetools.core.tool_manager`.
"""
from __future__ import annotations

import warnings

from .core.tool_manager import ToolManager  # re-export

warnings.warn(
    "eyetools.tool_manager is deprecated; import from eyetools.core.tool_manager instead",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ToolManager"]

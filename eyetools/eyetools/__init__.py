"""
EyeAgent Tools Package

This package provides a unified framework for various AI tools used in EyeAgent,
including classification, detection, segmentation, generation, and multimodal analysis.
"""

from .base import (
    ToolBase,
)

# Prefer core.* modules; keep legacy import paths as shims
from .core.registry import ToolRegistry  # noqa: F401

from .utils import (
    PathHandler,
    load_image_from_path,
    resolve_path,
    ensure_directory,
    get_file_info,
    validate_image_path,
)

__all__ = [
    "ToolBase",
    "ToolRegistry",
    "PathHandler",
    "load_image_from_path",
    "resolve_path",
    "ensure_directory",
    "get_file_info",
    "validate_image_path",
]

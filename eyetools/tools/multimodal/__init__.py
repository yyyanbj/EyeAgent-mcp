"""Multimodal conversion tools (fundus -> OCT / eye globe).

This package provides lightweight, dependency-friendly wrappers that perform
basic modality conversion tasks or produce informative placeholders when heavy
generation models are unavailable. Two variants are exposed via config:

- fundus2oct: produce a 32-slice pseudo-OCT volume montage, frames, and GIF
- fundus2eyeglobe: derive a simple point cloud PLY and static PNG visualization

For production-grade 3D generation, wire these wrappers to your model runtime
or adapt the implementation in `tool_impl.py` to call the appropriate pipelines.
"""

from .tool_impl import MultimodalTool  # noqa: F401

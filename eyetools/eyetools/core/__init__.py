"""Core framework components for EyeTools.

Modules:
  config_loader: Parse YAML/Python tool configs into normalized tool definitions.
  registry: In-memory registry of tool metadata.
  tool_base: New abstract base API (thin; can wrap legacy base if needed).
  env_manager: Manage uv-based environment resolution and execution wrappers.
  process_manager: Manage subprocess workers for tools running out-of-process.
  worker_protocol: JSON line protocol implementation for worker processes.
  loader: Filesystem discovery of tool packages and registration.
  utils: Helper functions (hashing, logging, resource usage, path ops).
"""

from .registry import ToolRegistry  # noqa: F401

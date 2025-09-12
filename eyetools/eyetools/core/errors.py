"""Centralized custom exception hierarchy for tool framework."""
from __future__ import annotations

class ToolError(Exception):
    """Base class for all tool related errors."""

class ConfigError(ToolError):
    pass

class RegistrationError(ToolError):
    pass

class ToolNotFoundError(ToolError):
    pass

class EnvironmentError(ToolError):  # environment setup / uv spawning
    pass

class WorkerInitError(ToolError):
    pass

class PredictionError(ToolError):
    pass

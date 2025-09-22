"""
Compatibility shim: DiagnosticBaseAgent now aliases BaseAgent.

This file previously contained a separate implementation that diverged from
BaseAgent. To avoid duplication and ensure consistent behavior (normalization,
standardization, tracing), we alias DiagnosticBaseAgent to BaseAgent.
"""

from .base_agent import BaseAgent as DiagnosticBaseAgent

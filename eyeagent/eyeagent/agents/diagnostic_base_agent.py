"""
Compatibility shim: DiagnosticBaseAgent now aliases BaseAgent.

This file previously contained a separate implementation that diverged from
BaseAgent. To avoid duplication and ensure consistent behavior (normalization,
standardization, trace), we alias DiagnosticBaseAgent to BaseAgent.

Deprecated: Prefer importing BaseAgent directly.
"""

import warnings

warnings.warn(
	"DiagnosticBaseAgent is deprecated; import BaseAgent instead.",
	DeprecationWarning,
	stacklevel=2,
)

from .base_agent import BaseAgent as DiagnosticBaseAgent

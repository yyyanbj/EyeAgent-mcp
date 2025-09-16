"""Tools namespace aggregator.

Added disease-specific classification (RETFound Dinov3) tools package.
"""

try:  # optional import
	from .disease_specific_cls import DiseaseSpecificClassificationTool  # noqa: F401
except Exception:  # pragma: no cover
	pass

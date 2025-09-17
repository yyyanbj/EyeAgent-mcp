from typing import Any, Dict, List, Optional

def get_capabilities(agent_class) -> Dict[str, Any]:
    """Return declared capabilities for an agent class (static or via method)."""
    # Prefer class attribute or method
    if hasattr(agent_class, "capabilities"):
        caps = getattr(agent_class, "capabilities")
        if callable(caps):
            return caps()
        return caps
    return {}

# Example capability structure for documentation and agent use
# {
#   "required_context": ["images", "patient"],
#   "expected_outputs": ["diagnoses", "lesions"],
#   "retry_policy": {"max_attempts": 2, "on_fail": "skip"},
#   "modalities": ["CFP", "OCT"],
#   "tools": ["classification:modality", ...]
# }

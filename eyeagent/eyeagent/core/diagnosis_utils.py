from __future__ import annotations
from typing import Dict, List


def get_candidate_diseases_from_probs(probs: Dict[str, float] | Dict[str, object] | None, threshold: float = 0.3, top_k: int = 5) -> List[str]:
    """Select candidate disease keys from a probabilities dict.

    - Cast values to float when possible
    - Filter by threshold
    - Sort by prob desc and take top_k
    """
    if not isinstance(probs, dict) or not probs:
        return []
    items: List[tuple[str, float]] = []
    for k, v in probs.items():
        try:
            fv = float(v) if v is not None else 0.0
        except Exception:
            fv = 0.0
        items.append((str(k), fv))
    items.sort(key=lambda kv: kv[1], reverse=True)
    return [k for k, v in items if v > threshold][:top_k]

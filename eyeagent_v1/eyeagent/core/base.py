from __future__ import annotations
from typing import Any, Dict, Optional

class Agent:
    """Synchronous base Agent.

    Contract:
    - init(config, trace)
    - run(context) -> dict (updates)
    """
    def __init__(self, config: Dict[str, Any], trace: Any):
        self.config = config or {}
        self.trace = trace
        self.name = getattr(self, "__agent_name__", self.__class__.__name__)

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


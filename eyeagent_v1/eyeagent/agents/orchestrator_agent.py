from __future__ import annotations
from typing import Any, Dict, List

from ..core.base import Agent
from ..core.registry import register_agent, create_agent


@register_agent("orchestrator")
class OrchestratorAgent(Agent):
    """Run a sequence of agents defined in config["pipeline"].

    Expected config shape:
    {
      "pipeline": [
         {"agent": "knowledge", "config": {...}},
         {"agent": "report",    "config": {...}}
      ]
    }
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pipeline: List[Dict[str, Any]] = self.config.get("pipeline", [])
        for step in pipeline:
            name = step.get("agent")
            cfg = step.get("config", {})
            if not name:
                continue
            self.trace.append_event(context["case_id"], {
                "type": "agent_step",
                "agent": name,
                "stage": "start",
                "input_keys": list(context.keys()),
            })
            agent = create_agent(name, config=cfg, trace=self.trace)
            updates = agent.run(context)
            if updates:
                context.update(updates)
            self.trace.append_event(context["case_id"], {
                "type": "agent_step",
                "agent": name,
                "stage": "end",
                "output_keys": list(updates.keys() if updates else []),
            })
        return context

from __future__ import annotations
from typing import Any, Dict

from ..core.base import Agent
from ..core.registry import register_agent


@register_agent("knowledge")
class KnowledgeAgent(Agent):
    """A stub knowledge agent that takes a prompt and returns a canned answer.
    Replace this with a real LLM/tool call per your needs (kept sync as requested).
    """

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (self.config.get("prompts", {}) or {}).get("system") or "Summarize input"
        user_input = context.get("input", "")
        answer = f"[knowledge] {prompt}: {user_input}".strip()
        self.trace.append_event(context["case_id"], {
            "type": "agent_output",
            "agent": self.name,
            "summary": answer[:200]
        })
        return {"knowledge_answer": answer}

from __future__ import annotations
from typing import Any, Dict

from ..core.base import Agent
from ..core.registry import register_agent


@register_agent("report")
class ReportAgent(Agent):
    """Assemble a simple final report using prior context outputs."""

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            "case_id": context.get("case_id"),
            "knowledge_answer": context.get("knowledge_answer", ""),
        }
        self.trace.write_final_report(context["case_id"], report)
        return {"final_report": report}

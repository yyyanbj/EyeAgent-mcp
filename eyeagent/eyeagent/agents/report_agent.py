from typing import Any, Dict, List
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent

@register_agent
class ReportAgent(DiagnosticBaseAgent):
    role = "report"
    name = "ReportAgent"
    allowed_tool_ids: List[str] = []  # no tools required
    system_prompt = (
        "You are the report agent. Consolidate all intermediate results into the final JSON report fragment including diagnoses, lesions, management, and reasoning."
        " Explicitly state any missing information."
    )

    # Capabilities declaration for report
    capabilities = {
        "required_context": ["image_analysis", "specialist", "follow_up"],
        "expected_outputs": ["diagnoses", "lesions", "management", "reasoning", "narrative", "conclusion"],
        "retry_policy": {"max_attempts": 1, "on_fail": "fail"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Coerce possibly None fields (engine may map selective outputs only)
        image_analysis = context.get("image_analysis") or {}
        specialist = context.get("specialist") or {}
        follow_up = context.get("follow_up") or {}
        knowledge = context.get("knowledge") or {}

        # Prefer per-image specialist grading if available
        per_sp = (specialist.get("per_image") or {}).get("disease_grades") or {}
        flat_sp = specialist.get("disease_grades", [])
        # Build diagnoses list (flattened for backward compatibility)
        diagnoses: List[Dict[str, Any]] = []
        for dg in flat_sp:
            diagnoses.append({
                "disease": dg.get("disease"),
                "grade": dg.get("grade"),
                "confidence": dg.get("confidence"),
                "evidence": []
            })
        # Lesions from IA per-image if present, else raw lesions
        per_ia = (image_analysis.get("per_image") or {})
        lesions = per_ia.get("lesions") if isinstance(per_ia, dict) and per_ia.get("lesions") is not None else image_analysis.get("lesions")
        management = follow_up.get("management")
        # Compose a narrative and explicit diagnostic conclusion
        diag_txt = ", ".join([f"{d.get('disease')} {d.get('grade')}" for d in diagnoses if d.get('disease') and d.get('grade')])
        mgmt_txt = None
        if isinstance(management, dict):
            sug = management.get("suggestion")
            months = management.get("follow_up_months")
            if sug and months is not None:
                mgmt_txt = f"{sug} (in {months} months)"
            elif sug:
                mgmt_txt = str(sug)
        # optional upstream narratives
        ia_narr = (context.get("image_analysis") or {}).get("narrative")
        sp_narr = (context.get("specialist") or {}).get("narrative")
        kn_narr = (context.get("knowledge") or {}).get("narrative")
        fu_narr = (context.get("follow_up") or {}).get("narrative")
        narrative_parts = []
        if ia_narr:
            narrative_parts.append(str(ia_narr))
        if sp_narr:
            narrative_parts.append(str(sp_narr))
        if kn_narr:
            narrative_parts.append(str(kn_narr))
        # If per-image narratives are ever added, they can be merged here as well
        # Compose final summary paragraph and conclusion, then polish via LLM
        final_sentence = []
        if diag_txt:
            final_sentence.append(f"Final impression: {diag_txt}.")
        if mgmt_txt:
            final_sentence.append(f"Suggested management: {mgmt_txt}.")
        if fu_narr:
            final_sentence.append(str(fu_narr))
        base_reasoning = " ".join(final_sentence) or "Consolidated findings into a final report."
        reasoning = self.gen_reasoning(base_reasoning)
        # Conclusion: primary diagnosis line with management if available
        conclusion = None
        if diag_txt and mgmt_txt:
            conclusion = f"{diag_txt}; {mgmt_txt}"
        elif diag_txt:
            conclusion = diag_txt
        elif mgmt_txt:
            conclusion = mgmt_txt
        outputs = {
            "diagnoses": diagnoses,
            "lesions": lesions,
            "management": management,
            "reasoning": reasoning,
            "narrative": reasoning,
            "conclusion": conclusion,
            # Pass through per-image blocks for downstream consumers
            "per_image": {
                "lesions": lesions if isinstance(lesions, dict) else None,
                "specialist": per_sp or None,
            },
            "knowledge": knowledge or None,
        }
        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": [], "reasoning": reasoning}

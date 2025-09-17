from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent

class ReportAgent(DiagnosticBaseAgent):
    role = "report"
    name = "ReportAgent"
    allowed_tool_ids: List[str] = []  # no tools required
    system_prompt = (
        "You are the report agent. Consolidate all intermediate results into the final JSON report fragment including diagnoses, lesions, management, and reasoning."
        " Explicitly state any missing information."
    )

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        image_analysis = context.get("image_analysis", {})
        specialist = context.get("specialist", {})
        follow_up = context.get("follow_up", {})

        diagnoses = []
        for dg in specialist.get("disease_grades", []):
            diagnoses.append({
                "disease": dg.get("disease"),
                "grade": dg.get("grade"),
                "confidence": dg.get("confidence"),
                "evidence": []
            })
        lesions = image_analysis.get("lesions")
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
        fu_narr = (context.get("follow_up") or {}).get("narrative")
        narrative_parts = []
        if ia_narr:
            narrative_parts.append(str(ia_narr))
        if sp_narr:
            narrative_parts.append(str(sp_narr))
        # Compose final summary paragraph and conclusion
        final_sentence = []
        if diag_txt:
            final_sentence.append(f"Final impression: {diag_txt}.")
        if mgmt_txt:
            final_sentence.append(f"Suggested management: {mgmt_txt}.")
        if fu_narr:
            final_sentence.append(str(fu_narr))
        reasoning = " ".join(final_sentence) or "Consolidated findings into a final report."
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
            "conclusion": conclusion
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

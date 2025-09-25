from typing import Any, Dict, List

from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent


@register_agent
class DecisionAgent(DiagnosticBaseAgent):
    """Aggregate upstream findings to form an explicit diagnostic decision.

    Inputs: image_analysis (optional), specialist (preferred), orchestrator_outputs (screening fallback)
    Tools: none
    Outputs: decision: {
        primary: List[{disease, grade, confidence}],
        candidates: List[str],
        notes: str
    }, narrative: str
    """

    role = "decision"
    name = "DecisionAgent"
    allowed_tool_ids: List[str] = [
        # optional knowledge tools
        "rag:query",
        "web_search:pubmed",
        "web_search:tavily",
    ]
    system_prompt = (
        "ROLE: Diagnostic decision aggregation.\n"
        "GOAL: Based on upstream grades/scores, produce an explicit diagnosis decision for downstream agents.\n"
        "INPUTS: specialist (preferred), image_analysis, orchestrator screening results.\n"
        "OUTPUTS: decision.primary (disease, grade, confidence), decision.candidates, decision.notes, and a concise narrative.\n"
        "CONSTRAINTS: You may optionally consult knowledge tools for brief evidence; remain conservative if uncertainty is high."
    )

    capabilities = {
        "required_context": ["patient"],
        "expected_outputs": ["decision", "narrative"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        specialist = context.get("specialist") or {}
        image_analysis = context.get("image_analysis") or {}
        orch = context.get("orchestrator_outputs") or {}

        # Collect primary diagnoses from specialist if available
        primary: List[Dict[str, Any]] = []
        try:
            for dg in (specialist.get("disease_grades") or []):
                if not isinstance(dg, dict):
                    continue
                primary.append({
                    "disease": dg.get("disease"),
                    "grade": dg.get("grade"),
                    "confidence": dg.get("confidence"),
                })
        except Exception:
            pass

        # Candidates from image_analysis probabilities as a fallback
        candidates: List[str] = []
        try:
            probs = image_analysis.get("diseases") if isinstance(image_analysis, dict) else None
            if isinstance(probs, dict):
                # top-5 by probability
                items = sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:5]
                candidates = [k for k, _ in items]
        except Exception:
            pass
        # Last resort: preliminary screening in orchestrator outputs
        if not candidates:
            try:
                scr = (orch.get("screening_results") or [])
                if scr:
                    out = (scr[0] or {}).get("output")
                    if isinstance(out, dict):
                        pr = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                        if isinstance(pr, dict):
                            candidates = list(pr.keys())[:5]
            except Exception:
                pass

        # Compose a short narrative via LLM from a deterministic summary
        prim_txt = ", ".join([f"{d.get('disease')} {d.get('grade')}" for d in primary if d.get("disease") and d.get("grade")])
        cand_txt = ", ".join([c for c in candidates])
        base_summary = (
            (f"Primary: {prim_txt}. " if prim_txt else "") +
            (f"Candidates: {cand_txt}." if cand_txt else "Candidates pending further confirmation.")
        ).strip()
        narrative = self.gen_reasoning(base_summary)

        # Optionally consult knowledge based on candidates
        kn_blocks: List[Dict[str, Any]] = []
        kn_query = None
        try:
            from ..config.tools_filter import filter_tool_ids  # lazy import to avoid cycles
            kn_allowed = filter_tool_ids(self.__class__.__name__, list(self.allowed_tool_ids))
        except Exception:
            kn_allowed = list(self.allowed_tool_ids)
        if kn_allowed and candidates:
            kn_query = ", ".join(candidates[:5])
            plan = await self.plan_tools(f"If helpful, fetch brief evidence for decision: {kn_query}", kn_allowed)
            async with self._client_ctx() as client:
                for step in (plan or []):
                    tid = step.get("tool_id")
                    if tid not in kn_allowed:
                        continue
                    if not self._knowledge_allowed():
                        break
                    args = self._apply_knowledge_defaults(tid, step.get("arguments"))
                    tc = await self._call_tool(client, tid, args)
                    self._note_knowledge_called()
                    tc["reasoning"] = step.get("reasoning")
                    # record but don't change primary decision
                    out = tc.get("output")
                    if isinstance(out, dict):
                        kn_blocks.append(out)

        outputs = {
            "decision": {
                "primary": primary,
                "candidates": candidates,
                "notes": None,
            },
            "narrative": narrative,
            "knowledge": {"query": kn_query, "items": kn_blocks} if kn_blocks else None,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": narrative,
        })

        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": [], "reasoning": narrative}

from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from ..tools.tool_registry import specialist_tools, get_tool, resolve_specialist_tools
from fastmcp import Client

class SpecialistAgent(DiagnosticBaseAgent):
    role = "specialist"
    name = "SpecialistAgent"
    allowed_tool_ids: List[str] = []  # decided at runtime based on candidate diseases
    system_prompt = (
        "You are the specialist agent. For candidate diseases, call corresponding grading models and return results with reasoning."
    )

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        candidate_diseases: List[str] = context.get("candidate_diseases", [])
        # Resolve candidate diseases to concrete disease-specific tools (robust mapping)
        tools_meta = resolve_specialist_tools(candidate_diseases)
        # Fallback: if nothing resolved but candidates exist, try simple substring search as backup
        if not tools_meta and candidate_diseases:
            tools_meta = specialist_tools(candidate_diseases)
        self.allowed_tool_ids = [m.get("tool_id") for m in tools_meta]
        # If still empty, emit a trace note and short-circuit with empty outputs
        if not self.allowed_tool_ids:
            note = {
                "type": "agent_step",
                "agent": self.name,
                "role": self.role,
                "outputs": {"disease_grades": [], "narrative": "No specialist tools matched candidate diseases."},
                "tool_calls": [],
                "reasoning": f"No tools resolved for: {candidate_diseases}"
            }
            self.trace_logger.append_event(self.case_id, note)
            return {"agent": self.name, "role": self.role, "outputs": note["outputs"], "tool_calls": [], "reasoning": note["reasoning"]}

        task_desc = f"For candidate diseases {candidate_diseases}, call grading models one by one."
        plan = await self.plan_tools(task_desc, self.allowed_tool_ids)
        if not plan:
            plan = [
                {"tool_id": tid, "arguments": None, "reasoning": "Run disease-specific grading."}
                for tid in self.allowed_tool_ids
            ]

        tool_calls = []
        results = []
        async with Client(self.mcp_url) as client:
            for step in plan:
                tool_id = step.get("tool_id")
                if tool_id not in self.allowed_tool_ids:
                    continue
                # Most disease-specific tools expect image_path
                img_path = (context.get("images") or [{}])[0].get("path") if context.get("images") else None
                args = step.get("arguments") or ({"image_path": img_path} if img_path else {})
                tc = await self._call_tool(client, tool_id, args)
                tc["reasoning"] = step.get("reasoning")
                tool_calls.append(tc)
                if tc.get("status") == "success":
                    meta = get_tool(tool_id) or {}
                    output = tc.get("output") or {}
                    results.append({
                        "disease": meta.get("disease") or tool_id.split(":")[-1],
                        "grade": output.get("grade") if isinstance(output, dict) else None,
                        "confidence": output.get("confidence") if isinstance(output, dict) else None,
                        "tool_id": tool_id
                    })
        # Narrative summary for specialist grading
        bits = []
        for r in results:
            d = r.get("disease")
            g = r.get("grade")
            if d and g:
                conf = r.get("confidence")
                s = f"{d} {g}"
                if conf is not None:
                    try:
                        s += f" ({float(conf)*100:.1f}%)"
                    except Exception:
                        s += f" ({conf})"
                bits.append(s)
        reasoning = ("Specialist grading summary: " + ", ".join(bits)) if bits else "Specialist grading completed."
        outputs = {"disease_grades": results, "narrative": reasoning}
        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}

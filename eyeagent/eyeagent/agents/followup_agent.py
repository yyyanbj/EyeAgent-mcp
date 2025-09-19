from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client

@register_agent
class FollowUpAgent(DiagnosticBaseAgent):
    role = "follow_up"
    name = "FollowUpAgent"
    allowed_tool_ids = ["classification:cfp_age"]
    system_prompt = (
        "You are the follow-up agent. Combine disease grades and age to produce management suggestions with reasoning."
    )

    # Capabilities declaration for follow-up
    capabilities = {
        "required_context": ["disease_grades", "images"],
        "expected_outputs": ["management", "narrative"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        disease_grades = context.get("disease_grades", [])
        images = context.get("images", [])
        plan = await self.plan_tools("Get or confirm age from images if needed, then generate management plan.", self.allowed_tool_ids)
        if not plan:
            plan = [
                {"tool_id": "classification:cfp_age", "arguments": None, "reasoning": "Estimate age from CFP if available."}
            ]

        tool_calls = []
        age_info = None
        async with self._client_ctx() as client:
            for step in plan:
                tool_id = step.get("tool_id")
                if tool_id not in self.allowed_tool_ids:
                    continue
                img_path = images[0].get("path") if images else None
                args = step.get("arguments") or ({"image_path": img_path} if img_path else {})
                tc = await self._call_tool(client, tool_id, args)
                tc["reasoning"] = step.get("reasoning")
                tool_calls.append(tc)
                if tc.get("status") == "success":
                    age_info = tc.get("output")

        # Simple rule-based example
        suggestion = "Annual follow-up"
        follow_up_months = 12
        for dg in disease_grades:
            dis = (dg.get("disease") or "").upper()
            grade = dg.get("grade") or ""
            if dis == "DR" and grade.upper().startswith("R3"):
                suggestion = "Follow-up in 3 months; consider referral if warranted"
                follow_up_months = 3
            if dis == "AMD" and grade:
                suggestion = "Follow-up in 6 months with OCT"
                follow_up_months = min(follow_up_months, 6)
        age_txt = None
        if isinstance(age_info, dict):
            age_pred = age_info.get("prediction") or age_info.get("age")
            if age_pred is not None:
                age_txt = str(age_pred)
        base_summary = (
            "Follow-up summary: "
            f"recommendation is '{suggestion}' with interval {follow_up_months} months"
            + (f", estimated age {age_txt}" if age_txt else "")
            + "."
        )
        reasoning = self.gen_reasoning(base_summary)
        outputs = {"management": {"suggestion": suggestion, "follow_up_months": follow_up_months, "age_info": age_info}, "narrative": reasoning}

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}

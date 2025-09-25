from typing import Any, Dict, List
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from ..config.tools_filter import filter_tool_ids, select_tool_ids
from fastmcp import Client

@register_agent
class FollowUpAgent(DiagnosticBaseAgent):
    role = "follow_up"
    name = "FollowUpAgent"
    allowed_tool_ids = ["classification:cfp_age", "rag:query", "web_search:pubmed", "web_search:tavily"]
    system_prompt = (
        "ROLE: Follow-up & management.\n"
        "GOAL: Combine disease grades and basic demographics (e.g., age) to produce a clear management recommendation and interval.\n"
    "TOOLS: classification:cfp_age (optional if age unknown); knowledge tools (rag:query, web_search:*) for evidence.\n"
    "INPUTS: disease_grades, images, patient; may consult knowledge inline instead of a separate knowledge agent.\n"
    "OUTPUTS: management (suggestion, follow_up_months, age_info, knowledge) and a brief narrative.\n"
    "CONSTRAINTS: Do not restate the final report; focus on actionable management guidance. Do NOT include any predicted age in the narrative; age may be used internally as a reference only."
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
        # Build a plan: optionally get age, then fetch brief knowledge evidence
        allowed = select_tool_ids(self.__class__.__name__, base_tool_ids=self.allowed_tool_ids, role=self.role)
        plan = await self.plan_tools("Optionally estimate age; then query brief knowledge evidence to support management.", allowed)
        if not plan:
            plan = [
                {"tool_id": "classification:cfp_age", "arguments": None, "reasoning": "Estimate age from CFP if available."},
                {"tool_id": "rag:query", "arguments": {"query": "ophthalmology follow-up management guidance", "top_k": 3}, "reasoning": "Fetch concise internal evidence for follow-up intervals."}
            ]

        tool_calls: List[Dict[str, Any]] = []
        age_info = None
        knowledge_blocks: List[Dict[str, Any]] = []
        knowledge_query = None
        async with self._client_ctx() as client:
            for step in plan:
                tool_id = step.get("tool_id")
                if tool_id not in allowed:
                    continue
                # Prefer image path when calling age tool; not used for RAG/web tools
                args = dict(step.get("arguments") or {})
                if tool_id == "classification:cfp_age":
                    img_path = images[0].get("path") if images else None
                    if img_path:
                        args["image_path"] = img_path
                else:
                    if not self._knowledge_allowed():
                        continue
                    knowledge_query = knowledge_query or args.get("query")
                    args = self._apply_knowledge_defaults(tool_id, args)
                tc = await self._call_tool(client, tool_id, args)
                if tool_id != "classification:cfp_age":
                    self._note_knowledge_called()
                tc["reasoning"] = step.get("reasoning")
                tool_calls.append(tc)
                if tc.get("status") == "success":
                    out = tc.get("output")
                    if tool_id == "classification:cfp_age":
                        age_info = out
                    elif isinstance(out, dict):
                        knowledge_blocks.append(out)

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
        # Build narrative WITHOUT stating predicted age; age may be considered internally only
        base_summary = (
            "Follow-up summary: "
            f"recommendation is '{suggestion}' with interval {follow_up_months} months."
        )
        reasoning = self.gen_reasoning(base_summary)
        outputs = {
            "management": {
                "suggestion": suggestion,
                "follow_up_months": follow_up_months,
                "age_info": age_info,
                "knowledge": {"query": knowledge_query, "items": knowledge_blocks} if knowledge_blocks else None,
            },
            "narrative": reasoning,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}

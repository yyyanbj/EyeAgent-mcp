from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client

@register_agent
class OrchestratorAgent(DiagnosticBaseAgent):
    role = "orchestrator"
    name = "OrchestratorAgent"
    allowed_tool_ids = ["classification:modality", "classification:laterality", "classification:multidis"]
    system_prompt = (
        "You are the orchestrator. Tasks: (1) infer imaging modality, (2) classify laterality, (3) plan downstream pipeline. "
        "Always provide reasoning."
    )

    # Capabilities declaration for orchestrator
    capabilities = {
        "required_context": ["images", "patient"],
        "expected_outputs": ["pipeline", "modality_results", "laterality_results"],
        "retry_policy": {"max_attempts": 2, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        task_desc = (
            "Given raw images, first ensure each image has modality (CFP/OCT/FFA) and eye (OD/OS)."
            " Then, based on modality, plan the pipeline (typically ImageAnalysis -> Specialist -> FollowUp -> Report)."
            " Return JSON array plan of tools as needed."
        )

        plan = await self.plan_tools(task_desc, self.allowed_tool_ids)
        if not plan:
            # Fallback minimal plan if planner disabled: attempt modality and laterality
            plan = [
                {"tool_id": "classification:modality", "arguments": None, "reasoning": "Baseline modality classification."},
                {"tool_id": "classification:laterality", "arguments": None, "reasoning": "Baseline laterality classification."},
            ]

        tool_calls = []
        async with Client(self.mcp_url) as client:
            for req in plan:
                tool_id = req.get("tool_id")
                if tool_id not in self.allowed_tool_ids:
                    continue
                # All classification tools expect a single image_path
                img_path = images[0].get("path") if images else None
                arguments = req.get("arguments") or ({"image_path": img_path} if img_path else {})
                tc = await self._call_tool(client, tool_id, arguments)
                tc["reasoning"] = req.get("reasoning")
                tool_calls.append(tc)

        modality_results = [c for c in tool_calls if c["tool_id"] == "classification:modality"]
        laterality_results = [c for c in tool_calls if c["tool_id"] == "classification:laterality"]

        # Optionally call multidis for quick screening when modality is CFP (or unknown)
        try:
            modality_label = None
            if modality_results:
                out = (modality_results[0] or {}).get("output")
                if isinstance(out, dict):
                    modality_label = out.get("label") or out.get("prediction")
            if (modality_label or "").upper() in ("", "CFP"):
                async with Client(self.mcp_url) as client:
                    img_path = images[0].get("path") if images else None
                    args = {"image_path": img_path} if img_path else {}
                    tc = await self._call_tool(client, "classification:multidis", args)
                    tc["reasoning"] = "Screen for multiple diseases at orchestration stage."
                    tool_calls.append(tc)
        except Exception:
            pass

        planned_pipeline = ["image_analysis", "specialist", "follow_up", "report"]
        reasoning = "Planned standard pipeline based on available modality and analysis goals."
        screening_results = [c for c in tool_calls if c["tool_id"] == "classification:multidis"]
        outputs = {
            "modality_results": modality_results,
            "laterality_results": laterality_results,
            "planned_pipeline": planned_pipeline,
            "screening_results": screening_results,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })

        return {
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        }

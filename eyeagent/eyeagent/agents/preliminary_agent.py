from typing import Any, Dict, List
from .base_agent import BaseAgent
from .registry import register_agent
from ..config.tools_filter import filter_tool_ids


@register_agent
class PreliminaryAgent(BaseAgent):
    role = "preliminary"
    name = "PreliminaryAgent"
    # Move screening tools here from Orchestrator
    allowed_tool_ids = [
        "classification:modality",
        "classification:laterality",
        "classification:multidis",
    ]
    system_prompt = (
        "ROLE: Preliminary screening.\n"
        "GOAL: Quickly determine modality (CFP/OCT/FFA), laterality (OD/OS), and a coarse multi-disease screening score.\n"
        "TOOLS: classification:modality, classification:laterality, classification:multidis (filtered by config).\n"
        "INPUTS: images, patient.\n"
        "OUTPUTS: modality_results (per-image), laterality_results (per-image), screening_results (at least 1 result).\n"
        "GUIDANCE: If multiple images exist, run modality and laterality per-image. Run multi-disease screening once (first image is enough). Provide brief reasoning only."
    )

    capabilities = {
        "required_context": ["images", "patient"],
        "expected_outputs": ["modality_results", "laterality_results", "screening_results"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])

        # Filter tools by config
        filtered_allowed = filter_tool_ids(self.__class__.__name__, list(self.allowed_tool_ids))

        tool_calls: List[Dict[str, Any]] = []
        modality_results: List[Dict[str, Any]] = []
        laterality_results: List[Dict[str, Any]] = []
        screening_results: List[Dict[str, Any]] = []

        # Always attempt modality and laterality for each image (if tools allowed)
        async with self._client_ctx() as client:
            if "classification:modality" in filtered_allowed:
                # Ensure image_path is passed per-image
                calls = await self.call_tool_per_image(client, "classification:modality", images, {})
                for c in calls:
                    c["reasoning"] = "Determine image modality"
                tool_calls.extend(calls)
                modality_results = calls
            if "classification:laterality" in filtered_allowed:
                # Ensure image_path is passed per-image
                calls = await self.call_tool_per_image(client, "classification:laterality", images, {})
                for c in calls:
                    c["reasoning"] = "Determine eye laterality"
                tool_calls.extend(calls)
                laterality_results = calls

            # Multi-disease screening once (on first image) if available
            if "classification:multidis" in filtered_allowed:
                args = {}
                if images and isinstance(images[0], dict) and images[0].get("path"):
                    args["image_path"] = images[0]["path"]
                tc = await self._call_tool(client, "classification:multidis", args)
                tc["reasoning"] = "Quick multi-disease screening"
                tool_calls.append(tc)
                screening_results = [tc]

        outputs = {
            "modality_results": modality_results,
            "laterality_results": laterality_results,
            "screening_results": screening_results,
        }

        # Append trace event for UI streaming consistency
        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": "Completed preliminary screening to support routing"
        })

        return {
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": "Completed preliminary screening to support routing",
        }

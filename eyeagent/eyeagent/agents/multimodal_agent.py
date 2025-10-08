from typing import Any, Dict, List
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from ..config.tools_filter import select_tool_ids


@register_agent
class MultimodalAgent(DiagnosticBaseAgent):
    role = "multimodal"
    name = "MultimodalAgent"
    # Default allowed tools; can be filtered via config.tools_filter
    allowed_tool_ids = [
        "multimodal:fundus2oct",
        "multimodal:fundus2eyeglobe",
    ]
    system_prompt = (
        "ROLE: Multimodal conversion agent.\n"
        "GOAL: Given CFP images, perform requested or suitable modality conversions using available multimodal tools.\n"
        "TOOLS: multimodal:fundus2oct (CFP→pseudo-OCT), multimodal:fundus2eyeglobe (CFP→eye globe).\n"
        "INPUTS: images, and optional parameters in context.multimodal_params (per tool).\n"
        "OUTPUTS: output_paths per image per tool (e.g., montage/gif/ply/png), concise narrative.\n"
        "CONSTRAINTS: Do not perform diagnosis; only generate converted artifacts."
    )

    capabilities = {
        "required_context": ["images"],
        "expected_outputs": ["conversions"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        params = context.get("multimodal_params") or {}
        # Decide which tools to run: prefer explicit request in params["tools"], else both
        requested: List[str] = []
        if isinstance(params, dict):
            t = params.get("tools")
            if isinstance(t, list):
                requested = [str(x) for x in t if isinstance(x, str)]
        base_allowed = self.allowed_tool_ids
        allowed = select_tool_ids(self.__class__.__name__, base_tool_ids=base_allowed, role=self.role)
        run_tools = [tid for tid in (requested or base_allowed) if tid in allowed]
        if not run_tools:
            run_tools = allowed

        tool_calls: List[Dict[str, Any]] = []
        conversions: Dict[str, Any] = {}
        async with self._client_ctx() as client:
            for tid in run_tools:
                # Per-image arguments: merge global per-tool params with image path
                per_tool_args = {}
                if isinstance(params.get(tid), dict):
                    per_tool_args = dict(params.get(tid) or {})
                calls = await self.call_tool_per_image(client, tid, images, per_tool_args)
                for c in calls:
                    tool_calls.append(c)
                    img_id = c.get("image_id") or "_"
                    conversions.setdefault(img_id, {}).setdefault(tid, c.get("output"))

        # Build a concise narrative
        num_imgs = len(images) if isinstance(images, list) else 0
        kinds = ", ".join([t.split(":", 1)[-1] for t in run_tools]) or "none"
        summary = f"Ran multimodal conversions ({kinds}) on {num_imgs} image(s); artifacts saved to output_paths."
        reasoning = self.gen_reasoning(summary)

        outputs = {
            "conversions": conversions,
            "tools_used": run_tools,
            "narrative": reasoning,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
        })

        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}

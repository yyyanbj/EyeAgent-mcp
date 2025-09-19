from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client
from loguru import logger
from ..config.tools_filter import filter_tool_ids

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

        # Apply config filter to allowed tools
        filtered_allowed = filter_tool_ids(self.__class__.__name__, list(self.allowed_tool_ids))
        plan = await self.plan_tools(task_desc, filtered_allowed)
        if not plan:
            # Fallback minimal plan if planner disabled: attempt modality and laterality
            plan = [
                {"tool_id": "classification:modality", "arguments": None, "reasoning": "Baseline modality classification."},
                {"tool_id": "classification:laterality", "arguments": None, "reasoning": "Baseline laterality classification."},
            ]

        tool_calls = []
        async with self._client_ctx() as client:
            for req in plan:
                tool_id = req.get("tool_id")
                if tool_id not in filtered_allowed:
                    continue
                # Run per image when available
                img_list = images if images else [None]
                for img in img_list:
                    img_path = (img or {}).get("path") if isinstance(img, dict) else None
                    arguments = dict(req.get("arguments") or {})
                    if img_path:
                        # force correct image_path into arguments, overriding any placeholder
                        arguments["image_path"] = img_path
                    tc = await self._call_tool(client, tool_id, arguments)
                    tc["reasoning"] = req.get("reasoning")
                    if isinstance(img, dict):
                        tc["image_id"] = img.get("image_id") or img.get("path")
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
                async with self._client_ctx() as client:
                    img_path = images[0].get("path") if images else None
                    args = {"image_path": img_path} if img_path else {}
                    tc = await self._call_tool(client, "classification:multidis", args)
                    tc["reasoning"] = "Screen for multiple diseases at orchestration stage."
                    tool_calls.append(tc)
        except Exception:
            pass

        # Dynamic pipeline decision based on modality and screening confidences
        planned_pipeline = ["image_analysis", "specialist", "follow_up", "report"]
        screening_results = [c for c in tool_calls if c["tool_id"] == "classification:multidis"]
        modality_label = None
        try:
            if modality_results:
                mout = (modality_results[0] or {}).get("output")
                if isinstance(mout, dict):
                    modality_label = mout.get("label") or mout.get("prediction")
        except Exception:
            modality_label = None
        # Parse screening confidences (assume dict of disease->prob)
        top_dis = None
        top_prob = 0.0
        try:
            if screening_results:
                s_out = (screening_results[0] or {}).get("output")
                if isinstance(s_out, dict):
                    # either probabilities dict or predictions list with probs
                    probs = s_out.get("probabilities") if isinstance(s_out.get("probabilities"), dict) else s_out
                    if isinstance(probs, dict) and probs:
                        for k, v in probs.items():
                            try:
                                fv = float(v)
                            except Exception:
                                continue
                            if fv > top_prob:
                                top_prob = fv
                                top_dis = k
                    elif isinstance(s_out, dict) and s_out:
                        # fallback: treat values as probs
                        for k, v in s_out.items():
                            try:
                                fv = float(v)
                            except Exception:
                                continue
                            if fv > top_prob:
                                top_prob = fv
                                top_dis = k
        except Exception:
            pass
        # Heuristics:
        # - If modality is OCT, skip CFP-specific follow_up possibly; still keep specialist/report
        # - If screening confidence is very low (< 0.15), skip specialist to save time
        # - If no images or bad quality likely, jump straight to report
        reasons: List[str] = []
        if (modality_label or "").upper() == "OCT":
            reasons.append("Modality is OCT; image_analysis will use OCT tools.")
        if top_prob and top_prob < 0.15:
            # remove specialist while keeping report
            if "specialist" in planned_pipeline:
                planned_pipeline.remove("specialist")
                reasons.append(f"Screening low confidence ({top_prob:.2f}); skipping specialist.")
        if not images:
            planned_pipeline = ["report"]
            reasons.append("No images provided; generating report from context only.")
        reasoning = "; ".join(reasons) if reasons else "Planned standard pipeline based on modality and screening."
        logger.info(f"[orchestrator] modality={modality_label} top_screen={top_dis}:{top_prob:.3f} pipeline={planned_pipeline}")
        outputs = {
            "modality_results": modality_results,
            "laterality_results": laterality_results,
            "planned_pipeline": planned_pipeline,
            "screening_results": screening_results,
        }

        # Add a routing message for global conversation context
        try:
            from ..diagnostic_workflow import _append_messages_from_result  # type: ignore
            # Synthesize a small result to inform the global messages
            _append_messages_from_result(context, {
                "agent": self.name,
                "role": self.role,
                "outputs": {"planned_pipeline": planned_pipeline, "modality": modality_label, "top_screen": {"label": top_dis, "prob": top_prob}},
                "tool_calls": [],
                "reasoning": f"Routing via pipeline: {', '.join(planned_pipeline)}"
            })
        except Exception:
            pass

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

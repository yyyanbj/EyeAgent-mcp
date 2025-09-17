from typing import Any, Dict, List
from .diagnostic_base_agent import DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client

@register_agent
class ImageAnalysisAgent(DiagnosticBaseAgent):
    role = "image_analysis"
    name = "ImageAnalysisAgent"
    # Collect all IA tools; selection by modality will happen at runtime
    allowed_tool_ids = [
        # CFP
        "classification:cfp_quality",
        "segmentation:cfp_DR",
        "segmentation:cfp_drusen",
        "segmentation:cfp_cnv",
        "segmentation:cfp_mh",
        "segmentation:cfp_rd",
        "segmentation:cfp_scar",
        "segmentation:cfp_laserscar",
        "segmentation:cfp_laserspots",
        "segmentation:cfp_membrane",
        "segmentation:cfp_edema",
        # "classification:multidis",
        # OCT
        "segmentation:oct_layer",
        "segmentation:oct_PMchovefosclera",
        "segmentation:oct_lesion",
        # FFA
        "segmentation:ffa_lesion",
    ]
    system_prompt = (
        "You are the image analysis agent. Based on the inferred modality (CFP/OCT/FFA), perform quality assessment when applicable, "
        "and run modality-appropriate lesion segmentation. Provide reasoning and structured outputs."
    )

    # Capabilities declaration for image analysis
    capabilities = {
        "required_context": ["images", "orchestrator_outputs"],
        "expected_outputs": ["quality", "lesions", "diseases"],
        "retry_policy": {"max_attempts": 2, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        # Try to infer modality from orchestrator outputs if available
        modality = None
        orch = context.get("orchestrator_outputs") or {}
        mod_calls = orch.get("modality_results") or []
        if mod_calls:
            try:
                # assume first modality result contains dict probs or label
                out = (mod_calls[0] or {}).get("output")
                if isinstance(out, dict):
                    # if label provided
                    modality = out.get("label") or out.get("prediction")
                    # else pick max prob key if probabilities
                    if not modality:
                        modality = max(out.items(), key=lambda kv: float(kv[1] or 0))[0]
            except Exception:
                modality = None

        task_desc = "Perform modality-appropriate analysis (quality if applicable, segmentation)."
        plan = await self.plan_tools(task_desc, self.allowed_tool_ids)
        if not plan:
            plan = []
            # CFP pipeline
            if (modality or "").upper() == "CFP" or modality is None:
                # If modality unknown, default to CFP-friendly pipeline (safe subset)
                plan.append({"tool_id": "classification:cfp_quality", "arguments": None, "reasoning": "Assess CFP quality."})
                for tid in self.allowed_tool_ids:
                    if tid.startswith("segmentation:cfp_"):
                        plan.append({"tool_id": tid, "arguments": None, "reasoning": "Run CFP lesion segmentation."})
            elif (modality or "").upper() == "OCT":
                for tid in self.allowed_tool_ids:
                    if tid.startswith("segmentation:oct_"):
                        plan.append({"tool_id": tid, "arguments": None, "reasoning": "Run OCT segmentation."})
            elif (modality or "").upper() == "FFA":
                plan.append({"tool_id": "segmentation:ffa_lesion", "arguments": None, "reasoning": "Run FFA lesion segmentation."})

        tool_calls = []
        async with Client(self.mcp_url) as client:
            for step in plan:
                tool_id = step.get("tool_id")
                if tool_id not in self.allowed_tool_ids:
                    continue
                img_path = images[0].get("path") if images else None
                args = step.get("arguments") or ({"image_path": img_path} if img_path else {})
                tc = await self._call_tool(client, tool_id, args)
                tc["reasoning"] = step.get("reasoning")
                tool_calls.append(tc)

        # Aggregate outputs (simplified)
        quality = next((c.get("output") for c in tool_calls if c.get("tool_id") == "classification:cfp_quality"), None)
        # lesions: collect any segmentation tool outputs
        lesions = {c.get("tool_id"): c.get("output") for c in tool_calls if (c.get("tool_id", "") or "").startswith("segmentation:")}
        # multi-disease only for CFP
        diseases = next((c.get("output") for c in tool_calls if c.get("tool_id") == "classification:multidis"), None)
        # Build a narrative summary
        def _fmt_prob(p):
            try:
                return f"{float(p)*100:.1f}%"
            except Exception:
                return str(p)
        # Quality summary
        q_txt = "unknown"
        if isinstance(quality, dict):
            q_pred = quality.get("prediction") or quality.get("quality") or quality.get("label")
            if q_pred:
                q_txt = str(q_pred)
        # Lesion summary: collect non-zero counts per tool
        lesion_bits = []
        if isinstance(lesions, dict):
            for tool_id, out in lesions.items():
                if isinstance(out, dict):
                    counts = out.get("counts")
                    if isinstance(counts, dict):
                        nz = [f"{k}:{v}" for k, v in counts.items() if isinstance(v, (int, float)) and float(v) > 0]
                        if nz:
                            lesion_bits.append(f"{tool_id.split(':')[-1]}({', '.join(nz)})")
        lesion_txt = ", ".join(lesion_bits) if lesion_bits else "no obvious lesions detected by the current models"
        # Diseases: top 3
        top_d_txt = ""
        if isinstance(diseases, dict) and diseases:
            try:
                tops = sorted(diseases.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:3]
            except Exception:
                tops = list(diseases.items())[:3]
            top_d_txt = ", ".join([f"{k} {_fmt_prob(v)}" for k, v in tops])
        # Prepare a concise context summary for reasoning; then let LLM polish it
        base_summary = (
            "Image analysis summary: "
            f"image quality is {q_txt}. "
            f"Segmentation suggests {lesion_txt}. "
            + (f"Top disease likelihoods: {top_d_txt}." if top_d_txt else "")
        ).strip()
        reasoning = self.gen_reasoning(base_summary)
        outputs = {"quality": quality, "lesions": lesions, "diseases": diseases, "narrative": reasoning}

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": reasoning}

from typing import Any, Dict, List, Tuple
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client
from eyeagent.core.tools_filter import filter_tool_ids, select_tool_ids
import asyncio
from loguru import logger

@register_agent
class ImageAnalysisAgent(DiagnosticBaseAgent):
    role = "image_analysis"
    name = "ImageAnalysisAgent"
    # Collect all IA tools; selection by modality will happen at runtime
    allowed_tool_ids = [
        # CFP
        "classification:cfp_quality",
        "classification:cfp_age",
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
        # optional knowledge tools
        "rag:query",
        "web_search:pubmed",
        "web_search:tavily",
    ]
    system_prompt = (
        "ROLE: Image analysis.\n"
        "GOAL: Given inferred modality from context (or CFP as safe default), perform quality assessment where applicable and run modality-appropriate segmentation/classification.\n"
        "TOOLS: CFP: classification:cfp_quality, classification:cfp_age, segmentation:cfp_*; OCT: segmentation:oct_*; FFA: segmentation:ffa_*.\n"
        "INPUTS: images, orchestrator_outputs (for modality/laterality hints).\n"
        "OUTPUTS: quality (per-image), age (optional), lesions (per-image per-tool), diseases (aggregated probabilities if present), narrative.\n"
        "CONSTRAINTS: Do not produce the final report or decide routing; limit reasoning to concise clinical interpretation of findings."
    )

    # Capabilities declaration for image analysis
    capabilities = {
        "required_context": ["images", "orchestrator_outputs"],
        "expected_outputs": ["quality", "age", "lesions", "diseases"],
        "retry_policy": {"max_attempts": 2, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": allowed_tool_ids,
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        incremental = bool(context.get("incremental"))
        new_ids_set = set(context.get("new_image_ids") or []) if incremental else set()
        prev_outputs = context.get("image_analysis") if incremental else None
        # 1) Build per-image modality map from preliminary outputs; default to CFP if unknown
        orch = context.get("orchestrator_outputs") or {}
        mod_calls = orch.get("preliminary", {}).get("modality_results") or orch.get("modality_results") or []
        id_to_mod: Dict[str, str] = {}
        try:
            for call in mod_calls:
                img_id = call.get("image_id") or None
                out = call.get("output")
                label = None
                if isinstance(out, dict):
                    label = out.get("label") or out.get("prediction")
                    if not label:
                        # choose max prob key
                        try:
                            label = max((out.get("probabilities") or out).items(), key=lambda kv: float(kv[1] or 0))[0]
                        except Exception as e:
                            logger.debug(f"Failed to infer modality label from probabilities: {e}")
                            label = None
                if img_id and label:
                    id_to_mod[str(img_id)] = str(label).upper()
        except Exception as e:
            logger.exception(f"Failed to build id_to_mod from preliminary modality results: {e}")
            id_to_mod = {}

        # 2) Resolve allowed tools from config once
        allowed = select_tool_ids(self.__class__.__name__, base_tool_ids=self.allowed_tool_ids, role=self.role)

        # 3) Build per-modality static plans (fallback when planner not used)
        def default_plan_for_mod(mod: str) -> List[Dict[str, Any]]:
            mod = (mod or "").upper()
            plan: List[Dict[str, Any]] = []
            if mod in ("", "CFP"):
                if "classification:cfp_quality" in allowed:
                    plan.append({"tool_id": "classification:cfp_quality", "arguments": None, "reasoning": "Assess CFP quality."})
                if "classification:cfp_age" in allowed:
                    plan.append({"tool_id": "classification:cfp_age", "arguments": None, "reasoning": "Estimate patient's age from CFP."})
                for tid in allowed:
                    if tid.startswith("segmentation:cfp_"):
                        plan.append({"tool_id": tid, "arguments": None, "reasoning": "Run CFP lesion segmentation."})
            elif mod == "OCT":
                for tid in allowed:
                    if tid.startswith("segmentation:oct_"):
                        plan.append({"tool_id": tid, "arguments": None, "reasoning": "Run OCT segmentation."})
            elif mod == "FFA":
                if "segmentation:ffa_lesion" in allowed:
                    plan.append({"tool_id": "segmentation:ffa_lesion", "arguments": None, "reasoning": "Run FFA lesion segmentation."})
            return plan

        # 4) Optionally allow the LLM planner to propose a general plan; we will still adapt it per-image by modality
        generic_plan = await self.plan_tools(
            "Perform modality-appropriate analysis (quality if applicable, segmentation).",
            allowed,
        )

        # 5) Execute per-image plans, only calling tools relevant to that image's modality
        tool_calls: List[Dict[str, Any]] = []

        async def run_plan_for_image(client: Client | None, img: Dict[str, Any]) -> List[Dict[str, Any]]:
            img_id = img.get("image_id") or img.get("path") or "_"
            mod = id_to_mod.get(str(img_id)) or "CFP"  # safe default
            # Derive an image-specific plan by filtering generic_plan to this modality; fallback to default
            plan_for_img: List[Dict[str, Any]] = []
            if generic_plan:
                for step in generic_plan:
                    tid = step.get("tool_id")
                    if not tid or tid not in allowed:
                        continue
                    if mod == "CFP" and (tid.startswith("classification:cfp_") or tid.startswith("segmentation:cfp_")):
                        plan_for_img.append(step)
                    elif mod == "OCT" and tid.startswith("segmentation:oct_"):
                        plan_for_img.append(step)
                    elif mod == "FFA" and tid == "segmentation:ffa_lesion":
                        plan_for_img.append(step)
            if not plan_for_img:
                plan_for_img = default_plan_for_mod(mod)

            calls: List[Dict[str, Any]] = []
            for step in plan_for_img:
                tid = step.get("tool_id")
                if not tid or tid not in allowed:
                    continue
                args = dict(step.get("arguments") or {})
                if img.get("path"):
                    args["image_path"] = img["path"]
                tc = await self._call_tool(client, tid, args)
                tc["image_id"] = img_id
                tc["reasoning"] = step.get("reasoning")
                calls.append(tc)
            return calls

        async with self._client_ctx() as client:
            # If no images present, we still allow running default CFP plan once with no image
            if not images:
                for step in default_plan_for_mod("CFP"):
                    tid = step.get("tool_id")
                    if tid in allowed:
                        tc = await self._call_tool(client, tid, step.get("arguments") or {})
                        tc["reasoning"] = step.get("reasoning")
                        tool_calls.append(tc)
            else:
                imgs_to_process = [img for img in images if isinstance(img, dict) and ((not new_ids_set) or ((str(img.get("image_id") or img.get("path")) in new_ids_set)))]
                # Strict sequential per-image execution to preserve order
                if incremental and not imgs_to_process:
                    pass  # nothing to do
                else:
                    for img in imgs_to_process:
                        try:
                            calls = await run_plan_for_image(client, img)
                            tool_calls.extend(calls)
                        except Exception as e:
                            logger.exception(f"Per-image plan execution failed: {e}")
                            tool_calls.append({"tool_id": "__internal__", "status": "failed", "error": str(e)})

            # Optional placeholder: knowledge tools planned later after aggregation
            # We compute the query based on aggregated outputs below

        # Aggregate outputs (simplified)
        # Aggregate per-image
        # Seed from previous outputs in incremental mode (per_image merge)
        quality = {}
        ages = {}
        lesions = {}
        diseases = {}
        if isinstance(prev_outputs, dict):
            per_prev = prev_outputs.get("per_image") or {}
            if isinstance(per_prev, dict):
                quality.update(per_prev.get("quality") or {})
                if isinstance(per_prev.get("age"), dict):
                    ages.update(per_prev.get("age") or {})
                if isinstance(per_prev.get("lesions"), dict):
                    # deep copy per tool map per image
                    for k, v in (per_prev.get("lesions") or {}).items():
                        if isinstance(v, dict):
                            lesions[k] = dict(v)
                if isinstance(per_prev.get("diseases"), dict):
                    diseases.update(per_prev.get("diseases") or {})
        for c in tool_calls:
            tid = c.get("tool_id")
            out = c.get("output")
            img_id = c.get("image_id") or "_"
            if tid == "classification:cfp_quality":
                quality[img_id] = out
            if tid == "classification:cfp_age":
                # Expect {prediction: number, unit: years}
                if isinstance(out, dict):
                    ages[img_id] = out.get("prediction")
            if isinstance(tid, str) and tid.startswith("segmentation:"):
                lesions.setdefault(img_id, {})[tid] = out
            if tid == "classification:multidis":
                diseases[img_id] = out
    # Build a narrative summary
        def _fmt_prob(p):
            try:
                return f"{float(p)*100:.1f}%"
            except Exception:
                return str(p)
        # Quality summary
        # Summaries across images
        def _quality_str(q):
            if isinstance(q, dict):
                return str(q.get("prediction") or q.get("quality") or q.get("label") or "unknown")
            return "unknown"
        if isinstance(quality, dict) and quality:
            q_txt = ", ".join([f"{k}:{_quality_str(v)}" for k, v in quality.items()])
        else:
            q_txt = "unknown"
        # Lesion summary: collect non-zero counts per tool
        lesion_bits = []
        if isinstance(lesions, dict):
            for img_id, tools in lesions.items():
                if isinstance(tools, dict):
                    for tool_id, out in tools.items():
                        if isinstance(out, dict):
                            counts = out.get("counts")
                            if isinstance(counts, dict):
                                nz = [f"{k}:{v}" for k, v in counts.items() if isinstance(v, (int, float)) and float(v) > 0]
                                if nz:
                                    lesion_bits.append(f"{img_id}:{tool_id.split(':')[-1]}({', '.join(nz)})")
        lesion_txt = ", ".join(lesion_bits) if lesion_bits else "no obvious lesions detected by the current models"
        # Diseases: top 3
        top_d_txt = ""
        if isinstance(diseases, dict) and diseases:
            # Flatten top diseases per image
            top_entries = []
            for img_id, probs in diseases.items():
                if isinstance(probs, dict) and probs:
                    try:
                        tops = sorted(probs.items(), key=lambda kv: float(kv[1] or 0), reverse=True)[:3]
                    except (TypeError, ValueError) as e:
                        logger.debug(f"Failed to sort disease probabilities for {img_id}: {e}")
                        tops = list(probs.items())[:3]
                    if tops:
                        top_entries.append(f"{img_id}: " + ", ".join([f"{k} {_fmt_prob(v)}" for k, v in tops]))
            top_d_txt = "; ".join(top_entries)
        # Prepare a concise context summary for reasoning; then let LLM polish it
        parts: List[str] = [f"Image analysis summary: image quality is {q_txt}."]
        if ages:
            age_bits = ", ".join([f"{k}:{v}" for k, v in ages.items()])
            parts.append(f"Estimated ages: {age_bits}.")
        parts.append(f"Segmentation suggests {lesion_txt}.")
        if top_d_txt:
            parts.append(f"Top disease likelihoods: {top_d_txt}.")
        base_summary = " ".join(parts).strip()
        reasoning = self.gen_reasoning(base_summary)
        # Optional knowledge fetch leveraging base_summary as a query
        knowledge_blocks: List[Dict[str, Any]] = []
        kn_allowed2 = select_tool_ids(self.__class__.__name__, base_tool_ids=["rag:query", "web_search:pubmed", "web_search:tavily"], role=self.role)
        knowledge_query = (top_d_txt or lesion_txt) if (top_d_txt or lesion_txt) else None
        if kn_allowed2 and knowledge_query:
            plan2 = await self.plan_tools(f"If needed, fetch brief knowledge for: {knowledge_query}", kn_allowed2)
            async with self._client_ctx() as client:
                for step in (plan2 or []):
                    tid = step.get("tool_id")
                    if tid not in kn_allowed2:
                        continue
                    if not self._knowledge_allowed():
                        break
                    args = self._apply_knowledge_defaults(tid, step.get("arguments"))
                    tc = await self._call_tool(client, tid, args)
                    self._note_knowledge_called()
                    tc["reasoning"] = step.get("reasoning")
                    tool_calls.append(tc)
                    out = tc.get("output")
                    if isinstance(out, dict):
                        knowledge_blocks.append(out)
        # Merge diseases across images into a single dict (max probability per disease) for backward compatibility
        merged_diseases: Dict[str, Any] = {}
        # include prior merged if available
        if isinstance(prev_outputs, dict) and isinstance(prev_outputs.get("diseases"), dict):
            for k, v in (prev_outputs.get("diseases") or {}).items():
                try:
                    merged_diseases[k] = float(v)
                except Exception:
                    pass
        if isinstance(diseases, dict):
            for _img, probs in diseases.items():
                if isinstance(probs, dict):
                    for k, v in probs.items():
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        merged_diseases[k] = max(merged_diseases.get(k, 0.0), fv)
        outputs = {
            "quality": quality,
            "age": ages if ages else None,
            "lesions": lesions,
            "diseases": merged_diseases or diseases,
            "per_image": {"quality": quality, "age": ages, "lesions": lesions, "diseases": diseases},
            "narrative": reasoning,
            "knowledge": {"query": knowledge_query, "items": knowledge_blocks} if knowledge_blocks else None,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": ("Incremental update: merged with prior outputs. " + reasoning) if incremental else reasoning
        })
        return {"agent": self.name, "role": self.role, "outputs": outputs, "tool_calls": tool_calls, "reasoning": ("Incremental update: merged with prior outputs. " + reasoning) if incremental else reasoning}

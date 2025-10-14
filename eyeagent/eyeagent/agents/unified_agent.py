from __future__ import annotations
from typing import Any, Dict, List, Optional
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from fastmcp import Client
from loguru import logger

from ..tools.tool_registry import (
    TOOL_REGISTRY,
    get_tool,
    role_tool_ids,
    specialist_tools,
)
from ..tools.tool_registry import resolve_specialist_tools  # type: ignore
from ..config.tools_filter import filter_tool_ids


@register_agent
class UnifiedAgent(DiagnosticBaseAgent):
    role = "unified"
    name = "UnifiedAgent"
    allowed_tool_ids: List[str] = []  # determined at runtime via config filters
    system_prompt = (
        "ROLE: Unified end-to-end diagnostic agent.\n"
        "GOAL: Perform orchestration, image analysis, specialist grading, and follow-up, then synthesize a concise final report.\n"
        "TOOLS: Determined by config filters; use modality-appropriate tools and disease-specific grading when needed.\n"
        "OUTPUTS: diagnoses, lesions, management, reasoning, narrative, conclusion.\n"
        "CONSTRAINTS: Keep reasoning concise and clinically relevant."
    )

    # Capabilities declaration
    capabilities = {
        "required_context": ["images", "patient"],
        "expected_outputs": [
            "diagnoses",
            "lesions",
            "management",
            "reasoning",
            "narrative",
            "conclusion",
        ],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],  # dynamic via config
    }

    def _all_tool_ids(self) -> List[str]:
        try:
            return list(TOOL_REGISTRY.keys())
        except Exception:
            return []

    # use base helper: call_tool_per_image

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images", [])
        # UnifiedAgent uses ALL tools; do not apply config-based filters here
        allowed = self._all_tool_ids()

        tool_calls: List[Dict[str, Any]] = []
        modality_label: Optional[str] = None
        screening_probs: Dict[str, float] = {}

        async with self._client_ctx() as client:
            # 1) Orchestration-lite: modality & laterality
            if any(t for t in ("classification:modality", "classification:laterality") if t in allowed):
                if "classification:modality" in allowed:
                    tool_calls += await self.call_tool_per_image(client, "classification:modality", images)
                    try:
                        out = (tool_calls[-1] or {}).get("output")
                        if isinstance(out, dict):
                            modality_label = out.get("label") or out.get("prediction")
                    except Exception:
                        modality_label = None
                if "classification:laterality" in allowed:
                    tool_calls += await self.call_tool_per_image(client, "classification:laterality", images)

            # Optional CFP screening
            if (modality_label or "").upper() in ("", "CFP") and "classification:multidis" in allowed:
                md = await self.call_tool_per_image(client, "classification:multidis", images[:1])
                tool_calls += md
                try:
                    out = (md[0] or {}).get("output")
                    if isinstance(out, dict):
                        probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                        if isinstance(probs, dict):
                            # Normalize float
                            screening_probs = {k: float(v) for k, v in probs.items() if v is not None}
                except Exception:
                    screening_probs = {}

            # 2) Image analysis based on modality
            ia_calls: List[Dict[str, Any]] = []
            if (modality_label or "").upper() in ("", "CFP"):
                for tid in [t for t in allowed if t == "classification:cfp_quality" or t.startswith("segmentation:cfp_")]:
                    ia_calls += await self.call_tool_per_image(client, tid, images)
            elif (modality_label or "").upper() == "OCT":
                for tid in [t for t in allowed if t.startswith("segmentation:oct_")]:
                    ia_calls += await self.call_tool_per_image(client, tid, images)
            elif (modality_label or "").upper() == "FFA":
                for tid in [t for t in allowed if t.startswith("segmentation:ffa_")]:
                    ia_calls += await self.call_tool_per_image(client, tid, images)
            tool_calls += ia_calls

            # Aggregate IA outputs
            quality: Dict[str, Any] = {}
            lesions: Dict[str, Dict[str, Any]] = {}
            diseases_img: Dict[str, Dict[str, float]] = {}
            for c in tool_calls:
                tid = c.get("tool_id")
                out = c.get("output")
                img_id = c.get("image_id") or "_"
                if tid == "classification:cfp_quality":
                    quality[img_id] = out
                if isinstance(tid, str) and tid.startswith("segmentation:"):
                    lesions.setdefault(img_id, {})[tid] = out
                if tid == "classification:multidis" and isinstance(out, dict):
                    probs = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                    if isinstance(probs, dict):
                        # ensure floats
                        diseases_img[img_id] = {k: float(v) for k, v in probs.items() if v is not None}

            # 3) Specialist grading: select tools either from screening or allow all specialist tools
            candidates: List[str] = []
            try:
                # Top candidates from screening
                all_probs = list(diseases_img.values())[0] if diseases_img else screening_probs
                if isinstance(all_probs, dict):
                    candidates = [k for k, v in sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            except Exception:
                candidates = []

            sp_tool_ids: List[str] = []
            try:
                metas = resolve_specialist_tools(candidates) if candidates else []
                if metas:
                    sp_tool_ids = [m.get("tool_id") for m in metas if isinstance(m, dict) and m.get("tool_id")]
            except Exception:
                sp_tool_ids = []
            if not sp_tool_ids:
                sp_tool_ids = [tid for tid in allowed if tid.startswith("disease_specific_cls:")]

            sp_calls: List[Dict[str, Any]] = []
            for tid in sp_tool_ids:
                sp_calls += await self.call_tool_per_image(client, tid, images)
            tool_calls += sp_calls

            # 4) Follow-up: optional age
            fu_output: Optional[Dict[str, Any]] = None
            if "classification:cfp_age" in allowed:
                age_calls = await self.call_tool_per_image(client, "classification:cfp_age", images[:1])
                tool_calls += age_calls
                try:
                    fu_output = age_calls[-1].get("output") if age_calls else None
                except Exception:
                    fu_output = None

        # Build specialist results
        results_flat: List[Dict[str, Any]] = []
        results_by_image: Dict[str, List[Dict[str, Any]]] = {}
        for tc in tool_calls:
            tid = tc.get("tool_id")
            if isinstance(tid, str) and tid.startswith("disease_specific_cls:") and tc.get("status") == "success":
                meta = get_tool(tid) or {}
                out = tc.get("output") or {}
                entry = {
                    "disease": meta.get("disease") or tid.split(":")[-1],
                    "grade": out.get("grade") if isinstance(out, dict) else None,
                    "confidence": out.get("confidence") if isinstance(out, dict) else None,
                    "tool_id": tid,
                }
                img_id = tc.get("image_id")
                if img_id:
                    entry["image_id"] = img_id
                    results_by_image.setdefault(img_id, []).append(entry)
                results_flat.append(entry)

        # Management suggestion (simple rules, similar to FollowUpAgent)
        suggestion = "Annual follow-up"
        follow_up_months = 12
        for dg in results_flat:
            dis = (dg.get("disease") or "").upper()
            grade = dg.get("grade") or ""
            if dis == "DR" and grade.upper().startswith("R3"):
                suggestion = "Follow-up in 3 months; consider referral if warranted"
                follow_up_months = 3
            if dis == "AMD" and grade:
                suggestion = "Follow-up in 6 months with OCT"
                follow_up_months = min(follow_up_months, 6)
        management = {"suggestion": suggestion, "follow_up_months": follow_up_months, "age_info": fu_output}

        # Aggregate IA summaries
        per_image = {
            "quality": quality,
            "lesions": lesions,
            "diseases": diseases_img,
            "specialist": results_by_image,
        }
        # Merge diseases across images (max probability per disease)
        merged_diseases: Dict[str, float] = {}
        for probs in (diseases_img.values() or []):
            for k, v in probs.items():
                merged_diseases[k] = max(merged_diseases.get(k, 0.0), float(v))

        # Compose final outputs akin to ReportAgent
        diag_txt = ", ".join([
            f"{d.get('disease')} {d.get('grade')}"
            for d in results_flat if d.get('disease') and d.get('grade')
        ])
        mgmt_txt = f"{suggestion} (in {follow_up_months} months)" if suggestion and follow_up_months else suggestion
        base_reasoning = (
            (f"Final impression: {diag_txt}. " if diag_txt else "") +
            (f"Suggested management: {mgmt_txt}." if mgmt_txt else "")
        ) or "Consolidated findings into a final report."
        reasoning = self.gen_reasoning(base_reasoning)
        conclusion = None
        if diag_txt and mgmt_txt:
            conclusion = f"{diag_txt}; {mgmt_txt}"
        elif diag_txt:
            conclusion = diag_txt
        elif mgmt_txt:
            conclusion = mgmt_txt

        outputs = {
            "diagnoses": results_flat,
            "lesions": lesions,
            "management": management,
            "reasoning": reasoning,
            "narrative": reasoning,
            "conclusion": conclusion,
            "per_image": per_image,
            "diseases": merged_diseases or diseases_img,
            "modality": modality_label,
        }

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
        })

        return {
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
        }

from typing import Any, Dict, List
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from loguru import logger
from eyeagent.core.settings import get_specialist_selection_settings, get_configured_agents, get_routing_strategy
from eyeagent.core.diagnosis_utils import get_candidate_diseases_from_probs
from eyeagent.llm.json_client import JsonLLM
from eyeagent.llm.models import RoutingDecision

@register_agent
class OrchestratorAgent(DiagnosticBaseAgent):
    role = "orchestrator"
    name = "OrchestratorAgent"
    # Orchestrator no longer calls tools directly; it only routes
    allowed_tool_ids: List[str] = []
    system_prompt = (
        "ROLE: Orchestrator/router. You never call tools.\n"
        "GOAL: Decide only the next agent to run based on current state and preliminary results.\n"
        "RECOMMENDED ORDER: symptom_triage (if symptom_text is available or needs collection) → preliminary → image_analysis → specialist → decision → follow_up → report.\n"
        "INPUTS: patient, images, and prior agent outputs (symptom_triage/preliminary/image_analysis/specialist/decision/follow_up).\n"
        "OUTPUTS: next_agent (string). Optionally provide brief routing_reasons.\n"
        "HEURISTICS: If no images → you may go to report. If images exist, preliminary and image_analysis should precede specialist/decision when enabled.\n"
        "CONSTRAINTS: Do not invoke tools; only route among available/enabled agents. Prefer running 'symptom_triage' early if enabled and not completed."
    )

    capabilities = {
        "required_context": ["patient", "images"],
        "expected_outputs": ["next_agent"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        import os as _os
        images = context.get("images") or []
        incremental = bool(context.get("incremental"))
        # Normalize images to list if an iterable-like is passed
        if not isinstance(images, list):
            try:
                images = list(images)
            except Exception:
                images = []
        try:
            sample = []
            for i in images[:3]:
                if isinstance(i, dict):
                    sample.append((i.get("image_id"), i.get("path")))
                else:
                    sample.append(str(i)[:120])
            logger.debug(f"[orchestrator] images_type={type(images).__name__} images_count={len(images)} sample={sample}")
        except Exception:
            pass
        # Gather state summary for LLM
        prelim = context.get("preliminary") or (context.get("orchestrator_outputs") or {}).get("preliminary")
        ia = context.get("image_analysis")
        specialist = context.get("specialist")
        knowledge = context.get("knowledge")
        follow_up = context.get("follow_up")
        triage = context.get("symptom_triage")
        configured = get_configured_agents()
        known = ["symptom_triage", "preliminary", "image_analysis", "specialist", "decision", "knowledge", "follow_up", "report"]
        if configured:
            enabled_roles = {k for k, v in configured.items() if isinstance(v, dict) and (v.get("enabled") is not False)}
            available = [r for r in known if r in enabled_roles or r == "report"]
        else:
            available = list(known)
        # If profile_steps present, restrict available roles to those steps (plus report)
        prof_steps = context.get("profile_steps")
        if isinstance(prof_steps, list) and prof_steps:
            try:
                prof_set = {str(x) for x in prof_steps}
                available = [r for r in available if (r in prof_set) or (r == "report")]
            except Exception:
                pass

        def _summarize(d: Any, limit_keys: int = 12) -> str:
            try:
                import json as _json
                if isinstance(d, dict):
                    keys = list(d.keys())[:limit_keys]
                    dd = {k: d.get(k) for k in keys}
                    return _json.dumps(dd, ensure_ascii=False)
                return _json.dumps(d, ensure_ascii=False)[:800]
            except Exception:
                return str(d)[:800]

        completed = []
        if prelim: completed.append("preliminary")
        if ia: completed.append("image_analysis")
        if specialist: completed.append("specialist")
        if context.get("decision"): completed.append("decision")
        if knowledge: completed.append("knowledge")
        if follow_up: completed.append("follow_up")
        if triage: completed.append("symptom_triage")

        # In incremental mode, force re-running image-related steps for new images
        if incremental:
            try:
                completed = [c for c in completed if c not in ("preliminary", "image_analysis", "specialist")]
            except Exception:
                pass

        sys = (
            "You are the OrchestratorAgent (orchestrator). Route agents; do not call tools. "
            "Return ONLY JSON following the exact schema. Rules: "
            "- Allowed roles: ['symptom_triage','preliminary','image_analysis','specialist','decision','follow_up','report']\n"
            "- Prefer 'symptom_triage' early when enabled and not completed\n"
            "- If images exist, do NOT jump to 'report' before 'preliminary' and 'image_analysis' unless they are disabled\n"
            "- next_agent must be one of allowed roles\n"
        )
        user = (
            f"Available roles (enabled): {available}\n"
            f"Images count: {len(images) if isinstance(images, list) else 0}\n"
            f"Completed steps: {completed}\n\n"
            f"Context summary (preliminary): {_summarize(prelim)}\n"
            f"Context summary (image_analysis): {_summarize(ia)}\n"
            f"Context summary (specialist): {_summarize(specialist)}\n"
            f"Context summary (decision): {_summarize(context.get('decision'))}\n"
            f"Context summary (knowledge): {_summarize(knowledge)}\n"
            f"Context summary (follow_up): {_summarize(follow_up)}\n"
            f"Context summary (symptom_triage): {_summarize(triage)}\n"
        )
        # Determine routing strategy: 'llm' (default) or 'config' (deterministic from profile)
        routing_strategy = get_routing_strategy(default="llm")

        llm = JsonLLM(agent_name=self.__class__.__name__)
        reasons: List[str] = []
        next_agent: str | None = None
        # Prefer LLM-guided next_agent selection but keep the schema minimal
        if routing_strategy != "config":
            try:
                schema = '{"next_agent": "...", "routing_reasons": ["..."]}'
                data = llm.invoke_json(system_prompt=sys, user_prompt=user, schema_hint=schema)
                if isinstance(data, dict):
                    na = data.get("next_agent")
                    if isinstance(na, str) and na in known and ((na in available) or (na == "report")):
                        next_agent = na
                    rr = data.get("routing_reasons")
                    if isinstance(rr, list):
                        reasons.extend([str(x) for x in rr][:4])
            except Exception as e:
                reasons.append(f"LLM routing failed: {str(e)[:160]}")

        # Deterministic heuristics as fallback or guardrails
        # Prefer symptom_triage first if enabled and not completed
        if (not next_agent) and ("symptom_triage" in available) and ("symptom_triage" not in completed):
            next_agent = "symptom_triage"
        # When images exist, ensure preliminary and image_analysis precede specialist/decision
        if images:
            for must in ("preliminary", "image_analysis"):
                if (not next_agent) and (must in available) and (must not in completed):
                    next_agent = must
        # Otherwise pick the first enabled and not completed, or default to report
        if not next_agent:
            order = ["specialist", "decision", "follow_up", "report"]
            for step in order:
                if (step in available) and (step not in completed):
                    next_agent = step
                    break
        if not next_agent:
            next_agent = "report"

        # Final safety: never route to a step that's already completed; advance to first incomplete by preferred order
        if next_agent in completed:
            preferred = ["symptom_triage", "preliminary", "image_analysis", "specialist", "decision", "follow_up", "report"]
            picked = None
            for step in preferred:
                if (step in available) and (step not in completed):
                    picked = step
                    break
            next_agent = picked or "report"

        outputs = {
            "next_agent": next_agent,
            "available_agents": available,
            "routing_reasons": reasons,
            "images_count": (len(images) if isinstance(images, list) else 0),
        }
        # Embed a short intake prompt when symptom text is missing and triage is available
        sym_text = (context.get("symptom_text") or "").strip()
        if (not sym_text) and ("symptom_triage" in available):
            outputs["symptom_prompt"] = (
                "Please briefly describe your main eye symptoms in English (1-3 sentences). "
                "Include: onset/duration, laterality (one eye or both), and key descriptors (e.g., blurred vision, pain, redness, flashes/floaters, central black spot)."
            )
            reasons.append("Collect symptom_text for triage")
        reasoning = "; ".join(reasons) if reasons else f"Next agent: {next_agent}"

        self.trace_logger.append_event(self.case_id, {
            "type": "agent_step",
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": reasoning,
        })

        return {
            "agent": self.name,
            "role": self.role,
            "outputs": outputs,
            "tool_calls": [],
            "reasoning": reasoning,
        }

from typing import Any, Dict, List
from .base_agent import BaseAgent as DiagnosticBaseAgent
from .registry import register_agent
from loguru import logger
from ..config.settings import get_specialist_selection_settings, get_configured_agents
from ..core.diagnosis_utils import get_candidate_diseases_from_probs
from ..llm.json_client import JsonLLM
from ..llm.models import RoutingDecision

@register_agent
class OrchestratorAgent(DiagnosticBaseAgent):
    role = "orchestrator"
    name = "OrchestratorAgent"
    # Orchestrator no longer calls tools directly; it only routes
    allowed_tool_ids: List[str] = []
    system_prompt = (
        "ROLE: Orchestrator/router. You never call tools.\n"
        "GOAL: Decide the next agent to run based on current state and preliminary results, and provide a planned pipeline.\n"
        "REQUIRED ORDER (when images exist): preliminary → image_analysis → specialist → decision → follow_up → report (knowledge optional and usually integrated by follow_up).\n"
        "INPUTS: patient, images, and prior agent outputs (preliminary/image_analysis/specialist/decision/knowledge/follow_up).\n"
        "OUTPUTS: planned_pipeline (list of agent roles in intended order), next_agent (string).\n"
        "HEURISTICS: If no images → go directly to report. If OCT modality → IA uses OCT tools. If screening confidence is very low, you may skip specialist.\n"
        "CONSTRAINTS: Do not invoke tools; only route among available/enabled agents. When images exist, never jump to report before running preliminary and image_analysis unless they are explicitly disabled. Re-evaluate after each step and finish with report."
    )

    capabilities = {
        "required_context": ["patient", "images"],
        "expected_outputs": ["planned_pipeline", "next_agent"],
        "retry_policy": {"max_attempts": 1, "on_fail": "skip"},
        "modalities": ["CFP", "OCT", "FFA"],
        "tools": [],
    }

    async def a_run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        images = context.get("images") or []
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
        configured = get_configured_agents()
        known = ["preliminary", "image_analysis", "specialist", "decision", "knowledge", "follow_up", "report"]
        if configured:
            enabled_roles = {k for k, v in configured.items() if isinstance(v, dict) and (v.get("enabled") is not False)}
            available = [r for r in known if r in enabled_roles or r == "report"]
        else:
            available = list(known)

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

        sys = (
            "You are the OrchestratorAgent (orchestrator). Route agents; do not call tools. "
            "Return ONLY JSON following the exact schema. Rules: "
            "- Allowed roles: ['preliminary','image_analysis','specialist','decision','knowledge','follow_up','report']\n"
            "- Always include 'report' at the END of planned_pipeline exactly once\n"
            "- If images exist, do NOT jump to 'report' before 'preliminary' and 'image_analysis' unless they are disabled\n"
            "- next_agent must be one of allowed roles and normally the first incomplete in planned_pipeline\n"
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
        )
        llm = JsonLLM(agent_name=self.__class__.__name__)
        routing: RoutingDecision | None = None
        reasons: List[str] = []
        try:
            routing = llm.invoke_structured(sys, user, RoutingDecision)  # type: ignore[assignment]
        except Exception:
            # Fallback to plain JSON with explicit schema hint
            try:
                schema = (
                    '{"planned_pipeline": ["preliminary","image_analysis","specialist","decision","follow_up","report"],' \
                    '"next_agent": "...", "routing_reasons": ["..."]}'
                )
                data = llm.invoke_json(system_prompt=sys, user_prompt=user, schema_hint=schema)
                routing = RoutingDecision(**data)  # type: ignore[arg-type]
            except Exception as e:
                reasons.append(f"LLM routing failed: {str(e)[:160]}")

        # Validate and normalize LLM output; fallback to minimal valid plan if needed
        planned_pipeline: List[str] = []
        next_agent: str | None = None
        if routing is not None:
            planned_pipeline = [r for r in routing.planned_pipeline if r in known]
            # Ensure report present exactly once and last
            planned_pipeline = [x for x in planned_pipeline if x != "report"] + ["report"]
            # Remove standalone knowledge step (knowledge will be consulted by follow_up when needed)
            planned_pipeline = [r for r in planned_pipeline if r != "knowledge"]
            # Constrain to available (enabled) roles, but always keep final 'report'
            planned_pipeline = [r for r in planned_pipeline if (r in available) or (r == "report")]
            next_agent = routing.next_agent if routing.next_agent in planned_pipeline else None
            if routing.routing_reasons:
                reasons.extend(list(routing.routing_reasons)[:4])

        # Hard guard: if images exist and prelim/IA are enabled+in plan, they must come first if incomplete
        if images and planned_pipeline:
            for must in ("preliminary", "image_analysis"):
                if must in planned_pipeline and must not in completed:
                    next_agent = must
                    break

        # Minimal fallback if still invalid
        if not planned_pipeline:
            planned_pipeline = [r for r in available if r in known]
            if "report" not in planned_pipeline:
                planned_pipeline.append("report")
        if not next_agent:
            for step in planned_pipeline:
                if step not in completed:
                    next_agent = step
                    break
        if not next_agent:
            next_agent = "report"

        # Final safety: never route to a step that's already completed; advance to first incomplete
        if next_agent in completed:
            picked = None
            for step in planned_pipeline:
                if step not in completed:
                    picked = step
                    break
            next_agent = picked or "report"

        outputs = {
            "planned_pipeline": planned_pipeline,
            "next_agent": next_agent,
            "available_agents": available,
            "routing_reasons": reasons,
            "images_count": (len(images) if isinstance(images, list) else 0),
        }
        reasoning = "; ".join(reasons) if reasons else f"Next agent: {next_agent} (LLM-guided)"

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

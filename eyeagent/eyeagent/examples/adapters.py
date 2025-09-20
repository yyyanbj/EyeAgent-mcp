from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
from uuid import uuid4

from ..tracing.trace_logger import TraceLogger
from ..agents.orchestrator_agent import OrchestratorAgent
from ..agents.image_analysis_agent import ImageAnalysisAgent
from ..agents.specialist_agent import SpecialistAgent
from ..agents.followup_agent import FollowUpAgent


def _ensure_runtime(state: Dict[str, Any]):
    ctx = state.setdefault("context", {})
    rt = ctx.setdefault("_runtime", {})
    if "trace" not in rt:
        rt["trace"] = TraceLogger()
    if "case_id" not in rt:
        rt["case_id"] = f"case-{uuid4().hex[:8]}"
    if "mcp_url" not in rt:
        rt["mcp_url"] = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
    return rt


def _merge_messages(state: Dict[str, Any], line: str):
    msgs = state.setdefault("messages", [])
    msgs.append(line)


def _top_k_from_probs(probs: Dict[str, Any], k: int = 3, min_prob: float = 0.0) -> List[str]:
    items: List[Tuple[str, float]] = []
    for d, v in (probs or {}).items():
        try:
            items.append((d, float(v)))
        except Exception:
            continue
    items.sort(key=lambda x: x[1], reverse=True)
    return [d for d, p in items if p >= min_prob][:k]


async def orchestrator_plan_adapter(state: Dict[str, Any]) -> Dict[str, Any] | None:
    rt = _ensure_runtime(state)
    ctx = state.get("context", {})
    agent = OrchestratorAgent(rt["mcp_url"], rt["trace"], rt["case_id"])
    res = await agent.a_run(ctx)
    outs = res.get("outputs", {}) if isinstance(res, dict) else {}
    ctx["orchestrator_outputs"] = outs
    ctx["planned_pipeline"] = outs.get("planned_pipeline")
    candidates: List[str] = []
    for tc in outs.get("screening_results") or []:
        if isinstance(tc, dict) and isinstance(tc.get("output"), dict):
            candidates.extend(_top_k_from_probs(tc["output"], k=3, min_prob=0.15))
    if not candidates:
        ia = (ctx.get("image_analysis_outputs") or {}).get("diseases")
        if isinstance(ia, dict):
            candidates = _top_k_from_probs(ia, k=3, min_prob=0.15)
    if candidates:
        ctx["candidate_diseases"] = sorted(set(candidates))
    mod = None
    try:
        mlist = outs.get("modality_results") or []
        mout = (mlist[0] or {}).get("output") if mlist else None
        if isinstance(mout, dict):
            mod = mout.get("label") or mout.get("prediction")
    except Exception:
        mod = None
    _merge_messages(state, f"Orchestrator: planned {ctx.get('planned_pipeline')} modality={mod} candidates={ctx.get('candidate_diseases')}")
    return {"context": ctx}


async def image_analysis_adapter(state: Dict[str, Any]) -> Dict[str, Any] | None:
    rt = _ensure_runtime(state)
    ctx = state.get("context", {})
    ctx_for_agent = {
        "images": ctx.get("images", []),
        "orchestrator_outputs": ctx.get("orchestrator_outputs", {}),
    }
    agent = ImageAnalysisAgent(rt["mcp_url"], rt["trace"], rt["case_id"])
    res = await agent.a_run(ctx_for_agent)
    outs = res.get("outputs", {}) if isinstance(res, dict) else {}
    ctx["image_analysis_outputs"] = outs
    if not ctx.get("candidate_diseases"):
        dis = outs.get("diseases")
        if isinstance(dis, dict):
            ctx["candidate_diseases"] = _top_k_from_probs(dis, k=3, min_prob=0.15)
    _merge_messages(state, "ImageAnalysis: completed IA with lesions/diseases summary")
    return {"context": ctx}


async def specialist_adapter(state: Dict[str, Any]) -> Dict[str, Any] | None:
    rt = _ensure_runtime(state)
    ctx = state.get("context", {})
    ctx_for_agent = {
        "images": ctx.get("images", []),
        "candidate_diseases": ctx.get("candidate_diseases", []),
    }
    agent = SpecialistAgent(rt["mcp_url"], rt["trace"], rt["case_id"])
    res = await agent.a_run(ctx_for_agent)
    outs = res.get("outputs", {}) if isinstance(res, dict) else {}
    ctx["specialist_outputs"] = outs
    dg = outs.get("disease_grades") or []
    if isinstance(dg, list):
        ctx["disease_grades"] = dg
    _merge_messages(state, "Specialist: graded candidate diseases")
    return {"context": ctx}


async def followup_adapter(state: Dict[str, Any]) -> Dict[str, Any] | None:
    rt = _ensure_runtime(state)
    ctx = state.get("context", {})
    ctx_for_agent = {
        "images": ctx.get("images", []),
        "disease_grades": ctx.get("disease_grades", []),
    }
    agent = FollowUpAgent(rt["mcp_url"], rt["trace"], rt["case_id"])
    res = await agent.a_run(ctx_for_agent)
    outs = res.get("outputs", {}) if isinstance(res, dict) else {}
    ctx["followup_outputs"] = outs
    ctx["management"] = outs.get("management")
    _merge_messages(state, "FollowUp: produced management suggestion")
    return {"context": ctx}


async def orchestrator_summarize_adapter(state: Dict[str, Any]) -> Dict[str, Any] | None:
    rt = _ensure_runtime(state)
    ctx = state.get("context", {})
    ia = ctx.get("image_analysis_outputs", {})
    sp = ctx.get("specialist_outputs", {})
    fu = ctx.get("followup_outputs", {})
    mod = None
    try:
        mlist = (ctx.get("orchestrator_outputs") or {}).get("modality_results") or []
        mout = (mlist[0] or {}).get("output") if mlist else None
        if isinstance(mout, dict):
            mod = mout.get("label") or mout.get("prediction")
    except Exception:
        mod = None
    parts: List[str] = []
    if mod:
        parts.append(f"modality={mod}")
    if ia:
        parts.append("IA done")
    if sp and isinstance(sp.get("disease_grades"), list):
        parts.append(f"grades={len(sp['disease_grades'])}")
    if fu and isinstance(fu.get("management"), dict):
        mg = fu["management"]
        parts.append(f"mg={mg.get('suggestion')}, {mg.get('follow_up_months')}m")
    base_summary = "Orchestrator final summary based on results: " + ", ".join(parts)

    orchestrator = OrchestratorAgent(rt["mcp_url"], rt["trace"], rt["case_id"])
    final_text = orchestrator.gen_reasoning(base_summary)
    ctx["final_summary"] = final_text
    _merge_messages(state, f"Orchestrator (final): {final_text}")
    return {"context": ctx}

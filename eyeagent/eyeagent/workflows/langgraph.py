"""LangGraph-only diagnostic workflow implementation.

This module defines the LangGraph-based orchestration with an Orchestrator that routes
between worker agents (preliminary, image_analysis, specialist, knowledge, follow_up)
until the final report is produced.
"""
from typing import Dict, Any, List, TypedDict
import os
import asyncio
from dotenv import load_dotenv
from ..core.logging import setup_logging

# Load environment variables from .env, if present
load_dotenv()
setup_logging()

from langgraph.graph import StateGraph, START, END  # type: ignore

from ..tracing.trace_logger import TraceLogger
from loguru import logger
from ..agents.registry import register_builtins, get_agent_class
from ..agents.capabilities import get_capabilities
from ..metrics.metrics import step_timer
from ..config.settings import get_specialist_selection_settings
from ..core.diagnosis_utils import get_candidate_diseases_from_probs

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
SCHEMA_VERSION = "1.0.0"

# Help diagnose which code path is running (local source vs installed package)
try:
    logger.info("[workflow] langgraph module path: {}", __file__)
except Exception:
    pass


class WorkflowState(TypedDict, total=False):
    # Core context
    patient: Dict[str, Any]
    images: List[Dict[str, Any]]
    case_id: str
    trace: Any
    messages: List[Dict[str, Any]]
    # Orchestration bookkeeping
    workflow: List[Dict[str, Any]]
    pipeline: List[str]
    next_agent: str
    completed_steps: List[str]
    last_agent: str
    _kickoff_emitted: bool
    orchestrator_outputs: Dict[str, Any]
    orchestrator_command: str
    # Agent outputs
    preliminary: Dict[str, Any]
    image_analysis: Dict[str, Any]
    specialist: Dict[str, Any]
    knowledge: Dict[str, Any]
    decision: Dict[str, Any]
    follow_up: Dict[str, Any]
    final_fragment: Dict[str, Any]
    # Derived
    candidate_diseases: List[str]


def _summarize_json(obj: Any, max_len: int = 800) -> str:
    try:
        import json as _json
        s = _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _append_messages_from_result(state: WorkflowState, result: Dict[str, Any]) -> None:
    msgs = state.setdefault("messages", [])
    agent = result.get("agent") or "Agent"
    role = result.get("role") or ""
    title = f"{agent} ({role})"
    reasoning = result.get("reasoning")
    case_id = state.get("case_id")
    if reasoning:
        msg = {"role": "assistant", "content": f"{title} reasoning: {reasoning}"}
        msgs.append(msg)
        try:
            (state.get("trace") or TraceLogger()).append_conversation_message(case_id, msg)  # type: ignore[arg-type]
        except Exception:
            pass
        logger.info(f"[agent case={case_id}] {title} reasoning -> {str(reasoning)[:200]}{'...' if len(str(reasoning))>200 else ''}")
    for tc in result.get("tool_calls", []) or []:
        tool_id = tc.get("tool_id") or "tool"
        status = tc.get("status") or ""
        args = tc.get("arguments")
        out = tc.get("output")
        msg = {
            "role": "assistant",
            "content": (
                f"{title} called {tool_id} -> {status}\n"
                f"args: {_summarize_json(args)}\n"
                f"output: {_summarize_json(out)}"
            )
        }
        msgs.append(msg)
        try:
            (state.get("trace") or TraceLogger()).append_conversation_message(case_id, msg)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            args_summary = _summarize_json(args, max_len=200)
        except Exception:
            args_summary = "?"
        try:
            out_summary = _summarize_json(out, max_len=400)
        except Exception:
            out_summary = "?"
        logger.info(
            f"[agent case={case_id}] {title} tool_call id={tool_id} status={status} args={args_summary} output={out_summary}"
        )
    outputs = result.get("outputs")
    if outputs:
        msg = {"role": "assistant", "content": f"{title} outputs: {_summarize_json(outputs)}"}
        msgs.append(msg)
        try:
            (state.get("trace") or TraceLogger()).append_conversation_message(case_id, msg)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            keys = ",".join(list(outputs.keys())[:6])
        except Exception:
            keys = "?"
        logger.info(f"[agent case={case_id}] {title} outputs keys=[{keys}]")


def get_pipeline_capabilities(pipeline: list) -> list:
    out = []
    for key in pipeline:
        cls = get_agent_class(key)
        if cls:
            try:
                caps = get_capabilities(cls)
            except Exception:
                caps = {}
            out.append({"key": key, "capabilities": caps})
    return out


async def node_orchestrator(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_orchestrator")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    # If images got dropped from in-memory state, recover from trace storage
    try:
        if not (state.get("images") and isinstance(state.get("images"), list) and len(state.get("images")) > 0):
            try:
                doc = trace.load_trace(case_id)
                imgs = doc.get("images") or []
                if imgs:
                    state["images"] = imgs
                    logger.debug("[workflow] recovered images from trace: count={}", len(imgs))
            except Exception:
                pass
    except Exception:
        pass
    # Kickoff event once per case for UI streaming context
    try:
        if not state.get("_kickoff_emitted"):
            trace.append_event(case_id, {
                "type": "agent_step",
                "agent": "Workflow",
                "role": "system",
                "outputs": {"status": "started", "images_count": len(state.get("images") or [])},
                "tool_calls": [],
                "reasoning": "Workflow started; invoking orchestrator to route next agent."
            })
            state["_kickoff_emitted"] = True
    except Exception:
        pass
    cls = get_agent_class("orchestrator")
    if not cls:
        logger.warning("[workflow] OrchestratorAgent not configured; default to report")
        state["pipeline"] = ["report"]
        state["next_agent"] = "report"
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    # Ensure images is a list for downstream agents
    try:
        imgs = state.get("images")
        if imgs is None:
            fixed = []
        elif isinstance(imgs, list):
            fixed = imgs
        else:
            try:
                fixed = list(imgs)
            except Exception:
                fixed = []
        context["images"] = fixed
        try:
            from loguru import logger as _logger
            _logger.debug(
                "[workflow] orchestrator pre-call images_count={} sample={}",
                len(fixed),
                [(getattr(i, 'get', lambda k: None)('image_id') if isinstance(i, dict) else None,
                  getattr(i, 'get', lambda k: None)('path') if isinstance(i, dict) else None) for i in fixed[:3]]
            )
        except Exception:
            pass
    except Exception:
        pass
    context["messages"] = state.get("messages", [])
    if "patient" in state and "instruction" in (state.get("patient") or {}):
        context["instruction"] = state["patient"]["instruction"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    outputs = res.get("outputs") or {}
    state["pipeline"] = outputs.get("planned_pipeline", [])
    # Compute completed steps from explicit flags and the helper list
    completed_list = list(dict.fromkeys((state.get("completed_steps") or [])))
    completed = set(completed_list)
    if state.get("preliminary"): completed.add("preliminary")
    if state.get("image_analysis"): completed.add("image_analysis")
    if state.get("specialist"): completed.add("specialist")
    if state.get("knowledge"): completed.add("knowledge")
    if state.get("follow_up"): completed.add("follow_up")
    # Avoid looping: if orchestrator suggests a step that's already completed, pick the next incomplete one
    suggested_next = outputs.get("next_agent")
    next_agent = suggested_next
    if isinstance(suggested_next, str) and suggested_next in completed:
        # walk planned pipeline to find the first incomplete step
        for step in (state.get("pipeline") or []):
            if step not in completed:
                next_agent = step
                break
        else:
            next_agent = "report"
    state["next_agent"] = next_agent
    state["orchestrator_command"] = context.get("instruction")
    state["orchestrator_outputs"] = outputs
    logger.debug(f"[workflow] orchestrator pipeline={state.get('pipeline')} next={state.get('next_agent')}")
    return state


async def node_preliminary(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_preliminary")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    # Safety: ensure images are available for this node too
    try:
        if not (state.get("images") and isinstance(state.get("images"), list) and len(state.get("images")) > 0):
            try:
                doc = trace.load_trace(case_id)
                imgs = doc.get("images") or []
                if imgs:
                    state["images"] = imgs
                    logger.debug("[workflow] recovered images (preliminary) from trace: count={}", len(imgs))
            except Exception:
                pass
    except Exception:
        pass
    cls = get_agent_class("preliminary")
    if not cls:
        logger.warning("[workflow] PreliminaryAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["preliminary"] = res.get("outputs")
    # Mark completion to help orchestrator skip already-done steps
    cs = state.setdefault("completed_steps", [])
    if "preliminary" not in cs:
        cs.append("preliminary")
    state["last_agent"] = "preliminary"
    state.setdefault("orchestrator_outputs", {})
    try:
        state["orchestrator_outputs"]["preliminary"] = res.get("outputs")
        # Back-compat: also mirror screening_results for older consumers
        sr = (res.get("outputs") or {}).get("screening_results")
        if sr is not None:
            state["orchestrator_outputs"]["screening_results"] = sr
    except Exception:
        pass
    logger.debug("[workflow] exit node_preliminary")
    return state


async def node_image_analysis(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_image_analysis")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("image_analysis")
    if not cls:
        logger.warning("[workflow] ImageAnalysisAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["image_analysis"] = res.get("outputs")
    cs = state.setdefault("completed_steps", [])
    if "image_analysis" not in cs:
        cs.append("image_analysis")
    state["last_agent"] = "image_analysis"
    logger.debug("[workflow] exit node_image_analysis")
    return state


async def node_specialist(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_specialist")
    ia = state.get("image_analysis", {})
    md = ia.get("diseases")
    # Fallback: if diseases not provided by IA, try to use preliminary screening probabilities
    if not (isinstance(md, dict) and md):
        try:
            # Look for screening results in multiple plausible locations
            prelim = state.get("preliminary")
            if not prelim and isinstance(state.get("orchestrator_outputs"), dict):
                prelim = (state.get("orchestrator_outputs") or {}).get("preliminary")
            scr = None
            if isinstance(prelim, dict):
                scr = prelim.get("screening_results")
            if not scr and isinstance(state.get("orchestrator_outputs"), dict):
                # Backward-compat: some versions stored screening_results at top-level orchestrator_outputs
                scr = (state.get("orchestrator_outputs") or {}).get("screening_results")
            scr = scr or []
            if scr:
                out = (scr[0] or {}).get("output")
                if isinstance(out, dict):
                    cand = out.get("probabilities") if isinstance(out.get("probabilities"), dict) else out
                    if isinstance(cand, dict):
                        md = cand
        except Exception:
            pass
    sel = get_specialist_selection_settings()
    candidates: List[str] = get_candidate_diseases_from_probs(md, threshold=sel["candidate_threshold"], top_k=sel["candidate_top_k"])
    state["candidate_diseases"] = candidates

    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("specialist")
    if not cls:
        logger.warning("[workflow] SpecialistAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "image_analysis" in state:
        context["image_analysis"] = state["image_analysis"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["specialist"] = res.get("outputs")
    cs = state.setdefault("completed_steps", [])
    if "specialist" not in cs:
        cs.append("specialist")
    state["last_agent"] = "specialist"
    logger.debug("[workflow] exit node_specialist")
    return state


async def node_decision(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_decision")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("decision")
    if not cls:
        logger.warning("[workflow] DecisionAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "image_analysis" in state:
        context["image_analysis"] = state["image_analysis"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["decision"] = res.get("outputs")
    cs = state.setdefault("completed_steps", [])
    if "decision" not in cs:
        cs.append("decision")
    state["last_agent"] = "decision"
    logger.debug("[workflow] exit node_decision")
    return state


async def node_knowledge(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_knowledge")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("knowledge")
    if not cls:
        logger.warning("[workflow] KnowledgeAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "image_analysis" in state:
        context["image_analysis"] = state["image_analysis"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["knowledge"] = res.get("outputs")
    cs = state.setdefault("completed_steps", [])
    if "knowledge" not in cs:
        cs.append("knowledge")
    state["last_agent"] = "knowledge"
    logger.debug("[workflow] exit node_knowledge")
    return state


async def node_followup(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_followup")
    state["disease_grades"] = (state.get("specialist") or {}).get("disease_grades", [])
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("follow_up")
    if not cls:
        logger.warning("[workflow] FollowUpAgent not configured; skipping")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    if "decision" in state:
        context["decision"] = state["decision"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["follow_up"] = res.get("outputs")
    cs = state.setdefault("completed_steps", [])
    if "follow_up" not in cs:
        cs.append("follow_up")
    state["last_agent"] = "follow_up"
    logger.debug("[workflow] exit node_followup")
    return state


async def node_report(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_report")
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("report")
    if not cls:
        logger.warning("[workflow] ReportAgent not configured; skipping final report assembly")
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "image_analysis" in state:
        context["image_analysis"] = state["image_analysis"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    if "knowledge" in state:
        context["knowledge"] = state["knowledge"]
    if "decision" in state:
        context["decision"] = state["decision"]
    if "follow_up" in state:
        context["follow_up"] = state["follow_up"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["final_fragment"] = res.get("outputs")
    logger.debug("[workflow] exit node_report")
    return state


def compile_graph():
    register_builtins()

    g = StateGraph(WorkflowState)  # type: ignore[call-arg]
    g.add_node("orchestrator", node_orchestrator)
    g.add_node("preliminary", node_preliminary)
    g.add_node("image_analysis", node_image_analysis)
    g.add_node("specialist", node_specialist)
    g.add_node("knowledge", node_knowledge)
    g.add_node("decision", node_decision)
    g.add_node("follow_up", node_followup)
    g.add_node("report", node_report)

    g.add_edge(START, "orchestrator")

    def route_next(state: WorkflowState):
        nxt = state.get("next_agent")
        if nxt in ("preliminary", "image_analysis", "specialist", "knowledge", "decision", "follow_up", "report"):
            return nxt
        return "report"

    g.add_conditional_edges(
        "orchestrator",
        route_next,
        {
            "preliminary": "preliminary",
            "image_analysis": "image_analysis",
            "specialist": "specialist",
            "knowledge": "knowledge",
            "decision": "decision",
            "follow_up": "follow_up",
            "report": "report",
        },
    )

    for worker in ("preliminary", "image_analysis", "specialist", "knowledge", "decision", "follow_up"):
        g.add_edge(worker, "orchestrator")

    g.add_edge("report", END)
    logger.info("[workflow] execution_mode=langgraph-only")
    return g.compile()


async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], trace: TraceLogger | None = None, case_id: str | None = None, messages: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    trace = trace or TraceLogger()
    case_id = case_id or trace.create_case(patient=patient, images=images)
    try:
        from loguru import logger as _logger
        _logger.debug(
            "[workflow] run_diagnosis_async case_id={} images_count={} sample={} ",
            case_id,
            len(images or []),
            [(i.get('image_id'), i.get('path')) for i in (images or [])[:3]]
        )
    except Exception:
        pass
    state: WorkflowState = {
        "patient": patient,
        "images": images,
        "case_id": case_id,
        "trace": trace,
        "workflow": [],
        "messages": list(messages or [])
    }

    # Ensure image file paths are explicitly present in the message history for downstream agents
    # try:
    #     paths = [itm.get("path") for itm in (images or []) if isinstance(itm, dict) and itm.get("path")]
    #     if paths:
    #         txt = "Images (paths):\n" + "\n".join([str(p) for p in paths])
    #         msg = {"role": "user", "content": txt}
    #         state["messages"].append(msg)
    #         try:
    #             (trace or TraceLogger()).append_conversation_message(case_id, msg)  # type: ignore[arg-type]
    #         except Exception:
    #             pass
    # except Exception:
    #     pass

    graph = compile_graph()
    result_state = await graph.ainvoke(state)
    if result_state is None:
        logger.warning("[workflow] graph returned None state; using current state")
        result_state = state

    final_fragment = result_state.get("final_fragment", {})
    workflow = result_state.get("workflow", [])

    trace_path = state["trace"]._trace_path(case_id)
    try:
        base_dir = state["trace"].base_dir
        rel_trace = os.path.relpath(trace_path, base_dir)
        trace_ref = f"cases/{case_id}/trace.json" if rel_trace == f"{case_id}/trace.json" else trace_path
    except Exception:
        trace_ref = trace_path

    final_report = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "patient": patient,
        "images": images,
        "workflow": workflow,
        "final_report": {
            "diagnoses": final_fragment.get("diagnoses"),
            "lesions": final_fragment.get("lesions"),
            "management": final_fragment.get("management"),
            "reasoning": final_fragment.get("reasoning")
        },
        "generated_at": None,
        "trace_log_path": trace_ref
    }
    trace.write_final_report(case_id, final_report)
    return final_report


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]]):
    return asyncio.run(run_diagnosis_async(patient, images))

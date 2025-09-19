"""Diagnostic workflow orchestrated with LangGraph, with a no-LangGraph fallback."""
from typing import Dict, Any, List
import os
import asyncio
from dotenv import load_dotenv
from .core.logging import setup_logging

# Load environment variables from .env, if present
load_dotenv()
setup_logging()

# Try to import LangGraph; if unavailable, fall back to a simple sequential runner
try:
    from langgraph.graph import StateGraph, START, END  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except Exception:  # ImportError or others
    StateGraph = None  # type: ignore
    START = END = None  # type: ignore
    _LANGGRAPH_AVAILABLE = False
from .tracing.trace_logger import TraceLogger
from loguru import logger
from .agents.registry import register_builtins, get_agent_class
from .agents.capabilities import get_capabilities
def get_pipeline_capabilities(pipeline: list) -> list:
    """Return a list of capabilities for the given pipeline (list of agent class keys)."""
    out = []
    for key in pipeline:
        cls = get_agent_class(key)
        if cls:
            out.append({"key": key, "capabilities": get_capabilities(cls)})
    return out
from .config.pipelines import get_profile_steps, step_should_run
from .config.settings import get_workflow_mode
from .metrics.metrics import step_timer
from .core.interaction_engine import InteractionEngine

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
SCHEMA_VERSION = "1.0.0"

class WorkflowState(dict):
    """Mutable state passed along nodes."""
    pass


def _summarize_json(obj: Any, max_len: int = 800) -> str:
    """Return a compact JSON string truncated for chat messages."""
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
            # write to conversation log with seq
            (state.get("trace") or TraceLogger()).append_conversation_message(case_id, msg)  # type: ignore[arg-type]
        except Exception:
            pass
        # Also write a concise log line indicating which agent produced text
        logger.info(f"[agent case={case_id}] {title} reasoning -> {reasoning[:200]}{'...' if len(str(reasoning))>200 else ''}")
    # Tool calls with arguments and outputs
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
        logger.info(f"[agent case={case_id}] {title} tool_call id={tool_id} status={status}")
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


def _load_interaction_spec() -> Dict[str, Any] | None:
    import json
    try:
        import yaml  # type: ignore
        _HAS_YAML = True
    except Exception:
        yaml = None  # type: ignore
        _HAS_YAML = False
    candidates: List[str] = []
    env_path = os.getenv("EYEAGENT_INTERACTION_FILE")
    if env_path:
        candidates.append(env_path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    eye_cfg = os.path.join(repo_root, "eyeagent", "config")
    legacy_cfg = os.path.join(repo_root, "config")
    candidates.append(os.path.join(eye_cfg, "interaction.yml"))
    candidates.append(os.path.join(eye_cfg, "interaction.yaml"))
    candidates.append(os.path.join(eye_cfg, "interaction.json"))
    candidates.append(os.path.join(legacy_cfg, "interaction.yml"))
    candidates.append(os.path.join(legacy_cfg, "interaction.yaml"))
    candidates.append(os.path.join(legacy_cfg, "interaction.json"))
    for p in candidates:
        if not os.path.isfile(p):
            continue
        try:
            text = open(p, "r", encoding="utf-8").read()
            if p.endswith(".yml") or p.endswith(".yaml"):
                if not _HAS_YAML:
                    logger.warning("interaction file is YAML but PyYAML not installed")
                    continue
                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
            if isinstance(data, dict) and data.get("nodes"):
                return data
        except Exception as e:
            logger.warning(f"failed to load interaction spec from {p}: {e}")
    return None

async def node_orchestrator(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_orchestrator")
    # Ensure trace and case_id exist in state
    trace = state.get("trace") or TraceLogger()
    state["trace"] = trace
    case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
    state["case_id"] = case_id
    cls = get_agent_class("orchestrator")
    if not cls:
        logger.warning("[workflow] OrchestratorAgent not configured; skipping and defaulting to report")
        state["pipeline"] = ["report"]
        return state
    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    # Always pass instruction to orchestrator
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "patient" in state and "instruction" in state["patient"]:
        context["instruction"] = state["patient"]["instruction"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["pipeline"] = res["outputs"].get("planned_pipeline", [])
    # Save orchestrator's plan/command for downstream
    state["orchestrator_command"] = context.get("instruction")
    state["orchestrator_outputs"] = res["outputs"]
    logger.debug(f"[workflow] orchestrator pipeline={state.get('pipeline')}")
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
    # Pass orchestrator command and outputs
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
    state["image_analysis"] = res["outputs"]
    logger.debug("[workflow] exit node_image_analysis")
    return state

async def node_specialist(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_specialist")
    # derive candidate diseases from multi-disease screening
    ia = state.get("image_analysis", {})
    md = ia.get("diseases")
    candidates: List[str] = []
    if isinstance(md, dict):
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return 0.0
        candidates = [k for k, v in md.items() if _to_float(v) > 0.3][:5]
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
    # Pass orchestrator command and outputs, and image_analysis outputs
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
    state["specialist"] = res["outputs"]
    logger.debug("[workflow] exit node_specialist")
    return state

async def node_followup(state: WorkflowState) -> WorkflowState:
    logger.debug("[workflow] enter node_followup")
    # requires disease_grades
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
    # Pass orchestrator command and outputs, and specialist outputs
    context = dict(state)
    context["messages"] = state.get("messages", [])
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["follow_up"] = res["outputs"]
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
    # Pass orchestrator command and outputs, and all upstream outputs
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
    if "follow_up" in state:
        context["follow_up"] = state["follow_up"]
    with step_timer(agent.__class__.__name__, agent.role):
        res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    _append_messages_from_result(state, res)
    state["final_fragment"] = res["outputs"]
    logger.debug("[workflow] exit node_report")
    return state

def compile_graph():
    use_langgraph = _LANGGRAPH_AVAILABLE and os.getenv("EYEAGENT_USE_LANGGRAPH", "1").lower() in ("1", "true", "yes")
    logger.debug(f"[workflow] compile_graph use_langgraph={use_langgraph}")
    # Always ensure built-ins are registered before graph assembly
    register_builtins()

    # Config-first mode selection
    mode = get_workflow_mode()
    # Small helper to avoid duplicating unified runner code
    def _make_unified_graph():
        from .agents.unified_agent import UnifiedAgent  # lazy import

        class _UnifiedGraph:
            async def ainvoke(self, state: WorkflowState) -> WorkflowState:
                trace = state.get("trace") or TraceLogger()
                state["trace"] = trace
                case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
                state["case_id"] = case_id
                state.setdefault("messages", [])
                agent = UnifiedAgent(MCP_SERVER_URL, trace, case_id)
                from .metrics.metrics import step_timer as _timer
                with _timer(agent.__class__.__name__, agent.role):
                    ctx = dict(state)
                    ctx["messages"] = state.get("messages", [])
                    res = await agent.a_run(ctx)
                state.setdefault("workflow", []).append(res)
                _append_messages_from_result(state, res)
                state["final_fragment"] = res.get("outputs") or {}
                return state

        return _UnifiedGraph()

    if mode == "unified":
        logger.info("[workflow] execution_mode=unified (config)")
        return _make_unified_graph()

    # Unified mode: single agent does end-to-end
    if os.getenv("EYEAGENT_UNIFIED", "0").lower() in ("1", "true", "yes"):
        logger.info("[workflow] execution_mode=unified")
        return _make_unified_graph()

    # Optional: static pipeline override (comma-separated class/role names), bypassing orchestrator
    pipeline_override = os.getenv("EYEAGENT_PIPELINE")
    if pipeline_override:
        steps = [s.strip() for s in pipeline_override.split(",") if s.strip()]
        logger.debug(f"[workflow] pipeline override steps={steps}")

        class _StaticGraph:
            async def ainvoke(self, state: WorkflowState) -> WorkflowState:
                # Ensure trace and case_id present
                trace = state.get("trace") or TraceLogger()
                state["trace"] = trace
                case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
                state["case_id"] = case_id
                state.setdefault("messages", [])
                for key in steps:
                    cls = get_agent_class(key)
                    if not cls:
                        logger.warning(f"[workflow] unknown pipeline step '{key}' – skipping")
                        continue
                    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
                    with step_timer(agent.__class__.__name__, agent.role):
                        ctx = dict(state)
                        ctx["messages"] = state.get("messages", [])
                        res = await agent.a_run(ctx)
                    state.setdefault("workflow", []).append(res)
                    _append_messages_from_result(state, res)
                    # merge known outputs by role for downstream usage
                    role = getattr(agent, "role", None)
                    if role == "image_analysis":
                        state["image_analysis"] = res.get("outputs")
                    elif role == "specialist":
                        state["specialist"] = res.get("outputs")
                    elif role == "follow_up":
                        state["follow_up"] = res.get("outputs")
                    elif role == "orchestrator":
                        state["pipeline"] = (res.get("outputs") or {}).get("planned_pipeline", [])
                return state
        logger.info("[workflow] execution_mode=static-pipeline (no graph), source=env-override")
        return _StaticGraph()

    # Optional: external interaction spec (preferred). If present, use engine instead of graph.
    spec = _load_interaction_spec() if mode != "unified" else None
    if spec:
        logger.debug("[workflow] using declarative interaction spec")
        logger.info("[workflow] execution_mode=interaction, source=spec")
        return InteractionEngine(spec)

    # Optional: profile-driven pipeline (YAML/JSON with conditions)
    profile = os.getenv("EYEAGENT_PIPELINE_PROFILE")
    if profile:
        steps = get_profile_steps(profile)
        logger.debug(f"[workflow] profile={profile} steps={[s.get('name') for s in steps]}")

        class _ProfileGraph:
            async def ainvoke(self, state: WorkflowState) -> WorkflowState:
                trace = state.get("trace") or TraceLogger()
                state["trace"] = trace
                case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
                state["case_id"] = case_id
                state.setdefault("messages", [])
                for s in steps:
                    name = s.get("name")
                    if not name:
                        continue
                    if not step_should_run(s, state):
                        logger.debug(f"[workflow] skip step={name} (condition false)")
                        continue
                    cls = get_agent_class(name)
                    if not cls:
                        logger.warning(f"[workflow] unknown step='{name}' – skipping")
                        continue
                    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore
                    logger.debug(f"[workflow] run step={name}")
                    with step_timer(agent.__class__.__name__, agent.role):
                        ctx = dict(state)
                        ctx["messages"] = state.get("messages", [])
                        res = await agent.a_run(ctx)
                    state.setdefault("workflow", []).append(res)
                    _append_messages_from_result(state, res)
                    role = getattr(agent, "role", None)
                    if role == "image_analysis":
                        state["image_analysis"] = res.get("outputs")
                    elif role == "specialist":
                        state["specialist"] = res.get("outputs")
                    elif role == "follow_up":
                        state["follow_up"] = res.get("outputs")
                    elif role == "orchestrator":
                        state["pipeline"] = (res.get("outputs") or {}).get("planned_pipeline", [])
                return state
    logger.info("[workflow] execution_mode=profile, source=pipelines.yml")
    return _ProfileGraph()
    if use_langgraph:
        g = StateGraph(WorkflowState)  # type: ignore[call-arg]
        g.add_node("orchestrator", node_orchestrator)
        g.add_node("image_analysis", node_image_analysis)
        g.add_node("specialist", node_specialist)
        g.add_node("follow_up", node_followup)
        g.add_node("report", node_report)

        g.add_edge(START, "orchestrator")
        # conditional edges based on orchestrator plan
        def next_stage(state: WorkflowState):
            pipeline = state.get("pipeline", [])
            # Execute IA -> Specialist -> FollowUp -> Report in order if present
            return "image_analysis" if "image_analysis" in pipeline else "report"

        g.add_conditional_edges("orchestrator", next_stage, {
            "image_analysis": "image_analysis",
            "report": "report"
        })
        # After IA, go to specialist if present
        def after_ia(state: WorkflowState):
            pipeline = state.get("pipeline", [])
            return "specialist" if "specialist" in pipeline else "report"
        g.add_conditional_edges("image_analysis", after_ia, {
            "specialist": "specialist",
            "report": "report"
        })
        # After specialist, go to follow_up if present
        def after_sp(state: WorkflowState):
            pipeline = state.get("pipeline", [])
            return "follow_up" if "follow_up" in pipeline else "report"
        g.add_conditional_edges("specialist", after_sp, {
            "follow_up": "follow_up",
            "report": "report"
        })
        g.add_edge("follow_up", "report")
        g.add_edge("report", END)
        return g.compile()

    # Fallback: simple sequential runner mimicking the planned pipeline
    class _FallbackGraph:
        def __init__(self):
            pass

        async def ainvoke(self, state: WorkflowState) -> WorkflowState:
            # Ensure base identifiers exist if caller didn't initialize
            trace = state.get("trace") or TraceLogger()
            state["trace"] = trace
            state["case_id"] = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
            state.setdefault("messages", [])
            # Orchestrator first
            state = await node_orchestrator(state)
            pipeline = state.get("pipeline", []) or []

            # If image_analysis not planned, jump directly to report to mimic conditional edge
            if "image_analysis" not in pipeline:
                state = await node_report(state)
                return state

            # Image Analysis
            state = await node_image_analysis(state)

            # Specialist if planned
            if "specialist" in pipeline:
                state = await node_specialist(state)

            # Follow-up if planned
            if "follow_up" in pipeline:
                state = await node_followup(state)

            # Report always last
            state = await node_report(state)
            return state

    logger.info("[workflow] execution_mode=fallback, source=code")
    return _FallbackGraph()

async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], trace: TraceLogger | None = None, case_id: str | None = None, messages: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    # Allow caller (e.g., UI) to inject an existing TraceLogger and case_id for streaming/replay
    trace = trace or TraceLogger()
    case_id = case_id or trace.create_case(patient=patient, images=images)
    state: WorkflowState = {
        "patient": patient,
        "images": images,
        "case_id": case_id,
        "trace": trace,
        "workflow": [],
        "messages": list(messages or [])
    }

    graph = compile_graph()
    result_state = await graph.ainvoke(state)
    if result_state is None:
        # some graph runtimes may return None; use current state to avoid crashes
        logger.warning("[workflow] graph returned None state; using current state")
        result_state = state

    final_fragment = result_state.get("final_fragment", {})
    workflow = result_state.get("workflow", [])

    # derive trace path from TraceLogger
    trace_path = state["trace"]._trace_path(case_id)
    # present relative path when under the same base dir
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

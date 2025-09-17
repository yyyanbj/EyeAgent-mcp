"""Diagnostic workflow orchestrated with LangGraph, with a no-LangGraph fallback."""
from typing import Dict, Any, List
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env, if present
load_dotenv()

# Try to import LangGraph; if unavailable, fall back to a simple sequential runner
try:
    from langgraph.graph import StateGraph, START, END  # type: ignore
    _LANGGRAPH_AVAILABLE = True
except Exception:  # ImportError or others
    StateGraph = None  # type: ignore
    START = END = None  # type: ignore
    _LANGGRAPH_AVAILABLE = False
from .tracing.trace_logger import TraceLogger
from .agents.orchestrator_agent import OrchestratorAgent
from .agents.image_analysis_agent import ImageAnalysisAgent
from .agents.specialist_agent import SpecialistAgent
from .agents.followup_agent import FollowUpAgent
from .agents.report_agent import ReportAgent

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
SCHEMA_VERSION = "1.0.0"

class WorkflowState(dict):
    """Mutable state passed along nodes."""
    pass

async def node_orchestrator(state: WorkflowState) -> WorkflowState:
    agent = OrchestratorAgent(MCP_SERVER_URL, state["trace"], state["case_id"])
    # Always pass instruction to orchestrator
    context = dict(state)
    if "patient" in state and "instruction" in state["patient"]:
        context["instruction"] = state["patient"]["instruction"]
    res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    state["pipeline"] = res["outputs"].get("planned_pipeline", [])
    # Save orchestrator's plan/command for downstream
    state["orchestrator_command"] = context.get("instruction")
    state["orchestrator_outputs"] = res["outputs"]
    return state

async def node_image_analysis(state: WorkflowState) -> WorkflowState:
    agent = ImageAnalysisAgent(MCP_SERVER_URL, state["trace"], state["case_id"])
    # Pass orchestrator command and outputs
    context = dict(state)
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    state["image_analysis"] = res["outputs"]
    return state

async def node_specialist(state: WorkflowState) -> WorkflowState:
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

    agent = SpecialistAgent(MCP_SERVER_URL, state["trace"], state["case_id"])
    # Pass orchestrator command and outputs, and image_analysis outputs
    context = dict(state)
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "image_analysis" in state:
        context["image_analysis"] = state["image_analysis"]
    res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    state["specialist"] = res["outputs"]
    return state

async def node_followup(state: WorkflowState) -> WorkflowState:
    # requires disease_grades
    state["disease_grades"] = (state.get("specialist") or {}).get("disease_grades", [])
    agent = FollowUpAgent(MCP_SERVER_URL, state["trace"], state["case_id"])
    # Pass orchestrator command and outputs, and specialist outputs
    context = dict(state)
    if "orchestrator_command" in state:
        context["orchestrator_command"] = state["orchestrator_command"]
    if "orchestrator_outputs" in state:
        context["orchestrator_outputs"] = state["orchestrator_outputs"]
    if "specialist" in state:
        context["specialist"] = state["specialist"]
    res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    state["follow_up"] = res["outputs"]
    return state

async def node_report(state: WorkflowState) -> WorkflowState:
    agent = ReportAgent(MCP_SERVER_URL, state["trace"], state["case_id"])
    # Pass orchestrator command and outputs, and all upstream outputs
    context = dict(state)
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
    res = await agent.a_run(context)
    state.setdefault("workflow", []).append(res)
    state["final_fragment"] = res["outputs"]
    return state

def compile_graph():
    use_langgraph = _LANGGRAPH_AVAILABLE and os.getenv("EYEAGENT_USE_LANGGRAPH", "0").lower() in ("1", "true", "yes")
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

    return _FallbackGraph()

async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], trace: TraceLogger | None = None, case_id: str | None = None) -> Dict[str, Any]:
    # Allow caller (e.g., UI) to inject an existing TraceLogger and case_id for streaming/replay
    trace = trace or TraceLogger()
    case_id = case_id or trace.create_case(patient=patient, images=images)
    state: WorkflowState = {
        "patient": patient,
        "images": images,
        "case_id": case_id,
        "trace": trace,
        "workflow": []
    }

    graph = compile_graph()
    result_state = await graph.ainvoke(state)

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

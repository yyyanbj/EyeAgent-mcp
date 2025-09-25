"""Profile-driven diagnostic workflow using the InteractionEngine and pipeline profiles.

This backend builds a simple sequential pipeline from a named profile in
`eyeagent/config/pipelines.yml` (or JSON) and runs it via the generic
InteractionEngine. It keeps the same public API as the default workflow.

Default profile can be set with env var EYEAGENT_PIPELINE_PROFILE (default: "default").
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import asyncio

from ..tracing.trace_logger import TraceLogger
from ..core.interaction_engine import InteractionEngine
from ..config.pipelines import get_profile_steps, step_should_run
from ..agents.registry import get_agent_class
from ..metrics.metrics import step_timer
from loguru import logger

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
SCHEMA_VERSION = "1.0.0"

# State type for consistency with langgraph backend
class WorkflowState(dict):
    pass


def _append_messages_from_result(state: WorkflowState, result: Dict[str, Any]) -> None:
    # Minimal helper: append reasoning and tool summaries to messages list
    msgs = state.setdefault("messages", [])
    agent = result.get("agent") or "Agent"
    role = result.get("role") or ""
    title = f"{agent} ({role})"
    reasoning = result.get("reasoning")
    if reasoning:
        msgs.append({"role": "assistant", "content": f"{title} reasoning: {reasoning}"})


def _build_engine_spec(profile: str) -> Dict[str, Any]:
    steps = get_profile_steps(profile)
    # Default start at orchestrator if present; else first step
    start = "orchestrator" if any(s.get("name") == "orchestrator" for s in steps) else (steps[0]["name"] if steps else "report")
    # Build nodes: each step maps to a node
    nodes = []
    for s in steps:
        name = s.get("name")
        if not name:
            continue
        nodes.append({
            "id": name,
            "agent": name,  # registry key matches name
            # Provide messages + state; allow agent to consume what it needs
            "inputs": ["patient", "images", "messages", "orchestrator_outputs", "image_analysis", "specialist", "knowledge", "follow_up"],
            # Map common outputs back into state when available
            "outputs": [
                {"from": "planned_pipeline", "to": "pipeline"},
                {"from": "next_agent", "to": "next_agent"},
                {"from": "diagnoses", "to": "diagnoses"},
                {"from": "lesions", "to": "lesions"},
                {"from": "management", "to": "management"},
            ],
            # Optional: when condition is checked dynamically during run (we'll handle it in loop)
            "when": s.get("when"),
        })
    # Build simple linear edges in declared order
    edges = []
    for i in range(len(nodes) - 1):
        edges.append({"from": nodes[i]["id"], "to": [{"next": nodes[i+1]["id"]}]})
    return {"start": start, "nodes": nodes, "edges": edges}


async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], trace: Optional[TraceLogger] = None, case_id: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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

    profile = os.getenv("EYEAGENT_PIPELINE_PROFILE", "default")
    spec = _build_engine_spec(profile)
    engine = InteractionEngine(spec)
    result_state = await engine.ainvoke(state)

    final_fragment = {
        "diagnoses": result_state.get("diagnoses"),
        "lesions": result_state.get("lesions"),
        "management": result_state.get("management"),
        "reasoning": (result_state.get("specialist") or {}).get("reasoning"),
    }

    # Write minimal final report for parity with default backend
    final_report = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "patient": patient,
        "images": images,
        "workflow": result_state.get("workflow", []),
        "final_report": final_fragment,
        "generated_at": None,
        "trace_log_path": trace._trace_path(case_id),
    }
    trace.write_final_report(case_id, final_report)
    return final_report


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]]):
    return asyncio.run(run_diagnosis_async(patient, images))

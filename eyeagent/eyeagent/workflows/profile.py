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

from eyeagent.trace.trace_logger import TraceLogger
from eyeagent.core.settings import get_pipeline_profile, get_mcp_server_url
from eyeagent.core.interaction_engine import InteractionEngine
from eyeagent.core.pipelines import get_profile_steps, step_should_run, get_profile_raw
from eyeagent.agents.registry import get_agent_class
from eyeagent.metrics.metrics import step_timer
from loguru import logger

MCP_SERVER_URL = get_mcp_server_url("http://localhost:8000/mcp/")
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
    raw = get_profile_raw(profile)
    steps = get_profile_steps(profile)
    # Default start at orchestrator if present; else first step
    start = "orchestrator" if any(s.get("name") == "orchestrator" for s in steps) else (steps[0]["name"] if steps else "report")
    # Build nodes: each step maps to a node; allow per-step inputs/outputs overrides from raw profile
    raw_steps = {s.get("name"): s for s in (raw.get("steps") or []) if isinstance(s, dict) and s.get("name")}
    nodes = []
    for s in steps:
        name = s.get("name")
        if not name:
            continue
        r = raw_steps.get(name) or {}
        inputs = r.get("inputs") or ["patient", "images", "messages", "orchestrator_outputs", "image_analysis", "specialist", "knowledge", "follow_up"]
        outputs = r.get("outputs") or [
            {"from": "planned_pipeline", "to": "pipeline"},
            {"from": "next_agent", "to": "next_agent"},
            {"from": "diagnoses", "to": "diagnoses"},
            {"from": "lesions", "to": "lesions"},
            {"from": "management", "to": "management"},
        ]
        nodes.append({
            "id": name,
            "agent": name,  # registry key matches name
            "inputs": inputs,
            "outputs": outputs,
            "when": s.get("when"),
        })
    # Use custom edges if provided in raw profile; else build simple linear edges
    edges = raw.get("edges") or []
    if not edges:
        for i in range(len(nodes) - 1):
            edges.append({"from": nodes[i]["id"], "to": [{"next": nodes[i+1]["id"]}]})
    return {"start": start, "nodes": nodes, "edges": edges}


async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], trace: Optional[TraceLogger] = None, case_id: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None, prior: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    # Hydrate prior incremental state if provided
    if prior and isinstance(prior, dict):
        for k in ("orchestrator_outputs","preliminary","image_analysis","specialist","knowledge","follow_up"):
            if prior.get(k) is not None:
                state[k] = prior.get(k)
        # Compute new_image_ids from previous images meta if provided
        try:
            prev_imgs = prior.get("images") or []
            prev_ids = {str(i.get("image_id") or i.get("path")) for i in prev_imgs if isinstance(i, dict)}
            cur_ids = {str(i.get("image_id") or i.get("path")) for i in images if isinstance(i, dict)}
            new_ids = [x for x in cur_ids if x not in prev_ids]
            if new_ids:
                state["incremental"] = True
                state["new_image_ids"] = new_ids
                state["prev_image_ids"] = list(prev_ids)
        except Exception:
            pass

    profile = get_pipeline_profile()
    # Expose profile info to agents via state
    try:
        steps_cfg = get_profile_steps(profile)
        state["profile_name"] = profile
        state["profile_steps"] = [s.get("name") for s in steps_cfg if isinstance(s, dict) and s.get("name")]
    except Exception:
        state["profile_name"] = profile
        state["profile_steps"] = []
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

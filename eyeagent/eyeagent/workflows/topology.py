"""Topology-driven diagnostic workflow using lightweight star/ring executors.

This backend runs agents according to simple topologies without requiring LangGraph.
Configuration is provided via environment variables or main settings:

Settings (eyeagent.yml):
workflow:
  backend: topology
  topology:
    type: ring
    agents: [preliminary, image_analysis, specialist, report]
    rounds: 1

Behavior:
- Ring: run agents in specified order for N rounds; leverages core.topologies.run_ring.
- Star: run selected agents concurrently (fan-out) then optionally call report at the end.

Public API mirrors other backends: run_diagnosis_async / run_diagnosis
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import asyncio

from loguru import logger

from eyeagent.trace.trace_logger import TraceLogger
from eyeagent.core.settings import get_mcp_server_url
from eyeagent.agents.registry import register_builtins, get_agent_class
from eyeagent.metrics.metrics import step_timer
from eyeagent.core.topologies import run_star, run_ring, TopologyState, AgentCallable

SCHEMA_VERSION = "1.0.0"
MCP_SERVER_URL = get_mcp_server_url("http://localhost:8000/mcp/")


def _append_messages_from_result(state: Dict[str, Any], result: Dict[str, Any]) -> None:
    try:
        # Reuse the existing helper if available
        from .langgraph import _append_messages_from_result as _impl  # type: ignore
        _impl(state, result)
    except Exception:
        # Minimal fallback: do nothing on failure
        pass


def _wrap_agent(role: str) -> AgentCallable:
    async def fn(state: TopologyState):
        cls = get_agent_class(role)
        if not cls:
            return {"context": {f"{role}_error": "unknown agent"}}
        trace = state.get("trace") or TraceLogger()
        case_id = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
        state["trace"] = trace
        state["case_id"] = case_id
        agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
        context = dict(state)  # shallow copy
        context["messages"] = state.get("messages", [])
        with step_timer(agent.__class__.__name__, getattr(agent, "role", role)):
            res = await agent.a_run(context)
        state.setdefault("workflow", []).append(res)
        _append_messages_from_result(state, res)
        outputs = (res or {}).get("outputs")
        # return update to merge under the agent role for convenience
        return {role: outputs}

    return fn


def _resolve_topology_settings() -> Dict[str, Any]:
    try:
        from eyeagent.core.settings import Settings
        cfg = Settings().load()
        wf = cfg.get("workflow") or {}
        topo = wf.get("topology") or {}
        agents = [str(x) for x in (topo.get("agents") or []) if isinstance(x, (str,))]
        r = topo.get("rounds")
        rounds = int(r) if r is not None else 1
        topo_type = str(topo.get("type") or "ring").strip().lower()
        if topo_type not in {"star", "ring"}:
            topo_type = "ring"
        return {"type": topo_type, "agents": agents, "rounds": rounds}
    except Exception:
        # Defaults
        return {"type": "ring", "agents": ["preliminary", "image_analysis", "specialist", "report"], "rounds": 1}


async def run_diagnosis_async(
    patient: Dict[str, Any],
    images: List[Dict[str, Any]],
    trace: Optional[TraceLogger] = None,
    case_id: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    register_builtins()

    trace = trace or TraceLogger()
    case_id = case_id or trace.create_case(patient=patient, images=images)
    state: TopologyState = {
        "patient": patient,
        "images": images,
        "case_id": case_id,
        "trace": trace,
        "workflow": [],
        "messages": list(messages or []),
        "context": {},
    }

    cfg = _resolve_topology_settings()
    roles = cfg["agents"] or ["preliminary", "image_analysis", "specialist", "report"]
    # Build callable map/order
    agent_map: Dict[str, AgentCallable] = {r: _wrap_agent(r) for r in roles}

    if cfg["type"] == "star":
        # Simple planner: run all configured agents (dedup, existing only)
        async def planner(s: TopologyState) -> List[str]:
            return [r for r in roles if r in agent_map]

        # Default aggregate: nothing special; keep per-agent updates in state
        await run_star(state, planner, agent_map)
        # Optional: if report is configured and not already executed as part of star, run it once
        if "report" in agent_map and "report" not in roles:
            try:
                await agent_map["report"](state)
            except Exception:
                pass
    else:
        # Ring execution with rounds
        ordered: List[Tuple[str, AgentCallable]] = [(r, agent_map[r]) for r in roles if r in agent_map]
        await run_ring(state, ordered, rounds=int(cfg.get("rounds", 1) or 1))

    # Prefer report outputs if present
    final_fragment = (
        (state.get("final_fragment") or {})
        if isinstance(state.get("final_fragment"), dict)
        else (state.get("report") or {})
    )
    if not final_fragment:
        # fallbacks
        final_fragment = (state.get("decision") or {}) or (state.get("specialist") or {})

    final_report = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "patient": patient,
        "images": images,
        "workflow": state.get("workflow", []),
        "final_report": {
            "diagnoses": (final_fragment or {}).get("diagnoses"),
            "lesions": (final_fragment or {}).get("lesions"),
            "management": (final_fragment or {}).get("management"),
            "reasoning": (final_fragment or {}).get("reasoning"),
        },
        "generated_at": None,
        "trace_log_path": trace._trace_path(case_id),
    }
    trace.write_final_report(case_id, final_report)
    return final_report


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]]):
    return asyncio.run(run_diagnosis_async(patient, images))

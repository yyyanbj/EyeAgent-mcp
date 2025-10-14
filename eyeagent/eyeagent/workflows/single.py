"""Single-agent diagnostic workflow backend.

This backend runs the UnifiedAgent directly without multi-agent orchestration.
It keeps the public API consistent with other backends by returning a final
report payload and writing traces via TraceLogger.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import asyncio

from ..tracing.trace_logger import TraceLogger
from ..agents.registry import register_builtins, get_agent_class
from ..metrics.metrics import step_timer
from loguru import logger

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/")
SCHEMA_VERSION = "1.0.0"


async def run_diagnosis_async(
    patient: Dict[str, Any],
    images: List[Dict[str, Any]],
    trace: Optional[TraceLogger] = None,
    case_id: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run diagnosis using a single UnifiedAgent.

    Returns a payload similar to other backends with a top-level final_report.
    For compatibility with some consumers, mirrors final_fragment fields at
    top-level under the key "final_fragment" as well.
    """
    trace = trace or TraceLogger()
    case_id = case_id or trace.create_case(patient=patient, images=images)

    # Ensure built-in agents are available (including UnifiedAgent)
    register_builtins()
    cls = get_agent_class("unified") or get_agent_class("UnifiedAgent")
    if not cls:
        raise RuntimeError("UnifiedAgent not found in registry; check configuration")

    context: Dict[str, Any] = {
        "patient": patient,
        "images": images,
        "messages": list(messages or []),
        "trace": trace,
        "case_id": case_id,
    }

    agent = cls(MCP_SERVER_URL, trace, case_id)  # type: ignore[call-arg]
    with step_timer(agent.__class__.__name__, getattr(agent, "role", "unified")):
        result = await agent.a_run(context)

    outputs = (result or {}).get("outputs") or {}

    # Build a final fragment compatible with downstream formatting
    final_fragment: Dict[str, Any] = {
        "diagnoses": outputs.get("diagnoses"),
        "lesions": outputs.get("lesions"),
        "management": outputs.get("management"),
        "reasoning": outputs.get("reasoning"),
        "narrative": outputs.get("narrative"),
        "conclusion": outputs.get("conclusion"),
        # Also pass through useful aggregates if present
        "per_image": outputs.get("per_image"),
        "diseases": outputs.get("diseases"),
        "modality": outputs.get("modality"),
    }

    final_report = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "patient": patient,
        "images": images,
        "workflow": [result],  # single step record for transparency
        "final_report": final_fragment,
        "generated_at": None,
        "trace_log_path": trace._trace_path(case_id),
        # For compatibility with some consumers that expect top-level fragment
        "final_fragment": final_fragment,
    }

    trace.write_final_report(case_id, final_report)
    return final_report


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]]):
    return asyncio.run(run_diagnosis_async(patient, images))

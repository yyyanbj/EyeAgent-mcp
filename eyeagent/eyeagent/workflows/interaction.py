"""Spec-driven diagnostic workflow using InteractionEngine.

This backend lets callers provide a custom spec (nodes/edges) for orchestration.
If no spec is provided, it falls back to a simple orchestrator-led sequence.

Public API mirrors the default workflow.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os
import asyncio

from ..tracing.trace_logger import TraceLogger
from ..core.interaction_engine import InteractionEngine
from loguru import logger

SCHEMA_VERSION = "1.0.0"

# Minimal state type to stay consistent
class WorkflowState(dict):
    pass


def _default_spec() -> Dict[str, Any]:
    # Simple sequence: orchestrator -> image_analysis -> specialist -> follow_up -> report
    return {
        "start": "orchestrator",
        "nodes": [
            {"id": "orchestrator", "agent": "orchestrator", "inputs": ["patient", "images", "messages"], "outputs": [{"from": "planned_pipeline", "to": "pipeline"}, {"from": "next_agent", "to": "next_agent"}]},
            {"id": "preliminary", "agent": "preliminary", "inputs": ["patient", "images", "messages"]},
            {"id": "image_analysis", "agent": "image_analysis", "inputs": ["patient", "images", "messages", "orchestrator_outputs"]},
            {"id": "specialist", "agent": "specialist", "inputs": ["patient", "images", "messages", "image_analysis", "orchestrator_outputs"]},
            {"id": "follow_up", "agent": "follow_up", "inputs": ["patient", "images", "messages", "specialist", "orchestrator_outputs"]},
            {"id": "knowledge", "agent": "knowledge", "inputs": ["patient", "images", "messages", "specialist", "image_analysis"]},
            {"id": "report", "agent": "report", "inputs": ["patient", "images", "messages", "image_analysis", "specialist", "knowledge", "follow_up", "orchestrator_outputs"]},
        ],
        "edges": [
            {"from": "orchestrator", "to": [{"next": "preliminary"}]},
            {"from": "preliminary", "to": [{"next": "image_analysis"}]},
            {"from": "image_analysis", "to": [{"next": "specialist"}]},
            {"from": "specialist", "to": [{"next": "follow_up"}]},
            {"from": "follow_up", "to": [{"next": "report"}]},
        ],
    }


async def run_diagnosis_async(patient: Dict[str, Any], images: List[Dict[str, Any]], spec: Optional[Dict[str, Any]] = None, trace: Optional[TraceLogger] = None, case_id: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
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

    engine = InteractionEngine(spec or _default_spec())
    result_state = await engine.ainvoke(state)

    final_fragment = {
        "diagnoses": result_state.get("diagnoses"),
        "lesions": result_state.get("lesions"),
        "management": result_state.get("management"),
        "reasoning": (result_state.get("specialist") or {}).get("reasoning"),
    }

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


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]], spec: Optional[Dict[str, Any]] = None):
    return asyncio.run(run_diagnosis_async(patient, images, spec=spec))

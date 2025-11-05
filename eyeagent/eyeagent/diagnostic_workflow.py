"""Unified entry that delegates to a selected workflow backend at runtime.

Default is LangGraph. You can switch by configuring `workflow.backend`
in `eyeagent/config/eyeagent.yml` or via env `EYEAGENT_WORKFLOW_BACKEND`.
Backends: langgraph | profile | interaction
"""
from typing import Dict, Any, List, Optional
import asyncio
from .workflows.langgraph import (
    WorkflowState,  # re-exported for compatibility
    TraceLogger,    # re-exported for type hints in signatures
)
from .workflows.langgraph import _append_messages_from_result  # backward-compat re-export
from .core.settings import get_workflow_backend


async def run_diagnosis_async(
    patient: Dict[str, Any],
    images: List[Dict[str, Any]],
    trace: TraceLogger | None = None,
    case_id: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    spec: Optional[Dict[str, Any]] = None,
    prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    backend = get_workflow_backend()
    if backend == "profile":
        from .workflows.profile import run_diagnosis_async as _impl
        return await _impl(patient, images, trace=trace, case_id=case_id, messages=messages, prior=prior)
    elif backend == "interaction":
        from .workflows.interaction import run_diagnosis_async as _impl
        return await _impl(patient, images, spec=spec, trace=trace, case_id=case_id, messages=messages, prior=prior)
    elif backend == "single":
        from .workflows.single import run_diagnosis_async as _impl
        return await _impl(patient, images, trace, case_id, messages)  # type: ignore[misc]
    elif backend == "topology":
        from .workflows.topology import run_diagnosis_async as _impl
        return await _impl(patient, images, trace, case_id, messages)  # type: ignore[misc]
    else:
        from .workflows.langgraph import run_diagnosis_async as _impl
        return await _impl(patient, images, trace, case_id, messages)  # type: ignore[misc]


def run_diagnosis(patient: Dict[str, Any], images: List[Dict[str, Any]]):
    return asyncio.run(run_diagnosis_async(patient, images))

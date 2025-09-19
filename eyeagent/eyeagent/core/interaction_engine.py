from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import os
from loguru import logger

from ..agents.registry import get_agent_class
from ..metrics.metrics import step_timer
from ..tracing.trace_logger import TraceLogger


class InteractionEngine:
    """
    A generic, declarative interaction engine to orchestrate agents based on a configurable spec.

    Spec shape (YAML/JSON):
      nodes:
        - id: orchestrator
          agent: OrchestratorAgent  # or registry key e.g. orchestrator
          when: { key: "patient.instruction", op: "exists" }
          inputs:
            # fields copied from state into agent context
            - images
            - patient
            - orchestrator_outputs
          outputs:
            # field mapping from result.outputs into state
            - { from: planned_pipeline, to: pipeline }
          on_result:
            # optional callback name for custom state transforms (not implemented here)
            - append_messages
      edges:
        - from: orchestrator
          to:
            - when: { key: "pipeline", op: "exists" }
              next: image_analysis
            - next: report
        - from: image_analysis
          to:
            - when: { key: "pipeline", op: "exists" }
              match: specialist
              next: specialist
            - next: report

    Engine evaluates nodes in a simple loop following edges; if no edge matches, it stops.
    """

    def __init__(self, spec: Dict[str, Any]):
        self.spec = spec or {}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        for n in (self.spec.get("nodes") or []):
            if isinstance(n, dict) and n.get("id"):
                self.nodes[n["id"]] = n
        self.edges: Dict[str, List[Dict[str, Any]]] = {}
        for e in (self.spec.get("edges") or []):
            src = e.get("from")
            if not src:
                continue
            self.edges.setdefault(src, []).append(e)

    @staticmethod
    def _get_by_path(state: Dict[str, Any], path: str) -> Any:
        cur: Any = state
        for part in (path or "").split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    @classmethod
    def _eval_condition(cls, cond: Dict[str, Any], state: Dict[str, Any]) -> bool:
        key = cond.get("key")
        op = cond.get("op", "==")
        val = cond.get("value")
        left = cls._get_by_path(state, key) if isinstance(key, str) else None
        try:
            if op == "==":
                return left == val
            if op == "!=":
                return left != val
            if op == ">":
                return float(left) > float(val)
            if op == ">=":
                return float(left) >= float(val)
            if op == "<":
                return float(left) < float(val)
            if op == "<=":
                return float(left) <= float(val)
            if op == "exists":
                return left is not None
            if op == "not_exists":
                return left is None
        except Exception:
            return False
        return False

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self.spec.get("start") or "orchestrator"
        visited = 0
        pipeline_order: Optional[List[str]] = None
        pipeline_idx = 0
        # ensure messages/trace/case_id exist
        trace = state.get("trace") or TraceLogger()
        state["trace"] = trace
        state["case_id"] = state.get("case_id") or trace.create_case(patient=state.get("patient", {}), images=state.get("images", []))
        state.setdefault("messages", [])

        while current and visited < 64:  # loop guard
            visited += 1
            node = self.nodes.get(current)
            if not node:
                logger.warning(f"[engine] missing node id={current}")
                break
            agent_key = node.get("agent") or current
            cls = get_agent_class(agent_key)
            if not cls:
                logger.warning(f"[engine] unknown agent={agent_key}")
                break
            # when condition
            when = node.get("when")
            if when and not self._eval_condition(when, state):
                logger.debug(f"[engine] skip node={current} (when false)")
                # follow edges even if skipped
                next_id = self._next_from_edges(current, state)
                current = next_id
                continue

            agent = cls(os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/"), state["trace"], state["case_id"])  # type: ignore
            # Build context
            ctx = dict(state)
            ctx["messages"] = state.get("messages", [])
            # filter inputs if provided
            inputs = node.get("inputs")
            if isinstance(inputs, list) and inputs:
                filtered = {k: state.get(k) for k in inputs}
                ctx.update(filtered)

            with step_timer(agent.__class__.__name__, getattr(agent, "role", "agent")):
                res = await agent.a_run(ctx)
            state.setdefault("workflow", []).append(res)
            # Append messages similar to workflow helper
            try:
                from ..diagnostic_workflow import _append_messages_from_result  # type: ignore
                _append_messages_from_result(state, res)
            except Exception:
                pass
            # map outputs
            outputs = (res or {}).get("outputs") or {}
            for m in (node.get("outputs") or []):
                try:
                    src = m.get("from")
                    dst = m.get("to")
                    if src and dst and isinstance(outputs, dict):
                        if src in outputs:
                            state[dst] = outputs[src]
                except Exception:
                    continue

            # After orchestrator, allow orchestrator-decided pipeline to take control
            if (current == "orchestrator") and isinstance(state.get("pipeline"), list) and state["pipeline"]:
                pipeline_order = [str(x) for x in state["pipeline"]]
                pipeline_idx = 0
                logger.info(f"[engine] routing via orchestrator pipeline: {pipeline_order}")
                # Advance to first node in pipeline
                current = pipeline_order[pipeline_idx]
                pipeline_idx += 1
                continue

            # If already following a pipeline, continue sequentially
            if pipeline_order is not None:
                if pipeline_idx < len(pipeline_order):
                    current = pipeline_order[pipeline_idx]
                    pipeline_idx += 1
                else:
                    current = None
                continue

            # Otherwise, follow declarative edges
            current = self._next_from_edges(current, state)

        return state

    def _next_from_edges(self, current: str, state: Dict[str, Any]) -> Optional[str]:
        arr = self.edges.get(current) or []
        # each edge: { from, to: [ { when?, next }...] }
        for e in arr:
            dests = e.get("to") or []
            for d in dests:
                cond = d.get("when")
                if cond and not self._eval_condition(cond, state):
                    continue
                nxt = d.get("next")
                if nxt:
                    return nxt
        return None

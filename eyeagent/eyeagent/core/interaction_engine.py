from __future__ import annotations
from typing import Any, Dict, List, Optional, Callable
import os
from loguru import logger

from eyeagent.agents.registry import get_agent_class
from eyeagent.metrics.metrics import step_timer
from eyeagent.trace.trace_logger import TraceLogger
from eyeagent.core.settings import get_mcp_server_url


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
        # Enforce strict cadence: route (orchestrator) -> execute (one agent) -> collect -> route ...
        # Start node: prefer spec.start; else orchestrator; else first defined node; else None
        current = None
        try:
            start = (self.spec.get("start") or "").strip()
            if start and start in self.nodes:
                current = start
            elif "orchestrator" in self.nodes:
                current = "orchestrator"
            elif self.nodes:
                current = next(iter(self.nodes.keys()))
            else:
                current = None
        except Exception:
            logger.warning("[engine] failed to determine start node")
            current = "orchestrator"
        visited = 0
        # When an agent (non-orchestrator) has just run, we always return to orchestrator
        return_to_orchestrator = False
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

            agent = cls(get_mcp_server_url("http://localhost:8000/mcp/"), state["trace"], state["case_id"])  # type: ignore
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
                from eyeagent.diagnostic_workflow import _append_messages_from_result  # type: ignore
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

            # Strict cadence control:
            if current == "orchestrator":
                # Prefer explicit next_agent from orchestrator outputs
                next_agent = None
                try:
                    next_agent = (outputs or {}).get("next_agent")
                except Exception:
                    next_agent = None
                if isinstance(next_agent, str) and next_agent:
                    # If orchestrator routes to report, we will run it once and then stop
                    current = next_agent
                    return_to_orchestrator = True if next_agent != "report" else False
                    # Map planned_pipeline to state for UI/debug but do not follow it automatically
                    state["pipeline"] = (outputs or {}).get("planned_pipeline") or state.get("pipeline")
                    continue
                # Fallback: use declarative edges if no next_agent provided
                current = self._next_from_edges("orchestrator", state)
                return_to_orchestrator = False
                continue
            else:
                # After any agent finishes, either stop (if it was report) or go back to orchestrator
                if current == "report":
                    current = None
                    continue
                # Return to orchestrator for re-routing
                current = "orchestrator" if "orchestrator" in self.nodes else (self.spec.get("start") or None)
                return_to_orchestrator = False
                continue

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

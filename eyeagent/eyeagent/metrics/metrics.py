from __future__ import annotations
import os
import time
from contextlib import contextmanager
from typing import Optional
from loguru import logger

_PROM = None


def _maybe_init_prom():
    global _PROM
    if _PROM is not None:
        return _PROM
    try:
        from prometheus_client import Counter, Histogram
        _PROM = {
            "llm_tokens": Counter("eyeagent_llm_tokens", "LLM tokens used", ["agent", "phase"]),
            "tool_latency": Histogram("eyeagent_tool_latency_seconds", "Tool call latency", ["tool_id", "status"]),
            "step_latency": Histogram("eyeagent_step_latency_seconds", "Agent step latency", ["agent", "role"]),
        }
        logger.debug("[metrics] prometheus initialized")
    except Exception as e:
        # Optional dependency; this is informational and not an error.
        logger.debug("[metrics] prometheus optional; client not installed – skipping export")
        _PROM = {}
    return _PROM


def add_tokens(agent: str, phase: str, n: int):
    prom = _maybe_init_prom()
    c = prom.get("llm_tokens")
    if c:
        try:
            c.labels(agent=agent, phase=phase).inc(n)
        except Exception:
            pass


@contextmanager
def tool_timer(tool_id: str):
    prom = _maybe_init_prom()
    h = prom.get("tool_latency")
    start = time.time()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        dur = time.time() - start
        if h:
            try:
                h.labels(tool_id=tool_id, status=status).observe(dur)
            except Exception:
                pass


@contextmanager
def step_timer(agent: str, role: str):
    prom = _maybe_init_prom()
    h = prom.get("step_latency")
    start = time.time()
    try:
        yield
    finally:
        dur = time.time() - start
        if h:
            try:
                h.labels(agent=agent, role=role).observe(dur)
            except Exception:
                pass

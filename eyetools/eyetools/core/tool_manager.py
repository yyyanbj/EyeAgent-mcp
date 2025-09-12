"""ToolManager: orchestrates lifecycle & dispatch (inproc + subprocess).

Adds lightweight lifecycle state tracking aligned with documented phases:
DISCOVERED -> REGISTERED -> LOADED -> IDLE -> UNLOADED (evicted) -> RELOADED.
"""
from __future__ import annotations
from typing import Dict, Any, Literal
from importlib import import_module
from .registry import ToolRegistry, ToolMeta
from .process_manager import ProcessManager
from .env_manager import EnvManager
from .errors import ToolNotFoundError, PredictionError
from .logging import core_logger
import time
import sys
from pathlib import Path


LifecycleState = Literal["DISCOVERED", "REGISTERED", "LOADED", "IDLE", "UNLOADED"]


class ToolManager:
    def __init__(self, registry: ToolRegistry, workspace_root=None, max_inproc: int = 32):
        self.registry = registry
        self._inproc_instances: Dict[str, Any] = {}
        self._workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._env_manager = EnvManager(self._workspace_root)
        self._process_manager = ProcessManager(self._workspace_root, env_manager=self._env_manager)
        self._max_inproc = max_inproc
        self._inproc_lru: Dict[str, float] = {}
        # metrics: per tool id
        self._metrics: Dict[str, Dict[str, Any]] = {}
        # lifecycle states per tool id
        self._lifecycle: Dict[str, LifecycleState] = {}

    def _import_entry(self, meta: ToolMeta):
        entry = meta.entry
        module_name, class_name = entry.split(":", 1)
        try:
            mod = import_module(module_name)
        except ModuleNotFoundError:
            # attempt to add tool root_dir to sys.path and retry
            root = Path(meta.root_dir)
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            mod = import_module(module_name)
        return getattr(mod, class_name)

    def get_or_create_inproc(self, meta: ToolMeta):
        inst = self._inproc_instances.get(meta.id)
        if inst:
            # mark as loaded if previously unloaded/idle
            prev = self._lifecycle.get(meta.id)
            if prev in ("IDLE", "UNLOADED", "REGISTERED"):
                self._lifecycle[meta.id] = "LOADED"
            return inst
        core_logger.debug(f"[inproc] create instance tool_id={meta.id} entry={meta.entry}")
        ToolCls = self._import_entry(meta)
        # Pass meta dict form & params
        inst = ToolCls(meta.__dict__, meta.params)
        # prepare lazy; model loaded on first predict
        if hasattr(inst, "prepare"):
            try:
                inst.prepare()
            except Exception as e:  # noqa
                core_logger.exception(f"prepare failed tool_id={meta.id}: {e}")
                raise
        core_logger.debug(f"[inproc] ready tool_id={meta.id}")
        self._inproc_instances[meta.id] = inst
        # first time load transition
        self._lifecycle[meta.id] = "LOADED"
        return inst

    def predict(self, tool_id: str, request: Dict[str, Any]):
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        load_mode = meta.runtime.get("load_mode", "auto")
        # For now treat auto as inproc
        if load_mode in ("auto", "inproc"):
            # ensure lifecycle registration baseline
            self._lifecycle.setdefault(meta.id, "REGISTERED")
            inst = self.get_or_create_inproc(meta)
            if meta.model.get("lazy", True) and hasattr(inst, "ensure_model_loaded"):
                before_load = time.time()
                inst.ensure_model_loaded()
                core_logger.debug(f"lazy model load tool_id={meta.id} took={(time.time()-before_load)*1000:.1f}ms")
            self._inproc_lru[meta.id] = time.time()
            # LRU eviction if over limit
            if len(self._inproc_instances) > self._max_inproc:
                # evict oldest non-pinned
                oldest = sorted(self._inproc_lru.items(), key=lambda x: x[1])
                for tid, _ts in oldest:
                    if tid != meta.id:  # keep current
                        self._inproc_instances.pop(tid, None)
                        self._inproc_lru.pop(tid, None)
                        self._lifecycle[tid] = "UNLOADED"
                        break
            start = time.time()
            try:
                result = inst.predict(request)
            except Exception as e:  # noqa
                raise PredictionError(str(e)) from e
            latency = (time.time() - start) * 1000.0
            core_logger.debug(f"predict inproc tool_id={meta.id} latency_ms={latency:.2f}")
            m = self._metrics.setdefault(meta.id, {"predict_count": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0})
            m["predict_count"] += 1
            m["total_latency_ms"] += latency
            m["last_latency_ms"] = latency
            m["avg_latency_ms"] = m["total_latency_ms"] / m["predict_count"]
            # mark idle baseline (caller of maintenance can flip to IDLE later)
            self._lifecycle[meta.id] = "LOADED"
            return result
        elif load_mode == "subprocess":
            # ensure worker running & initialized
            self._process_manager.ensure_init(meta)
            start = time.time()
            resp = self._process_manager.request(meta.id, {"cmd": "PREDICT", "request": request})
            if not resp.get("ok"):
                raise PredictionError(resp.get("error"))
            latency = (time.time() - start) * 1000.0
            core_logger.debug(f"predict subprocess tool_id={meta.id} latency_ms={latency:.2f}")
            m = self._metrics.setdefault(meta.id, {"predict_count": 0, "total_latency_ms": 0.0, "last_latency_ms": 0.0})
            m["predict_count"] += 1
            m["total_latency_ms"] += latency
            m["last_latency_ms"] = latency
            m["avg_latency_ms"] = m["total_latency_ms"] / m["predict_count"]
            return resp.get("data")
        else:
            raise ValueError(f"Unknown load_mode {load_mode}")

    def lifecycle_states(self) -> Dict[str, LifecycleState]:
        """Return a snapshot of lifecycle state per tool."""
        return self._lifecycle.copy()

    def mark_idle(self, older_than_s: float = 300.0):
        """Mark LOADED tools as IDLE if last used before threshold (no unload)."""
        now = time.time()
        for tid, last_ts in list(self._inproc_lru.items()):
            if now - last_ts > older_than_s and self._lifecycle.get(tid) == "LOADED":
                self._lifecycle[tid] = "IDLE"

    def unload_idle(self, older_than_s: float = 900.0):
        """Unload IDLE tools whose last use exceeds threshold to free memory."""
        now = time.time()
        for tid, last_ts in list(self._inproc_lru.items()):
            if now - last_ts > older_than_s:
                inst = self._inproc_instances.pop(tid, None)
                if inst and hasattr(inst, "release"):
                    try:
                        inst.release()
                    except Exception:  # noqa
                        core_logger.exception(f"release failed tool_id={tid}")
                self._inproc_lru.pop(tid, None)
                self._lifecycle[tid] = "UNLOADED"

    def get_metrics(self, tool_id: str | None = None):
        """Return metrics for a single tool or all tools.
        If tool_id is None returns dict of all metrics plus aggregate summary under key '__aggregate__'."""
        if tool_id:
            return self._metrics.get(tool_id, {}).copy()
        # build aggregate
        aggregate = {"tools": len(self._metrics), "predict_total": 0, "avg_latency_ms": 0.0}
        total_latency = 0.0
        for mid, m in self._metrics.items():
            aggregate["predict_total"] += m.get("predict_count", 0)
            total_latency += m.get("total_latency_ms", 0.0)
        if aggregate["predict_total"]:
            aggregate["avg_latency_ms"] = total_latency / aggregate["predict_total"]
        all_metrics = {tid: data.copy() for tid, data in self._metrics.items()}
        all_metrics["__aggregate__"] = aggregate
        return all_metrics

__all__ = ["ToolManager"]

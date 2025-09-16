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
        # last used timestamp
        self._last_used = {}
        # telemetry collected from subprocess LOAD_MODEL responses
        self._subprocess_telemetry = {}
        # last warmup timestamps
        self._last_warmup = {}

    def resource_snapshot(self) -> Dict[str, Any]:
        """Return a lightweight resource usage snapshot.
        Includes:
          - counts: discovered, loaded_inproc, subprocess_workers
          - memory (RSS in MB) if psutil available
          - aggregate metrics summary
        GPU memory is omitted unless torch + cuda available; then reports total + allocated.
        """
        snap: Dict[str, Any] = {}
        snap["counts"] = {
            "discovered": len(self.registry.list()),
            "loaded_inproc": len(self._inproc_instances),
            "subprocess_workers": len(getattr(self._process_manager, "_workers", {})),
        }
        # attach subprocess telemetry (non-huge scalar values only)
        if getattr(self, "_subprocess_telemetry", None):
            telem_out = {}
            for k, v in self._subprocess_telemetry.items():
                if isinstance(v, dict):
                    compact = {sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str, type(None), bool))}
                    telem_out[k] = compact
            # add last warmup timestamps (seconds since epoch)
            warm = {k: self._last_warmup.get(k) for k in telem_out.keys() if k in self._last_warmup}
            snap["subprocess"] = {"telemetry": telem_out, "last_warmup": warm}
        # metrics aggregate
        agg = self.get_metrics().get("__aggregate__", {})
        snap["metrics"] = agg
        # process RSS
        try:  # pragma: no cover - psutil optional
            import psutil  # type: ignore
            proc = psutil.Process()
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            snap["memory"] = {"rss_mb": round(rss_mb, 2)}
        except Exception:  # noqa
            snap["memory"] = {"rss_mb": None}
        # GPU (best-effort)
        try:  # pragma: no cover - optional
            import torch  # type: ignore
            if torch.cuda.is_available():
                snap["gpu"] = {
                    "device_count": torch.cuda.device_count(),
                    "allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
                    "reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
                }
        except Exception:  # noqa
            pass
        return snap

    def _import_entry(self, meta: ToolMeta):
        entry = meta.entry
        module_name, class_name = entry.split(":", 1)
        # Always ensure tool root dir precedes import to stabilize relative imports
        root = Path(meta.root_dir)
        # Insert just before existing workspace root paths to keep deterministic order
        if str(root) not in sys.path:
            # place early but after cwd to avoid shadowing top-level packages
            insertion_index = 1 if sys.path and sys.path[0] == '' else 0
            sys.path.insert(insertion_index, str(root))
            core_logger.debug(
                f"[import] inserted tool root into sys.path index={insertion_index} root={root} module={module_name}"
            )
        else:
            core_logger.debug(
                f"[import] tool root already in sys.path root={root} module={module_name}"
            )
        try:
            mod = import_module(module_name)
        except ModuleNotFoundError:
            # fallback: if module name is a simple file (e.g., tool_impl) load via spec directly later
            # This branch keeps previous behavior but root already inserted.
            mod = import_module(module_name)
        if not hasattr(mod, class_name):
            # Fallback: load by explicit file path (avoid name collision like multiple tool_impl.py on sys.path)
            candidate = Path(meta.root_dir) / f"{module_name}.py"
            if candidate.exists():
                import importlib.util as _ilu
                safe_name = f"eyetools_dyn_{meta.id.replace(':','_').replace('-','_')}"
                spec = _ilu.spec_from_file_location(safe_name, str(candidate))
                if spec and spec.loader:
                    dyn_mod = _ilu.module_from_spec(spec)
                    sys.modules[safe_name] = dyn_mod
                    spec.loader.exec_module(dyn_mod)  # type: ignore
                    if hasattr(dyn_mod, class_name):
                        return getattr(dyn_mod, class_name)
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
            self._last_used[meta.id] = self._inproc_lru[meta.id]
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

    # ------------------------------------------------------------------
    # Manual warmup
    def warmup_tool(self, tool_id: str) -> Dict[str, Any]:
        """Manually warm up a tool.

        Subprocess: send WARMUP command (worker handles absence gracefully).
        Inproc: call .warmup() if available else ensure_model_loaded().
        Returns status + optional telemetry.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        mode = meta.runtime.get("load_mode", "auto")
        result: Dict[str, Any] = {"tool_id": tool_id, "runtime": mode}
        try:
            if mode == "subprocess":
                self._process_manager.ensure_init(meta)
                resp = self._process_manager.request(tool_id, {"cmd": "WARMUP"}, timeout=600)
                if resp.get("ok"):
                    data = resp.get("data") or {}
                    if isinstance(data, dict):
                        tele = self._subprocess_telemetry.setdefault(tool_id, {})
                        tele.update({k: v for k, v in data.items() if isinstance(v, (int, float, str, type(None), bool))})
                        self._last_warmup[tool_id] = time.time()
                    result.update({"status": "ok", "warmup": data})
                else:
                    result.update({"status": "error", "error": resp.get("error")})
            else:  # inproc
                inst = self.get_or_create_inproc(meta)
                tele: Dict[str, Any] = {}
                if hasattr(inst, "warmup"):
                    try:
                        data = inst.warmup()  # type: ignore
                        if isinstance(data, dict):
                            tele.update({k: v for k, v in data.items() if isinstance(v, (int, float, str, type(None), bool))})
                            self._last_warmup[tool_id] = time.time()
                    except Exception as e:  # noqa
                        result.update({"status": "error", "error": str(e)})
                        return result
                else:
                    if hasattr(inst, "ensure_model_loaded"):
                        inst.ensure_model_loaded()
                    tele["warmed_up"] = True
                    self._last_warmup[tool_id] = time.time()
                result.update({"status": "ok", "warmup": tele})
        except Exception as e:  # noqa
            result.update({"status": "error", "error": str(e)})
        return result

    def warmup_all(self, runtime: str | None = None) -> Dict[str, Any]:
        """Warm up all tools optionally filtered by runtime ('subprocess' or 'inproc')."""
        results = []
        for meta in self.registry.list():
            rmode = meta.runtime.get("load_mode", "auto")
            if runtime and runtime != rmode and not (runtime == "inproc" and rmode in ("auto", "inproc")):
                continue
            results.append(self.warmup_tool(meta.id))
        return {"count": len(results), "results": results}

    def lifecycle_states(self) -> Dict[str, LifecycleState]:
        """Return a snapshot of lifecycle state per tool."""
        return self._lifecycle.copy()

    def lifecycle_states_detailed(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for tid, state in self._lifecycle.items():
            out[tid] = {
                "state": state,
                "last_used": self._last_used.get(tid),
                "loaded": tid in self._inproc_instances,
            }
        return out

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

    def unload(self, tool_id: str) -> bool:
        """Explicitly unload a single inproc tool. Returns True if unloaded."""
        inst = self._inproc_instances.pop(tool_id, None)
        if not inst:
            return False
        if hasattr(inst, "release"):
            try:
                inst.release()
            except Exception:  # noqa
                core_logger.exception(f"release failed tool_id={tool_id}")
        self._inproc_lru.pop(tool_id, None)
        self._lifecycle[tool_id] = "UNLOADED"
        return True

    def unload_any(self, tool_id: str) -> bool:
        """Unload either an inproc tool or a subprocess worker.

        Returns True if something was unloaded/terminated, False otherwise.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            return False
        mode = meta.runtime.get("load_mode", "auto")
        if mode in ("auto", "inproc"):
            return self.unload(tool_id)
        elif mode == "subprocess":
            # terminate worker if running
            workers = getattr(self._process_manager, "_workers", {})
            if tool_id in workers:
                try:
                    self._process_manager.stop(tool_id)
                except Exception:  # noqa
                    core_logger.exception(f"stop worker failed tool_id={tool_id}")
                self._lifecycle[tool_id] = "UNLOADED"
                return True
            return False
        return False

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

    def ensure_loaded(self, tool_id: str, include_model: bool = True):
        """Ensure a tool is loaded/initialized based on its load_mode.

        For inproc/auto: instantiate (if needed) and optionally force lazy model load.
        For subprocess: spawn worker & perform INIT (model load is deferred unless protocol extends).
        Returns dict with {mode, tool_id, status}.
        """
        meta = self.registry.get(tool_id)
        if not meta:
            raise ToolNotFoundError(f"Tool {tool_id} not found")
        mode = meta.runtime.get("load_mode", "auto")
        if mode in ("auto", "inproc"):
            inst = self.get_or_create_inproc(meta)
            if include_model and meta.model.get("lazy", True) and hasattr(inst, "ensure_model_loaded"):
                inst.ensure_model_loaded()
            return {"tool_id": tool_id, "mode": mode, "status": "loaded"}
        elif mode == "subprocess":
            self._process_manager.ensure_init(meta)
            # Optionally could add LOAD_MODEL command in future if protocol extended
            self._lifecycle.setdefault(tool_id, "REGISTERED")
            self._lifecycle[tool_id] = "LOADED"
            return {"tool_id": tool_id, "mode": mode, "status": "worker_started"}
        else:
            raise ValueError(f"Unknown load_mode {mode}")

    def maintenance(self, mark_idle_s: float | None = None, unload_idle_s: float | None = None) -> Dict[str, Any]:
        """Run maintenance operations (mark idle, unload idle) and return a summary.

        mark_idle_s: if provided, tools not used for > seconds are transitioned LOADED->IDLE.
        unload_idle_s: if provided, tools not used for > seconds are fully unloaded (inproc) or subprocess workers stopped.
        Note: subprocess worker auto-stop currently not time-based here (could be extended); only inproc unloads.
        """
        before_states = self.lifecycle_states().copy()
        now = time.time()
        actions: Dict[str, Any] = {"marked_idle": [], "unloaded": []}
        if mark_idle_s is not None:
            # replicate mark_idle logic with explicit threshold
            for tid, last_ts in list(self._inproc_lru.items()):
                if (now - last_ts) > mark_idle_s and self._lifecycle.get(tid) == "LOADED":
                    self._lifecycle[tid] = "IDLE"
                    actions["marked_idle"].append(tid)
        if unload_idle_s is not None:
            for tid, last_ts in list(self._inproc_lru.items()):
                if (now - last_ts) > unload_idle_s:
                    inst = self._inproc_instances.pop(tid, None)
                    if inst and hasattr(inst, "release"):
                        try:
                            inst.release()
                        except Exception:  # noqa
                            core_logger.exception(f"release failed tool_id={tid}")
                    self._inproc_lru.pop(tid, None)
                    self._lifecycle[tid] = "UNLOADED"
                    actions["unloaded"].append(tid)
        after_states = self.lifecycle_states().copy()
        return {"before": before_states, "after": after_states, "actions": actions}

    # ------------------------------------------------------------------
    def preload_all(self, include_subprocess: bool = False, parallel_subprocess: bool = False, max_workers: int = 4):
        """Instantiate all tools and (if lazy) load their models.

        include_subprocess: if True also spin up subprocess tools (may be expensive).
        For subprocess tools we currently only INIT the worker; model weights may still
        be lazily loaded on first PREDICT unless/until a future LOAD_MODEL protocol
        extension is issued. (A placeholder will be added if handler exists.)
        Exceptions during individual tool preload are logged and skipped so one
        broken tool doesn't block server startup.
        """
        from .logging import core_logger as _logger
        count = 0
        subprocess_metas = []
        regular_metas = []
        for meta in self.registry.list():
            if meta.runtime.get("load_mode", "auto") == "subprocess":
                subprocess_metas.append(meta)
            else:
                regular_metas.append(meta)

        # Optionally handle subprocess tools in parallel
        if include_subprocess and parallel_subprocess and subprocess_metas:
            from concurrent.futures import ThreadPoolExecutor, as_completed  # pragma: no cover (timing sensitive)

            def _load_subproc(meta):
                from .logging import core_logger as _plog
                try:
                    wi = self._process_manager.ensure_init(meta)
                    self._lifecycle[meta.id] = "LOADED"
                    try:
                        resp = self._process_manager.request(meta.id, {"cmd": "LOAD_MODEL"}, timeout=300)
                        if resp.get("ok") is False:
                            _plog.warning("[preload] LOAD_MODEL failed tool_id=%s resp=%s", meta.id, resp)
                        else:
                            data = resp.get("data") or {}
                            if isinstance(data, dict):
                                loaded = data.get("loaded", True)
                                if loaded:
                                    self._subprocess_telemetry[meta.id] = data
                                    _plog.info(
                                        "[preload] subprocess model loaded tool_id=%s (parallel) cuda=%s device=%s mem_delta=%s reserved_delta=%s",
                                        meta.id,
                                        data.get("cuda"),
                                        data.get("device_name"),
                                        data.get("mem_delta"),
                                        data.get("reserved_delta"),
                                    )
                                    if data.get("cuda") and (data.get("mem_delta") in (0, None)):
                                        _plog.warning(
                                            "[preload] tool_id=%s CUDA loaded but mem_delta=%s (may allocate later)",
                                            meta.id,
                                            data.get("mem_delta"),
                                        )
                                else:
                                    _plog.warning("[preload] subprocess model NOT loaded tool_id=%s", meta.id)
                            else:
                                _plog.info("[preload] subprocess model loaded tool_id=%s (parallel unstructured)", meta.id)
                    except Exception as e:  # noqa
                        _plog.debug("[preload] subprocess model lazy-load later tool_id=%s error=%s", meta.id, e)
                    return True
                except Exception as e:  # noqa
                    _plog.exception("[preload] worker init failed tool_id=%s error=%s", meta.id, e)
                    return False

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_load_subproc, m): m for m in subprocess_metas}
                for f in as_completed(futures):  # pragma: no cover
                    if f.result():
                        count += 1

        # Sequential handling (remaining subprocess if not parallel, and all regular)
        seq_list = []
        if include_subprocess and not parallel_subprocess:
            seq_list.extend(subprocess_metas)
        seq_list.extend(regular_metas)
        for meta in seq_list:
            mode = meta.runtime.get("load_mode", "auto")
            if mode == "subprocess":
                if not include_subprocess:
                    _logger.info("[preload] skip subprocess tool_id=%s (include_subprocess=False)", meta.id)
                    continue
                # start worker (INIT)
                try:
                    wi = self._process_manager.ensure_init(meta)
                    self._lifecycle[meta.id] = "LOADED"
                    # Attempt forced model load if protocol supports it
                    try:  # pragma: no cover (depends on model availability)
                        resp = self._process_manager.request(meta.id, {"cmd": "LOAD_MODEL"}, timeout=300)
                        if resp.get("ok") is False:
                            _logger.warning("[preload] LOAD_MODEL failed tool_id=%s resp=%s", meta.id, resp)
                        else:
                            data = resp.get("data") or {}
                            if isinstance(data, dict):
                                loaded = data.get("loaded", True)
                                cuda_flag = data.get("cuda")
                                device_name = data.get("device_name")
                                mem_delta = data.get("mem_delta")
                                mem_before = data.get("mem_before")
                                mem_after = data.get("mem_after")
                                if loaded:
                                    # store telemetry
                                    self._subprocess_telemetry[meta.id] = data
                                    _logger.info(
                                        "[preload] subprocess model loaded tool_id=%s cuda=%s device=%s mem_delta=%s bytes(before=%s after=%s) reserved_delta=%s first_param_device=%s params=%s cuda_param_bytes=%s",
                                        meta.id,
                                        cuda_flag,
                                        device_name,
                                        mem_delta,
                                        mem_before,
                                        mem_after,
                                        data.get("reserved_delta"),
                                        data.get("first_param_device"),
                                        data.get("param_count"),
                                        data.get("cuda_param_bytes"),
                                    )
                                    if cuda_flag and (mem_delta is None or (isinstance(mem_delta, (int,float)) and mem_delta <= 0)):
                                        _logger.warning(
                                            "[preload] tool_id=%s reported loaded on CUDA but mem_delta=%s (may allocate later during first predict)",
                                            meta.id,
                                            mem_delta,
                                        )
                                else:
                                    _logger.warning("[preload] subprocess model NOT loaded (weights missing?) tool_id=%s", meta.id)
                            else:
                                _logger.info("[preload] subprocess model loaded tool_id=%s (unstructured data)", meta.id)
                    except Exception as e:  # noqa
                        _logger.debug("[preload] subprocess model lazy-load will occur on first predict tool_id=%s error=%s", meta.id, e)
                    count += 1
                except Exception as e:  # noqa
                    _logger.exception("[preload] worker init failed tool_id=%s error=%s", meta.id, e)
                continue
            try:
                inst = self.get_or_create_inproc(meta) if mode in ("auto", "inproc") else None
                if inst and meta.model.get("lazy", True) and hasattr(inst, "ensure_model_loaded"):
                    inst.ensure_model_loaded()
                # Optional warmup request
                warm = meta.warmup.get("request") if hasattr(meta, "warmup") else None
                if warm and inst:
                    try:
                        inst.predict(warm)
                    except Exception:  # pragma: no cover
                        _logger.exception("[preload] warmup failed tool_id=%s", meta.id)
                count += 1
            except Exception as e:  # noqa
                _logger.exception("[preload] failed tool_id=%s error=%s", meta.id, e)
        _logger.info("[preload] completed count=%d parallel_subprocess=%s", count, parallel_subprocess)
        return count

__all__ = ["ToolManager"]

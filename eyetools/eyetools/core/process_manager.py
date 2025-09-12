"""Process manager for out-of-process tool workers (with uv support)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import subprocess
import sys
import threading
import json
import time
from pathlib import Path
from .env_manager import EnvManager
from .errors import WorkerInitError, EnvironmentError
from .logging import core_logger


@dataclass
class WorkerInfo:
    tool_id: str
    process: subprocess.Popen
    started: float
    last_used: float
    status: str = "starting"


class ProcessManager:
    def __init__(self, workspace_root: Path, env_manager: Optional[EnvManager] = None):
        self.workspace_root = workspace_root
        self.env_manager = env_manager or EnvManager(workspace_root)
        self._workers: Dict[str, WorkerInfo] = {}
        self._init_done: Dict[str, bool] = {}
        self._lock = threading.RLock()

    def spawn(self, tool_id: str, meta=None):
        with self._lock:
            if tool_id in self._workers:
                return self._workers[tool_id]
            # Build uv command if python/extra requires specified
            entrypoint: List[str]
            if meta and (getattr(meta, 'python', None) or getattr(meta, 'extra_requires', None)):
                extra = meta.extra_requires or []
                with_arg = ",".join(extra) if extra else None
                entrypoint = ["uv", "run"]
                if with_arg:
                    entrypoint += ["--with", with_arg]
                if getattr(meta, 'python', None):
                    entrypoint += [f"--python={meta.python}"]
                entrypoint += ["-m", "eyetools.core.worker_entry"]
            else:
                entrypoint = [sys.executable, "-m", "eyetools.core.worker_entry"]
            core_logger.debug(f"spawn worker tool_id={tool_id} cmd={' '.join(entrypoint)}")
            try:
                proc = subprocess.Popen(entrypoint, cwd=str(self.workspace_root), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            except Exception as e:  # noqa
                raise EnvironmentError(f"Failed spawning worker for {tool_id}: {e}") from e
            wi = WorkerInfo(tool_id=tool_id, process=proc, started=time.time(), last_used=time.time())
            self._workers[tool_id] = wi
            return wi

    def _send(self, wi: WorkerInfo, payload: Dict):
        line = json.dumps(payload) + "\n"
        assert wi.process.stdin is not None
        wi.process.stdin.write(line)
        wi.process.stdin.flush()

    def _recv(self, wi: WorkerInfo, timeout: float = 30.0) -> Dict:
        assert wi.process.stdout is not None
        start = time.time()
        while True:
            if (time.time() - start) > timeout:
                raise TimeoutError("Timeout waiting for worker response")
            line = wi.process.stdout.readline()
            if line:
                return json.loads(line)
            time.sleep(0.01)

    def request(self, tool_id: str, payload: Dict, timeout: float = 30.0) -> Dict:
        wi = self._workers.get(tool_id)
        if not wi:
            raise WorkerInitError(f"Worker for {tool_id} not running")
        self._send(wi, payload)
        wi.last_used = time.time()
        return self._recv(wi, timeout)

    def ensure_init(self, tool_meta):
        wi = self.spawn(tool_meta.id, meta=tool_meta)
        if not self._init_done.get(tool_meta.id):
            core_logger.debug(f"init worker tool_id={tool_meta.id}")
            resp = self.request(tool_meta.id, {"cmd": "INIT", "meta": tool_meta.__dict__})
            if not resp.get("ok", False):
                raise WorkerInitError(f"INIT failed: {resp}")
            self._init_done[tool_meta.id] = True
        return wi

    def stop(self, tool_id: str):
        wi = self._workers.pop(tool_id, None)
        if wi and wi.process.poll() is None:
            try:
                self.request(tool_id, {"cmd": "SHUTDOWN"}, timeout=5)
            except Exception:  # noqa: BLE001
                pass
            wi.process.terminate()

    def cleanup_idle(self, idle_seconds: float):
        now = time.time()
        to_stop = []
        for tid, wi in self._workers.items():
            if (now - wi.last_used) > idle_seconds:
                to_stop.append(tid)
        for tid in to_stop:
            self.stop(tid)

__all__ = ["ProcessManager", "WorkerInfo"]

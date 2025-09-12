"""New minimal ToolBase abstraction (process-agnostic)."""
from __future__ import annotations
from typing import Any, Dict
import time


class ToolBase:
    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        self.meta = meta
        self.params = params or {}
        self._model_loaded = False
        self._load_started = False

    # Lifecycle hooks -------------------------------------------------
    def prepare(self):
        """Lightweight initialization (parse params, path checks)."""
        return True

    def load_model(self):
        """Heavy model load. Override in subclass."""
        self._model_loaded = True

    def predict(self, request: Dict[str, Any]):  # pragma: no cover - base
        raise NotImplementedError

    def shutdown(self):  # pragma: no cover - base
        pass

    # Helper wrappers -------------------------------------------------
    def ensure_model_loaded(self):
        if not self._model_loaded:
            if not self._load_started:
                self._load_started = True
                start = time.time()
                self.load_model()
                self.meta.setdefault("metrics", {})["model_load_seconds"] = round(time.time() - start, 3)
        return True

__all__ = ["ToolBase"]

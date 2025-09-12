import time
from typing import Any, Dict

# Placeholder for future centralized ToolBase (new core). For now simple class.
class DemoTemplateTool:
    """Demo template tool implementation.

    Methods here mirror the planned ToolBase API subset so the framework
    can later swap in the real base class without changing template logic.
    """
    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        self.meta = meta
        self.params = params or {}
        self.model_loaded = False
        self._model_obj = None

    def prepare(self):
        # lightweight checks
        return True

    def load_model(self):
        if self.model_loaded:
            return
        # Simulate heavy load
        time.sleep(0.1)
        self._model_obj = {"loaded": True, "weights": self.meta.get("model", {}).get("weights")}
        self.model_loaded = True

    def predict(self, request: Dict[str, Any]):
        if self.meta.get("model", {}).get("lazy", True) and not self.model_loaded:
            self.load_model()
        prefix = self.params.get("msg_prefix", "DEMO")
        inp = request.get("inputs", {}).get("input")
        return {
            "status": "ok",
            "outputs": {"message": f"[{prefix}] processed {type(inp).__name__}"},
            "meta": {"tool_id": self.meta.get("id"), "model_loaded": self.model_loaded},
        }

    def shutdown(self):
        self._model_obj = None
        self.model_loaded = False

__all__ = ["DemoTemplateTool"]

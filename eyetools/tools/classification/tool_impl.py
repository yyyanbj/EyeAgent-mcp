from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import torch, torch.nn as nn
from torchvision import transforms
from PIL import Image

# Support execution when the classification directory is added directly to sys.path
try:  # normal package relative imports
    from .class_lists import TASK_CLASS_MAP, MULTIDIS_SIGNS, DEFAULT_SHOW_N  # type: ignore
    from .model_factory import create_model  # type: ignore
except Exception:  # noqa
    # Fallback to absolute module names if user appended root dir of tool to sys.path
    try:
        from class_lists import TASK_CLASS_MAP, MULTIDIS_SIGNS, DEFAULT_SHOW_N  # type: ignore
        from model_factory import create_model  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(f"Failed to import classification dependencies: {e}")

@dataclass
class ToolVariantMeta:
    task: str
    threshold: float = 0.3


class ClassificationTool:
    """Unified constructor signature: (meta_dict, params_dict)

    This aligns with ToolManager expectations so preload works.
    Backwards-compatible helper load_tool still returns an initialized instance.
    """

    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        # meta/params stored
        self.meta = meta
        self.params = params or {}
        # Resolve task/threshold from params or meta
        self.task = self.params.get("task") or meta.get("params", {}).get("task")
        if not self.task:
            raise ValueError("ClassificationTool requires 'task' in params")
        self.threshold = float(self.params.get("threshold", meta.get("params", {}).get("threshold", 0.3)))
        self.weights_root = self.params.get("weights_root", "weights/classification")
        self.device = torch.device(self.params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.img_size = 224
        self.transform = None
        self._model_loaded = False

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Return static output description for this tool variant.

        Does not load any weights. Provides categories and field explanations.
        """
        try:
            from .class_lists import TASK_CLASS_MAP  # type: ignore
        except Exception:  # pragma: no cover
            TASK_CLASS_MAP = {}
        task = (params or {}).get("task") or (meta.get("params", {}) or {}).get("task")
        cats = TASK_CLASS_MAP.get(task)
        out: Dict[str, Any] = {"schema": (meta.get("io") or {}).get("output_schema", {})}
        if task == "cfp_age":
            out["fields"] = {
                "prediction": "predicted age (years)",
                "unit": "years",
                "inference_time": "seconds",
            }
        else:
            out["fields"] = {
                "predictions": "top-N predicted categories",
                "probabilities": "map of category -> score",
                "inference_time": "seconds",
            }
            if cats:
                out["categories"] = cats
        return out

    # For consistency with ToolBase style
    def ensure_model_loaded(self):
        if self._model_loaded:
            return
        self._load_model_internal()

    def _load_model_internal(self):
        self.classes = TASK_CLASS_MAP[self.task]
        self.model, self.img_size = create_model(self.task, self.classes, self.weights_root)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        self._model_loaded = True

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, request: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        # Backward compatibility: if a string path is passed treat as direct image_path
        if isinstance(request, str):
            image_path = request
        else:
            # Accept formats: {"inputs":{"image_path":...}} or {"image_path":...}
            inputs = request.get("inputs") if isinstance(request, dict) else None
            image_path = (inputs or request).get("image_path") if isinstance(request, dict) else None
        if not image_path:
            raise ValueError("image_path missing for classification predict")
        self.ensure_model_loaded()
        t = self._load_image(image_path)
        start = time.time()
        with torch.no_grad():
            logits = self.model(t)
        dur = time.time() - start
        if self.task == "cfp_age":
            val = logits.squeeze().cpu().item()
            val = max(40, min(val, 70))
            return {"task": self.task, "prediction": val, "unit": "years", "inference_time": round(dur,4)}
        multilabel = self.task == "multidis"
        # CSRA outputs raw logits; for single-label tasks (non-multidis) apply softmax
        probs = (torch.sigmoid(logits) if multilabel else torch.softmax(logits, dim=1)).squeeze().cpu().tolist()
        pairs = list(zip(self.classes, probs))
        if multilabel:
            sign_set = set(MULTIDIS_SIGNS.get("sign", []))
            pairs = [p for p in pairs if p[0] not in sign_set]
            filtered = [p for p in pairs if p[1] >= self.threshold]
            if len(filtered) < DEFAULT_SHOW_N[self.task]:
                filtered = sorted(pairs, key=lambda x: x[1], reverse=True)[: DEFAULT_SHOW_N[self.task]]
        else:
            filtered = sorted(pairs, key=lambda x: x[1], reverse=True)[: DEFAULT_SHOW_N[self.task]]
        return {"task": self.task, "predictions": [p[0] for p in filtered], "probabilities": {k: round(float(v),3) for k,v in filtered}, "inference_time": round(dur,4)}

def load_tool(task: str, threshold: float = 0.3, **kw):
    """Backward-compatible factory for direct usage in tests / scripts."""
    if task not in TASK_CLASS_MAP:
        raise ValueError(f"Unsupported task {task}")
    meta = {"id": f"classification:{task}", "entry": "tool_impl:ClassificationTool", "version": "0.1.0", "params": {"task": task, "threshold": threshold}}
    params = {"task": task, "threshold": threshold, **kw}
    tool = ClassificationTool(meta, params)
    tool.ensure_model_loaded()
    return tool

__all__ = ["ClassificationTool", "load_tool"]
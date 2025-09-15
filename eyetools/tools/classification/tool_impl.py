from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
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
    def __init__(self, variant: ToolVariantMeta, weights_root: str = "weights/classification", device: Optional[str] = None):
        self.task = variant.task
        self.threshold = variant.threshold
        self.classes = TASK_CLASS_MAP[self.task]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.img_size = create_model(self.task, self.classes, weights_root)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, image_path: str) -> Dict[str, Any]:
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
    if task not in TASK_CLASS_MAP:
        raise ValueError(f"Unsupported task {task}")
    return ClassificationTool(ToolVariantMeta(task=task, threshold=threshold), **kw)

__all__ = ["ClassificationTool", "load_tool"]
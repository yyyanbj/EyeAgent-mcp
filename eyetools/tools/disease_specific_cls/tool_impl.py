from __future__ import annotations
"""Disease-specific classification tool (RETFound fine-tuned Dinov3 ViT-B/16).

Each variant corresponds to a folder under `weights/disease-specific/` containing a
`checkpoint-best.pth` (training script output). We dynamically inspect the
checkpoint head to determine number of output classes (binary vs multi-class).

Prediction contract (minimal):
    inputs: {"image_path": str}
    outputs: {
        "disease": <disease_name>,
        "probability": float,            # probability of positive class
        "predicted": bool,               # probability >= threshold
        "all_probabilities": {class: prob, ...},
        "inference_time": seconds
    }

Assumptions:
 1. Binary tasks dominate; if model head has output dim == 1 we apply sigmoid.
 2. If output dim > 1 we apply softmax and treat the positive disease class as
    the non-"normal" / non-"control" class if present, else the max probability.
 3. Dinov3 backbone weights are embedded inside the checkpoint. If torch.hub
    loading of upstream repository is unavailable offline, we will attempt a
    lightweight ViT-B/16 timm fallback to host the state dict keys that match
    (best effort). Any missing keys will be ignored (strict=False).
"""

import os
import time
import logging
import inspect
import argparse
import re
from typing import Any, Dict, Union, List

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


def _disease_from_variant(variant: str) -> str:
    # strip training mode suffixes
    base = variant
    for suf in ("_finetune", "_lp"):
        if base.endswith(suf):
            base = base[:-len(suf)]
            break
    return base


def _read_modalities(weights_root: str, variant: str) -> List[str] | None:
    """Read modalities.txt for a variant directory.

    Accepts either one modality per line, or a single line with comma / semicolon / pipe separated values.
    Returns None if file doesn't exist or parsing yields no tokens.
    """
    path = os.path.join(weights_root, variant, "modalities.txt")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            return None
        if len(lines) == 1 and re.search(r"[,;|]", lines[0]):
            # split by common delimiters
            tokens = [t.strip() for t in re.split(r"[,;|]", lines[0]) if t.strip()]
            lines = tokens or lines
        return lines or None
    except Exception:  # pragma: no cover
        return None


def _build_dinov3_head(num_classes: int, skip_hub: bool = False) -> nn.Module:
    """Create a ViT-B/16 backbone and attach linear head.

    We first attempt to load official dinov3 via torch.hub. If it fails we fall
    back to a timm ViT-B/16 (different pretraining) just to allow state dict
    partial load; performance may be reduced but tool remains usable.
    """
    model = None
    errors: List[str] = []
    # attempt hub unless skipped
    if not skip_hub:
        try:  # pragma: no cover (network dependent)
            model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",
                model="dinov3_vitb16",
                pretrained=False,
                trust_repo=True,
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"hub:{e}")
    if model is None:
        try:
            import timm  # type: ignore
            model = timm.create_model("vit_base_patch16_224", pretrained=False)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to construct Dinov3 fallback model: {';'.join(errors)}; timm:{e}")
    # Replace head
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    if feat_dim is None:
        # try common attribute for timm
        feat_dim = getattr(model, "head", None)
    model.head = nn.Linear(int(feat_dim), num_classes)
    return model


logger = logging.getLogger(__name__)


class DiseaseSpecificClassificationTool:
    def __init__(self, meta: Dict[str, Any], params: Dict[str, Any]):
        self.meta = meta
        self.params = params or {}
        self.variant: str = params.get("variant") or meta.get("variant")
        if not self.variant:
            raise ValueError("variant parameter required (e.g. 'AMD_finetune')")
        self.threshold: float = float(params.get("threshold", 0.5))
        self.weights_root = params.get("weights_root", "weights/disease-specific")
        self.device = torch.device(params.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ckpt_path = os.path.join(self.weights_root, self.variant, "checkpoint-best.pth")
        self.allow_missing = bool(params.get("allow_missing", True))
        self.skip_hub = bool(params.get("skip_hub", False))
        # Security toggle: allow unsafe pickle execution path if needed.
        # Only set this True if you fully trust the checkpoint source.
        self.allow_unsafe_checkpoint = bool(params.get("allow_unsafe_checkpoint", False))
        # only raise if strictly required
        if not os.path.isfile(self.ckpt_path) and not self.allow_missing:
            raise FileNotFoundError(f"Checkpoint not found: {self.ckpt_path}")
        self.model: nn.Module | None = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.classes: List[str] | None = None
        self.modalities: List[str] | None = None
        self.disease_name = _disease_from_variant(self.variant)
        self._loaded = False

    @staticmethod
    def describe_outputs(meta: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a static description of outputs for this disease-specific variant.

        - Does not load model weights.
        - Reads label.txt if present to list class names; otherwise provides a
          best-effort default based on the variant name.
        """
        variant = (params or {}).get("variant") or meta.get("variant")
        disease_name = _disease_from_variant(variant) if variant else None
        weights_root = (params or {}).get("weights_root", "weights/disease-specific")
        classes: List[str] | None = None
        modalities: List[str] | None = None
        if variant:
            label_file = os.path.join(weights_root, variant, "label.txt")
            if os.path.isfile(label_file):
                try:
                    with open(label_file, "r", encoding="utf-8") as lf:
                        raw_lines = [l.strip() for l in lf.readlines() if l.strip()]
                    if raw_lines and raw_lines[0].isdigit() and int(raw_lines[0]) == len(raw_lines) - 1:
                        raw_lines = raw_lines[1:]
                    parsed: List[str] = []
                    for entry in raw_lines:
                        part = entry.split(",") if "," in entry else entry.split("\t")
                        if len(part) >= 1 and part[0].strip():
                            parsed.append(part[0].strip())
                    if parsed:
                        classes = parsed
                except Exception:
                    classes = None
        # Minimal fields/scheme description
        out: Dict[str, Any] = {
            "schema": (meta.get("io") or {}).get("output_schema", {}),
            "fields": {
                "disease": "disease name (derived from variant)",
                "probability": "probability of positive class",
                "predicted": "probability >= threshold",
                "all_probabilities": "map of class -> probability",
                "probabilities": "alias of all_probabilities (map of class -> probability)",
                "prediction": "predicted label (top-1)",
                "label": "alias of prediction",
                "inference_time": "seconds",
            },
        }
        if disease_name:
            out["disease"] = disease_name
        if classes:
            out["classes"] = classes
        if variant:
            modalities = _read_modalities(weights_root, variant)
        if modalities:
            out["modalities"] = modalities
        return out

    # Compatibility with framework pattern
    def ensure_model_loaded(self):
        if not self._loaded:
            self._load_model()

    def _load_model(self):
        if not os.path.isfile(self.ckpt_path):
            # graceful path: fabricate 1-class head for tests / discovery
            self.model = _build_dinov3_head(1, skip_hub=self.skip_hub)
            self.classes = [self.disease_name]
            self.model.to(self.device).eval()
            # Load modalities metadata (graceful if absent)
            self.modalities = _read_modalities(self.weights_root, self.variant)
            self._loaded = True
            return
        # attempt to read explicit label list (one per line) if present
        label_file = os.path.join(self.weights_root, self.variant, "label.txt")
        explicit_labels: List[str] | None = None
        if os.path.isfile(label_file):
            try:
                with open(label_file, "r", encoding="utf-8") as lf:
                    raw_lines = [l.strip() for l in lf.readlines() if l.strip()]
                lines: List[str] = []
                # Detect numeric first line pattern
                if raw_lines and raw_lines[0].isdigit() and int(raw_lines[0]) == len(raw_lines) - 1:
                    candidate = raw_lines[1:]
                else:
                    candidate = raw_lines
                # Support formats:
                # 1) "label" (one per line)
                # 2) "label,index"
                # 3) "label\tindex"
                for entry in candidate:
                    part = entry.split(",") if "," in entry else entry.split("\t")
                    if len(part) >= 1:
                        label_name = part[0].strip()
                        if label_name:
                            lines.append(label_name)
                if lines:
                    explicit_labels = lines
            except Exception:  # pragma: no cover
                explicit_labels = None
        state = self._load_checkpoint_multi_strategy(self.ckpt_path)
        # Accept typical wrappers
        if isinstance(state, dict) and "model" in state:
            model_state = state.get("model") or state
        elif isinstance(state, dict) and "state_dict" in state:
            model_state = state["state_dict"]
        else:
            model_state = state
        # Detect classifier head size by scanning keys ending with weight
        head_keys = [k for k in model_state.keys() if "head" in k and k.endswith("weight")]
        num_classes = None
        for k in head_keys:
            w = model_state[k]
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                num_classes = w.shape[0]
                break
        if num_classes is None:
            num_classes = 1  # default to binary logistic
        self.model = _build_dinov3_head(num_classes, skip_hub=self.skip_hub)
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        # Derive classes list
        if explicit_labels:
            if len(explicit_labels) == num_classes:
                self.classes = explicit_labels
            else:
                # Warn about mismatch between label file and checkpoint head
                logger.warning(
                    "Label file class count (%d) does not match checkpoint head (%d) for variant %s (ckpt=%s). Truncating or padding heuristically.",
                    len(explicit_labels), num_classes, self.variant, self.ckpt_path,
                )
                # Truncate if label list longer; if shorter, extend with generic names
                if len(explicit_labels) > num_classes:
                    self.classes = explicit_labels[:num_classes]
                else:  # pad
                    padded = explicit_labels + [f"cls_{i}" for i in range(len(explicit_labels), num_classes)]
                    self.classes = padded
        else:
            if num_classes == 1:
                self.classes = [self.disease_name]
            elif num_classes == 2:
                self.classes = [self.disease_name, "normal"]
            else:
                self.classes = [f"cls_{i}" for i in range(num_classes)]
        self.model.to(self.device).eval()
        # Attach modalities metadata once variant directory confirmed
        self.modalities = _read_modalities(self.weights_root, self.variant)
        self._loaded = True

    # ------------------------------------------------------------------
    # Robust checkpoint loading (handles PyTorch 2.6+ weights_only=True default)
    # ------------------------------------------------------------------
    def _torch_load_supports(self, param: str) -> bool:
        try:
            sig = inspect.signature(torch.load)
            return param in sig.parameters
        except Exception:  # pragma: no cover
            return False

    def _attempt_torch_load(self, path: str, *, weights_only: bool | None) -> Any:
        kwargs = {"map_location": "cpu"}
        if weights_only is not None and self._torch_load_supports("weights_only"):
            kwargs["weights_only"] = weights_only
        try:
            return torch.load(path, **kwargs)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            raise e

    def _load_checkpoint_multi_strategy(self, path: str) -> Any:
        """Attempt secure-first loading with fallbacks.

        Strategy order:
          1. Try default (which on PyTorch>=2.6 is weights_only=True).
          2. If fails with unsupported global referring to argparse.Namespace, add it to safe globals and retry weights_only=True.
          3. If still fails AND user explicitly allows unsafe, retry with weights_only=False (pickle execution risk) and log warning.
        """
        # Step 1: initial attempt
        try:
            return self._attempt_torch_load(path, weights_only=None)  # defer to framework default
        except Exception as e1:  # noqa: BLE001
            msg = str(e1)
            insecure_needed = "Weights only load failed" in msg or "Unsupported global" in msg
            if not insecure_needed:
                # unrelated error; propagate
                raise
            # Step 2: add safe globals for argparse.Namespace if mentioned
            try:
                if "argparse.Namespace" in msg:
                    try:
                        from torch.serialization import add_safe_globals  # type: ignore
                        add_safe_globals([argparse.Namespace])  # type: ignore[arg-type]
                    except Exception:  # pragma: no cover
                        pass
                # retry with explicit weights_only=True if supported
                return self._attempt_torch_load(path, weights_only=True)
            except Exception as e2:  # noqa: BLE001
                # Step 3: optional unsafe fallback
                if self.allow_unsafe_checkpoint:
                    logger.warning(
                        "Unsafe checkpoint load fallback engaged (weights_only=False) for %s due to: %s. Ensure you trust this file.",
                        path, e2,
                    )
                    try:
                        return self._attempt_torch_load(path, weights_only=False)
                    except Exception as e3:  # noqa: BLE001
                        raise RuntimeError(
                            f"Failed to load checkpoint with unsafe fallback as well: {e3}"  # noqa: EM101
                        ) from e3
                # Not allowed to perform unsafe load; re-raise original secure error with guidance
                raise RuntimeError(
                    "Secure checkpoint load failed and unsafe fallback disabled. "
                    "Set allow_unsafe_checkpoint=True ONLY if you trust the source. Original error: " + msg
                ) from e2

    def _load_image(self, path: str):
        img = Image.open(path).convert("RGB")
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict(self, request: Union[str, Dict[str, Any]]):
        if isinstance(request, str):
            image_path = request
        else:
            inputs = request.get("inputs") if isinstance(request, dict) else None
            image_path = (inputs or request).get("image_path") if isinstance(request, dict) else None
        if not image_path:
            raise ValueError("image_path missing for prediction")
        self.ensure_model_loaded()
        tensor = self._load_image(image_path)
        start = time.time()
        with torch.no_grad():
            logits = self.model(tensor)
        dur = time.time() - start
        logits = logits.squeeze(0)
        if logits.ndim == 0:  # single scalar
            prob = torch.sigmoid(logits).item()
            return {
                "disease": self.disease_name,
                "probability": float(prob),
                "predicted": bool(prob >= self.threshold),
                "all_probabilities": {self.disease_name: float(prob)},
                "probabilities": {self.disease_name: float(prob)},
                "prediction": self.disease_name,
                "label": self.disease_name,
                "inference_time": round(dur, 4),
            }
        # vector output
        if logits.shape[0] == 1:
            prob = torch.sigmoid(logits[0]).item()
            probs_map = {self.classes[0]: float(prob)}
            return {
                "disease": self.disease_name,
                "probability": float(prob),
                "predicted": bool(prob >= self.threshold),
                "all_probabilities": probs_map,
                "probabilities": dict(probs_map),
                "prediction": self.disease_name,
                "label": self.disease_name,
                "inference_time": round(dur, 4),
            }
        if logits.shape[0] == 2:
            probs = torch.softmax(logits, dim=0).tolist()
            disease_index = 0  # we put disease first
            prob = probs[disease_index]
            probs_map = {self.classes[i]: float(p) for i, p in enumerate(probs)}
            return {
                "disease": self.disease_name,
                "probability": float(prob),
                "predicted": bool(prob >= self.threshold),
                "all_probabilities": probs_map,
                "probabilities": dict(probs_map),
                "prediction": self.disease_name,
                "label": self.disease_name,
                "inference_time": round(dur, 4),
            }
        # multi-class
        probs = torch.softmax(logits, dim=0).tolist()
        top_idx = int(torch.argmax(logits).item())
        probs_map = {self.classes[i]: float(p) for i, p in enumerate(probs)}
        return {
            "disease": self.disease_name,
            "probability": float(probs[top_idx]),
            "predicted": bool(probs[top_idx] >= self.threshold),
            "all_probabilities": probs_map,
            "probabilities": dict(probs_map),
            "prediction": self.classes[top_idx],
            "label": self.classes[top_idx],
            "inference_time": round(dur, 4),
        }


def load_tool(variant: str, **params):  # convenience
    meta = {"id": f"disease-specific:{variant}", "variant": variant, "entry": "tool_impl:DiseaseSpecificClassificationTool"}
    params = {"variant": variant, **params}
    tool = DiseaseSpecificClassificationTool(meta, params)
    tool.ensure_model_loaded()
    return tool

__all__ = ["DiseaseSpecificClassificationTool", "load_tool"]

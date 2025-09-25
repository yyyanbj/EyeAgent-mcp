from __future__ import annotations
import os, glob
from typing import Tuple, List, Optional
import torch, torch.nn as nn

# Mapping task -> weights folder name
FOLDER_MAP = {
    "modality": "modality23_resnet101_224_12-28-1602",
    "cfp_quality": "cfp_quality7_resnet101_384_12-28-2221",
    "laterality": "laterality2_resnet101_224_12-30-1455",
    "multidis": "multidis260_resnet101_224_05-15-2006",
    # Age weights may live under either 'cfp_age' or legacy 'fundus2age'
    "cfp_age": "cfp_age",
}


def _select_checkpoint(weight_dir: str) -> str | None:
    if not os.path.isdir(weight_dir):
        return None
    # Prefer explicitly named best / last checkpoints
    patterns = ["best*.pth", "last*.pth", "*.pth"]
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(weight_dir, pat)))
        if files:
            return files[0]
    return None


def _find_age_checkpoint(weights_root: str, folder_candidates: List[str]) -> Optional[str]:
    """Find age regression checkpoint, supporting legacy folder/layout.

    Search order:
    - nested fundus2age path used historically
    - best*.pth / last*.pth / *.pth / *.pth.tar in the provided folders
    """
    # 1) Known nested legacy path
    legacy_rel = os.path.join(
        "ckpts_morph",
        "age_prediction",
        "res2net50d_regression_384",
        "replicate0",
        "best_model.pth.tar",
    )
    for folder in folder_candidates:
        base = os.path.join(weights_root, folder)
        legacy_path = os.path.join(base, legacy_rel)
        if os.path.isfile(legacy_path):
            return legacy_path
    # 2) Common patterns at the folder root
    patterns = ["best*.pth", "last*.pth", "*.pth", "*.pth.tar"]
    for folder in folder_candidates:
        base = os.path.join(weights_root, folder)
        if not os.path.isdir(base):
            continue
        for pat in patterns:
            files = sorted(glob.glob(os.path.join(base, pat)))
            if files:
                return files[0]
    return None


def _load_state(model: nn.Module, ckpt_path: str | None):
    if not ckpt_path:
        raise FileNotFoundError("No checkpoint path provided for model load")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        # common nesting
        if "state_dict" in state:
            state = state["state_dict"]
    # Allow non-strict to support minor key mismatches while keeping deterministic load
    model.load_state_dict(state, strict=False)


def _build_model(task: str, num_classes: int) -> nn.Module:
    if task == "cfp_age":
        # Lazy import to avoid requiring timm unless age task is used
        try:
            try:
                from .models.cfp_age_model import CfpAgeModel  # type: ignore
            except Exception:  # noqa
                from models.cfp_age_model import CfpAgeModel  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(f"cfp_age model unavailable (timm installed?): {e}")
        return CfpAgeModel(pretrained=False)
    # Multi-label (multidis) & multi-class tasks share CSRA backbone; for binary laterality still use num_classes
    depth = 101
    try:
        try:
            from .models.resnet_csra import ResNet_CSRA  # type: ignore
        except Exception:  # noqa
            from models.resnet_csra import ResNet_CSRA  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(f"CSRA backbone import failed: {e}")
    # Use num_heads=1 to match legacy checkpoints for consistent outputs across tools
    return ResNet_CSRA(num_heads=1, lam=0.1, num_classes=num_classes, depth=depth)


def create_model(task: str, classes: List[str], weights_root: str) -> Tuple[nn.Module, int]:
    folder = FOLDER_MAP.get(task, task)
    # For cfp_age, support both new and legacy folder names
    folder_candidates = [folder]
    if task == "cfp_age" and folder != "fundus2age":
        folder_candidates.append("fundus2age")
    weight_dir = os.path.join(weights_root, folder)
    img_size = 384 if task in ("cfp_quality", "cfp_age") else 224
    num_classes = 1 if task == "cfp_age" else len(classes)
    model = _build_model(task, num_classes)
    if task == "cfp_age":
        # Age regression uses a different checkpoint layout and may use .pth.tar
        ckpt = _find_age_checkpoint(weights_root, folder_candidates)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found for cfp_age under {[os.path.join(weights_root, f) for f in folder_candidates]}"
            )
        # Prefer specialized loader if available on the model
        load_fn = getattr(model, "load_checkpoint", None)
        if callable(load_fn):
            load_fn(ckpt, torch.device("cpu"))
        else:
            _load_state(model, ckpt)
    else:
        ckpt = _select_checkpoint(weight_dir)
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint found in {weight_dir} (looked for best*.pth/last*.pth/*.pth)")
        # Try to read img_size metadata from checkpoint (legacy behavior)
        meta = torch.load(ckpt, map_location="cpu")
        if isinstance(meta, dict):
            img_size = int(meta.get("img_size", img_size))
        _load_state(model, ckpt)
    return model, img_size

__all__ = ["create_model"]
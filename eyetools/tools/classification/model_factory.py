from __future__ import annotations
import os, glob
from typing import Tuple, List
import torch, torch.nn as nn

from .models.resnet_csra import ResNet_CSRA
from .models.cfp_age_model import CfpAgeModel

# Mapping task -> weights folder name
FOLDER_MAP = {
    "modality": "modality23_resnet101_224_12-28-1602",
    "cfp_quality": "cfp_quality7_resnet101_384_12-28-2221",
    "laterality": "laterality2_resnet101_224_12-30-1455",
    "multidis": "multidis260_resnet101_224_05-15-2006",
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


def _load_state(model: nn.Module, ckpt_path: str | None):
    if not ckpt_path:
        return
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            # common nesting
            if "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)
    except Exception:
        # Silent fail to keep tool usable without weights
        pass


def _build_model(task: str, num_classes: int) -> nn.Module:
    if task == "cfp_age":
        return CfpAgeModel(pretrained=False)
    # Multi-label (multidis) & multi-class tasks share CSRA backbone; for binary laterality still use num_classes
    depth = 101  # fixed based on weight naming
    # For CSRA we treat all classification tasks as multi-label logits; tool_impl decides activation
    return ResNet_CSRA(num_heads=2, lam=0.1, num_classes=num_classes, depth=depth)


def create_model(task: str, classes: List[str], weights_root: str) -> Tuple[nn.Module, int]:
    folder = FOLDER_MAP.get(task, task)
    weight_dir = os.path.join(weights_root, folder)
    img_size = 384 if task in ("cfp_quality", "cfp_age") else 224
    num_classes = 1 if task == "cfp_age" else len(classes)
    model = _build_model(task, num_classes)
    ckpt = _select_checkpoint(weight_dir)
    _load_state(model, ckpt)
    return model, img_size

__all__ = ["create_model"]
"""Lightweight placeholder classification functions for tests.
Return simple deterministic values without heavy ML dependencies.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

CLASSES = ["class1", "class2", "class3"]


def _validate_image(path: str | Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))


def classify_image(path: str) -> str:
    _validate_image(path)
    return "class1"


def classify_image_resnet(path: str) -> str:
    _validate_image(path)
    return "resnet:class2"


def classify_image_vit(path: str) -> str:
    _validate_image(path)
    return "vit:class3"


def classify_image_fundus(path: str) -> str:
    _validate_image(path)
    return "fundus:normal"


def modality(path: str) -> str:
    _validate_image(path)
    return "fundus"


def cfp_quality(path: str) -> str:
    _validate_image(path)
    return "good"


def laterality(path: str) -> str:
    _validate_image(path)
    return "left"


def pre_analyze(path: str) -> list:
    _validate_image(path)
    return ["class1", "class2"]


def vis_prob(probabilities: List[float], classes: List[str]):  # pragma: no cover (visual)
    # simplified: just ensure same length
    assert len(probabilities) == len(classes)
    return None


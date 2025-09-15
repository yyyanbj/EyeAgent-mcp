#!/usr/bin/env python
"""Segmentation demo script.

Modes:
  real      - attempt real nnUNet inference (requires weights/segmentation present)
  fallback  - generate synthetic mask (no heavy model load) for quick visual output

Usage:
    # Direct in-process (current interpreter):
    uv run python scripts/demo_segmentation.py --variant cfp_artifact --image examples/test_images/Artifact.jpg --mode real
    uv run python scripts/demo_segmentation.py --variant cfp_artifact --image examples/test_images/Artifact.jpg --mode fallback

    # Via ToolManager/EnvManager (ensures env deps resolved from envs/py312-seg):
    uv run python scripts/demo_segmentation.py --variant cfp_artifact --image examples/test_images/Artifact.jpg --mode real --manager
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, cv2

# Tool imports
import sys
from pathlib import Path as _Path

# Robust import path handling: add repo root (parent of scripts/) to sys.path
_THIS_FILE = _Path(__file__).resolve()
_CANDIDATE_ROOT = _THIS_FILE.parent.parent  # parent of scripts
if str(_CANDIDATE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CANDIDATE_ROOT))

try:  # local package import
    from tools.segmentation.tool_impl import SegmentationTool
    from eyetools.core.registry import ToolRegistry
    from eyetools.core.loader import discover_tools
    from eyetools.core.tool_manager import ToolManager
except ModuleNotFoundError as e:  # pragma: no cover
    print("[WARN] Could not import core components. sys.path=", sys.path, file=sys.stderr)
    raise


def build_tool(variant: str, mode: str, base_path: Path) -> SegmentationTool:
    meta = {"id": f"segmentation:{variant}", "entry": "tool_impl:SegmentationTool"}
    params = {"task": variant, "base_path": str(base_path), "weights_root": "weights/segmentation"}
    tool = SegmentationTool(meta, params)
    if mode == "fallback":
        class DummyPredictor:
            def predict_from_files_sequential(self, inputs, out_dir, *a, **k):
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                tgt = Path(inputs[0][0])
                seg_path = Path(out_dir) / f"{tgt.stem}.png"
                img = cv2.imread(str(tgt), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(tgt)
                h, w = img.shape[:2]
                # create a synthetic circular mask
                yy, xx = np.ogrid[:h, :w]
                cy, cx = h // 2, w // 2
                r = min(h, w) // 4
                mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype("uint8")
                cv2.imwrite(str(seg_path), mask * 255)
                return True
        def fake_load():
            tool.lesions = {"artifact": (255, 255, 255)}  # minimal for chosen variant
            tool.model_id = 0
            tool.predictor = DummyPredictor()
            tool._model_loaded = True
        tool.load_model = lambda : fake_load()  # type: ignore
    return tool


def run_demo(variant: str, image: Path, mode: str, use_manager: bool = False):
    image = image.resolve()
    if not image.exists():
        print(f"Image not found: {image}", file=sys.stderr)
        sys.exit(2)
    out_base = Path("temp/demo_seg") / variant / mode
    if use_manager:
        # Discover tools and use ToolManager so EnvManager resolves environment dependencies (py312-seg)
        reg = ToolRegistry()
        discover_tools([Path('tools')], reg, [])
        # pick the segmentation meta for the variant
        try:
            meta = next(m for m in reg.list() if m.package == 'segmentation' and m.variant == variant)
        except StopIteration:
            print(f"Variant {variant} not found in discovered tools.", file=sys.stderr)
            sys.exit(3)
        manager = ToolManager(registry=reg, workspace_root=Path('.'))
        # Use ToolManager.predict; EnvManager inside will handle env resolution if implemented for subprocess/overlay
        result = manager.predict(meta.id, {"inputs": {"image_path": str(image)}})
    else:
        tool = build_tool(variant, mode, out_base)
        tool.prepare()
        result = tool.predict({"inputs": {"image_path": str(image)}})
    print(json.dumps(result, indent=2))
    print("Output artifacts folder:", out_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="Segmentation variant, e.g. cfp_artifact")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--mode", choices=["real", "fallback"], default="fallback")
    ap.add_argument("--manager", action="store_true", help="Route through ToolManager/EnvManager pipeline")
    args = ap.parse_args()
    run_demo(args.variant, Path(args.image), args.mode, use_manager=args.manager)


if __name__ == "__main__":
    main()

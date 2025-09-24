#!/usr/bin/env python
"""Multimodal demo script (fundus2oct, fundus2eyeglobe).

Two ways to run:
  - Direct in-process (fast placeholder path works without heavy deps)
  - Via ToolManager (uses subprocess + envs/py12-multimodal if configured)

Examples:
  uv run python scripts/run_multimodal_demo.py --variant fundus2oct --image examples/test_images/retinal vessel.jpg --mode fallback
  uv run python scripts/run_multimodal_demo.py --variant fundus2eyeglobe --image examples/test_images/retinal vessel.jpg --mode fallback

  # Route via ToolManager/EnvManager (spawns subprocess with configured env)
  uv run python scripts/run_multimodal_demo.py --variant fundus2oct --image examples/test_images/retinal vessel.jpg --manager
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

# Robust import path handling: add repo root (parent of scripts/) to sys.path
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

try:
	from tools.multimodal.tool_impl import MultimodalTool
	from eyetools.core.registry import ToolRegistry
	from eyetools.core.loader import discover_tools
	from eyetools.core.tool_manager import ToolManager
except ModuleNotFoundError as e:  # pragma: no cover
	print("[WARN] Could not import core components. sys.path=", sys.path, file=sys.stderr)
	raise


def build_tool(variant: str, base_path: Path) -> MultimodalTool:
	meta = {"id": f"multimodal:{variant}", "entry": "tool_impl:MultimodalTool"}
	params = {"task": variant, "base_path": str(base_path), "weights_root": "weights/multimodal"}
	tool = MultimodalTool(meta, params)
	return tool


def run_demo(variant: str, image: Path, mode: str, use_manager: bool = False, extra: dict | None = None):
	image = image.resolve()
	if not image.exists():
		print(f"Image not found: {image}", file=sys.stderr)
		sys.exit(2)
	out_base = Path("temp/demo_multimodal") / variant / ("manager" if use_manager else mode)
	out_base.mkdir(parents=True, exist_ok=True)

	request_inputs = {"image_path": str(image)}
	if extra:
		request_inputs.update(extra)

	if use_manager:
		reg = ToolRegistry()
		discover_tools([Path('tools')], reg, [])
		try:
			meta = next(m for m in reg.list() if m.package == 'multimodal' and m.variant == variant)
		except StopIteration:
			print(f"Variant {variant} not found in discovered tools.", file=sys.stderr)
			sys.exit(3)
		tm = ToolManager(registry=reg, workspace_root=Path('.'))
		result = tm.predict(meta.id, {"inputs": request_inputs})
	else:
		tool = build_tool(variant, out_base)
		tool.prepare()
		# Placeholder/real path determined internally by tool.load_model()
		result = tool.predict({"inputs": request_inputs})

	print(json.dumps(result, indent=2, ensure_ascii=False))
	print("Artifacts written under:", out_base)


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--variant", required=True, choices=["fundus2oct", "fundus2eyeglobe"], help="Multimodal variant")
	ap.add_argument("--image", required=True, help="Path to input fundus image")
	ap.add_argument("--mode", choices=["fallback", "real"], default="fallback", help="In-process run mode; Tool will fallback to placeholder if real deps missing")
	ap.add_argument("--manager", action="store_true", help="Route via ToolManager/EnvManager (subprocess)")
	ap.add_argument("--eye_category", choices=["OD", "OS"], default="OD")
	ap.add_argument("--sampling_steps", type=int, default=32)
	ap.add_argument("--SE", type=float)
	ap.add_argument("--AL", type=float)
	args = ap.parse_args()

	extra = {"eye_category": args.eye_category, "sampling_steps": args.sampling_steps}
	if args.SE is not None:
		extra["SE"] = args.SE
	if args.AL is not None:
		extra["AL"] = args.AL

	run_demo(args.variant, Path(args.image), args.mode, use_manager=args.manager, extra=extra)


if __name__ == "__main__":
	main()


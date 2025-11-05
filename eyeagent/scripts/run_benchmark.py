#!/usr/bin/env python3
"""
Batch benchmark runner for EyeAgent.

Primary UI entry remains the Streamlit app (eyeagent/ui/app.py), but this tool
allows headless batch runs for benchmarking, with CLI args that override env vars.

Examples:
  # Triage version, profile backend, LLM routing; input as JSONL
  python bin/run_benchmark.py \
    --config-file eyeagent/eyeagent/config/eyeagent.triage.yml \
    --profile triage --backend profile --routing llm \
    --input-jsonl ./cases_list.jsonl \
    --output-dir ./cases

  # Imaging-only version; scan a directory of images as one-case-per-file
  python bin/run_benchmark.py \
    --config-file eyeagent/eyeagent/config/eyeagent.imaging.yml \
    --profile imaging --backend profile --routing llm \
    --images-dir ./datasets/fundus --glob "*.jpg" \
    --output-dir ./benchmark_out

Input JSONL format (one JSON object per line):
  {
    "id": "case-001",
    "patient": {"age": 60, "sex": "F"},
    "images": [
      {"image_id": "OD", "path": "/path/to/od.jpg"},
      {"image_id": "OS", "path": "/path/to/os.jpg"}
    ]
  }
You may also pass images as a list of strings ["/path/od.jpg", "/path/os.jpg"].
"""
from __future__ import annotations
import os
import sys
import argparse
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eyeagent.diagnostic_workflow import run_diagnosis  # type: ignore
from eyeagent.core.settings import set_overrides


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="EyeAgent benchmark runner (batch mode)")
    # Config overrides
    ap.add_argument("--config-file", type=str, default=None, help="Path to eyeagent config file (override)")
    ap.add_argument("--profile", type=str, default=None, help="Pipeline profile name (override)")
    ap.add_argument("--backend", type=str, default=None, choices=["langgraph","profile","interaction","single"], help="Workflow backend (override)")
    ap.add_argument("--routing", type=str, default=None, choices=["llm","config"], help="Orchestrator routing strategy (override)")
    # Inputs
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input-jsonl", type=str, help="Path to JSONL file, one case per line")
    g.add_argument("--images-dir", type=str, help="Directory of images to scan (one case per file)")
    ap.add_argument("--glob", type=str, default="*.jpg", help="Glob pattern for --images-dir (default: *.jpg)")
    # Output
    ap.add_argument("--output-dir", type=str, required=True, help="Directory to save final_report.json per case")
    return ap.parse_args()


def apply_env_overrides(args: argparse.Namespace) -> None:
    # Legacy name kept for compatibility; now sets process-local overrides (no env)
    set_overrides(
        config_file=args.config_file or None,
        workflow_backend=args.backend or None,
        pipeline_profile=args.profile or None,
        routing_strategy=args.routing or None,
    )


def _coerce_images(images: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(images, list):
        for i, it in enumerate(images):
            if isinstance(it, str):
                out.append({"image_id": f"IMG{i+1}", "path": it})
            elif isinstance(it, dict):
                iid = it.get("image_id") or f"IMG{i+1}"
                p = it.get("path") or it.get("file") or it.get("uri")
                if p:
                    out.append({"image_id": iid, "path": p})
    return out


def _iter_cases_from_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            cid = str(data.get("id") or data.get("case_id") or data.get("name") or "case")
            patient = data.get("patient") or {}
            images = _coerce_images(data.get("images"))
            if not images:
                continue
            yield cid, patient, images


def _iter_cases_from_dir(images_dir: Path, pattern: str):
    files = sorted(images_dir.rglob(pattern)) if ("**" in pattern) else sorted(images_dir.glob(pattern))
    for i, p in enumerate(files):
        if not p.is_file():
            continue
        cid = f"img-{i+1}"
        patient: Dict[str, Any] = {"id": cid}
        images = [{"image_id": cid, "path": str(p)}]
        yield cid, patient, images


def _save_report(output_dir: Path, case_id: str, report: Dict[str, Any]) -> None:
    case_dir = output_dir / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / "final_report.json"
    try:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report.get("final_report") or report, f, ensure_ascii=False, indent=2)
    except Exception:
        # Fail silently to continue batch
        pass


def main():
    args = parse_args()
    apply_env_overrides(args)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        iterator = _iter_cases_from_jsonl(Path(args.input_jsonl))
    else:
        iterator = _iter_cases_from_dir(Path(args.images_dir), args.glob)

    total = 0
    ok = 0
    for cid, patient, images in iterator:
        try:
            report = run_diagnosis(patient, images)
            _save_report(out_dir, cid, report)
            ok += 1
        except Exception as e:
            # Print a short error and continue
            print(f"[run_benchmark] case={cid} failed: {str(e)[:200]}")
        total += 1

    print(f"[run_benchmark] done: {ok}/{total} cases succeeded -> {out_dir}")


if __name__ == "__main__":
    main()

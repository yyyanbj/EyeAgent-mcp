#!/usr/bin/env python3
"""Utility to rerun only failed EyeAgent benchmark cases and refresh results.

This helper scans an existing benchmark output directory, locates any case
results whose status is not "success", reruns them using the original
benchmark configuration, and regenerates aggregate metrics/reports without
reprocessing successful samples.

Example usage:
    python scripts/rerun_failed_cases.py \
        --config benchmark/examples/vqa_cfp_real.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

# Ensure repository root is on the path so `benchmark` can be imported when the
# script is executed from anywhere inside the repo.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from loguru import logger

from benchmark import BenchmarkConfig, BenchmarkRunner
from benchmark.metrics import MetricsCalculator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun failed EyeAgent benchmark cases without touching successes.",
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the benchmark YAML configuration used for the original run.",
    )
    parser.add_argument(
        "--cases-dir",
        help="Override the case results directory (defaults to <output_dir>/case_results).",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="Do not delete older case JSON files after a successful rerun.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip rerunning failures; only recompute metrics from existing results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args()


def _configure_environment(config: BenchmarkConfig) -> None:
    """Mirror the environment tweaks performed during a full benchmark run."""
    if config.model.dry_run:
        os.environ["EYEAGENT_DRY_RUN"] = "1"
    else:
        os.environ.pop("EYEAGENT_DRY_RUN", None)

    if config.model.mcp_server_url:
        os.environ["MCP_SERVER_URL"] = config.model.mcp_server_url
        try:  # pragma: no cover - defensive sync of module-level constant
            import eyeagent.workflows.langgraph as _langgraph  # type: ignore

            _langgraph.MCP_SERVER_URL = config.model.mcp_server_url
        except Exception as exc:  # pragma: no cover - best-effort sync
            logger.warning("Unable to propagate MCP server URL override: {}", exc)


def _load_latest_case_results(case_dir: Path) -> Dict[int, MutableMapping[str, Any]]:
    """Return the most recent JSON blob for each case index."""
    results: Dict[int, MutableMapping[str, Any]] = {}
    for json_path in sorted(case_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Skipping unreadable case file %s: %s", json_path, exc)
            continue
        idx = data.get("index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except Exception:
            logger.warning("Unexpected non-integer case index in %s", json_path)
            continue
        mtime = json_path.stat().st_mtime
        previous = results.get(idx)
        if previous is None or mtime >= previous.get("_mtime", 0):
            data["_source_path"] = str(json_path)
            data["_mtime"] = mtime
            results[idx] = data
    return results


def _persist_case_result(case_dir: Path, result: Dict[str, Any]) -> Path:
    """Write a single case result JSON file and return its path."""
    index = int(result.get("index", -1))
    case_id = result.get("case_id") or f"case_{index if index >= 0 else 'unknown'}"
    filename = f"{index:04d}_{case_id}.json" if index >= 0 else f"{case_id}.json"
    output_path = case_dir / filename
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
    logger.debug("Persisted case %s to %s", case_id, output_path)
    return output_path


def _purge_old_case_files(case_dir: Path, index: int, keep_path: Path) -> None:
    pattern = f"{index:04d}_*.json"
    for candidate in case_dir.glob(pattern):
        if candidate.resolve() != keep_path.resolve():
            try:
                candidate.unlink()
                logger.debug("Removed outdated case file %s", candidate)
            except Exception as exc:
                logger.warning("Failed to delete %s: %s", candidate, exc)


async def _rerun_case(
    runner: BenchmarkRunner,
    index: int,
    image_path: str,
    true_label: str,
) -> Dict[str, Any]:
    logger.info("Re-running case index=%s | image=%s", index, Path(image_path).name)
    runner.start_time = time.time()
    return await runner._run_single_diagnostic(image_path, true_label, index)  # pylint: disable=protected-access


async def rerun_failed_cases(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = BenchmarkConfig.from_yaml(str(config_path))
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    _configure_environment(config)

    runner = BenchmarkRunner(config)
    image_paths, labels = runner.dataset_loader.load()
    dataset_size = len(image_paths)
    overall_start_time = time.time()

    case_dir = Path(args.cases_dir) if args.cases_dir else Path(config.output.output_dir) / config.output.case_results_subdir
    case_dir.mkdir(parents=True, exist_ok=True)

    stored_results = _load_latest_case_results(case_dir)
    if not stored_results:
        logger.warning("No existing case results found in %s", case_dir)

    failed_indices_all = sorted(
        idx for idx, payload in stored_results.items() if payload.get("status") != "success"
    )
    rerun_targets = list(failed_indices_all)
    if args.dry_run:
        logger.info("Dry-run requested: skipping reruns, will only recompute metrics.")
        rerun_targets = []

    rerun_records: Dict[int, Dict[str, Any]] = {}

    for idx in rerun_targets:
        if idx >= dataset_size:
            logger.warning(
                "Skipping case index %s because it exceeds dataset size (%s)", idx, dataset_size
            )
            continue
        image_path = image_paths[idx]
        true_label = labels[idx]

        try:
            rerun_result = await _rerun_case(runner, idx, image_path, true_label)
            rerun_records[idx] = rerun_result
            saved_path = _persist_case_result(case_dir, rerun_result)
            rerun_result["_source_path"] = str(saved_path)
            rerun_result["_mtime"] = saved_path.stat().st_mtime
            stored_results[idx] = rerun_result
            if not args.keep_history:
                _purge_old_case_files(case_dir, idx, saved_path)
        except Exception as exc:  # pragma: no cover - rerun failure path
            logger.error("Rerun failed for index %s: %s", idx, exc)
            failure_record = {
                "index": idx,
                "image_path": image_path,
                "true_label": true_label,
                "status": "failed",
                "error": str(exc),
                "rerun_timestamp": datetime.now().isoformat(),
            }
            saved_path = _persist_case_result(case_dir, failure_record)
            failure_record["_source_path"] = str(saved_path)
            failure_record["_mtime"] = saved_path.stat().st_mtime
            stored_results[idx] = failure_record

    # Build final ordered results list (best-effort covering dataset order)
    ordered_indices: List[int] = sorted(stored_results.keys())
    final_results: List[Dict[str, Any]] = [stored_results[idx] for idx in ordered_indices]

    # Recompute metrics using the refreshed result set
    runner.results = final_results
    runner.start_time = overall_start_time
    runner.end_time = time.time()

    if final_results:
        metrics = runner._calculate_metrics()  # pylint: disable=protected-access
        runner._final_metrics = metrics  # pylint: disable=protected-access
        report = runner._generate_report(metrics)  # pylint: disable=protected-access
    else:
        metrics = {}
        report = {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path(config.output.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    detailed_results_path = summary_dir / f"rerun_results_{timestamp}.json"
    with open(detailed_results_path, "w", encoding="utf-8") as fh:
        json.dump(final_results, fh, indent=2, ensure_ascii=False, default=str)

    metrics_path: Optional[Path] = None
    if metrics:
        metrics_calc = MetricsCalculator(config.metrics, runner.dataset_loader.class_names)
        metrics_path = summary_dir / f"rerun_metrics_{timestamp}.json"
        metrics_calc.save_metrics_report(metrics, str(metrics_path))

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "config_path": str(config_path),
        "output_dir": config.output.output_dir,
        "case_results_dir": str(case_dir),
        "dataset_size": dataset_size,
        "processed_cases": len(final_results),
    "initial_failed_indices": failed_indices_all,
    "rerun_indices": rerun_targets,
        "remaining_failures": [
            idx for idx, payload in stored_results.items() if payload.get("status") != "success"
        ],
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("f1_score"),
    "macro_precision": metrics.get("precision") if metrics else None,
    "macro_recall": metrics.get("recall") if metrics else None,
        "config_snapshot": asdict(config),
        "report": report,
        "detailed_results_path": str(detailed_results_path),
    "metrics_report_path": str(metrics_path) if metrics_path else None,
    }

    summary_path = summary_dir / f"rerun_summary_{timestamp}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2, ensure_ascii=False, default=str)

    logger.info("Rerun complete: %s", summary_path)
    print("\n=== RERUN SUMMARY ===")
    print(f"Config: {config_path}")
    print(f"Case directory: {case_dir}")
    print(f"Initial failed indices: {failed_indices_all}")
    print(f"Failed indices rerun: {rerun_targets}")
    print(f"Remaining failures: {summary_payload['remaining_failures']}")
    print(f"Accuracy: {summary_payload['accuracy']:.3f}" if summary_payload["accuracy"] is not None else "Accuracy: N/A")
    print(f"Macro F1: {summary_payload['macro_f1']:.3f}" if summary_payload["macro_f1"] is not None else "Macro F1: N/A")
    print(f"Detailed results: {detailed_results_path}")
    print(f"Metrics report: {metrics_path}")
    print(f"Summary file: {summary_path}\n")

    return summary_payload


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(rerun_failed_cases(args))
    except KeyboardInterrupt:  # pragma: no cover - user initiated stop
        print("\nAborted by user.")


if __name__ == "__main__":  # pragma: no cover
    main()

"""Main class for running diagnostic benchmarks."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, MutableMapping
from dataclasses import asdict

from tqdm import tqdm

from .config import BenchmarkConfig
from .dataset_loader import DatasetLoader
from .metrics import MetricsCalculator, extract_predictions_from_results
from .format_agent import FormatAgent
from eyeagent.diagnostic_workflow import run_diagnosis_async
from eyeagent.tracing.trace_logger import TraceLogger

import logging

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Benchmark evaluation runner."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_loader = DatasetLoader(config.dataset)
        self.results: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        # Ensure output directory exists
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        self.case_results_dir: Optional[str] = None
        if self.config.output.save_case_results:
            self.case_results_dir = os.path.join(
                self.config.output.output_dir,
                self.config.output.case_results_subdir,
            )
            os.makedirs(self.case_results_dir, exist_ok=True)

    def _configure_environment(self) -> None:
        """Configure environment variables for diagnostic execution."""
        if self.config.model.dry_run:
            os.environ["EYEAGENT_DRY_RUN"] = "1"
        else:
            os.environ.pop("EYEAGENT_DRY_RUN", None)

        # Select workflow backend (langgraph | profile | interaction | single)
        if self.config.model.workflow_backend:
            os.environ["EYEAGENT_WORKFLOW_BACKEND"] = str(self.config.model.workflow_backend)
        else:
            os.environ.pop("EYEAGENT_WORKFLOW_BACKEND", None)

        if self.config.model.mcp_server_url:
            os.environ["MCP_SERVER_URL"] = self.config.model.mcp_server_url
            try:
                import eyeagent.workflows.langgraph as _langgraph

                _langgraph.MCP_SERVER_URL = self.config.model.mcp_server_url
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Unable to synchronize langgraph MCP URL override: %s", exc
                )

    async def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark evaluation.

        Returns:
            Dictionary containing benchmark results and metrics
        """
        logger.info("Starting EyeAgent benchmark evaluation")
        self.start_time = time.time()

        try:
            # Load dataset
            image_paths, labels = self.dataset_loader.load()
            logger.info(f"Loaded {len(image_paths)} samples for evaluation")

            # Run diagnostic workflow on each sample
            await self._run_diagnostics(image_paths, labels)

            # Calculate metrics
            metrics = self._calculate_metrics()

            # Generate reports
            report = self._generate_report(metrics)

            # Save results
            if self.config.output.save_predictions:
                await self._save_results()

            self.end_time = time.time()
            total_time = self.end_time - self.start_time

            logger.info(f"Benchmark completed in {total_time:.2f} seconds")
            logger.info(f"Overall accuracy: {metrics.get('accuracy', 0):.3f}")

            return {
                "config": asdict(self.config),
                "metrics": metrics,
                "report": report,
                "results": self.results if self.config.output.save_predictions else [],
                "runtime": total_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _run_diagnostics(self, image_paths: List[str], labels: List[str]) -> None:
        """Run diagnostic workflow on all samples."""
        logger.info("Running diagnostic workflow on samples...")

        # Set environment variables for the diagnostic workflow
        self._configure_environment()

        cached_successes: Dict[int, Dict[str, Any]] = {}
        skipped_count = 0

        if (
            self.config.runner.skip_existing_results
            and self.config.output.save_case_results
            and self.case_results_dir
        ):
            case_dir = Path(self.case_results_dir)
            if case_dir.exists():
                stored_results = _load_latest_case_results(case_dir)
                dataset_size = len(image_paths)
                for idx, payload in stored_results.items():
                    try:
                        idx_int = int(idx)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Encountered cached case with non-integer index %s; ignoring",
                            idx,
                        )
                        continue
                    if payload.get("status") == "success" and 0 <= idx_int < dataset_size:
                        cached_successes[idx_int] = payload
                if cached_successes:
                    logger.info(
                        "Found %d cached successful case results; reusing them (skip_existing_results=True)",
                        len(cached_successes),
                    )
            else:
                logger.debug(
                    "Case results directory %s does not exist yet; no cached results to reuse",
                    case_dir,
                )

        # Progress bar
        pbar = tqdm(total=len(image_paths), desc="Processing samples")

        for i, (image_path, true_label) in enumerate(zip(image_paths, labels)):
            postfix = {"current": f"{i+1}/{len(image_paths)}", "label": true_label}

            cached_result = None
            if self.config.runner.skip_existing_results:
                cached_result = cached_successes.get(i)
                if cached_result:
                    if (
                        cached_result.get("image_path") == image_path
                        and cached_result.get("true_label") == true_label
                    ):
                        skipped_count += 1
                        postfix["status"] = "cached"
                        self.results.append(cached_result)
                        if self.config.output.verbose:
                            predicted = cached_result.get("extracted_diagnosis", "Unknown")
                            logger.info(
                                "Skipping sample %d (index=%d): reused cached prediction '%s'",
                                i + 1,
                                i,
                                predicted,
                            )
                        pbar.set_postfix(postfix)
                        pbar.update(1)
                        continue
                    else:
                        logger.warning(
                            "Cached result for index %d does not match current dataset entry; rerunning",
                            i,
                        )

            pbar.set_postfix(postfix)

            try:
                # Run single diagnostic
                result = await self._run_single_diagnostic(image_path, true_label, i)
                self.results.append(result)
                self._save_case_result(result)

                if self.config.output.verbose and i % 10 == 0:
                    predicted = result.get("extracted_diagnosis", "Unknown")
                    logger.info(f"Sample {i+1}: True={true_label}, Predicted={predicted}")

            except Exception as e:
                logger.error(f"Failed to process sample {i+1} ({image_path}): {e}")
                # Add failed result to maintain index alignment
                failure_result = {
                    "index": i,
                    "image_path": image_path,
                    "true_label": true_label,
                    "status": "failed",
                    "error": str(e),
                    "formatted_diagnosis": "The diagnosis of this image is Normal",
                    "extracted_diagnosis": "Normal",
                }
                self.results.append(failure_result)
                self._save_case_result(failure_result)

            pbar.update(1)

        pbar.close()

        if skipped_count:
            logger.info(
                "Reused %d cached successful case results out of %d samples",
                skipped_count,
                len(image_paths),
            )

        successful_count = sum(1 for r in self.results if r.get("status") != "failed")
        logger.info(f"Completed {successful_count}/{len(image_paths)} samples successfully")

    async def _run_single_diagnostic(self, image_path: str, true_label: str, index: int) -> Dict[str, Any]:
        """Run diagnostic workflow on a single sample."""

        # Create trace logger for this case
        case_id = f"benchmark_{index:04d}_{int(time.time())}"
        trace_logger = TraceLogger()

        # Prepare input for diagnostic workflow
        patient = {
            "age": 65,  # Default age for benchmark
            "gender": "unknown",
            "medical_history": [],
            "symptoms": [],
        }

        images = [
            {
                "path": image_path,
                "modality": "CFP",  # Assume color fundus photography
                "eye": "unknown",
                "timestamp": datetime.now().isoformat(),
            }
        ]

        # Run diagnostic workflow
        diagnostic_result = await run_diagnosis_async(
            patient=patient, images=images, trace=trace_logger, case_id=case_id
        )

        # Format output if enabled
        formatted_output = ""
        extracted_diagnosis = "Normal"

        if self.config.model.enable_format_agent:
            try:
                format_agent = FormatAgent(
                    mcp_url=self.config.model.mcp_server_url,
                    trace_logger=trace_logger,
                    case_id=case_id,
                    class_names=self.dataset_loader.class_names,
                )

                # Prefer the final fragment section when available
                final_fragment = (
                    diagnostic_result.get("final_report")
                    if isinstance(diagnostic_result, dict) and diagnostic_result.get("final_report")
                    else diagnostic_result
                )
                format_context = {"final_fragment": final_fragment}
                format_result = await format_agent.a_run(format_context)

                formatted_output = format_result.get("outputs", {}).get("formatted_diagnosis", "")
                extracted_diagnosis = format_result.get("outputs", {}).get("extracted_diagnosis", "Normal")

            except Exception as e:
                logger.warning(f"Format agent failed for sample {index}: {e}")
                # Fallback formatting
                from .format_agent import format_diagnosis_for_evaluation

                formatted_output = format_diagnosis_for_evaluation(
                    diagnostic_result, self.dataset_loader.class_names
                )
                extracted_diagnosis = self._extract_diagnosis_fallback(formatted_output)
        else:
            # Use simple extraction without format agent
            from .format_agent import format_diagnosis_for_evaluation

            formatted_output = format_diagnosis_for_evaluation(
                diagnostic_result, self.dataset_loader.class_names
            )
            extracted_diagnosis = self._extract_diagnosis_fallback(formatted_output)

        # Extract additional information
        confidence = self._extract_confidence(diagnostic_result)
        # probabilities = self._extract_probabilities(diagnostic_result)

        return {
            "index": index,
            "case_id": case_id,
            "image_path": image_path,
            "true_label": true_label,
            "formatted_diagnosis": formatted_output,
            "extracted_diagnosis": extracted_diagnosis,
            "confidence": confidence,
            # "prediction_probabilities": probabilities,
            "diagnostic_result": diagnostic_result if self.config.output.save_detailed_report else None,
            "status": "success",
            "processing_time": time.time() - self.start_time if self.start_time else 0,
        }

    def _save_case_result(self, result: Dict[str, Any]) -> None:
        """Persist individual case result to disk immediately."""
        if not self.config.output.save_case_results or not self.case_results_dir:
            return

        try:
            case_id = result.get("case_id") or f"case_{result.get('index', 'unknown')}"
            file_name = f"{result.get('index', 0):04d}_{case_id}.json"
            path = os.path.join(self.case_results_dir, file_name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save case result for index {result.get('index')}: {e}")

    def _extract_diagnosis_fallback(self, formatted_output: str) -> str:
        """Fallback diagnosis extraction from formatted output."""
        import re

        pattern = r"The diagnosis of this image is\s+(.+?)(?:\.|$)"
        match = re.search(pattern, formatted_output, re.IGNORECASE)

        if match:
            diagnosis = match.group(1).strip()
            # Normalize to valid class name
            for class_name in self.dataset_loader.class_names:
                if diagnosis.lower() == class_name.lower():
                    return class_name

        return "Normal"

    def _extract_confidence(self, diagnostic_result: Dict[str, Any]) -> float:
        """Extract confidence score from diagnostic result."""
        # Look for confidence in various places
        if "final_fragment" in diagnostic_result:
            final = diagnostic_result["final_fragment"]
            if isinstance(final, dict):
                if "confidence" in final:
                    try:
                        return float(final["confidence"])
                    except (ValueError, TypeError):
                        pass

        return 0.5  # Default confidence

    def _extract_probabilities(self, diagnostic_result: Dict[str, Any]) -> List[float]:
        """Extract class probabilities from diagnostic result."""
        num_classes = len(self.dataset_loader.class_names)

        # Try to extract from specialist outputs
        if "specialist" in diagnostic_result:
            specialist = diagnostic_result["specialist"]
            if isinstance(specialist, dict) and "all_probabilities" in specialist:
                probs_dict = specialist["all_probabilities"]
                if isinstance(probs_dict, dict):
                    # Map probabilities to class order
                    probs = []
                    for class_name in self.dataset_loader.class_names:
                        prob = probs_dict.get(class_name, 0.0)
                        try:
                            probs.append(float(prob))
                        except (ValueError, TypeError):
                            probs.append(0.0)

                    # Normalize if needed
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                        return probs

        # Default uniform probabilities
        uniform_prob = 1.0 / num_classes
        return [uniform_prob] * num_classes

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics from results."""
        logger.info("Calculating evaluation metrics...")

        # Extract predictions
        true_labels, predicted_labels, prediction_probs = extract_predictions_from_results(
            self.results, self.dataset_loader.class_names
        )

        # Initialize metrics calculator
        metrics_calc = MetricsCalculator(self.config.metrics, self.dataset_loader.class_names)

        # Calculate all metrics
        metrics = metrics_calc.calculate_all_metrics(true_labels, predicted_labels, prediction_probs)

        # Generate plots if enabled
        if self.config.output.save_confusion_matrix:
            cm_path = os.path.join(self.config.output.output_dir, "confusion_matrix.png")
            try:
                metrics_calc.plot_confusion_matrix(true_labels, predicted_labels, cm_path)
            except Exception as e:
                logger.warning(f"Failed to generate confusion matrix plot: {e}")

        # Generate ROC curves for multi-class
        if len(self.dataset_loader.class_names) > 2 and self.config.metrics.compute_auc:
            roc_path = os.path.join(self.config.output.output_dir, "roc_curves.png")
            try:
                metrics_calc.plot_roc_curves(true_labels, prediction_probs, roc_path)
            except Exception as e:
                logger.warning(f"Failed to generate ROC curves: {e}")

        return metrics

    def _generate_report(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""

        # Create metrics calculator for summary table
        metrics_calc = MetricsCalculator(self.config.metrics, self.dataset_loader.class_names)
        summary_table = metrics_calc.create_summary_table(metrics)

        report = {
            "dataset_info": {
                "name": self.config.dataset.name,
                "path": self.config.dataset.path,
                "total_samples": len(self.results),
                "successful_samples": sum(1 for r in self.results if r.get("status") != "failed"),
                "failed_samples": sum(1 for r in self.results if r.get("status") == "failed"),
                "class_distribution": self.dataset_loader.get_class_distribution(),
            },
            "model_info": {
                "workflow_backend": self.config.model.workflow_backend,
                "mcp_server_url": self.config.model.mcp_server_url,
                "dry_run": self.config.model.dry_run,
                "format_agent_enabled": self.config.model.enable_format_agent,
            },
            "performance_summary": {
                "accuracy": metrics.get("accuracy", 0),
                "macro_f1": metrics.get("f1_score", 0),
                "macro_precision": metrics.get("precision", 0),
                "macro_recall": metrics.get("recall", 0),
                "auc_roc": metrics.get("auc_roc_macro", metrics.get("auc_roc", "N/A")),
            },
            "summary_table": summary_table.to_dict("records"),
            "runtime_info": {
                "total_time": self.end_time - self.start_time if self.end_time and self.start_time else 0,
                "avg_time_per_sample": (self.end_time - self.start_time) / len(self.results)
                if self.end_time and self.start_time and self.results else 0,
            },
        }

        return report

    async def _save_results(self) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_path = os.path.join(
            self.config.output.output_dir, f"benchmark_results_{timestamp}.json"
        )
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        logger.info(f"Detailed results saved to {results_path}")

        # Save configuration
        config_path = os.path.join(
            self.config.output.output_dir, f"benchmark_config_{timestamp}.yaml"
        )
        self.config.to_yaml(config_path)

        # Save metrics report
        if hasattr(self, "_final_metrics"):
            metrics_path = os.path.join(
                self.config.output.output_dir, f"metrics_report_{timestamp}.json"
            )
            metrics_calc = MetricsCalculator(self.config.metrics, self.dataset_loader.class_names)
            metrics_calc.save_metrics_report(self._final_metrics, metrics_path)


async def run_benchmark_from_config(config_path: str) -> Dict[str, Any]:
    """Run benchmark from configuration file."""
    config = BenchmarkConfig.from_yaml(config_path)
    runner = BenchmarkRunner(config)
    return await runner.run_benchmark()


def run_benchmark_sync(config: BenchmarkConfig) -> Dict[str, Any]:
    """Synchronous wrapper for running benchmark."""
    runner = BenchmarkRunner(config)
    return asyncio.run(runner.run_benchmark())


def _load_latest_case_results(case_dir: Path) -> Dict[int, MutableMapping[str, Any]]:
    """Return the most recent JSON payload for each case index."""
    results: Dict[int, MutableMapping[str, Any]] = {}
    for json_path in sorted(case_dir.glob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            logger.warning("Skipping unreadable case file %s: %s", json_path, exc)
            continue

        idx = data.get("index")
        if idx is None:
            continue
        try:
            idx = int(idx)
        except (TypeError, ValueError):
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
    """Persist a single case result and return the saved path."""
    index = result.get("index")
    try:
        index_int = int(index) if index is not None else -1
    except (TypeError, ValueError):
        index_int = -1

    case_id = result.get("case_id") or (
        f"case_{index_int}" if index_int >= 0 else "case_unknown"
    )
    filename = f"{index_int:04d}_{case_id}.json" if index_int >= 0 else f"{case_id}.json"
    output_path = case_dir / filename
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=str)
    logger.debug("Persisted case %s to %s", case_id, output_path)
    return output_path


def _purge_old_case_files(case_dir: Path, index: int, keep_path: Path) -> None:
    pattern = f"{index:04d}_*.json"
    for candidate in case_dir.glob(pattern):
        if candidate.resolve() == keep_path.resolve():
            continue
        try:
            candidate.unlink()
            logger.debug("Removed outdated case file %s", candidate)
        except Exception as exc:
            logger.warning("Failed to delete %s: %s", candidate, exc)


async def rerun_failed_cases(
    config: BenchmarkConfig,
    *,
    cases_dir: Optional[str] = None,
    keep_history: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Rerun failed benchmark cases without touching successful ones."""

    if verbose:
        logger.setLevel(logging.DEBUG)

    runner = BenchmarkRunner(config)
    runner._configure_environment()  # type: ignore[attr-defined]
    image_paths, labels = runner.dataset_loader.load()
    dataset_size = len(image_paths)
    overall_start_time = time.time()

    case_dir = Path(cases_dir) if cases_dir else Path(config.output.output_dir) / config.output.case_results_subdir
    case_dir.mkdir(parents=True, exist_ok=True)

    stored_results = _load_latest_case_results(case_dir)
    if not stored_results:
        logger.warning("No existing case results found in %s", case_dir)

    failed_indices_all = sorted(
        idx for idx, payload in stored_results.items() if payload.get("status") != "success"
    )
    rerun_targets = [] if dry_run else list(failed_indices_all)

    for idx in rerun_targets:
        if idx >= dataset_size:
            logger.warning(
                "Skipping case index %s because it exceeds dataset size (%s)", idx, dataset_size
            )
            continue

        image_path = image_paths[idx]
        true_label = labels[idx]

        try:
            rerun_result = await runner._run_single_diagnostic(  # pylint: disable=protected-access
                image_path,
                true_label,
                idx,
            )
            saved_path = _persist_case_result(case_dir, rerun_result)
            rerun_result["_source_path"] = str(saved_path)
            rerun_result["_mtime"] = saved_path.stat().st_mtime
            stored_results[idx] = rerun_result
            if not keep_history:
                _purge_old_case_files(case_dir, idx, saved_path)
        except Exception as exc:
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

    ordered_indices = sorted(idx for idx in stored_results.keys() if idx < dataset_size)
    final_results = [stored_results[idx] for idx in ordered_indices]

    runner.results = final_results
    runner.start_time = overall_start_time
    runner.end_time = time.time()

    metrics: Dict[str, Any] = {}
    report: Dict[str, Any] = {}
    metrics_path: Optional[Path] = None

    if final_results:
        metrics = runner._calculate_metrics()  # pylint: disable=protected-access
        runner._final_metrics = metrics  # type: ignore[attr-defined]
        report = runner._generate_report(metrics)  # pylint: disable=protected-access

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = Path(config.output.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    detailed_results_path = summary_dir / f"rerun_results_{timestamp}.json"
    with open(detailed_results_path, "w", encoding="utf-8") as handle:
        json.dump(final_results, handle, indent=2, ensure_ascii=False, default=str)

    if metrics:
        metrics_calc = MetricsCalculator(config.metrics, runner.dataset_loader.class_names)
        metrics_path = summary_dir / f"rerun_metrics_{timestamp}.json"
        metrics_calc.save_metrics_report(metrics, str(metrics_path))

    summary_path = summary_dir / f"rerun_summary_{timestamp}.json"

    summary_payload = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "output_dir": config.output.output_dir,
        "case_results_dir": str(case_dir),
        "dataset_size": dataset_size,
        "processed_cases": len(final_results),
        "initial_failed_indices": failed_indices_all,
        "rerun_indices": rerun_targets,
        "remaining_failures": [
            idx for idx, payload in stored_results.items() if payload.get("status") != "success"
        ],
        "accuracy": metrics.get("accuracy") if metrics else None,
        "macro_f1": metrics.get("f1_score") if metrics else None,
        "macro_precision": metrics.get("precision") if metrics else None,
        "macro_recall": metrics.get("recall") if metrics else None,
        "detailed_results_path": str(detailed_results_path),
        "metrics_report_path": str(metrics_path) if metrics_path else None,
        "report": report,
        "summary_path": str(summary_path),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Rerun complete. Summary written to %s. Remaining failures: %s",
        summary_path,
        summary_payload["remaining_failures"],
    )

    return summary_payload


if __name__ == "__main__":
    # Example usage
    from .config import get_default_config

    # Create default config
    config = get_default_config()

    # Run benchmark
    results = run_benchmark_sync(config)

    print(f"Benchmark completed with accuracy: {results['metrics']['accuracy']:.3f}")
"""Main class for running diagnostic benchmarks."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
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
        if self.config.model.dry_run:
            os.environ["EYEAGENT_DRY_RUN"] = "1"
        else:
            os.environ.pop("EYEAGENT_DRY_RUN", None)

        # Ensure workflow modules pick up the configured MCP server URL
        if self.config.model.mcp_server_url:
            os.environ["MCP_SERVER_URL"] = self.config.model.mcp_server_url
            try:
                import eyeagent.workflows.langgraph as _langgraph

                _langgraph.MCP_SERVER_URL = self.config.model.mcp_server_url
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning(
                    "Unable to synchronize langgraph MCP URL override: %s", exc
                )

        # Progress bar
        pbar = tqdm(total=len(image_paths), desc="Processing samples")

        for i, (image_path, true_label) in enumerate(zip(image_paths, labels)):
            pbar.set_postfix({"current": f"{i+1}/{len(image_paths)}", "label": true_label})

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

                format_context = {"final_fragment": diagnostic_result}
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
        probabilities = self._extract_probabilities(diagnostic_result)

        return {
            "index": index,
            "case_id": case_id,
            "image_path": image_path,
            "true_label": true_label,
            "formatted_diagnosis": formatted_output,
            "extracted_diagnosis": extracted_diagnosis,
            "confidence": confidence,
            "prediction_probabilities": probabilities,
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


if __name__ == "__main__":
    # Example usage
    from .config import get_default_config

    # Create default config
    config = get_default_config()

    # Run benchmark
    results = run_benchmark_sync(config)

    print(f"Benchmark completed with accuracy: {results['metrics']['accuracy']:.3f}")
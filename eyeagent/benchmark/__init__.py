"""
EyeAgent Benchmark Module

This module provides functionality for benchmarking the diagnostic accuracy
of EyeAgent's multiagent framework using classification datasets.

Key Components:
- DatasetLoader: Load and parse classification datasets
- FormatAgent: Standardize agent outputs for evaluation
- MetricsCalculator: Compute accuracy, AUC, F1 scores
- BenchmarkRunner: Orchestrate the benchmarking process
"""

from .dataset_loader import DatasetLoader
from .format_agent import FormatAgent
from .metrics import MetricsCalculator
from .runner import BenchmarkRunner, run_benchmark_from_config, run_benchmark_sync
from .config import (
    BenchmarkConfig,
    DatasetConfig,
    ModelConfig,
    MetricsConfig,
    OutputConfig,
    RunnerConfig,
)

__all__ = [
    "DatasetLoader",
    "FormatAgent", 
    "MetricsCalculator",
    "BenchmarkRunner",
    "BenchmarkConfig",
    "DatasetConfig",
    "ModelConfig",
    "MetricsConfig",
    "OutputConfig",
    "RunnerConfig",
    "run_benchmark_from_config",
    "run_benchmark_sync",
]

__version__ = "1.0.0"
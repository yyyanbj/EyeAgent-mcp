# EyeAgent Benchmark Configuration Examples

This directory contains example configurations for different benchmarking scenarios.

## Available Examples

### 1. basic_benchmark.yaml
Basic benchmark configuration for a simple classification dataset.

### 2. dr_screening_benchmark.yaml
Configuration specifically for diabetic retinopathy screening evaluation.

### 3. multi_disease_benchmark.yaml  
Configuration for multi-disease classification benchmarking.

### 4. dry_run_benchmark.yaml
Configuration for testing the benchmark pipeline without actual model inference.

## Usage

```python
from eyeagent.benchmark import run_benchmark_from_config

# Run benchmark with specific config
results = await run_benchmark_from_config("examples/basic_benchmark.yaml")
```

## Configuration Fields

See `../config.py` for detailed documentation of all configuration options.
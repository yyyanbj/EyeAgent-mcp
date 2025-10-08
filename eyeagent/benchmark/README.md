# EyeAgent Benchmark Module

The EyeAgent Benchmark module provides comprehensive evaluation capabilities for the multiagent diagnostic framework. It enables systematic assessment of diagnostic accuracy using classification datasets with various metrics including accuracy, AUC, F1-score, precision, and recall.

## Overview

The benchmark module consists of several key components:

- **DatasetLoader**: Load and preprocess classification datasets
- **FormatAgent**: Standardize diagnostic outputs for evaluation  
- **MetricsCalculator**: Compute evaluation metrics
- **BenchmarkRunner**: Orchestrate the complete benchmarking process
- **Configuration**: Flexible configuration system for different scenarios

## Key Features

- Support for multiple dataset formats (CSV, JSON, JSONL)
- Standardized output formatting: "The diagnosis of this image is XXX"
- Comprehensive metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC
- Per-class and overall performance evaluation
- Confusion matrix and ROC curve visualization
- Configurable evaluation scenarios
- Dry-run mode for testing
- Detailed reporting and result persistence

## Quick Start

### 1. Basic Usage

```python
import asyncio
from eyeagent.benchmark import BenchmarkConfig, BenchmarkRunner
from eyeagent.benchmark.config import DatasetConfig

# Create configuration
config = BenchmarkConfig(
    dataset=DatasetConfig(
        name="my_dataset",
        path="./data/classification_dataset.csv",
        image_column="image_path",
        label_column="diagnosis",
        class_names=["Normal", "DR", "AMD", "Glaucoma"]
    )
)

# Run benchmark
runner = BenchmarkRunner(config)
results = await runner.run_benchmark()

print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
```

### 2. Configuration File Usage

```python
from eyeagent.benchmark import run_benchmark_from_config

# Run from YAML config
results = await run_benchmark_from_config("config/benchmark.yaml")
```

### 3. Dataset Format

Your dataset should be a CSV file with columns for image paths and labels:

```csv
image_path,diagnosis,patient_id
./images/001.jpg,Normal,P001
./images/002.jpg,DR,P002
./images/003.jpg,AMD,P003
```

## Configuration

### Dataset Configuration

```yaml
dataset:
  name: "my_dataset"
  path: "./data/dataset.csv"
  image_column: "image_path"
  label_column: "diagnosis"
  class_names: ["Normal", "DR", "AMD", "Glaucoma"]
  max_samples: 100  # Optional: limit samples
```

### Model Configuration

```yaml
model:
  workflow_backend: "langgraph"
  mcp_server_url: "http://localhost:8000/mcp/"
  dry_run: false
  enable_format_agent: true
```

### Metrics Configuration

```yaml
metrics:
  compute_accuracy: true
  compute_auc: true
  compute_f1: true
  compute_precision: true
  compute_recall: true
  average: "macro"  # macro, micro, weighted
```

### Output Configuration

```yaml
output:
  output_dir: "./benchmark_results"
  save_predictions: true
  save_detailed_report: true
  save_confusion_matrix: true
  verbose: true
```

## Architecture

### Data Flow

1. **Dataset Loading**: Load images and labels from dataset file
2. **Diagnostic Workflow**: Run EyeAgent diagnostic workflow on each image
3. **Output Formatting**: Standardize outputs using FormatAgent
4. **Metrics Calculation**: Compute evaluation metrics
5. **Report Generation**: Create comprehensive reports and visualizations

### Format Agent

The FormatAgent ensures consistent output formatting for evaluation:

- Extracts primary diagnosis from complex workflow outputs
- Standardizes to format: "The diagnosis of this image is [DIAGNOSIS]"
- Maps diagnoses to valid class names
- Handles multiple extraction strategies for robustness

### Metrics

Supported evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class and macro/micro/weighted precision
- **Recall**: Per-class and macro/micro/weighted recall
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve (binary and multi-class)
- **Confusion Matrix**: Detailed classification breakdown

## Examples

See the `examples/` directory for various configuration examples:

- `basic_benchmark.yaml`: Basic classification evaluation
- `dr_screening_benchmark.yaml`: Diabetic retinopathy screening
- `multi_disease_benchmark.yaml`: Multi-disease classification
- `dry_run_benchmark.yaml`: Testing without model inference

### Running Examples

```bash
# Run the example script
cd eyeagent/benchmark/examples
python run_example.py
```

## Output

The benchmark generates several output files:

- `benchmark_results_[timestamp].json`: Detailed per-sample results
- `metrics_report_[timestamp].json`: Comprehensive metrics
- `benchmark_config_[timestamp].yaml`: Configuration used
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curves.png`: ROC curves for multi-class problems

### Sample Output Structure

```json
{
  "config": {...},
  "metrics": {
    "accuracy": 0.85,
    "f1_score": 0.82,
    "precision": 0.84,
    "recall": 0.81,
    "per_class": {
      "Normal": {"precision": 0.90, "recall": 0.88, "f1_score": 0.89},
      "DR": {"precision": 0.78, "recall": 0.75, "f1_score": 0.76}
    },
    "confusion_matrix": [[45, 2, 1, 0], [3, 38, 2, 1], ...]
  },
  "runtime": 125.3
}
```

## Best Practices

### Dataset Preparation

1. Ensure image paths are correct (absolute or relative to dataset file)
2. Use consistent class naming
3. Balance dataset if possible
4. Validate image files are readable
5. Use appropriate train/validation/test splits

### Configuration

1. Start with dry-run mode for testing
2. Use smaller sample sizes during development
3. Enable verbose output for debugging
4. Configure appropriate metrics for your use case
5. Set reasonable timeouts for large datasets

### Performance Optimization

1. Use `max_samples` to limit dataset size during development
2. Disable detailed reports for large-scale evaluations
3. Consider batch processing for very large datasets
4. Monitor resource usage during evaluation

## Troubleshooting

### Common Issues

1. **Image files not found**: Check image paths in dataset
2. **Format agent errors**: Verify diagnostic outputs contain expected fields
3. **Memory issues**: Reduce batch size or use sampling
4. **MCP connection errors**: Ensure MCP server is running

### Debug Mode

Enable verbose logging and dry-run mode for debugging:

```python
config.model.dry_run = True
config.output.verbose = True
```

## Integration

The benchmark module integrates seamlessly with the existing EyeAgent framework:

- Uses the same diagnostic workflow
- Leverages existing agent configurations
- Compatible with all workflow backends
- Supports existing tool ecosystem

## Future Enhancements

Planned improvements:

- Support for additional dataset formats
- Integration with popular ML datasets
- Advanced visualization options
- Multi-modal evaluation support
- Automated hyperparameter tuning
- Cross-validation support

## API Reference

For detailed API documentation, see:

- `config.py`: Configuration classes and utilities
- `dataset_loader.py`: Dataset loading and preprocessing
- `format_agent.py`: Output formatting and standardization
- `metrics.py`: Evaluation metrics calculation
- `runner.py`: Main benchmark orchestration
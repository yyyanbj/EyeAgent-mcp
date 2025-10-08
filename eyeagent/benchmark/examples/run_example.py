#!/usr/bin/env python3
"""
Example script demonstrating how to use the EyeAgent benchmark module.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import eyeagent
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eyeagent.benchmark import (
    BenchmarkConfig, 
    BenchmarkRunner,
    DatasetConfig,
    ModelConfig,
    MetricsConfig,
    OutputConfig
)
from eyeagent.benchmark.dataset_loader import create_sample_dataset


async def run_example_benchmark():
    """Run an example benchmark with synthetic data."""
    
    print("EyeAgent Benchmark Example")
    print("=" * 50)
    
    # Create sample dataset
    dataset_path = "./sample_data/test_dataset.csv"
    print(f"Creating sample dataset at {dataset_path}")
    create_sample_dataset(dataset_path, num_samples=5)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset=DatasetConfig(
            name="example_test",
            path=dataset_path,
            image_column="image_path",
            label_column="diagnosis",
            class_names=["Normal", "DR", "AMD", "Glaucoma"],
            max_samples=5
        ),
        model=ModelConfig(
            dry_run=True,  # Use dry run for this example
            enable_format_agent=True
        ),
        metrics=MetricsConfig(
            compute_auc=False  # Skip AUC for small dataset
        ),
        output=OutputConfig(
            output_dir="./example_results",
            verbose=True
        )
    )
    
    print("\nConfiguration:")
    print(f"  Dataset: {config.dataset.name}")
    print(f"  Samples: {config.dataset.max_samples}")
    print(f"  Classes: {config.dataset.class_names}")
    print(f"  Dry run: {config.model.dry_run}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    runner = BenchmarkRunner(config)
    results = await runner.run_benchmark()
    
    # Print results
    print("\nResults:")
    print("=" * 30)
    print(f"Total samples: {len(results['results'])}")
    print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    print(f"F1-score: {results['metrics']['f1_score']:.3f}")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    
    # Print per-class results
    print("\nPer-class metrics:")
    for class_name, metrics in results['metrics']['per_class'].items():
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1-score: {metrics['f1_score']:.3f}")
    
    print(f"\nDetailed results saved to: {config.output.output_dir}")
    
    return results


async def run_config_based_benchmark():
    """Run benchmark from configuration file."""
    
    print("\nRunning benchmark from configuration file...")
    
    # Use the basic benchmark config
    config_path = Path(__file__).parent / "basic_benchmark.yaml"
    
    # Update the config to use dry run and sample data
    from eyeagent.benchmark import run_benchmark_from_config
    
    try:
        results = await run_benchmark_from_config(str(config_path))
        print(f"Config-based benchmark completed with accuracy: {results['metrics']['accuracy']:.3f}")
        return results
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print("Skipping config-based benchmark")
        return None


def demonstrate_metrics_calculation():
    """Demonstrate metrics calculation with sample data."""
    
    print("\nDemonstrating metrics calculation...")
    
    from eyeagent.benchmark.metrics import MetricsCalculator, MetricsConfig
    
    # Sample predictions
    y_true = ["Normal", "DR", "AMD", "Normal", "Glaucoma", "DR", "Normal", "AMD"]
    y_pred = ["Normal", "DR", "Normal", "Normal", "Glaucoma", "DR", "DR", "AMD"]
    
    class_names = ["Normal", "DR", "AMD", "Glaucoma"]
    
    # Calculate metrics
    config = MetricsConfig()
    calculator = MetricsCalculator(config, class_names)
    metrics = calculator.calculate_all_metrics(y_true, y_pred)
    
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-score: {metrics['f1_score']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    
    return metrics


def main():
    """Main function to run all examples."""
    
    print("Starting EyeAgent Benchmark Examples")
    print("=" * 60)
    
    # Ensure output directories exist
    os.makedirs("./sample_data", exist_ok=True)
    os.makedirs("./example_results", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Basic benchmark
        results1 = asyncio.run(run_example_benchmark())
        
        # Example 2: Metrics calculation
        metrics = demonstrate_metrics_calculation()
        
        # Example 3: Config-based benchmark (optional)
        results2 = asyncio.run(run_config_based_benchmark())
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
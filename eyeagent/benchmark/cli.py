#!/usr/bin/env python3
"""
Command-line interface for EyeAgent Benchmark module.

This script provides a convenient way to run benchmarks from the command line
using configuration files or command-line arguments.
"""

import argparse
import asyncio
import sys
import os
from pathlib import Path
import yaml
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import (
    BenchmarkConfig,
    BenchmarkRunner,
    DatasetConfig,
    ModelConfig,
    MetricsConfig,
    OutputConfig,
    RunnerConfig,
    rerun_failed_cases,
)
from benchmark.dataset_loader import create_sample_dataset


def create_parser():
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="EyeAgent Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with config file
  python cli.py run --config examples/basic_benchmark.yaml
  
  # Run with command-line arguments
  python cli.py run --dataset ./data/test.csv --classes Normal DR AMD --output ./results
  
  # Create sample dataset
  python cli.py create-dataset --output ./sample_data/test.csv --samples 20
  
  # Generate default config
  python cli.py generate-config --output ./my_config.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run benchmark command
    run_parser = subparsers.add_parser('run', help='Run benchmark evaluation')
    run_parser.add_argument('--config', '-c', type=str, help='Path to YAML configuration file')
    run_parser.add_argument('--dataset', '-d', type=str, help='Path to dataset file')
    run_parser.add_argument('--image-column', type=str, default='image_path', 
                           help='Column name for image paths')
    run_parser.add_argument('--label-column', type=str, default='label',
                           help='Column name for labels')
    run_parser.add_argument('--classes', nargs='+', 
                           help='List of class names (e.g., Normal DR AMD)')
    run_parser.add_argument('--max-samples', type=int, help='Maximum number of samples to evaluate')
    run_parser.add_argument('--output', '-o', type=str, default='./benchmark_results',
                           help='Output directory for results')
    run_parser.add_argument('--dry-run', action='store_true', 
                           help='Run in dry-run mode (mock outputs)')
    run_parser.add_argument('--mcp-url', type=str, default='http://localhost:8000/mcp/',
                           help='MCP server URL')
    run_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Enable verbose output')
    run_parser.add_argument('--no-format-agent', action='store_true',
                           help='Disable format agent')
    run_parser.add_argument('--concurrency', type=int,
                           help='Number of samples to evaluate in parallel (0 = auto)')
    run_parser.add_argument('--force-rerun', action='store_true',
                           help='Ignore cached results and force a fresh run')
    
    # Rerun failed cases command
    rerun_parser = subparsers.add_parser('rerun-failed', help='Rerun only failed benchmark cases')
    rerun_parser.add_argument('--config', '-c', required=True, help='Path to YAML configuration file')
    rerun_parser.add_argument('--cases-dir', type=str, help='Override case results directory (defaults to <output_dir>/case_results)')
    rerun_parser.add_argument('--keep-history', action='store_true', help='Preserve previous case JSON files alongside refreshed ones')
    rerun_parser.add_argument('--dry-run', action='store_true', help='Skip diagnostics and only recompute metrics from existing results')
    rerun_parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging output')

    # Create dataset command
    create_parser = subparsers.add_parser('create-dataset', help='Create sample dataset')
    create_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output path for dataset file')
    create_parser.add_argument('--samples', '-n', type=int, default=10,
                              help='Number of samples to generate')
    
    # Generate config command
    config_parser = subparsers.add_parser('generate-config', help='Generate configuration file')
    config_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output path for configuration file')
    config_parser.add_argument('--template', type=str, default='basic',
                              choices=['basic', 'dr', 'multi-disease', 'dry-run'],
                              help='Configuration template to use')
    
    return parser


async def run_benchmark_cli(args):
    """Run benchmark from CLI arguments."""
    
    if args.config:
        # Use configuration file
        logger.info(f"Running benchmark with config: {args.config}")
        config = BenchmarkConfig.from_yaml(args.config)

        # Ensure backend env is set according to config for this run
        if config.model.workflow_backend:
            os.environ["EYEAGENT_WORKFLOW_BACKEND"] = str(config.model.workflow_backend)

        if args.concurrency is not None:
            config.runner.concurrency = args.concurrency
        if args.force_rerun:
            config.runner.skip_existing_results = False

        runner = BenchmarkRunner(config)
        results = await runner.run_benchmark()

    else:
        # Build config from CLI arguments
        if not args.dataset:
            logger.error("Either --config or --dataset must be specified")
            return 1
        
        if not args.classes:
            logger.error("--classes must be specified when not using config file")
            return 1
        
        logger.info(f"Running benchmark with dataset: {args.dataset}")
        
        runner_config = RunnerConfig()
        if args.concurrency is not None:
            runner_config.concurrency = args.concurrency
        if args.force_rerun:
            runner_config.skip_existing_results = False

        config = BenchmarkConfig(
            dataset=DatasetConfig(
                name=Path(args.dataset).stem,
                path=args.dataset,
                image_column=args.image_column,
                label_column=args.label_column,
                class_names=args.classes,
                max_samples=args.max_samples
            ),
            model=ModelConfig(
                mcp_server_url=args.mcp_url,
                dry_run=args.dry_run,
                enable_format_agent=not args.no_format_agent
            ),
            output=OutputConfig(
                output_dir=args.output,
                verbose=args.verbose
            ),
            runner=runner_config,
        )
        if config.model.workflow_backend:
            os.environ["EYEAGENT_WORKFLOW_BACKEND"] = str(config.model.workflow_backend)
        
        runner = BenchmarkRunner(config)
        results = await runner.run_benchmark()
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"Dataset: {results['config']['dataset']['name']}")
    print(f"Samples: {len(results['results'])}")
    print(f"Classes: {results['config']['dataset']['class_names']}")
    print(f"Concurrency: {results['config']['runner']['concurrency']}")
    print(f"Runtime: {results['runtime']:.2f} seconds")
    print()
    print("METRICS:")
    metrics = results['metrics']
    print(f"  Accuracy:  {metrics.get('accuracy', 0):.3f}")
    print(f"  F1-Score:  {metrics.get('f1_score', 0):.3f}")
    print(f"  Precision: {metrics.get('precision', 0):.3f}")
    print(f"  Recall:    {metrics.get('recall', 0):.3f}")
    if 'auc_roc_macro' in metrics:
        print(f"  AUC-ROC:   {metrics['auc_roc_macro']:.3f}")
    
    print("\nPER-CLASS METRICS:")
    for class_name, class_metrics in metrics.get('per_class', {}).items():
        print(f"  {class_name}:")
        print(f"    F1: {class_metrics['f1_score']:.3f}, "
              f"Precision: {class_metrics['precision']:.3f}, "
              f"Recall: {class_metrics['recall']:.3f}")
    
    print(f"\nDetailed results saved to: {results['config']['output']['output_dir']}")
    print("="*60)
    
    return 0


async def rerun_failed_cli(args):
    """Rerun only failed cases from a previous benchmark run."""

    logger.info(f"Rerunning failed cases with config: {args.config}")
    config = BenchmarkConfig.from_yaml(args.config)

    summary = await rerun_failed_cases(
        config,
        cases_dir=args.cases_dir,
        keep_history=args.keep_history,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    print("\n" + "=" * 60)
    print("FAILED CASE RERUN SUMMARY")
    print("=" * 60)
    print(f"Config: {summary['config']['dataset']['name']}")
    print(f"Case directory: {summary['case_results_dir']}")
    print(f"Initial failed indices: {summary['initial_failed_indices']}")
    print(f"Failed indices rerun: {summary['rerun_indices']}")
    print(f"Remaining failures: {summary['remaining_failures']}")
    accuracy = summary.get('accuracy')
    f1 = summary.get('macro_f1')
    print(f"Accuracy: {accuracy:.3f}" if accuracy is not None else "Accuracy: N/A")
    print(f"Macro F1: {f1:.3f}" if f1 is not None else "Macro F1: N/A")
    print(f"Detailed results: {summary['detailed_results_path']}")
    metrics_path = summary.get('metrics_report_path')
    if metrics_path:
        print(f"Metrics report: {metrics_path}")
    detailed_results_path = Path(summary['detailed_results_path'])
    summary_path = summary.get('summary_path')
    if summary_path:
        print(f"Summary file: {summary_path}")
    else:
        summary_file_name = detailed_results_path.name.replace('results', 'summary')
        fallback_summary_path = Path(summary['output_dir']) / summary_file_name
        print(f"Summary file: {fallback_summary_path}")
    print("=" * 60 + "\n")

    return 0


def create_dataset_cli(args):
    """Create sample dataset from CLI arguments."""
    logger.info(f"Creating sample dataset: {args.output}")
    
    try:
        create_sample_dataset(args.output, args.samples)
        print(f"Sample dataset created successfully at: {args.output}")
        print(f"Number of samples: {args.samples}")
        return 0
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}")
        return 1


def generate_config_cli(args):
    """Generate configuration file from CLI arguments."""
    logger.info(f"Generating config template: {args.template}")
    
    # Configuration templates
    templates = {
        'basic': {
            'dataset': {
                'name': 'my_dataset',
                'path': './data/dataset.csv',
                'image_column': 'image_path',
                'label_column': 'diagnosis',
                'class_names': ['Normal', 'DR', 'AMD', 'Glaucoma'],
                'max_samples': 100
            },
            'model': {
                'workflow_backend': 'langgraph',
                'mcp_server_url': 'http://localhost:8000/mcp/',
                'dry_run': False,
                'enable_format_agent': True
            },
            'metrics': {
                'compute_accuracy': True,
                'compute_auc': True,
                'compute_f1': True,
                'compute_precision': True,
                'compute_recall': True,
                'average': 'macro'
            },
            'output': {
                'output_dir': './benchmark_results',
                'save_predictions': True,
                'save_detailed_report': True,
                'save_confusion_matrix': True,
                'verbose': True
            }
        },
        'dr': {
            'dataset': {
                'name': 'dr_dataset',
                'path': './data/dr_dataset.csv',
                'image_column': 'image_path',
                'label_column': 'dr_grade',
                'class_names': ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'PDR']
            },
            'model': {'enable_format_agent': True},
            'metrics': {'average': 'macro'},
            'output': {'output_dir': './dr_results'}
        },
        'multi-disease': {
            'dataset': {
                'name': 'multi_disease',
                'path': './data/eye_diseases.csv',
                'class_names': [
                    'Normal', 'Diabetic Retinopathy', 'Age-related Macular Degeneration',
                    'Glaucoma', 'Cataract', 'Hypertensive Retinopathy'
                ]
            },
            'metrics': {'average': 'weighted'},
            'output': {'output_dir': './multi_disease_results'}
        },
        'dry-run': {
            'dataset': {
                'name': 'test_dataset',
                'path': './data/test.csv',
                'max_samples': 10
            },
            'model': {'dry_run': True},
            'metrics': {'compute_auc': False},
            'output': {'output_dir': './test_results', 'save_detailed_report': False}
        }
    }
    
    if args.template not in templates:
        logger.error(f"Unknown template: {args.template}")
        return 1
    
    try:
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save template
        template_config = templates[args.template]
        with open(args.output, 'w') as f:
            yaml.dump(template_config, f, default_flow_style=False, indent=2)
        
        print(f"Configuration template '{args.template}' saved to: {args.output}")
        print("Edit the configuration file to match your dataset and requirements.")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate config: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Configure logging
    logger.remove()  # Remove default handler
    verbose = getattr(args, 'verbose', False)
    info_commands = {'run', 'rerun-failed'}
    level = "INFO" if (args.command in info_commands and verbose) else "WARNING"
    logger.add(sys.stderr, level=level)
    
    try:
        if args.command == 'run':
            return asyncio.run(run_benchmark_cli(args))
        elif args.command == 'rerun-failed':
            return asyncio.run(rerun_failed_cli(args))
        elif args.command == 'create-dataset':
            return create_dataset_cli(args)
        elif args.command == 'generate-config':
            return generate_config_cli(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
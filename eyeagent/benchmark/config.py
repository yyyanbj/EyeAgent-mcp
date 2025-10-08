"""
Configuration for EyeAgent Benchmark module.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import yaml
import os


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    name: str
    path: str
    image_column: str = "image_path"
    label_column: str = "label" 
    class_names: Optional[List[str]] = None
    split: str = "test"  # test, validation, or train
    max_samples: Optional[int] = None  # Limit number of samples for testing


@dataclass
class ModelConfig:
    """Configuration for the agent model being benchmarked."""
    workflow_backend: str = "langgraph"
    mcp_server_url: str = "http://localhost:8000/mcp/"
    dry_run: bool = False
    enable_format_agent: bool = True


@dataclass 
class MetricsConfig:
    """Configuration for evaluation metrics."""
    compute_accuracy: bool = True
    compute_auc: bool = True
    compute_f1: bool = True
    compute_precision: bool = True
    compute_recall: bool = True
    average: str = "macro"  # macro, micro, weighted for multi-class metrics
    

@dataclass
class OutputConfig:
    """Configuration for benchmark output and reporting."""
    output_dir: str = "./benchmark_results"
    save_predictions: bool = True
    save_detailed_report: bool = True
    save_confusion_matrix: bool = True
    verbose: bool = True
    save_case_results: bool = True
    case_results_subdir: str = "case_results"


@dataclass
class BenchmarkConfig:
    """Main configuration class for benchmarking."""
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'BenchmarkConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            dataset=DatasetConfig(**config_dict['dataset']),
            model=ModelConfig(**config_dict.get('model', {})),
            metrics=MetricsConfig(**config_dict.get('metrics', {})),
            output=OutputConfig(**config_dict.get('output', {}))
        )
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'dataset': self.dataset.__dict__,
            'model': self.model.__dict__,
            'metrics': self.metrics.__dict__,
            'output': self.output.__dict__
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        dataset=DatasetConfig(
            name="example_dataset",
            path="./data/example_dataset.csv",
            image_column="image_path",
            label_column="diagnosis",
            class_names=["Normal", "DR", "AMD", "Glaucoma"]
        )
    )
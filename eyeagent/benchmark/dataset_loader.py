"""
Dataset loader for EyeAgent benchmark module.

Supports loading classification datasets in various formats including:
- CSV files with image paths and labels
- JSONL files
- Custom ophthalmology dataset formats
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from loguru import logger

from .config import DatasetConfig


class DatasetLoader:
    """Load and preprocess classification datasets for benchmarking."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.class_names: Optional[List[str]] = None
        
    def load(self) -> Tuple[List[str], List[str]]:
        """
        Load dataset and return image paths and labels.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        logger.info(f"Loading dataset: {self.config.name} from {self.config.path}")
        
        if not os.path.exists(self.config.path):
            raise FileNotFoundError(f"Dataset file not found: {self.config.path}")
        
        # Determine file format and load accordingly
        if self.config.path.endswith('.csv'):
            self.data = self._load_csv()
        elif self.config.path.endswith('.jsonl'):
            self.data = self._load_jsonl()
        elif self.config.path.endswith('.json'):
            self.data = self._load_json()
        else:
            raise ValueError(f"Unsupported file format: {self.config.path}")
        
        # Extract image paths and labels
        image_paths = self.data[self.config.image_column].tolist()
        labels = self.data[self.config.label_column].tolist()
        
        # Convert relative paths to absolute paths
        image_paths = self._resolve_image_paths(image_paths)
        
        # Validate images exist
        image_paths, labels = self._validate_images(image_paths, labels)
        
        # Apply sample limit if specified
        if self.config.max_samples and len(image_paths) > self.config.max_samples:
            logger.info(f"Limiting dataset to {self.config.max_samples} samples")
            image_paths = image_paths[:self.config.max_samples]
            labels = labels[:self.config.max_samples]
        
        # Set class names
        if self.config.class_names:
            self.class_names = self.config.class_names
        else:
            self.class_names = sorted(list(set(labels)))
        
        logger.info(f"Loaded {len(image_paths)} samples with {len(self.class_names)} classes")
        logger.info(f"Classes: {self.class_names}")
        
        return image_paths, labels
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV file."""
        return pd.read_csv(self.config.path)
    
    def _load_jsonl(self) -> pd.DataFrame:
        """Load JSONL file."""
        data = []
        with open(self.config.path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return pd.DataFrame(data)
    
    def _load_json(self) -> pd.DataFrame:
        """Load JSON file."""
        with open(self.config.path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                return pd.DataFrame(data['data'])
            else:
                return pd.DataFrame([data])
        else:
            raise ValueError("Invalid JSON format")
    
    def _resolve_image_paths(self, image_paths: List[str]) -> List[str]:
        """Convert relative paths to absolute paths."""
        dataset_dir = os.path.dirname(os.path.abspath(self.config.path))
        resolved_paths = []
        
        for path in image_paths:
            if os.path.isabs(path):
                resolved_paths.append(path)
            else:
                # Try relative to dataset file
                abs_path = os.path.join(dataset_dir, path)
                if os.path.exists(abs_path):
                    resolved_paths.append(abs_path)
                else:
                    # Try as-is (might be relative to current working directory)
                    resolved_paths.append(path)
        
        return resolved_paths
    
    def _validate_images(self, image_paths: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """Validate that image files exist and are readable."""
        valid_paths = []
        valid_labels = []
        invalid_count = 0
        
        for path, label in zip(image_paths, labels):
            if os.path.exists(path):
                try:
                    # Try to open image to verify it's valid
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_labels.append(label)
                except Exception as e:
                    logger.warning(f"Invalid image file {path}: {e}")
                    invalid_count += 1
            else:
                logger.warning(f"Image file not found: {path}")
                invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Excluded {invalid_count} invalid/missing images")
        
        return valid_paths, valid_labels
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        return self.data[self.config.label_column].value_counts().to_dict()
    
    def get_sample_info(self, index: int) -> Dict[str, Any]:
        """Get detailed information about a specific sample."""
        if self.data is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        if index >= len(self.data):
            raise IndexError(f"Index {index} out of range")
        
        sample = self.data.iloc[index]
        return {
            'image_path': sample[self.config.image_column],
            'label': sample[self.config.label_column],
            'index': index,
            'additional_fields': {k: v for k, v in sample.items() 
                                if k not in [self.config.image_column, self.config.label_column]}
        }


def create_sample_dataset(output_path: str, num_samples: int = 10) -> None:
    """
    Create a sample dataset file for testing.
    
    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of sample entries to create
    """
    import random
    
    # Generate sample data
    diseases = ["Normal", "DR", "AMD", "Glaucoma"]
    data = []
    
    for i in range(num_samples):
        data.append({
            "image_path": f"sample_images/image_{i:03d}.jpg",
            "diagnosis": random.choice(diseases),
            "patient_id": f"P{i:04d}",
            "age": random.randint(30, 80),
            "eye": random.choice(["left", "right"])
        })
    
    # Save as CSV
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created sample dataset with {num_samples} entries at {output_path}")


if __name__ == "__main__":
    # Create a sample dataset for testing
    create_sample_dataset("./sample_data/test_dataset.csv")
"""
Base dataset class for EyeAgent datasets.

This module provides a base class that all EyeAgent datasets should inherit from,
ensuring consistent interface and functionality across different dataset types.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import warnings

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Base class for all EyeAgent datasets.
    
    This abstract class provides common functionality and interface
    that all dataset implementations should follow.
    """
    
    def __init__(self, 
                 root_path: str, 
                 image_path: str, 
                 file_path: str,
                 image_format: str = "jpg"):
        """
        Initialize the base dataset.
        
        Args:
            root_path: Root directory of the dataset
            image_path: Relative path to image directory from root_path
            file_path: Relative path to CSV/data file from root_path
            image_format: Image file format (jpg, png, etc.) or "none" for auto-detect
        """
        self.root_path = Path(root_path)
        self.image_path = image_path
        self.file_path = file_path
        self.image_format = image_format
        
        # Validate paths
        if not self.root_path.exists():
            raise FileNotFoundError(f"Root path does not exist: {root_path}")
        
        data_file = self.root_path / file_path
        if not data_file.exists():
            raise FileNotFoundError(f"Data file does not exist: {data_file}")
        
        # Load the dataset
        self.df = self._load_dataframe(data_file)
        
        # Validate dataset
        self._validate_dataset()
    
    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """Load dataframe from file."""
        if file_path.suffix.lower() == '.csv':
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            return pd.read_json(file_path)
        elif file_path.suffix.lower() == '.jsonl':
            return pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _validate_dataset(self):
        """Validate the loaded dataset."""
        if self.df.empty:
            raise ValueError("Dataset is empty")
        
        # Check required columns
        required_columns = self.get_required_columns()
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """
        Return list of required column names for this dataset type.
        
        Returns:
            List of required column names
        """
        pass
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.df)
    
    def get_image_path(self, image_id: str) -> str:
        """
        Get the full path to an image file.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Full path to the image file
        """
        image_dir = self.root_path / self.image_path
        
        if self.image_format == "none":
            # Auto-detect image format
            image_files = list(image_dir.glob(f"{image_id}.*"))
            if len(image_files) == 0:
                warnings.warn(f"Image file for {image_id} not found. Using default format.")
                return str(image_dir / f"{image_id}.jpg")
            else:
                if len(image_files) > 1:
                    warnings.warn(f"Multiple image files found for {image_id}. Using the first one.")
                # Prefer common image formats (jpg, png, etc.)
                image_files = sorted(image_files, key=lambda x: self._get_format_priority(x.suffix))
                return str(image_files[0])
        else:
            return str(image_dir / f"{image_id}.{self.image_format}")
    
    def _get_format_priority(self, suffix: str) -> int:
        """Get priority for image format selection."""
        priority_map = {
            '.jpg': 0, '.jpeg': 1, '.png': 2, '.bmp': 3, '.tiff': 4, '.tif': 5
        }
        return priority_map.get(suffix.lower(), 999)
    
    def validate_image_exists(self, image_path: str) -> bool:
        """
        Check if an image file exists.
        
        Args:
            image_path: Path to image file
            
        Returns:
            True if image exists, False otherwise
        """
        return Path(image_path).exists()
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample information
        """
        if idx >= len(self.df):
            raise IndexError(f"Index {idx} out of range")
        
        sample = self.df.iloc[idx]
        return {
            'index': idx,
            'raw_data': sample.to_dict(),
            'dataset_info': {
                'root_path': str(self.root_path),
                'image_path': self.image_path,
                'total_samples': len(self.df)
            }
        }
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            'total_samples': len(self.df),
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict()
        }
        
        # Add column-specific stats
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'string']:
                stats[f'{col}_unique_values'] = self.df[col].nunique()
                if self.df[col].nunique() < 20:  # Show values if not too many
                    stats[f'{col}_values'] = self.df[col].value_counts().to_dict()
        
        return stats
    
    def filter_samples(self, condition: str) -> 'BaseDataset':
        """
        Filter samples based on a condition.
        
        Args:
            condition: Pandas query condition string
            
        Returns:
            New dataset instance with filtered samples
        """
        filtered_df = self.df.query(condition)
        
        # Create new instance with filtered data
        new_dataset = self.__class__(
            root_path=str(self.root_path),
            image_path=self.image_path,
            file_path=self.file_path,
            image_format=self.image_format
        )
        new_dataset.df = filtered_df
        
        return new_dataset
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        pass
    
    @abstractmethod
    def generate_prompt(self, sample: pd.Series) -> tuple:
        """
        Generate prompt for the given sample.
        
        Args:
            sample: Pandas Series containing sample data
            
        Returns:
            Tuple of (prompt, extra_info)
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (f"{self.__class__.__name__}("
                f"root_path='{self.root_path}', "
                f"samples={len(self.df)}, "
                f"image_format='{self.image_format}')")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.__class__.__name__} with {len(self.df)} samples"
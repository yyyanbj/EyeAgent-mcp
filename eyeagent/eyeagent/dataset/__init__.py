"""
EyeAgent Dataset Module

This module provides dataset classes for various eye disease datasets 
and visual question answering (VQA) tasks in ophthalmology.

Supported dataset types:
- VQA datasets for eye disease diagnosis
- Classification datasets for benchmarking
- Multi-modal datasets combining different imaging modalities
"""

from .vqa_dataset import VQADataset
from .base_dataset import BaseDataset

__all__ = [
    "VQADataset",
    "BaseDataset"
]

__version__ = "1.0.0"
"""
VQA (Visual Question Answering) Dataset for EyeAgent.

This module provides a dataset class for visual question answering tasks
in ophthalmology, compatible with EyeAgent's diagnostic framework.
"""

import os
import pandas as pd
from pathlib import Path
import warnings
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset

from .base_dataset import BaseDataset


class VQADataset(BaseDataset):
    """
    Visual Question Answering Dataset for eye disease diagnosis.
    
    This dataset is designed for VQA tasks where the model needs to answer
    questions about eye images, often requiring diagnostic reasoning.
    """
    
    def __init__(self, 
                 root_path: str, 
                 image_path: str, 
                 file_path: str, 
                 image_format: str = "jpg"):
        """
        Initialize the VQADataset.
        
        Args:
            root_path: Root directory of the dataset
            image_path: Relative path to image directory from root_path
            file_path: Relative path to CSV/data file from root_path
            image_format: Image file format (jpg, png, etc.) or "none" for auto-detect
        """
        super().__init__(root_path, image_path, file_path, image_format)
    
    def get_required_columns(self) -> List[str]:
        """
        Return list of required column names for VQA dataset.
        
        Returns:
            List of required column names
        """
        return ['imid', 'question', 'answer']
    
    def generate_prompt(self, sample: pd.Series) -> Tuple[str, str]:
        """
        Generate a prompt for the given sample.
        
        Args:
            sample: Pandas Series containing sample data
            
        Returns:
            Tuple of (prompt, extra_info)
        """
        prompt = ""
        extra_info = ""

        prompt += f"Use tools to answer the following question based on the image.\n"
        prompt += f"Question: {sample['question']}\n"

        extra_info += "Briefly answer the question based on the findings, but do not use tools any more. Use the following format:\n"
        extra_info += "The diagnosis of this image is <diagnosis>\n"

        return prompt, extra_info
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Generate a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        # Get the sample data for the current index
        sample = self.df.iloc[idx]

        # Extract required fields
        image_id = sample['imid']
        question = sample['question']
        reference_answer = sample['answer']
        
        # Handle optional modality field
        modality = sample.get('modality', 'CFP')  # Default to CFP if not specified
        
        # Get image path
        image_path = self.get_image_path(str(image_id))
        
        # Validate image exists
        if not self.validate_image_exists(image_path):
            warnings.warn(f"Image file not found: {image_path}")
        
        # Generate prompts
        prompt, extra_info = self.generate_prompt(sample)

        return {
            "image_id": f"{idx}_{image_id}",
            "image_path": image_path,
            "modality": modality,
            "question": question,
            "answer": reference_answer,
            "prompt": prompt,
            "extra_info": extra_info,
        }
    
    def get_questions_by_type(self) -> Dict[str, int]:
        """
        Get distribution of questions by type or category.
        
        Returns:
            Dictionary mapping question types to counts
        """
        # Simple categorization based on question content
        question_types = {}
        
        for question in self.df['question']:
            question_lower = question.lower()
            
            # Categorize questions
            if any(word in question_lower for word in ['diagnosis', 'disease', 'condition']):
                category = 'diagnosis'
            elif any(word in question_lower for word in ['grade', 'severity', 'stage']):
                category = 'grading'
            elif any(word in question_lower for word in ['lesion', 'abnormality', 'finding']):
                category = 'finding_detection'
            elif any(word in question_lower for word in ['quality', 'gradable']):
                category = 'quality_assessment'
            elif any(word in question_lower for word in ['count', 'number', 'how many']):
                category = 'counting'
            else:
                category = 'other'
            
            question_types[category] = question_types.get(category, 0) + 1
        
        return question_types
    
    def get_answer_distribution(self) -> Dict[str, int]:
        """
        Get distribution of answers in the dataset.
        
        Returns:
            Dictionary mapping answers to counts
        """
        return self.df['answer'].value_counts().to_dict()
    
    def filter_by_modality(self, modality: str) -> 'VQADataset':
        """
        Filter dataset by imaging modality.
        
        Args:
            modality: Imaging modality (e.g., 'CFP', 'OCT', 'FFA')
            
        Returns:
            New VQADataset instance with filtered samples
        """
        if 'modality' not in self.df.columns:
            warnings.warn("Modality column not found in dataset")
            return self
        
        return self.filter_samples(f"modality == '{modality}'")
    
    def filter_by_question_type(self, question_keywords: List[str]) -> 'VQADataset':
        """
        Filter dataset by question type based on keywords.
        
        Args:
            question_keywords: List of keywords to filter questions
            
        Returns:
            New VQADataset instance with filtered samples
        """
        # Create condition for filtering
        conditions = [f"question.str.contains('{keyword}', case=False, na=False)" 
                     for keyword in question_keywords]
        condition = " | ".join(conditions)
        
        return self.filter_samples(condition)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the VQA dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = super().get_dataset_stats()
        
        # Add VQA-specific stats
        stats['question_types'] = self.get_questions_by_type()
        stats['answer_distribution'] = self.get_answer_distribution()
        
        # Question length statistics
        question_lengths = self.df['question'].str.len()
        stats['question_length_stats'] = {
            'mean': question_lengths.mean(),
            'std': question_lengths.std(),
            'min': question_lengths.min(),
            'max': question_lengths.max()
        }
        
        # Answer length statistics
        answer_lengths = self.df['answer'].str.len()
        stats['answer_length_stats'] = {
            'mean': answer_lengths.mean(),
            'std': answer_lengths.std(),
            'min': answer_lengths.min(),
            'max': answer_lengths.max()
        }
        
        return stats
    
    def create_diagnosis_prompt(self, sample: pd.Series, include_context: bool = True) -> str:
        """
        Create a diagnosis-focused prompt for EyeAgent integration.
        
        Args:
            sample: Pandas Series containing sample data
            include_context: Whether to include additional context
            
        Returns:
            Formatted prompt string
        """
        base_prompt, extra_info = self.generate_prompt(sample)
        
        if include_context:
            # Add context about the image and modality
            modality = sample.get('modality', 'fundus photograph')
            context = f"You are analyzing a {modality} image. "
            base_prompt = context + base_prompt
        
        return base_prompt + "\n\n" + extra_info
    
    def export_for_benchmark(self, output_path: str, format: str = 'csv') -> None:
        """
        Export dataset in format suitable for benchmarking.
        
        Args:
            output_path: Path to save the exported dataset
            format: Export format ('csv', 'json', 'jsonl')
        """
        # Prepare data for export
        export_data = []
        
        for idx in range(len(self.df)):
            sample_data = self[idx]
            export_data.append({
                'image_path': sample_data['image_path'],
                'question': sample_data['question'],
                'ground_truth_answer': sample_data['answer'],
                'prompt': sample_data['prompt'],
                'modality': sample_data.get('modality', 'CFP')
            })
        
        export_df = pd.DataFrame(export_data)
        
        # Save in specified format
        if format == 'csv':
            export_df.to_csv(output_path, index=False)
        elif format == 'json':
            export_df.to_json(output_path, orient='records', indent=2)
        elif format == 'jsonl':
            export_df.to_json(output_path, orient='records', lines=True)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        print(f"Dataset exported to {output_path} in {format} format")


def create_sample_vqa_dataset(output_dir: str, num_samples: int = 20) -> None:
    """
    Create a sample VQA dataset for testing purposes.
    
    Args:
        output_dir: Directory to save the sample dataset
        num_samples: Number of samples to generate
    """
    import random
    
    # Sample questions and answers for eye diseases
    sample_data = [
        ("What is the primary diagnosis visible in this fundus image?", "Diabetic Retinopathy"),
        ("Is there evidence of macular degeneration?", "Yes"),
        ("What grade of diabetic retinopathy is present?", "Moderate NPDR"),
        ("Are there any signs of glaucoma?", "No"),
        ("What is the quality of this fundus photograph?", "Good"),
        ("Is the optic disc clearly visible?", "Yes"),
        ("Are there any cotton wool spots present?", "Yes"),
        ("What is the condition of the macula?", "Normal"),
        ("Is there evidence of hypertensive retinopathy?", "No"),
        ("What abnormalities are visible in this image?", "Microaneurysms and hemorrhages")
    ]
    
    modalities = ['CFP', 'OCT', 'FFA']
    
    # Generate dataset
    data = []
    for i in range(num_samples):
        question, answer = random.choice(sample_data)
        data.append({
            'imid': f'sample_{i:03d}',
            'question': question,
            'answer': answer,
            'modality': random.choice(modalities)
        })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, 'sample_vqa_dataset.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"Sample VQA dataset created at {csv_path}")
    print(f"Generated {num_samples} samples")
    print("Questions distribution:")
    print(df['question'].value_counts())


if __name__ == "__main__":
    # Example usage
    create_sample_vqa_dataset("./sample_data", num_samples=50)
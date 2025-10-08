"""
Metrics calculation module for EyeAgent benchmark.

Provides functions to compute various evaluation metrics including:
- Accuracy
- Precision, Recall, F1-score
- AUC-ROC
- Confusion Matrix
- Per-class metrics
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
import os

from .config import MetricsConfig


class MetricsCalculator:
    """Calculate evaluation metrics for classification results."""
    
    def __init__(self, config: MetricsConfig, class_names: List[str]):
        self.config = config
        self.class_names = class_names
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(class_names)
        
    def calculate_all_metrics(self, 
                            y_true: List[str], 
                            y_pred: List[str],
                            y_proba: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Calculate all configured metrics.
        
        Args:
            y_true: True labels as class names
            y_pred: Predicted labels as class names  
            y_proba: Prediction probabilities (optional, needed for AUC)
            
        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        # Convert string labels to numeric
        y_true_encoded = self.label_encoder.transform(y_true)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        metrics = {}
        
        # Basic accuracy
        if self.config.compute_accuracy:
            metrics['accuracy'] = accuracy_score(y_true_encoded, y_pred_encoded)
        
        # Precision, Recall, F1
        if any([self.config.compute_precision, self.config.compute_recall, self.config.compute_f1]):
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_encoded, y_pred_encoded, average=self.config.average, zero_division=0
            )
            
            if self.config.compute_precision:
                metrics['precision'] = precision
            if self.config.compute_recall:
                metrics['recall'] = recall
            if self.config.compute_f1:
                metrics['f1_score'] = f1
        
        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(y_true_encoded, y_pred_encoded)
        metrics['per_class'] = per_class_metrics
        
        # AUC-ROC (if probabilities provided)
        if self.config.compute_auc and y_proba is not None:
            try:
                auc_metrics = self._calculate_auc_metrics(y_true_encoded, y_proba)
                metrics.update(auc_metrics)
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['auc_roc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(
            y_true_encoded,
            y_pred_encoded,
            labels=list(range(len(self.class_names)))
        )
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        report = classification_report(
            y_true_encoded,
            y_pred_encoded,
            target_names=self.class_names,
            labels=list(range(len(self.class_names))),
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Summary statistics
        metrics['summary'] = {
            'total_samples': len(y_true),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_distribution': self._get_class_distribution(y_true),
            'prediction_distribution': self._get_class_distribution(y_pred)
        }
        
        logger.info(f"Calculated metrics: Accuracy={metrics.get('accuracy', 0):.3f}, "
                   f"F1={metrics.get('f1_score', 0):.3f}")
        
        return metrics
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each class individually."""
        per_class = {}
        
        for i, class_name in enumerate(self.class_names):
            # Create binary labels for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_binary, y_pred_binary, average='binary', zero_division=0
            )
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            
            per_class[class_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'support': int(np.sum(y_true_binary))
            }
        
        return per_class
    
    def _calculate_auc_metrics(self, y_true: np.ndarray, y_proba: List[List[float]]) -> Dict[str, float]:
        """Calculate AUC-ROC metrics."""
        auc_metrics = {}
        
        # Convert probabilities to numpy array
        y_proba_array = np.array(y_proba)
        
        if len(self.class_names) == 2:
            # Binary classification
            auc_roc = roc_auc_score(y_true, y_proba_array[:, 1])
            auc_metrics['auc_roc'] = float(auc_roc)
        else:
            # Multi-class classification
            # Binarize the labels for multi-class AUC
            y_true_binarized = label_binarize(y_true, classes=range(len(self.class_names)))
            
            # Macro-average AUC
            try:
                auc_roc_macro = roc_auc_score(y_true_binarized, y_proba_array, average='macro', multi_class='ovr')
                auc_metrics['auc_roc_macro'] = float(auc_roc_macro)
            except Exception:
                auc_metrics['auc_roc_macro'] = None
            
            # Per-class AUC
            per_class_auc = {}
            for i, class_name in enumerate(self.class_names):
                try:
                    if y_true_binarized.shape[1] > i:
                        auc = roc_auc_score(y_true_binarized[:, i], y_proba_array[:, i])
                        per_class_auc[class_name] = float(auc)
                except Exception:
                    per_class_auc[class_name] = None
            
            auc_metrics['auc_roc_per_class'] = per_class_auc
        
        return auc_metrics
    
    def _get_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """Get distribution of class labels."""
        from collections import Counter
        return dict(Counter(labels))
    
    def plot_confusion_matrix(self, y_true: List[str], y_pred: List[str], 
                            save_path: Optional[str] = None, 
                            normalize: bool = False) -> None:
        """Plot confusion matrix."""
        # Convert to encoded labels
        y_true_encoded = self.label_encoder.transform(y_true)
        y_pred_encoded = self.label_encoder.transform(y_pred)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true_encoded, y_pred_encoded)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curves(self, y_true: List[str], y_proba: List[List[float]], 
                       save_path: Optional[str] = None) -> None:
        """Plot ROC curves for multi-class classification."""
        if len(self.class_names) < 2:
            logger.warning("Cannot plot ROC curve with less than 2 classes")
            return
        
        # Convert labels
        y_true_encoded = self.label_encoder.transform(y_true)
        y_proba_array = np.array(y_proba)
        
        # Binarize labels for multi-class
        y_true_binarized = label_binarize(y_true_encoded, classes=range(len(self.class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            if y_true_binarized.shape[1] > i:
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba_array[:, i])
                auc = roc_auc_score(y_true_binarized[:, i], y_proba_array[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_metrics_report(self, metrics: Dict[str, Any], save_path: str) -> None:
        """Save detailed metrics report to file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = self._make_json_serializable(metrics)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        logger.info(f"Metrics report saved to {save_path}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def create_summary_table(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create a summary table of key metrics."""
        summary_data = []
        
        # Overall metrics
        overall = {
            'Class': 'Overall',
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'AUC-ROC': metrics.get('auc_roc_macro', metrics.get('auc_roc', 'N/A')),
            'Support': metrics.get('summary', {}).get('total_samples', 0)
        }
        summary_data.append(overall)
        
        # Per-class metrics
        per_class = metrics.get('per_class', {})
        for class_name in self.class_names:
            if class_name in per_class:
                class_metrics = per_class[class_name]
                row = {
                    'Class': class_name,
                    'Accuracy': class_metrics.get('accuracy', 0),
                    'Precision': class_metrics.get('precision', 0),
                    'Recall': class_metrics.get('recall', 0),
                    'F1-Score': class_metrics.get('f1_score', 0),
                    'AUC-ROC': metrics.get('auc_roc_per_class', {}).get(class_name, 'N/A'),
                    'Support': class_metrics.get('support', 0)
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)


def extract_predictions_from_results(results: List[Dict[str, Any]], 
                                   class_names: List[str]) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Extract predictions and probabilities from benchmark results.
    
    Args:
        results: List of benchmark result dictionaries
        class_names: List of valid class names
        
    Returns:
        Tuple of (true_labels, predicted_labels, prediction_probabilities)
    """
    true_labels = []
    predicted_labels = []
    prediction_probs = []
    
    for result in results:
        # Extract true label
        true_label = result.get('true_label', 'Normal')
        true_labels.append(true_label)
        
        # Extract predicted label from formatted output
        formatted_diagnosis = result.get('formatted_diagnosis', '')
        predicted_label = extract_diagnosis_from_formatted_output(formatted_diagnosis, class_names)
        predicted_labels.append(predicted_label)
        
        # Extract probabilities (if available)
        probs = result.get('prediction_probabilities')
        if probs is None:
            # Create uniform probabilities if not available
            uniform_prob = 1.0 / len(class_names)
            probs = [uniform_prob] * len(class_names)
        
        prediction_probs.append(probs)
    
    return true_labels, predicted_labels, prediction_probs


def extract_diagnosis_from_formatted_output(formatted_output: str, class_names: List[str]) -> str:
    """Extract diagnosis from formatted output string."""
    # Pattern: "The diagnosis of this image is XXX"
    import re
    
    pattern = r"The diagnosis of this image is\s+(.+?)(?:\.|$)"
    match = re.search(pattern, formatted_output, re.IGNORECASE)
    
    if match:
        diagnosis = match.group(1).strip()
        
        # Normalize to valid class name
        for class_name in class_names:
            if diagnosis.lower() == class_name.lower():
                return class_name
    
    # Default fallback
    return "Normal"
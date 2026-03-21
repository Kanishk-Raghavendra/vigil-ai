"""Metrics and evaluation utilities for VIGIL system."""

from typing import List, Dict, Any, Tuple
import json
from pathlib import Path
from src.utils.logger import Logger


logger = Logger(__name__)


class VerificationResult:
    """Data structure for a verification result."""
    
    def __init__(self, claim: str, score: float, decision: str, explanation: str = ""):
        """
        Initialize verification result.
        
        Args:
            claim: The verified claim
            score: Similarity score [0.0, 1.0]
            decision: "TRUST" or "WARNING"
            explanation: Human-readable explanation
        """
        self.claim = claim
        self.score = score
        self.decision = decision
        self.explanation = explanation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "claim": self.claim,
            "score": round(self.score, 3),
            "decision": self.decision,
            "explanation": self.explanation
        }


class MetricsComputer:
    """Compute evaluation metrics for VIGIL system."""
    
    @staticmethod
    def make_decision(score: float, threshold: float = 0.25) -> str:
        """
        Make TRUST/WARNING decision based on score.
        
        Args:
            score: Similarity score [0.0, 1.0]
            threshold: Decision threshold
            
        Returns:
            "TRUST" or "WARNING"
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {score}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        return "TRUST" if score >= threshold else "WARNING"
    
    @staticmethod
    def compute_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute accuracy of predictions.
        
        Args:
            predictions: List of predictions ("TRUST" or "WARNING")
            ground_truth: List of ground truth labels
            
        Returns:
            Accuracy [0.0, 1.0]
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        if len(predictions) == 0:
            return 0.0
        
        correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
        return correct / len(predictions)
    
    @staticmethod
    def compute_precision_recall(predictions: List[str], ground_truth: List[str]) -> Tuple[float, float]:
        """
        Compute precision and recall for TRUST class.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Tuple of (precision, recall)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        tp = sum(1 for p, gt in zip(predictions, ground_truth) 
                if p == "TRUST" and gt == "TRUST")
        fp = sum(1 for p, gt in zip(predictions, ground_truth) 
                if p == "TRUST" and gt == "WARNING")
        fn = sum(1 for p, gt in zip(predictions, ground_truth) 
                if p == "WARNING" and gt == "TRUST")
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    @staticmethod
    def compute_fpr(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute False Positive Rate (critical for safety).
        
        FPR = FP / (FP + TN)
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            False positive rate [0.0, 1.0]
        """
        fp = sum(1 for p, gt in zip(predictions, ground_truth) 
                if p == "TRUST" and gt == "WARNING")
        tn = sum(1 for p, gt in zip(predictions, ground_truth) 
                if p == "WARNING" and gt == "WARNING")
        
        return fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    @staticmethod
    def compute_f1_score(precision: float, recall: float) -> float:
        """
        Compute F1 score.
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score [0.0, 1.0]
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def compute_all_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of predictions
            ground_truth: List of ground truth labels
            
        Returns:
            Dictionary of metric names to values
        """
        accuracy = MetricsComputer.compute_accuracy(predictions, ground_truth)
        precision, recall = MetricsComputer.compute_precision_recall(predictions, ground_truth)
        fpr = MetricsComputer.compute_fpr(predictions, ground_truth)
        f1 = MetricsComputer.compute_f1_score(precision, recall)
        
        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "f1_score": round(f1, 4)
        }
    
    @staticmethod
    def save_metrics(metrics: Dict[str, Any], output_path: Path):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")

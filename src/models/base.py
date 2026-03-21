"""Abstract base classes for VIGIL models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union
from pathlib import Path


class CaptionModel(ABC):
    """Abstract base class for caption generation models."""
    
    @abstractmethod
    def generate_caption(self, image_path: Union[str, Path]) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            str: Generated caption
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model name, version, etc.
        """
        pass


class VerifierModel(ABC):
    """Abstract base class for claim verification models."""
    
    @abstractmethod
    def verify_claim(self, image_path: Union[str, Path], claim: str) -> float:
        """
        Verify a claim by computing image-text similarity.
        
        Args:
            image_path: Path to input image
            claim: Text claim to verify
            
        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model name, version, etc.
        """
        pass

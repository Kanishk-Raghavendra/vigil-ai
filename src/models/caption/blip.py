"""BLIP Caption Generation Model."""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union

from src.models.base import CaptionModel
from src.utils.logger import Logger


logger = Logger(__name__)


class BLIPCaptionModel(CaptionModel):
    """
    BLIP (Bootstrapping Language-Image Pre-training) caption generation model.
    
    Uses Salesforce/blip-image-captioning-base for lightweight, efficient inference.
    """
    
    def __init__(self):
        """Initialize BLIP model and processor."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"BLIP Model: Using device {self.device}")
        
        try:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info("BLIP Model: Successfully loaded")
        except Exception as e:
            logger.error(f"BLIP Model: Failed to load - {e}")
            raise
    
    def generate_caption(self, image_path: Union[str, Path]) -> str:
        """
        Generate caption for image using BLIP.
        
        Args:
            image_path: Path to input image
            
        Returns:
            str: Generated caption text
            
        Raises:
            FileNotFoundError: If image file not found
            RuntimeError: If caption generation fails
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")
        
        try:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"BLIP Model: Generated caption for {image_path.name}")
            return caption
        except Exception as e:
            logger.error(f"BLIP Model: Caption generation failed - {e}")
            raise RuntimeError(f"Caption generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get BLIP model information."""
        return {
            "name": "BLIP",
            "variant": "blip-image-captioning-base",
            "source": "Salesforce",
            "device": str(self.device),
            "task": "image-to-text"
        }

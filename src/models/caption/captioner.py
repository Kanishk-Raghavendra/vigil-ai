"""
Mobile Vision-Language Captioning Model.

This module provides caption generation using MobileVLM-3B, optimized for
edge inference on Apple Silicon MPS and resource-constrained devices.

MobileVLM replaces BLIP for edge deployment, maintaining the same
generate_caption(image) interface for pipeline compatibility.

Device routing:
- Automatically uses Apple Silicon MPS if available (torch.backends.mps.is_available())
- Falls back to CPU if MPS is not available
- No CUDA dependency required
"""

import time
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union

from src.models.base import CaptionModel
from src.utils.logger import Logger


logger = Logger(__name__)


class MobileVLMCapcioner(CaptionModel):
    """
    MobileVLM-3B caption generation model for edge inference.
    
    MobileVLM is a lightweight vision-language model optimized for
    mobile and edge devices with Apple Silicon MPS acceleration.
    
    Attributes:
        device: torch.device (MPS or CPU)
        processor: AutoProcessor for image and text preprocessing
        model: AutoModelForVision2Seq (MobileVLM-3B)
        load_time_ms: Model load time in milliseconds
    """
    
    # HuggingFace checkpoint for MobileVLM-3B
    MOBILEVLM_CHECKPOINT = "mtgv/MobileVLM_V2-3B"
    
    def __init__(self):
        """Initialize MobileVLM-3B model and processor with edge device routing."""
        # Device routing: MPS > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("MobileVLM Model: Using Apple Silicon MPS backend")
        else:
            self.device = torch.device("cpu")
            logger.info("MobileVLM Model: Apple Silicon MPS not available, using CPU")
        
        start_time = time.time()
        
        try:
            # Load processor (handles image preprocessing)
            self.processor = AutoProcessor.from_pretrained(
                self.MOBILEVLM_CHECKPOINT
            )
            
            # Load MobileVLM model
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.MOBILEVLM_CHECKPOINT,
                torch_dtype=torch.float16 if str(self.device) == "mps" else torch.float32,
                device_map=str(self.device)
            )
            
            # Set to eval mode
            self.model.eval()
            
            # Record load time
            self.load_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"MobileVLM Model: Successfully loaded {self.MOBILEVLM_CHECKPOINT} "
                f"in {self.load_time_ms:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"MobileVLM Model: Failed to load - {e}")
            raise RuntimeError(f"Failed to load MobileVLM: {e}")

    def generate_caption(self, image_path: Union[str, Path]) -> str:
        """
        Generate caption for image using MobileVLM-3B.
        
        Args:
            image_path: Path to input image (PIL-compatible format)
            
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
            start_time = time.time()
            
            # Prepare inputs
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=1,
                    temperature=0.7
                )
            
            # Decode caption
            caption = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            inference_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"MobileVLM Model: Generated caption for {image_path.name} "
                f"({inference_time_ms:.2f}ms)"
            )
            
            return caption
            
        except Exception as e:
            logger.error(f"MobileVLM Model: Caption generation failed - {e}")
            raise RuntimeError(f"Caption generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get MobileVLM model information and metadata."""
        return {
            "model_name": "MobileVLM-3B",
            "checkpoint": self.MOBILEVLM_CHECKPOINT,
            "device": str(self.device),
            "load_time_ms": self.load_time_ms,
            "dtype": str(next(self.model.parameters()).dtype),
            "framework": "transformers",
            "purpose": "edge-optimized vision-language caption generation"
        }

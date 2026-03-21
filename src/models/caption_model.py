"""
Caption Generation Module - VIGIL System

This module handles loading and using the BLIP model for generating
descriptive captions from images.
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


class CaptionGenerator:
    """
    Load and generate captions using BLIP model.
    Models are loaded once to avoid redundant loading.
    """
    
    def __init__(self):
        """Initialize BLIP model and processor."""
        # Detect device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CaptionGenerator] Using device: {self.device}")
        
        try:
            # Load BLIP processor and model from HuggingFace
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print("[CaptionGenerator] BLIP model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load BLIP model: {e}")
    
    def generate_caption(self, image_path: str) -> str:
        """
        Generate a caption for the given image.
        
        Args:
            image_path: Path to the input image file
            
        Returns:
            str: Generated caption text
            
        Raises:
            FileNotFoundError: If image file does not exist
            RuntimeError: If caption generation fails
        """
        try:
            # Load and convert image to RGB
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        try:
            # Process image for BLIP
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption (disable gradients for inference)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=50)
            
            # Decode generated tokens to text
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            raise RuntimeError(f"Failed to generate caption: {e}")

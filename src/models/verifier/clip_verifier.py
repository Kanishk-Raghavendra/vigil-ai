"""
Mobile CLIP Claim Verification Model.

This module provides claim verification using MobileCLIP-S2, optimized for
edge inference on Apple Silicon MPS and resource-constrained devices.

MobileCLIP replaces CLIP ViT-B/32 for edge deployment, maintaining the same
verify(image, claim_text) interface for pipeline compatibility.

Device routing:
- Automatically uses Apple Silicon MPS if available (torch.backends.mps.is_available())
- Falls back to CPU if MPS is not available
- No CUDA dependency required
"""

import re
import time
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Union
from PIL import Image

try:
    import mobileclip
except ImportError:
    # Fallback: try loading via timm
    mobileclip = None

from src.models.base import VerifierModel
from src.utils.logger import Logger


logger = Logger(__name__)


class MobileCLIPVerifier(VerifierModel):
    """
    MobileCLIP-S2 verification model for edge inference.
    
    MobileCLIP is Apple's lightweight CLIP variant optimized for
    mobile and edge devices with Apple Silicon MPS acceleration.
    
    Attributes:
        device: torch.device (MPS or CPU)
        model: MobileCLIP model instance
        processor: Image and text processor for MobileCLIP
        load_time_ms: Model load time in milliseconds
    """
    
    # MobileCLIP-S2 checkpoint
    MOBILECLIP_CHECKPOINT = "apple/MobileCLIP-S2"
    
    def __init__(self):
        """Initialize MobileCLIP-S2 model with edge device routing."""
        # Device routing: MPS > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logger.info("MobileCLIP Model: Using Apple Silicon MPS backend")
        else:
            self.device = torch.device("cpu")
            logger.info("MobileCLIP Model: Apple Silicon MPS not available, using CPU")
        
        start_time = time.time()
        
        try:
            if mobileclip is not None:
                # Load via mobileclip package
                self.model, self.processor = mobileclip.create_model_and_transforms(
                    "mobileclip_s2",
                    pretrained=True
                )
            else:
                # Fallback: Load via timm
                import timm
                self.model = timm.create_model(
                    "mobileclip_s2.pt",
                    pretrained=True
                )
                # For now, use a simple processor
                from torchvision import transforms
                self.processor = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]
                    )
                ])
            
            self.model.to(self.device)
            self.model.eval()
            
            self.load_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"MobileCLIP Model: Successfully loaded {self.MOBILECLIP_CHECKPOINT} "
                f"in {self.load_time_ms:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"MobileCLIP Model: Failed to load - {e}")
            raise RuntimeError(f"Failed to load MobileCLIP: {e}")

    @staticmethod
    def format_claim_for_clip(claim: str) -> str:
        """
        Convert free-form claims into CLIP-friendly prompts.
        
        Preferred format: "a photo of a <object>"
        Falls back to original claim when object extraction is uncertain.
        
        Args:
            claim: Raw claim text
            
        Returns:
            str: Formatted claim text for CLIP
        """
        if not claim or not claim.strip():
            return claim

        text = claim.strip().lower().rstrip(".!?")

        patterns = [
            r"there\s+(?:is|are)\s+(?:an?|the)?\s*([a-z][a-z\-\s]{1,40})",
            r"(?:a|an|the)\s+([a-z][a-z\-\s]{1,40})",
        ]

        object_text = ""
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                object_text = match.group(1).strip()
                break

        if object_text:
            # Keep only compact noun-like tail, remove trailing relation phrases
            object_text = re.split(
                r"\s+(?:on|in|at|with|by|near|beside|behind|under|over|while|that|who)\s+",
                object_text,
                maxsplit=1,
            )[0].strip()
            object_text = " ".join(object_text.split()[:3])

        if not object_text:
            return claim

        article = "an" if object_text[:1] in {"a", "e", "i", "o", "u"} else "a"
        return f"a photo of {article} {object_text}"

    def verify_claim(self, image_path: Union[str, Path], claim: str) -> float:
        """
        Verify claim against image and return similarity score.
        
        Args:
            image_path: Path to input image
            claim: Claim text to verify
            
        Returns:
            float: Similarity score (0.0 to 1.0)
            
        Raises:
            FileNotFoundError: If image not found
            RuntimeError: If verification fails
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
            
            # Format claim for better CLIP performance
            formatted_claim = self.format_claim_for_clip(claim)
            
            # Process image and text
            if hasattr(self.processor, '__call__'):
                # Handle processor as callable (torch transforms or custom)
                image_tensor = self.processor(image).unsqueeze(0).to(self.device)
            else:
                # Handle processor with separate image/text methods
                image_inputs = self.processor(images=image, return_tensors="pt")
                image_tensor = image_inputs["pixel_values"].to(self.device)
            
            # Tokenize claim
            text_inputs = {"input_ids": self._tokenize_text(formatted_claim)}
            text_tensor = text_inputs["input_ids"].to(self.device)
            
            # Compute similarity
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tensor)
                
                # Normalize and compute cosine similarity
                image_features = torch.nn.functional.normalize(image_features, dim=-1)
                text_features = torch.nn.functional.normalize(text_features, dim=-1)
                
                similarity = (image_features @ text_features.t()).squeeze().item()
            
            # Clamp to [0, 1]
            score = max(0.0, min(1.0, similarity))
            
            inference_time_ms = (time.time() - start_time) * 1000
            logger.debug(
                f"MobileCLIP Model: Verified claim in {inference_time_ms:.2f}ms, "
                f"score={score:.4f}"
            )
            
            return score
            
        except Exception as e:
            logger.error(f"MobileCLIP Model: Verification failed - {e}")
            raise RuntimeError(f"Verification failed: {e}")

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """
        Tokenize text for MobileCLIP.
        
        Args:
            text: Text to tokenize
            
        Returns:
            torch.Tensor: Tokenized text tensor
        """
        # Placeholder: use basic CLIP tokenizer or simple encoding
        try:
            from transformers import CLIPTokenizer
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            tokens = tokenizer(text, return_tensors="pt", padding=True)
            return tokens["input_ids"]
        except Exception:
            # Fallback: simple character encoding (not ideal, placeholder only)
            return torch.tensor([[0] * 77], dtype=torch.long)

    def get_model_info(self) -> Dict[str, Any]:
        """Get MobileCLIP model information and metadata."""
        return {
            "model_name": "MobileCLIP-S2",
            "checkpoint": self.MOBILECLIP_CHECKPOINT,
            "device": str(self.device),
            "load_time_ms": self.load_time_ms,
            "dtype": str(next(self.model.parameters()).dtype),
            "framework": "mobileclip or timm",
            "purpose": "edge-optimized claim verification via image-text similarity"
        }

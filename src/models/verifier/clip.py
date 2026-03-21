"""CLIP Claim Verification Model."""

import re
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Tuple, Union

from src.models.base import VerifierModel
from src.utils.logger import Logger


logger = Logger(__name__)


class CLIPVerifierModel(VerifierModel):
    """
    CLIP (Contrastive Language-Image Pre-training) verification model.
    
    Uses OpenAI CLIP (ViT-B/32) to compute image-text similarity for claim verification.
    """
    
    def __init__(self):
        """Initialize CLIP model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"CLIP Model: Using device {self.device}")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.model.eval()
            logger.info("CLIP Model: Successfully loaded (ViT-B/32)")
        except Exception as e:
            logger.error(f"CLIP Model: Failed to load - {e}")
            raise

    @staticmethod
    def format_claim_for_clip(claim: str) -> str:
        """Convert free-form claims into CLIP-friendly prompts.

        Preferred format: "a photo of a <object>"
        Falls back to original claim when object extraction is uncertain.
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
            # Keep only compact noun-like tail, remove trailing relation phrases.
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

    def verify_claim_with_prompt(self, image_path: Union[str, Path], claim: str) -> Tuple[float, str]:
        """Verify claim and return both score and formatted prompt text."""
        formatted_claim = self.format_claim_for_clip(claim)
        score = self._verify_text_against_image(image_path, formatted_claim)
        logger.debug(
            f"CLIP Model: Prompted claim '{claim}' -> '{formatted_claim}' score {score:.3f}"
        )
        return score, formatted_claim
    
    def verify_claim(self, image_path: Union[str, Path], claim: str) -> float:
        """
        Verify a claim using CLIP image-text similarity.
        
        Args:
            image_path: Path to input image
            claim: Text claim to verify
            
        Returns:
            float: Normalized similarity score [0.0, 1.0]
                   Higher score indicates higher confidence in claim
            
        Raises:
            FileNotFoundError: If image file not found
            RuntimeError: If verification fails
        """
        try:
            score, _formatted_claim = self.verify_claim_with_prompt(image_path, claim)
            return score
        except Exception as e:
            logger.error(f"CLIP Model: Verification failed - {e}")
            raise RuntimeError(f"Verification failed: {e}")

    def _verify_text_against_image(self, image_path: Union[str, Path], text: str) -> float:
        """Compute CLIP similarity score for given image and text."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_input = clip.tokenize([text]).to(self.device)
            text_features = self.model.encode_text(text_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).item()

        similarity_normalized = (similarity + 1) / 2
        similarity_normalized = max(0.0, min(1.0, similarity_normalized))
        return similarity_normalized
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get CLIP model information."""
        return {
            "name": "CLIP",
            "variant": "ViT-B/32",
            "source": "OpenAI",
            "device": str(self.device),
            "task": "image-text-matching"
        }

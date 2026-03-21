"""BLIP-VQA direct yes/no baseline for claim verification."""

from pathlib import Path
from typing import Dict, Any, Union

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

from src.utils.logger import Logger


logger = Logger(__name__)


class BLIPVQABaseline:
    """Zero-shot VQA baseline.

    The model is asked directly whether an image supports a claim:
    "Does this image show that <claim>? Answer yes or no."
    """

    def __init__(self, model_name: str = "Salesforce/blip-vqa-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        logger.info(f"BLIP-VQA Baseline: Using device {self.device}")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info(f"BLIP-VQA Baseline: Loaded model {model_name}")

    @staticmethod
    def _build_question(claim: str) -> str:
        claim_text = claim.strip().rstrip(".?!")
        return f"Does this image show that {claim_text}? Answer yes or no."

    def answer(self, image_path: Union[str, Path], claim: str) -> str:
        """Generate VQA yes/no answer text for claim and image."""
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        question = self._build_question(claim)

        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=5)

        answer = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
        return answer

    def predict_binary(self, image_path: Union[str, Path], claim: str) -> int:
        """Return 1 for yes/trust, 0 for no/warning."""
        answer = self.answer(image_path, claim)

        # Conservative mapping: only explicit yes variants become positive.
        yes_tokens = ("yes", "yeah", "yep", "true")
        return 1 if answer.startswith(yes_tokens) else 0

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "BLIP-VQA",
            "variant": self.model_name,
            "device": str(self.device),
            "task": "yes-no-vqa",
        }

"""VIGIL Pipeline - Orchestrator for the verification system."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from src.models.base import CaptionModel, VerifierModel
from src.utils.claim_extractor import ClaimExtractor
from src.utils.metrics import VerificationResult
from src.utils.decision import Decision
from src.utils.logger import Logger


logger = Logger(__name__)


class VIGILPipeline:
    """
    Main VIGIL verification pipeline.
    
    Orchestrates the full process:
    Image → Caption → Claim Extraction → Verification → Decision
    """
    
    def __init__(
        self,
        caption_model: CaptionModel,
        verifier_model: VerifierModel,
        threshold: float = 0.25,
        cache_dir: Optional[Path] = None,
        aggregation_mode: str = "balanced",
    ):
        """
        Initialize VIGIL pipeline.
        
        Args:
            caption_model: CaptionModel instance
            verifier_model: VerifierModel instance
            threshold: Decision threshold
            cache_dir: Optional directory for caching intermediates
            aggregation_mode: Global aggregation calibration mode ("balanced" or "safe")
        """
        if aggregation_mode not in {"balanced", "safe"}:
            raise ValueError(
                f"Invalid aggregation_mode '{aggregation_mode}'. Use 'balanced' or 'safe'."
            )

        self.caption_model = caption_model
        self.verifier_model = verifier_model
        self.threshold = threshold
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.aggregation_mode = aggregation_mode
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"VIGIL Pipeline initialized (threshold={threshold}, aggregation_mode={aggregation_mode})"
        )
        logger.info(f"Caption model: {caption_model.get_model_info()['name']}")
        logger.info(f"Verifier model: {verifier_model.get_model_info()['name']}")
    
    def run(self, image_path: str, cache: bool = True) -> Dict[str, Any]:
        """
        Run the full VIGIL pipeline on an image.
        
        Args:
            image_path: Path to input image
            cache: Whether to cache intermediate outputs
            
        Returns:
            Dictionary containing pipeline outputs
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Starting VIGIL pipeline for {image_path.name}")
        
        # Initialize result structure
        result = {
            "image": str(image_path),
            "timestamp": datetime.now().isoformat(),
            "caption": None,
            "claims": [],
            "extracted_claims": [],
            "verifications": [],
            "threshold": self.threshold,
            "adaptive_threshold": None,
            "aggregation_mode": self.aggregation_mode,
            "global_trust": {},
            "statistics": {}
        }
        
        try:
            # Step 1: Generate caption
            logger.info("Step 1/4: Generating caption...")
            caption = self.caption_model.generate_caption(str(image_path))
            result["caption"] = caption
            logger.info(f"Caption: {caption}")
            
            # Step 2: Extract claims
            logger.info("Step 2/4: Extracting claims...")
            claims = ClaimExtractor.extract_claims(caption)
            claims = ClaimExtractor.filter_claims(claims)
            result["extracted_claims"] = [c.to_dict() for c in claims]
            logger.info(f"Extracted {len(claims)} claims")
            
            # Step 3 & 4: Verify claims and make decisions
            logger.info("Step 3/4 & 4/4: Verifying claims...")
            verifications = []
            claim_outputs = []
            trust_count = 0
            warning_count = 0

            scored_claims = []

            for claim in claims:
                try:
                    if hasattr(self.verifier_model, "verify_claim_with_prompt"):
                        score, formatted_claim = self.verifier_model.verify_claim_with_prompt(
                            str(image_path), claim.text
                        )
                    else:
                        score = self.verifier_model.verify_claim(str(image_path), claim.text)
                        formatted_claim = claim.text

                    scored_claims.append(
                        {
                            "claim": claim,
                            "score": score,
                            "formatted_claim": formatted_claim,
                        }
                    )
                except Exception as e:
                    logger.error(f"Verification failed for claim '{claim.text}': {e}")

            adaptive_threshold = Decision.compute_adaptive_threshold(
                [item["score"] for item in scored_claims]
            )
            result["adaptive_threshold"] = adaptive_threshold
            logger.info(f"Adaptive threshold for image: {adaptive_threshold:.3f}")
            
            for item in scored_claims:
                try:
                    claim = item["claim"]
                    score = item["score"]
                    formatted_claim = item["formatted_claim"]
                    
                    # Make decision
                    decision = Decision.make_decision(score, adaptive_threshold)
                    
                    # Generate explanation
                    explanation = self._generate_explanation(score, adaptive_threshold, decision)
                    
                    # Store result
                    verification = VerificationResult(
                        claim=claim.text,
                        score=score,
                        decision=decision,
                        explanation=explanation
                    )
                    verifications.append(verification)
                    claim_outputs.append(
                        {
                            "claim": claim.text,
                            "formatted_claim": formatted_claim,
                            "score": round(score, 3),
                            "decision": decision,
                        }
                    )
                    
                    # Update counts
                    if decision == "TRUST":
                        trust_count += 1
                    else:
                        warning_count += 1
                    
                    logger.info(f"  [{decision}] {claim.text} (score={score:.3f})")
                
                except Exception as e:
                    logger.error(f"Verification failed for claim '{claim.text}': {e}")
            
            # Store verifications
            result["claims"] = claim_outputs
            result["verifications"] = [v.to_dict() for v in verifications]
            result["global_trust"] = Decision.compute_global_trust(
                claim_outputs, mode=self.aggregation_mode
            )
            
            # Compute statistics
            total_claims = len(verifications)
            if total_claims > 0:
                result["statistics"] = {
                    "total_claims": total_claims,
                    "trusted": trust_count,
                    "warnings": warning_count,
                    "trust_rate": round(trust_count / total_claims, 3),
                    "average_score": round(
                        sum(v["score"] for v in result["verifications"]) / total_claims, 3
                    ),
                    "global_decision": result["global_trust"]["final_decision"],
                }
            
            logger.info(f"Pipeline completed: {trust_count} trusted, {warning_count} warnings")
            return result
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    @staticmethod
    def _generate_explanation(score: float, threshold: float, decision: str) -> str:
        """
        Generate human-readable explanation for decision.
        
        Args:
            score: Similarity score
            threshold: Decision threshold
            decision: "TRUST" or "WARNING"
            
        Returns:
            Explanation string
        """
        score_pct = score * 100
        threshold_pct = threshold * 100
        
        if decision == "TRUST":
            if score > 0.7:
                return f"High confidence alignment (score {score_pct:.1f}% > threshold {threshold_pct:.0f}%)"
            else:
                return f"Adequate confidence alignment (score {score_pct:.1f}% >= threshold {threshold_pct:.0f}%)"
        else:
            if score < 0.2:
                return f"Low confidence alignment (score {score_pct:.1f}% << threshold {threshold_pct:.0f}%)"
            else:
                return f"Insufficient confidence (score {score_pct:.1f}% < threshold {threshold_pct:.0f}%)"
    
    def set_threshold(self, threshold: float):
        """
        Set decision threshold.
        
        Args:
            threshold: New threshold value [0.0, 1.0]
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        
        self.threshold = threshold
        logger.info(f"Threshold updated to {threshold}")
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """
        Save pipeline results to JSON file.
        
        Args:
            results: Pipeline results dictionary
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

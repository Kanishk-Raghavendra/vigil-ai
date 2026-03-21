"""
Decision Module - VIGIL System

This module makes trust/warning decisions based on claim verification scores.
"""

import math
from typing import Any, Dict, List


class Decision:
    """Decision logic for VIGIL verification system."""
    
    # Default confidence threshold
    DEFAULT_THRESHOLD = 0.25
    
    # Decision categories
    TRUST = "TRUST"
    WARNING = "WARNING"

    @staticmethod
    def compute_adaptive_threshold(scores: List[float]) -> float:
        """Compute per-image adaptive threshold from score distribution.

        Formula:
        threshold = mean(scores) - 0.5 * std(scores)

        The returned value is clamped into [0.2, 0.8] for stability.
        """
        if not scores:
            return Decision.DEFAULT_THRESHOLD

        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_score = math.sqrt(variance)

        threshold = mean_score - 0.5 * std_score
        threshold = max(0.2, min(0.8, threshold))

        return round(threshold, 4)
    
    @staticmethod
    def make_decision(score: float, threshold: float = DEFAULT_THRESHOLD) -> str:
        """
        Make a trust/warning decision based on verification score.
        
        Decision logic:
        - score >= threshold: TRUST (claim is likely accurate)
        - score < threshold:  WARNING (claim may be inaccurate)
        
        Args:
            score: Similarity score between 0.0 and 1.0
            threshold: Confidence threshold for trust decision
            
        Returns:
            str: "TRUST" or "WARNING"
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {score}")
        
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        return Decision.TRUST if score >= threshold else Decision.WARNING

    @staticmethod
    def compute_global_trust(
        claim_results: List[Dict[str, Any]], mode: str = "balanced"
    ) -> Dict[str, Any]:
        """Aggregate per-claim outputs into a single global trust judgment.

        Soft aggregation uses:
        weighted_score = 0.7 * average_score + 0.3 * trusted_ratio

        Adaptive global threshold uses:
        global_threshold = mean(scores) - 0.3 * std(scores), clamped to [0.3, 0.7]

        Calibration mode:
        - balanced: threshold as-is
        - safe: threshold + 0.05 (clamped to 0.7)
        """
        if mode not in {"balanced", "safe"}:
            raise ValueError(f"Unsupported mode: {mode}. Use 'balanced' or 'safe'.")

        if not claim_results:
            default_threshold = 0.5
            if mode == "safe":
                default_threshold = min(0.7, default_threshold + 0.05)
            return {
                "score": 0.0,
                "global_score": 0.0,
                "average_score": 0.0,
                "trusted_ratio": 0.0,
                "threshold": round(default_threshold, 4),
                "mode": mode,
                "final_decision": "UNTRUSTED OUTPUT",
            }

        scores = [float(result.get("score", 0.0)) for result in claim_results]
        trusted_count = sum(
            1 for result in claim_results if result.get("decision") == Decision.TRUST
        )
        total = len(claim_results)

        average_score = sum(scores) / total
        trusted_ratio = trusted_count / total

        variance = sum((score - average_score) ** 2 for score in scores) / total
        std_score = math.sqrt(variance)

        # Adaptive global threshold, clamped for stability.
        global_threshold = average_score - 0.3 * std_score
        global_threshold = max(0.3, min(0.7, global_threshold))
        if mode == "safe":
            global_threshold = min(0.7, global_threshold + 0.05)

        # Soft aggregation score combines evidence strength and agreement.
        weighted_score = 0.7 * average_score + 0.3 * trusted_ratio

        final_decision = (
            "TRUSTED OUTPUT" if weighted_score >= global_threshold else "UNTRUSTED OUTPUT"
        )

        return {
            "score": round(weighted_score, 4),
            "global_score": round(weighted_score, 4),
            "average_score": round(average_score, 4),
            "trusted_ratio": round(trusted_ratio, 4),
            "threshold": round(global_threshold, 4),
            "mode": mode,
            "final_decision": final_decision,
        }

    @staticmethod
    def compute_global_trust_with_mode(
        claim_results: List[Dict[str, Any]], mode: str = "balanced"
    ) -> Dict[str, Any]:
        """Compute global trust with optional calibration mode.

        Modes:
        - balanced: use adaptive threshold as-is.
        - safe: increase threshold by +0.05 (clamped to 0.7).
        """
        return Decision.compute_global_trust(claim_results, mode=mode)
    
    @staticmethod
    def format_decision(claim: str, score: float, decision: str, threshold: float = DEFAULT_THRESHOLD) -> str:
        """
        Format decision result as human-readable string.
        
        Args:
            claim: The claim being verified
            score: The similarity score
            decision: The decision ("TRUST" or "WARNING")
            threshold: The threshold used for decision
            
        Returns:
            str: Formatted decision string
        """
        status_symbol = "✓" if decision == Decision.TRUST else "✗"
        return f"{status_symbol} [{decision:7s}] Score: {score:.2%} | Claim: {claim}"

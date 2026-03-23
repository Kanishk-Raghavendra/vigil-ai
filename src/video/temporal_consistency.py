"""
Temporal Consistency Checking for Video Hallucinations.

This module implements temporal consistency verification for CCTV video streams.
A hallucination naturally occurs when a person leaves the frame but the caption
still mentions them, when lighting changes cause misidentification, or when
partial occlusion causes wrong attribute assignment.

By comparing verification scores between consecutive keyframes, we can detect
and flag claims that were trusted in one frame but fail verification in the next.
This is a novel contribution unique to video-stream CCTV processing.

Typical usage:
    checker = TemporalConsistencyChecker()
    result_n = {..., "claims": [...]}  # Frame N verification result
    result_n1 = {..., "claims": [...]}  # Frame N+1 verification result
    augmented = checker.check_consistency(result_n, result_n1)
    # augmented includes "temporal_flags" and temporal_fpr metric
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import Logger


logger = Logger(__name__)


@dataclass
class TemporalInconsistency:
    """
    Record of a claim that failed temporal consistency.
    
    Attributes:
        claim_text: The claim that was inconsistent
        frame_n_score: Verification score in frame N
        frame_n_decision: Decision in frame N (e.g., "TRUSTED")
        frame_n1_score: Verification score in frame N+1
        frame_n1_decision: Decision in frame N+1 (e.g., "REJECTED")
        confidence_drop: Absolute score drop (frame_n_score - frame_n1_score)
    """
    claim_text: str
    frame_n_score: float
    frame_n_decision: str
    frame_n1_score: float
    frame_n1_decision: str
    confidence_drop: float


class TemporalConsistencyChecker:
    """
    Detect hallucinations that appear valid in one frame but fail in the next.
    
    CCTV hallucinations often manifest as temporal inconsistencies:
    - A person is detected in frame N but leaves before frame N+1
      (caption still mentions them, but verification fails in N+1)
    - Lighting changes cause attribute misidentification in N+1
    - Partial occlusion causes wrong object assignment that corrects in N+1
    
    The temporal consistency checker catches these by verifying that claims
    trusted in frame N maintain sufficient confidence in frame N+1.
    
    Attributes:
        consistency_threshold: Minimum score for a claim in N+1 if it was
                              trusted in frame N (default 0.5)
    """
    
    def __init__(self, consistency_threshold: float = 0.5):
        """
        Initialize temporal consistency checker.
        
        Args:
            consistency_threshold: Minimum verification score in frame N+1
                                  for a claim trusted in frame N (default 0.5)
        """
        self.consistency_threshold = consistency_threshold
        logger.info(
            f"TemporalConsistencyChecker: Initialized with "
            f"consistency_threshold={consistency_threshold}"
        )
    
    def check_consistency(
        self,
        result_frame_n: Dict[str, Any],
        result_frame_n1: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check temporal consistency between two consecutive frame results.
        
        Args:
            result_frame_n: Verification result for frame N, containing:
                - "claims": List of claim dicts with keys:
                  - "text": claim text
                  - "score": verification score (0.0 to 1.0)
                  - "decision": decision string (e.g., "TRUSTED" or "REJECTED")
            
            result_frame_n1: Verification result for frame N+1
                           (same structure as result_frame_n)
        
        Returns:
            Dict with augmented results including:
            - "temporal_flags": List of TemporalInconsistency records
            - "temporal_fpr": Ratio of trusted claims that failed consistency
            - "total_trusted_frame_n": Count of trusted claims in frame N
            - "failed_consistency": Count of claims that failed consistency check
        
        Raises:
            ValueError: If results do not contain required keys
        """
        # Validate input structure
        if "claims" not in result_frame_n or "claims" not in result_frame_n1:
            raise ValueError(
                "Both results must contain 'claims' key with list of claim dicts"
            )
        
        claims_n = result_frame_n["claims"]
        claims_n1 = result_frame_n1["claims"]
        
        # Build claim text -> score/decision maps for frame N+1
        n1_map: Dict[str, Tuple[float, str]] = {}
        for claim in claims_n1:
            claim_text = claim.get("text", "").strip().lower()
            score = claim.get("score", 0.0)
            decision = claim.get("decision", "UNKNOWN")
            n1_map[claim_text] = (score, decision)
        
        # Check consistency for each trusted claim in frame N
        temporal_flags: List[TemporalInconsistency] = []
        total_trusted = 0
        failed_consistency = 0
        
        for claim in claims_n:
            claim_text = claim.get("text", "").strip().lower()
            decision_n = claim.get("decision", "UNKNOWN")
            score_n = claim.get("score", 0.0)
            
            # Only check consistency for trusted claims
            if decision_n != "TRUSTED":
                continue
            
            total_trusted += 1
            
            # Look up claim in frame N+1
            if claim_text not in n1_map:
                # Claim disappeared in next frame (likely hallucination)
                temporal_flags.append(
                    TemporalInconsistency(
                        claim_text=claim_text,
                        frame_n_score=score_n,
                        frame_n_decision=decision_n,
                        frame_n1_score=0.0,
                        frame_n1_decision="MISSING",
                        confidence_drop=score_n - 0.0
                    )
                )
                failed_consistency += 1
                continue
            
            score_n1, decision_n1 = n1_map[claim_text]
            
            # Check if score dropped below threshold
            if score_n1 < self.consistency_threshold:
                temporal_flags.append(
                    TemporalInconsistency(
                        claim_text=claim_text,
                        frame_n_score=score_n,
                        frame_n_decision=decision_n,
                        frame_n1_score=score_n1,
                        frame_n1_decision=decision_n1,
                        confidence_drop=score_n - score_n1
                    )
                )
                failed_consistency += 1
        
        # Calculate temporal false positive rate
        temporal_fpr = (
            failed_consistency / total_trusted
            if total_trusted > 0
            else 0.0
        )
        
        # Build augmented result
        augmented_result = {
            **result_frame_n,
            "temporal_flags": [
                {
                    "claim_text": flag.claim_text,
                    "frame_n_score": flag.frame_n_score,
                    "frame_n_decision": flag.frame_n_decision,
                    "frame_n1_score": flag.frame_n1_score,
                    "frame_n1_decision": flag.frame_n1_decision,
                    "confidence_drop": flag.confidence_drop,
                }
                for flag in temporal_flags
            ],
            "temporal_fpr": temporal_fpr,
            "total_trusted_frame_n": total_trusted,
            "failed_consistency": failed_consistency,
        }
        
        logger.info(
            f"TemporalConsistencyChecker: Checked {total_trusted} trusted claims, "
            f"found {failed_consistency} inconsistencies (FPR={temporal_fpr:.4f})"
        )
        
        return augmented_result
    
    def compute_video_temporal_stat(
        self,
        temporal_flags_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute aggregated temporal statistics across all frames in a video.
        
        Args:
            temporal_flags_list: List of temporal flag dicts from each frame
            
        Returns:
            Dict with aggregated statistics:
            - "total_frames_with_flags": Frames with temporal inconsistencies
            - "total_inconsistent_claims": Total count across all frames
            - "avg_confidence_drop": Average score drop
            - "max_confidence_drop": Maximum score drop observed
        """
        if not temporal_flags_list:
            return {
                "total_frames_with_flags": 0,
                "total_inconsistent_claims": 0,
                "avg_confidence_drop": 0.0,
                "max_confidence_drop": 0.0,
            }
        
        frames_with_flags = sum(1 for flags in temporal_flags_list if flags)
        all_drops = []
        
        for flags in temporal_flags_list:
            for flag in flags:
                all_drops.append(flag.get("confidence_drop", 0.0))
        
        avg_drop = sum(all_drops) / len(all_drops) if all_drops else 0.0
        max_drop = max(all_drops) if all_drops else 0.0
        
        return {
            "total_frames_with_flags": frames_with_flags,
            "total_inconsistent_claims": len(all_drops),
            "avg_confidence_drop": avg_drop,
            "max_confidence_drop": max_drop,
        }

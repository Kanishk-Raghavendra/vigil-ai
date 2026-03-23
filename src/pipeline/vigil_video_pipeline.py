"""
Edge-Optimized CCTV Video Pipeline Orchestrator.

This module orchestrates the end-to-end verification pipeline for CCTV video streams,
integrating keyframe sampling, caption generation, claim extraction, verification,
temporal consistency checking, and profiling into a single coordinated workflow.

This is the primary inference entry point for edge CCTV processing on video clips.

Typical usage:
    pipeline = VIGILVideoOrcestrator()
    results = pipeline.process_video(
        video_path="data/virat/clips/VIRAT_S_000000.mp4",
        interval_seconds=3.0,
        max_frames=50,
        output_path="results/video_results.json"
    )
"""

from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json

from src.models.caption.captioner import MobileVLMCapcioner
from src.models.verifier.clip_verifier import MobileCLIPVerifier
from src.video.keyframe_sampler import KeyframeSampler
from src.video.temporal_consistency import TemporalConsistencyChecker
from src.utils.profiler import EdgeProfiler
from src.utils.claim_extractor import ClaimExtractor
from src.utils.decision import AdaptiveDecisionEngine
from src.utils.logger import Logger


logger = Logger(__name__)


class VIGILVideoOrcestrator:
    """
    Orchestrate VIGIL pipeline for CCTV video stream processing.
    
    This orchestrator coordinates:
    1. Keyframe extraction at configurable intervals
    2. Caption generation for each keyframe using MobileVLM-3B
    3. Claim extraction from captions
    4. Claim verification against image using MobileCLIP-S2
    5. Adaptive threshold decision-making
    6. Temporal consistency checking between consecutive frames
    7. Comprehensive performance profiling
    
    The output is a structured JSON report with per-frame results and
    video-level aggregation metrics.
    
    Attributes:
        caption_model: MobileVLMCapcioner instance
        verifier_model: MobileCLIPVerifier instance
        claim_extractor: ClaimExtractor instance
        decision_engine: AdaptiveDecisionEngine instance
        temporal_checker: TemporalConsistencyChecker instance
        profiler: EdgeProfiler instance
    """
    
    def __init__(
        self,
        temporal_consistency_threshold: float = 0.5,
        adaptive_threshold_mode: str = "balanced"
    ):
        """
        Initialize the VIGIL video orchestrator.
        
        Args:
            temporal_consistency_threshold: Minimum score for claim consistency
                                          between consecutive frames
            adaptive_threshold_mode: Decision mode ("balanced" or "safe")
        """
        logger.info("VIGILVideoOrcestrator: Initializing models and components...")
        
        # Initialize models
        self.caption_model = MobileVLMCapcioner()
        self.verifier_model = MobileCLIPVerifier()
        self.claim_extractor = ClaimExtractor()
        self.decision_engine = AdaptiveDecisionEngine(mode=adaptive_threshold_mode)
        self.temporal_checker = TemporalConsistencyChecker(
            consistency_threshold=temporal_consistency_threshold
        )
        self.profiler = EdgeProfiler()
        
        logger.info("VIGILVideoOrcestrator: All components initialized successfully")
    
    def process_video(
        self,
        video_path: Union[str, Path],
        interval_seconds: float = 3.0,
        max_frames: int = 50,
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process a complete CCTV video through the VIGIL pipeline.
        
        Args:
            video_path: Path to input video file
            interval_seconds: Keyframe sampling interval in seconds (default 3.0)
            max_frames: Maximum frames to process (default 50)
            output_path: Path to save results JSON (optional)
        
        Returns:
            Dict containing:
            - "frames": List of per-frame results
            - "video_summary": Aggregated statistics
            - "profiler_summary": Performance metrics
        
        Raises:
            FileNotFoundError: If video file not found
            RuntimeError: If processing fails
        """
        video_path = Path(video_path)
        logger.info(f"VIGILVideoOrcestrator: Starting video processing {video_path}")
        
        try:
            # Step 1: Extract keyframes
            logger.info(f"VIGILVideoOrcestrator: Extracting keyframes (interval={interval_seconds}s)")
            sampler = KeyframeSampler(video_path)
            keyframes = sampler.extract_keyframes(
                interval_seconds=interval_seconds,
                max_frames=max_frames
            )
            
            logger.info(f"VIGILVideoOrcestrator: Extracted {len(keyframes)} keyframes")
            
            # Step 2: Process each keyframe
            frame_results: List[Dict[str, Any]] = []
            previous_frame_result: Optional[Dict[str, Any]] = None
            
            for frame_idx, (timestamp, image) in enumerate(keyframes):
                logger.info(
                    f"VIGILVideoOrcestrator: Processing frame {frame_idx + 1}/{len(keyframes)} "
                    f"(t={timestamp:.1f}s)"
                )
                
                # Process this frame
                frame_result = self._process_single_frame(
                    image,
                    timestamp,
                    frame_idx
                )
                
                # Check temporal consistency with previous frame
                if previous_frame_result is not None:
                    frame_result = self.temporal_checker.check_consistency(
                        previous_frame_result,
                        frame_result
                    )
                    logger.debug(
                        f"VIGILVideoOrcestrator: Temporal check complete for frame {frame_idx}"
                    )
                
                frame_results.append(frame_result)
                previous_frame_result = frame_result
                
                # Record frame processing
                self.profiler.record_frame()
            
            # Step 3: Aggregate results
            video_summary = self._compute_video_summary(frame_results)
            profiler_summary = self.profiler.get_summary()
            
            # Step 4: Build final output
            output = {
                "video_path": str(video_path),
                "interval_seconds": interval_seconds,
                "frames": frame_results,
                "video_summary": video_summary,
                "profiler_summary": profiler_summary,
            }
            
            # Step 5: Save if requested
            if output_path:
                self._save_results(output, output_path)
            
            logger.info(
                f"VIGILVideoOrcestrator: Video processing complete. "
                f"Processed {len(keyframes)} frames in "
                f"{profiler_summary['avg_latency_per_frame_ms']:.2f}ms avg per frame"
            )
            
            return output
            
        except Exception as e:
            logger.error(f"VIGILVideoOrcestrator: Processing failed - {e}")
            raise RuntimeError(f"Video processing failed: {e}")
    
    def _process_single_frame(
        self,
        image,
        timestamp: float,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Process a single keyframe through the full pipeline.
        
        Args:
            image: PIL Image
            timestamp: Timestamp in seconds
            frame_idx: Frame index
        
        Returns:
            Dict with frame results including claims, decisions, and scores
        """
        # Save temp image for processing
        temp_image_path = Path("/tmp") / f"temp_keyframe_{frame_idx}.jpg"
        image.save(temp_image_path)
        
        try:
            # Caption generation
            self.profiler.start_stage("captioning")
            caption = self.caption_model.generate_caption(temp_image_path)
            self.profiler.end_stage("captioning")
            
            # Claim extraction
            self.profiler.start_stage("claim_extraction")
            claims = self.claim_extractor.extract_claims(caption)
            self.profiler.end_stage("claim_extraction")
            
            # Claim verification
            self.profiler.start_stage("verification")
            verified_claims = []
            for claim in claims:
                score = self.verifier_model.verify_claim(temp_image_path, claim)
                verified_claims.append({
                    "text": claim,
                    "score": score
                })
            self.profiler.end_stage("verification")
            
            # Decision making
            self.profiler.start_stage("decision")
            decisions = self.decision_engine.make_decisions(verified_claims)
            self.profiler.end_stage("decision")
            
            # Merge claims with decisions
            for claim_dict, decision_dict in zip(verified_claims, decisions):
                claim_dict["decision"] = decision_dict["decision"]
                claim_dict["explanation"] = decision_dict.get("explanation", "")
            
            # Count statistics
            trusted_count = sum(1 for c in verified_claims if c["decision"] == "TRUSTED")
            rejected_count = len(verified_claims) - trusted_count
            avg_score = sum(c["score"] for c in verified_claims) / len(verified_claims) \
                       if verified_claims else 0.0
            
            result = {
                "frame_idx": frame_idx,
                "timestamp_seconds": timestamp,
                "caption": caption,
                "claims": verified_claims,
                "statistics": {
                    "total_claims": len(verified_claims),
                    "trusted_claims": trusted_count,
                    "rejected_claims": rejected_count,
                    "avg_claim_score": avg_score,
                }
            }
            
            return result
            
        finally:
            # Clean up temp image
            if temp_image_path.exists():
                temp_image_path.unlink()
    
    def _compute_video_summary(self, frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute video-level aggregate statistics.
        
        Args:
            frame_results: List of per-frame results
        
        Returns:
            Dict with aggregated metrics
        """
        if not frame_results:
            return {
                "total_frames": 0,
                "frames_trusted": 0,
                "frames_rejected": 0,
                "avg_global_score": 0.0,
                "temporal_inconsistency_rate": 0.0,
            }
        
        # Aggregate statistics
        total_trusted = sum(
            f["statistics"]["trusted_claims"]
            for f in frame_results
        )
        total_claims = sum(
            f["statistics"]["total_claims"]
            for f in frame_results
        )
        avg_score = sum(
            f["statistics"]["avg_claim_score"]
            for f in frame_results
        ) / len(frame_results)
        
        # Count temporal inconsistencies
        temporal_flags = [
            f.get("temporal_flags", [])
            for f in frame_results
            if "temporal_flags" in f
        ]
        flat_flags = [item for sublist in temporal_flags for item in sublist]
        
        temporal_inconsistency_rate = (
            len(flat_flags) / total_trusted
            if total_trusted > 0
            else 0.0
        )
        
        return {
            "total_frames": len(frame_results),
            "frames_trusted": sum(
                1 for f in frame_results
                if f["statistics"]["trusted_claims"] > f["statistics"]["rejected_claims"]
            ),
            "frames_rejected": sum(
                1 for f in frame_results
                if f["statistics"]["rejected_claims"] >= f["statistics"]["trusted_claims"]
            ),
            "avg_global_score": avg_score,
            "temporal_inconsistency_rate": temporal_inconsistency_rate,
            "total_claims_across_frames": total_claims,
        }
    
    def _save_results(self, results: Dict[str, Any], output_path: Union[str, Path]):
        """
        Save results to JSON file.
        
        Args:
            results: Results dictionary
            output_path: Path to save JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"VIGILVideoOrcestrator: Saved results to {output_path}")
        except Exception as e:
            logger.error(f"VIGILVideoOrcestrator: Failed to save results - {e}")

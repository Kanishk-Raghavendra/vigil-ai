"""
VIGIL-Edge: Explainable Hallucination Verification for CCTV Vision-Language Systems
Main entry point supporting both single-image inference and edge-optimized video processing.

This system provides trustworthy verification for vision-language model outputs
through model-agnostic, modular architecture with two inference modes:

IMAGE MODE (legacy):
1. Load image
2. Generate caption using BLIP (pretrained LVLM)
3. Extract verifiable claims from caption
4. Verify each claim using CLIP similarity
5. Output structured decisions with explanations

VIDEO MODE (edge-optimized):
1. Extract keyframes from video at configurable interval
2. For each keyframe:
   - Generate caption using MobileVLM-3B (edge-optimized)
   - Extract verifiable claims from caption
   - Verify each claim using MobileCLIP-S2 (edge-optimized)
   - Check temporal consistency with previous frame
3. Aggregate results and output comprehensive report
"""

import sys
import argparse
import json
from pathlib import Path

# Import VIGIL components
from src.models.caption.blip import BLIPCaptionModel
from src.models.verifier.clip import CLIPVerifierModel
from src.pipeline.vigil_pipeline import VIGILPipeline
from src.pipeline.vigil_video_pipeline import VIGILVideoOrcestrator
from src.utils.config import Config
from src.utils.logger import Logger


# Initialize logger
logger = Logger(__name__)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="VIGIL-Edge: Explainable Hallucination Verification for CCTV Systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (Image Mode):
  python main.py image.jpg
  python main.py image.jpg --threshold 0.3
  python main.py image.jpg --config ./configs/config.yaml --output results.json

Examples (Video Mode):
  python main.py --video data/virat/clips/VIRAT_S_000000.mp4 --interval 3.0
  python main.py --video data/virat/clips/VIRAT_S_000000.mp4 --interval 3.0 --output results/video_results.json
  python main.py --video data/virat/clips/VIRAT_S_000000.mp4 --interval 3.0 --device mps
        """,
        prog="VIGIL-Edge"
    )
    
    # Video mode flag
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (enables video mode)"
    )
    
    # Image path (required for image mode, not used in video mode)
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Path to the input image file (required for image mode, omit for video mode)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Keyframe interval in seconds for video mode (default: 3.0)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cpu"],
        default="auto",
        help="Device for inference (auto=detect MPS if available, default: auto)"
    )
    
    # Image mode arguments
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for trust decisions in image mode (default: 0.25)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to configuration file (default: ./configs/config.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON (optional)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate pipeline based on mode
    if args.video:
        return _run_video_pipeline(args)
    else:
        return _run_image_pipeline(args)


def _run_image_pipeline(args) -> int:
    """Run single-image inference pipeline (legacy mode)."""
    if not args.image_path:
        logger.error("Image mode requires image_path argument")
        print("Error: Image mode requires image_path argument")
        return 2
    
    try:
        # Load configuration
        config_path = Path(args.config)
        if config_path.exists():
            config = Config.load(config_path)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.warning(f"Config not found at {config_path}, using defaults")
            config = {}
        
        # Initialize models
        logger.info("Initializing VIGIL image pipeline...")
        caption_model = BLIPCaptionModel()
        verifier_model = CLIPVerifierModel()
        
        # Initialize pipeline
        pipeline = VIGILPipeline(
            caption_model=caption_model,
            verifier_model=verifier_model,
            threshold=args.threshold,
            cache_dir=Path("./results/cache")
        )
        
        # Run pipeline
        logger.info(f"Processing image: {args.image_path}")
        results = pipeline.run(args.image_path)
        
        # Print results
        print("\n" + "="*70)
        print("VIGIL VERIFICATION RESULTS")
        print("="*70 + "\n")
        
        print(f"Image: {Path(args.image_path).name}")
        print(f"Caption: {results['caption']}")
        print(f"Threshold: {args.threshold:.0%}\n")
        
        if results['verifications']:
            print("Claims Verified:")
            print("-" * 70)
            for v in results['verifications']:
                status = "✓ TRUST" if v['decision'] == 'TRUST' else "✗ WARNING"
                print(f"{status:12s} | Score: {v['score']:.1%} | {v['claim']}")
            
            print("\n" + "-" * 70)
            stats = results['statistics']
            print(f"Summary: {stats['trusted']} trusted, {stats['warnings']} warnings")
            print(f"Average confidence: {stats['average_score']:.1%}")
        else:
            print("No claims extracted from caption.")
        
        print("="*70 + "\n")
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            pipeline.save_results(results, output_path)
            logger.info(f"Results saved to {output_path}")
        
        # Return exit code based on results
        warning_count = results['statistics'].get('warnings', 0) if results['statistics'] else 0
        return 0 if warning_count == 0 else 1
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 3


def _run_video_pipeline(args) -> int:
    """Run edge-optimized video inference pipeline."""
    try:
        logger.info("Initializing VIGIL-Edge video pipeline...")
        
        # Initialize video orchestrator
        orchestrator = VIGILVideoOrcestrator()
        
        # Process video
        logger.info(
            f"Processing video: {args.video} "
            f"(interval={args.interval}s)"
        )
        
        results = orchestrator.process_video(
            video_path=args.video,
            interval_seconds=args.interval,
            max_frames=50,
            output_path=args.output
        )
        
        # Print summary
        print("\n" + "="*70)
        print("VIGIL-EDGE VIDEO VERIFICATION RESULTS")
        print("="*70 + "\n")
        
        print(f"Video: {Path(args.video).name}")
        
        summary = results.get("video_summary", {})
        print(f"Total frames processed: {summary.get('total_frames', 0)}")
        print(f"Frames trusted: {summary.get('frames_trusted', 0)}")
        print(f"Frames rejected: {summary.get('frames_rejected', 0)}")
        print(f"Average global score: {summary.get('avg_global_score', 0):.4f}")
        print(f"Temporal inconsistency rate: {summary.get('temporal_inconsistency_rate', 0):.4f}")
        
        profiler = results.get("profiler_summary", {})
        print(f"\nPerformance:")
        print(f"  Average latency per frame: {profiler.get('avg_latency_per_frame_ms', 0):.2f}ms")
        print(f"  Peak memory usage: {profiler.get('peak_memory_mb', 0):.2f}MB")
        
        print("\n" + "="*70 + "\n")
        
        # Results already saved by orchestrator if output_path provided
        if args.output:
            logger.info(f"Results saved to {args.output}")
        
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"Video file not found: {e}")
        print(f"Error: Video file not found: {e}")
        return 2
    except Exception as e:
        logger.error(f"Unexpected error in video processing: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

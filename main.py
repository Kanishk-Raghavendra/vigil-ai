"""
VIGIL: Explainable Cross-Modal Verification Framework
Main entry point for research-grade prototype.

This system provides trustworthy verification for vision-language model outputs
through model-agnostic, modular architecture.

Pipeline:
1. Load image
2. Generate caption using BLIP (pretrained LVLM)
3. Extract verifiable claims from caption
4. Verify each claim using CLIP similarity
5. Output structured decisions with explanations
"""

import sys
import argparse
import json
from pathlib import Path

# Import VIGIL components
from src.models.caption.blip import BLIPCaptionModel
from src.models.verifier.clip import CLIPVerifierModel
from src.pipeline.vigil_pipeline import VIGILPipeline
from src.utils.config import Config
from src.utils.logger import Logger


# Initialize logger
logger = Logger(__name__)


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="VIGIL: Explainable Cross-Modal Verification Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image.jpg
  python main.py image.jpg --threshold 0.3
  python main.py image.jpg --config ./configs/config.yaml --output results.json
        """,
        prog="VIGIL"
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the input image file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for trust decisions (default: 0.25)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
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
        logger.info("Initializing VIGIL system...")
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


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

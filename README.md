# VIGIL-Edge: Explainable On-Device Hallucination Verification for CCTV Vision-Language Systems

## 1) Project Description

VIGIL-Edge is a production-grade edge verification layer for CCTV vision-language systems. It is designed around a safety-first principle and operates on-device: treat generated captions as untrusted claims, then verify each claim against video evidence at configurable keyframe intervals before final acceptance. The framework runs entirely on Apple Silicon MPS with no internet requirement, enabling privacy-preserving, real-time hallucination detection in field deployments.

The framework targets hallucination reduction in CCTV video streams and provides structured explanations, calibrated confidence, safety-aware aggregation, and temporal consistency checking (a novel contribution for video processing).

## Key Results

Custom benchmark (100 images / 600 claims):

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Prompt + Adaptive | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Full VIGIL (Balanced) | 0.7033 | 0.6723 | 0.7933 | 0.3867 |
| Full VIGIL (Safe) | 0.6567 | 0.6621 | 0.6400 | 0.3267 |

POPE-COCO adversarial (500 images / 3000 claims):

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Full_VIGIL_Balanced | 0.7290 | 0.6929 | 0.8227 | 0.3647 |
| Full_VIGIL_Safe | 0.6890 | 0.7529 | 0.5627 | 0.1847 |
| Baseline_CLIP_Fixed | 0.5000 | 0.5000 | 1.0000 | 1.0000 |

Safe mode reduces FPR to 18.5% on adversarial data with a controllable recall trade-off.

## Edge Deployment

**Target device**: Apple Silicon (MPS) / resource-constrained edge devices  
**Caption model**: MobileVLM-3B (lightweight, optimized for edge)  
**Verifier**: MobileCLIP-S2 (Apple's lightweight CLIP)  
**Input**: Video clips (VIRAT Ground Dataset 2.0)  
**Keyframe interval**: Configurable (default 3 seconds)  
**Processing**: Entirely on-device, no internet required  
**Novel capability**: Temporal consistency checking across consecutive frames to detect video hallucinations

## 2) Problem Statement

Modern vision-language models can produce fluent but unsupported statements. These hallucinations are risky in high-stakes contexts (education, healthcare, accessibility, and automated decision support).

Challenge:
- Generated language appears plausible.
- End users cannot easily inspect claim-level validity.
- Single-threshold acceptance is brittle across heterogeneous images.

VIGIL addresses this by introducing a separate cross-modal verifier that checks generated claims against visual evidence.

## 3) Core Idea

VIGIL treats the captioning model as an untrusted generator and adds an independent verification pipeline:

Image -> Caption Generation -> Claim Extraction -> Claim Verification -> Trust Decision + Explanation

## 4) Key Features

- Adaptive thresholding (per image score distribution)
- Soft aggregation (global trust from claim confidence and agreement)
- Prompt engineering for CLIP verification robustness
- Explainability with claim-level scores, decisions, and global summary
- Safety-aware operation modes:
	- Balanced mode for general usage
	- Safe mode for lower false positive behavior

## 5) Architecture Overview

1. Caption generation
- BLIP generates an initial image caption.

2. Claim extraction
- Caption text is converted into verifiable atomic claims.

3. Cross-modal verification
- CLIP computes image-text alignment scores per claim.
- Prompted and non-prompted forms are supported.

4. Decision logic
- Adaptive claim threshold: mean(score) - 0.5*std(score), clamped.
- Soft global aggregation combines:
	- Average claim score
	- Trusted-claim ratio
- Global threshold is adaptive and mode-calibrated.

5. Explainable output
- Per-claim and global metadata are serialized to JSON for auditability.

## 6) Folder Structure

vigil-ai/
- main.py
- requirements.txt
- configs/
- data/
	- sample_images/
	- expanded_annotations.json
	- virat/
		- clips/  (place VIRAT .mp4 files here)
- results/
	- ablation_results.json
	- mode_comparison.json
	- threshold_sweep.json
	- video_results.json
	- profiler_summary.json
- src/
	- data/
		- generate_dataset.py
	- evaluation/
		- run_ablation.py
	- models/
		- caption/
			- captioner.py (MobileVLM-3B)
		- verifier/
			- clip_verifier.py (MobileCLIP-S2)
	- pipeline/
		- vigil_pipeline.py (image mode)
		- vigil_video_pipeline.py (video mode)
	- utils/
		- profiler.py (edge profiling)
	- video/
		- keyframe_sampler.py (video extraction)
		- temporal_consistency.py (temporal verification)
- demo/
	- app.py (Gradio demo)

## 7) Installation

1. Create and activate environment

macOS/Linux:
- python3 -m venv .venv
- source .venv/bin/activate

2. Install dependencies
- pip install -r requirements.txt

## 8) How to Run

### A) Single Image Inference (Legacy Mode)

- python main.py data/sample_images/images/000000000025.jpg --output results/pipeline_soft_aggregation_sample.json

### B) Video Inference (Edge CCTV Mode)

- python main.py --video data/virat/clips/VIRAT_S_000000.mp4 --interval 3.0 --output results/video_results.json

Parameters:
- --video: Path to video file (MP4, AVI, MOV)
- --interval: Keyframe interval in seconds (default 3.0)
- --device: Device selection (auto/mps/cpu, default auto)
- --output: Output JSON path (optional)

### C) Interactive Demo App

- python demo/app.py

Opens a Gradio interface at http://localhost:7860 for real-time video processing and visualization.

### D) Dataset Generation (100 images, 600 claims target)

- python -m src.data.generate_dataset --images-dir data/sample_images/images --output data/expanded_annotations.json --target-images 100 --seed 42

### E) Ablation Study

- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced

### F) Mode Comparison (Balanced vs Safe)

- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced --compare-modes

## 9) Example Outputs

Key generated files (image mode):
- results/pipeline_soft_aggregation_sample.json
- results/ablation_results.json
- results/statistics.json
- results/claim_type_analysis.json
- results/mode_comparison.json
- results/mode_comparison.csv

Video mode outputs:
- results/video_results.json (per-frame claims, temporal flags, video summary)
- results/profiler_summary.json (latency and memory metrics)

Global trust output (example fields):
- global_score
- average_score
- trusted_ratio
- threshold
- mode
- final_decision

## 10) Key Results Summary

Current scaled benchmark (100 images / 600 claims):

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Prompt + Adaptive | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Full VIGIL (Balanced) | 0.7033 | 0.6723 | 0.7933 | 0.3867 |
| Full VIGIL (Safe) | 0.6567 | 0.6621 | 0.6400 | 0.3267 |

Interpretation:
- Balanced mode reduces FPR versus adaptive-only verification while maintaining comparable accuracy.
- Safe mode further reduces FPR at a stronger recall trade-off.
- POPE adversarial benchmark (500 images / 3000 claims): Full_VIGIL_Balanced reaches 0.7290 accuracy, while Full_VIGIL_Safe cuts FPR from 0.3647 to 0.1847 with a recall trade-off.

## 11) Additional Documentation

- Setup walkthrough: SETUP_GUIDE.md
- Research-style report: PROJECT_REPORT.md
- Experiment details: EXPERIMENTS.md
- Development timeline: CONTRIBUTION_LOG.md
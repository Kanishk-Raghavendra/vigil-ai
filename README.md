# VIGIL: An Explainable Cross-Modal Verification Framework for Vision-Language Systems

## 1) Project Description

VIGIL is a research-grade verification layer for vision-language systems. It is designed around a safety-first principle: treat generated captions as untrusted claims, then verify each claim against the image before final acceptance.

The framework targets hallucination reduction in image-to-text outputs and provides structured explanations, calibrated confidence, and safety-aware aggregation.

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
- results/
	- ablation_results.json
	- mode_comparison.json
	- threshold_sweep.json
- src/
	- data/
		- generate_dataset.py
	- evaluation/
		- run_ablation.py
	- models/
		- caption/
		- verifier/
	- pipeline/
		- vigil_pipeline.py
	- utils/

## 7) Installation

1. Create and activate environment

macOS/Linux:
- python3 -m venv .venv
- source .venv/bin/activate

2. Install dependencies
- pip install -r requirements.txt

## 8) How to Run

### A) Single Image Inference

- python main.py data/sample_images/images/000000000025.jpg --output results/pipeline_soft_aggregation_sample.json

### B) Dataset Generation (100 images, 600 claims target)

- python -m src.data.generate_dataset --images-dir data/sample_images/images --output data/expanded_annotations.json --target-images 100 --seed 42

### C) Ablation Study

- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced

### D) Mode Comparison (Balanced vs Safe)

- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced --compare-modes

## 9) Example Outputs

Key generated files:
- results/pipeline_soft_aggregation_sample.json
- results/ablation_results.json
- results/statistics.json
- results/claim_type_analysis.json
- results/mode_comparison.json
- results/mode_comparison.csv

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
| Full VIGIL (Balanced) | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Full VIGIL (Safe) | 0.6950 | 0.6657 | 0.7833 | 0.3933 |

Interpretation:
- Balanced mode preserves top-line performance.
- Safe mode reduces FPR with a modest recall trade-off.

## 11) Additional Documentation

- Setup walkthrough: SETUP_GUIDE.md
- Research-style report: PROJECT_REPORT.md
- Experiment details: EXPERIMENTS.md
- Development timeline: CONTRIBUTION_LOG.md
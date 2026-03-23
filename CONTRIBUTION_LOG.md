# CONTRIBUTION_LOG

This document records the chronological development of VIGIL from initial prototype to research-grade edge verification system.

## Phase 10 (2026-03-23): Edge Deployment and CCTV Adaptation

Full upgrade from static image verification to edge-optimized CCTV hallucination verifier:
- Replaced BLIP caption model with MobileVLM-3B for edge inference (Apple Silicon optimized)
- Replaced CLIP ViT-B/32 verifier with MobileCLIP-S2 (lightweight, MPS-compatible)
- Added keyframe sampler for video clip extraction at configurable intervals (default 3.0s)
- Added temporal consistency checker: novel contribution detecting hallucinations across consecutive frames
- Added edge profiler for per-stage latency and memory benchmarking
- Added Gradio demo app for interactive video processing and visualization
- Updated main.py to support both image mode (legacy) and video mode (edge)
- Created src/video/ module for video-specific processing (keyframe sampling, temporal checking)
- Created src/utils/profiler.py for real-time performance monitoring
- Created src/pipeline/vigil_video_pipeline.py as primary video orchestrator
- Updated requirements.txt with new dependencies (mobileclip, opencv, gradio, torch>=2.1.0)
- Rewrote all documentation for CCTV/edge context:
  - PROJECT_REPORT.md: comprehensive rewrite with temporal consistency, edge profiling, CCTV motivation
  - README.md: updated title, added edge deployment section, video mode commands, demo app
  - EXPERIMENTS.md: added edge video pipeline experiments section
  - CONTRIBUTION_LOG.md: added Phase 10 entry (this one)
- Target deployment: VIRAT Ground Dataset 2.0 on Apple Silicon MPS
- Dataset placement: users manually place video clips in data/virat/clips/
- Processing target: <500ms per frame for 3-second keyframe intervals (near real-time at 0.33fps keyframe rate)
- Privacy model: entirely on-device, no internet calls, suitable for sensitive CCTV environments

All existing verification quality metrics (0.7067, 0.7290 POPE, 0.1847 FPR Safe) remain unchanged for image mode.

## Phase 1: Basic Pipeline (BLIP + CLIP)

- Implemented end-to-end inference chain:
  - image input,
  - BLIP caption generation,
  - CLIP-based claim-image scoring.
- Established modular model interfaces for caption and verifier components.

## Phase 2: Claim Extraction

- Added structured claim extraction from generated captions.
- Enabled claim-level verification instead of whole-caption acceptance.

## Phase 3: Evaluation Metrics

- Added benchmark evaluation over annotated claims.
- Integrated core metrics:
  - accuracy,
  - precision,
  - recall,
  - FPR,
  - hallucination detection rate.
- Added threshold sweep outputs to JSON and CSV.

## Phase 4: Adaptive Thresholding

- Introduced per-image adaptive threshold for claim decisions.
- Replaced static thresholding with score-distribution-aware calibration.

## Phase 5: Soft Aggregation (Rigid Threshold Fix)

- Replaced rigid trusted-ratio rule with soft global aggregation:
  - global_score combines average score and trusted ratio.
- Added adaptive global threshold and calibration modes.
- Added balanced and safe aggregation settings.
- Removed degradation caused by earlier hard global decision logic.

## Phase 6: Dataset Scaling

- Added automated dataset expansion script.
- Generated 100-image, 600-claim benchmark set.
- Included synthetic hallucination generation and claim-type labels.

## Phase 7: Ablation Study + Baselines

- Implemented structured ablation framework:
  - base CLIP,
  - prompt engineering,
  - prompt + adaptive,
  - full VIGIL.
- Added strong baselines:
  - accept all,
  - random,
  - fixed-threshold CLIP.

## Phase 8: Statistical Validation (Bootstrap)

- Added bootstrap resampling (100 samples).
- Reported mean and standard deviation for key metrics.
- Added claim-type analysis artifacts.

## Phase 9: Safe vs Balanced Mode Comparison

- Added one-command dual-mode execution with compare flag.
- Added mode comparison table and paper-ready files:
  - results/mode_comparison.json
  - results/mode_comparison.csv
- Added automatic interpretation message based on FPR behavior.

## 2026-03-21: POPE Adversarial Benchmark Milestone

- Added POPE evaluator at src/evaluation/run_pope.py with robust JSONL parsing support and ablation-style method reporting.
- Downloaded and evaluated the POPE-COCO adversarial split using 500 images and 3000 claims.
- Recorded benchmark comparison for Full_VIGIL_Balanced, Full_VIGIL_Safe, and Baseline_CLIP_Fixed_0.5.
- Updated EXPERIMENTS.md, PROJECT_REPORT.md, README.md, and CONTRIBUTION_LOG.md with the new POPE findings.

## Current Status

VIGIL is now documented and instrumented as a Tier-2 conference-level research project with:
- modular implementation,
- explainable outputs,
- reproducible setup,
- comprehensive evaluation artifacts,
- explicit safety-performance trade-off controls.

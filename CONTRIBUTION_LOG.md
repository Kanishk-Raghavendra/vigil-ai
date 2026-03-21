# CONTRIBUTION_LOG

This document records the chronological development of VIGIL from initial prototype to research-grade evaluation system.

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

## Current Status

VIGIL is now documented and instrumented as a Tier-2 conference-level research project with:
- modular implementation,
- explainable outputs,
- reproducible setup,
- comprehensive evaluation artifacts,
- explicit safety-performance trade-off controls.

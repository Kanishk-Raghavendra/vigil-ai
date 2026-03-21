# VIGIL Setup Guide

This guide provides a complete, reproducible setup for the VIGIL project.

## 1) Clone Repository

- git clone <your-repository-url>
- cd vigil-ai

## 2) Create Virtual Environment

macOS/Linux:
- python3 -m venv .venv
- source .venv/bin/activate

Optional check:
- python --version
- pip --version

## 3) Install Dependencies

- pip install --upgrade pip
- pip install -r requirements.txt

## 4) Data Options

### Option A: Use Provided Sample Images (Recommended Quick Start)

Use the existing dataset at:
- data/sample_images/images
- data/sample_images/annotations.json

No additional download needed.

### Option B: Use COCO-128 (Lightweight External Dataset)

If you want an external lightweight benchmark, prepare COCO-128 under data/ and then point scripts to that image directory.

Example expected command style once images are present:
- python -m src.data.generate_dataset --images-dir data/coco128/images/train2017 --output data/expanded_annotations.json --target-images 100 --seed 42

## 5) Generate Expanded Dataset

Default project command (100 images):
- python -m src.data.generate_dataset --images-dir data/sample_images/images --output data/expanded_annotations.json --target-images 100 --seed 42

Expected output:
- data/expanded_annotations.json

## 6) Run Single-Image Pipeline

- python main.py data/sample_images/images/000000000025.jpg --output results/pipeline_soft_aggregation_sample.json

This produces explainable per-claim and global trust outputs.

## 7) Run Annotation Evaluation

- python -m src.evaluate_annotations --annotations data/sample_images/annotations.json --images-dir data/sample_images/images --limit 5 --threshold 0.6 --output results/metrics/benchmark_5_images.json --sweep-json results/threshold_sweep.json --sweep-csv results/threshold_sweep.csv --failure-output results/failure_cases.json

## 8) Run Ablation Study

Balanced mode:
- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced

Safe mode:
- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode safe

## 9) Run Mode Comparison (Balanced vs Safe)

One-command comparison:
- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced --compare-modes

Generated files:
- results/mode_comparison.json
- results/mode_comparison.csv

## 10) Key Output Files

- results/ablation_results.json
- results/ablation_results.csv
- results/statistics.json
- results/statistics.csv
- results/claim_type_analysis.json
- results/claim_type_analysis.csv
- results/mode_comparison.json
- results/mode_comparison.csv
- results/evaluation_log.txt

## 11) Troubleshooting

- Model download/caching issues:
  - Re-run command after ensuring internet access.
- Slow CPU execution:
  - Use smaller limits first (for example --max-images in ablation script).
- Missing files:
  - Confirm paths for images and annotations before running evaluation.

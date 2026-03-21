# EXPERIMENTS

## 1. Experimental Goals

The experiments evaluate whether VIGIL improves claim reliability in vision-language outputs while preserving explainability and operational safety.

Primary goals:
- measure gains from adaptive thresholding,
- validate soft aggregation behavior,
- compare balanced and safe safety profiles,
- benchmark against strong baselines.

## 2. Experiment Workflow

1. Construct expanded dataset from images.
2. Score all claims with CLIP (raw and prompt-engineered text).
3. Run ablation variants and baselines.
4. Compute metrics and bootstrap statistics.
5. Run balanced-safe mode comparison.
6. Export JSON/CSV for reporting.

## 3. Dataset Construction Process

Command:
- python -m src.data.generate_dataset --images-dir data/sample_images/images --output data/expanded_annotations.json --target-images 100 --seed 42

Process:
- BLIP generates captions.
- Claim extractor selects top correct claims.
- Synthetic hallucinations are created via object replacement and relation swaps.
- Claim types are tagged (object, relation, attribute).

Final size:
- 100 images
- 600 claims (300 correct, 300 hallucinated)

## 4. Ablation Study Design

Command:
- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced

Variants:
- A_Base_CLIP_Fixed
- B_Prompt_Engineering
- C_Prompt_Adaptive
- D_Full_VIGIL

## 5. Baselines

Included baselines:
- Baseline_Accept_All
- Baseline_Random
- Baseline_CLIP_Fixed_0.5

These cover naive optimistic, random, and fixed-threshold verifier behaviors.

## 6. Threshold Sweep

Command:
- python -m src.evaluate_annotations --annotations data/sample_images/annotations.json --images-dir data/sample_images/images --limit 5 --threshold 0.6 --output results/metrics/benchmark_5_images.json --sweep-json results/threshold_sweep.json --sweep-csv results/threshold_sweep.csv --failure-output results/failure_cases.json

Sweep behavior:
- threshold range from 0.10 to 0.90 with fixed increments.
- outputs include accuracy, precision, recall, FPR, and hallucination detection rate.

## 7. Mode Comparison (Balanced vs Safe)

Command:
- python -m src.evaluation.run_ablation --annotations data/expanded_annotations.json --images-dir data/sample_images/images --bootstrap-samples 100 --seed 42 --aggregation-mode balanced --compare-modes

Generated outputs:
- results/mode_comparison.json
- results/mode_comparison.csv

## 8. Results Tables

### 8.1 Ablation and Baselines

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| A_Base_CLIP_Fixed | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| B_Prompt_Engineering | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| C_Prompt_Adaptive | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| D_Full_VIGIL (Balanced) | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Baseline_Accept_All | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| Baseline_Random | 0.4967 | 0.4966 | 0.4867 | 0.4933 |
| Baseline_CLIP_Fixed_0.5 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |

### 8.2 Mode Comparison

| Mode | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Balanced | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Safe | 0.6950 | 0.6657 | 0.7833 | 0.3933 |

### POPE Adversarial Benchmark (COCO, 500 images / 3000 claims)

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Full_VIGIL_Balanced | 0.7290 | 0.6929 | 0.8227 | 0.3647 |
| Full_VIGIL_Safe | 0.6890 | 0.7529 | 0.5627 | 0.1847 |
| Baseline_CLIP_Fixed_0.5 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |

Safe mode reduces FPR by about 50% versus Balanced (0.365 -> 0.185), at the cost of recall. This demonstrates a controllable safety-recall trade-off curve for deployment tuning.

## 9. Key Findings

- Adaptive thresholding provides a major lift over fixed-threshold CLIP variants.
- Prompt + Adaptive is the strongest non-aggregation configuration.
- Soft aggregation resolves prior rigid-threshold degradation issues.
- Safe mode lowers FPR and demonstrates a clear safety-recall trade-off.

## 10. Artifact Index

Core files:
- results/ablation_results.json
- results/ablation_results.csv
- results/statistics.json
- results/statistics.csv
- results/claim_type_analysis.json
- results/claim_type_analysis.csv
- results/mode_comparison.json
- results/mode_comparison.csv
- results/threshold_sweep.json
- results/threshold_sweep.csv
- results/evaluation_log.txt

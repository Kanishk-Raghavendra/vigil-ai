# PROJECT_REPORT

## Title

VIGIL: An Explainable Cross-Modal Verification Framework for Vision-Language Systems

## 1. Introduction

Vision-language models can generate fluent image descriptions, but they may include unsupported content (hallucinations). This project develops VIGIL, a verification-first framework that audits generated claims against visual evidence before final trust decisions.

## 2. Problem Statement

The central problem is reliability in image-grounded language generation:
- Generated text may be coherent but factually ungrounded.
- End users need interpretable confidence and safety behavior.
- A static decision rule often fails across diverse scenes.

## 3. Motivation

Hallucination mitigation is essential for trustworthy AI systems. A practical verifier must:
- operate independently from the caption model,
- provide explainable claim-level evidence,
- support configurable safety profiles.

VIGIL is designed as a post-generation verification layer with explicit trust calibration.

## 4. Related Work (Brief)

Relevant areas include:
- Vision-language generation (captioning and multimodal foundation models)
- Cross-modal retrieval/verification with CLIP-like encoders
- Hallucination analysis in multimodal systems
- Post-hoc explainability and calibration methods

VIGIL combines these strands into a modular verification pipeline with safety-aware aggregation modes.

## 5. Proposed Method

### 5.1 Caption Generation

A BLIP-based caption model generates an initial image description.

### 5.2 Claim Extraction

The caption is decomposed into atomic, verifiable claims to avoid binary whole-caption acceptance.

### 5.3 Cross-Modal Verification

A CLIP verifier scores image-claim alignment. Prompt engineering is used to improve text-query consistency for visual concepts.

### 5.4 Adaptive Thresholding

Claim decisions are made using a per-image adaptive threshold, computed from the score distribution:
- threshold = mean(scores) - 0.5 * std(scores), clamped for stability.

### 5.5 Soft Aggregation

Global trust combines both evidence strength and decision agreement:
- average_score = mean(claim scores)
- trusted_ratio = trusted_claims / total_claims
- global_score = 0.7 * average_score + 0.3 * trusted_ratio

Adaptive global threshold:
- global_threshold = mean(scores) - 0.3 * std(scores)
- clamped to [0.3, 0.7]

Mode calibration:
- balanced: default adaptive threshold
- safe: threshold + 0.05 (for stricter acceptance)

## 6. Architecture Diagram (Textual)

Image
-> BLIP Caption
-> Claim Extractor
-> CLIP Verifier (raw + prompted claim text)
-> Adaptive Claim Decisions
-> Soft Global Aggregation (balanced/safe)
-> Explainable JSON Output

## 7. Novel Contributions

1. Adaptive thresholding for per-image calibration.
2. Soft aggregation replacing rigid ratio-based logic.
3. Safety-aware mode control (balanced vs safe).
4. Research-oriented evaluation stack with ablation, baselines, bootstrap, and mode comparison.

## 8. Experimental Setup

Dataset:
- 100 images
- 600 total claims
- 300 positive (correct) and 300 negative (hallucinated)

Metrics:
- Accuracy
- Precision
- Recall
- FPR

Evaluation artifacts:
- Ablation and baseline comparison
- Bootstrap statistics (100 samples)
- Claim-type analysis
- Balanced vs safe mode comparison

## 9. Results

### 9.1 Ablation and Baselines

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| A_Base_CLIP_Fixed | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| B_Prompt_Engineering | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| C_Prompt_Adaptive | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| D_Full_VIGIL (Balanced) | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Baseline_Random | 0.4967 | 0.4966 | 0.4867 | 0.4933 |

### 9.2 Mode Comparison

| Mode | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Balanced | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| Safe | 0.6950 | 0.6657 | 0.7833 | 0.3933 |

### 9.3 POPE Adversarial Benchmark

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Full_VIGIL_Balanced | 0.7290 | 0.6929 | 0.8227 | 0.3647 |
| Full_VIGIL_Safe | 0.6890 | 0.7529 | 0.5627 | 0.1847 |
| Baseline_CLIP_Fixed_0.5 | 0.5000 | 0.5000 | 1.0000 | 1.0000 |

VIGIL reaches 0.729 accuracy on the POPE adversarial split while operating as a post-hoc verifier over generated claims. Safe mode provides a meaningful false-positive reduction relative to Balanced mode. Together, the safe-balanced trade-off enables deployment-time safety control without retraining.

## 10. Discussion

### 10.1 Trade-off Analysis

Safe mode reduces false positives (lower FPR) with a moderate recall drop. This is expected in safety-oriented deployments where over-trusting is costly.

### 10.2 Failure Cases

Observed failure patterns include:
- relation-heavy claims with limited visual cues,
- object confusion in complex scenes,
- sparse attribute coverage in generated synthetic annotations.

These indicate opportunities for richer claim generation and relation-focused verification prompts.

## 11. Conclusion

VIGIL demonstrates that a modular verification layer can significantly improve reliability over naive fixed-threshold verification. The framework now supports:
- adaptive, explainable decision making,
- robust ablation and baseline reporting,
- explicit balanced-safe operating modes.

The current pipeline is suitable for Tier-2 conference-level demonstration and further refinement toward publication-quality experiments.
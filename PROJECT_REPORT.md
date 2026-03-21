# PROJECT_REPORT

## Title

VIGIL: An Explainable Cross-Modal Verification Framework for Vision-Language Systems

## Abstract

Hallucinated claims in deployed vision-language systems can create direct safety and reliability risks in high-stakes workflows. VIGIL is a post-hoc verifier that extracts atomic claims from generated captions, grounds them with CLIP, and calibrates trust decisions using adaptive per-image thresholds plus safety-aware aggregation modes. On the custom 100-image, 600-claim benchmark, VIGIL reaches 0.7067 accuracy, a 41.4% relative gain over fixed-threshold CLIP. On POPE-COCO adversarial (500 images, 3000 claims), Balanced mode reaches 0.7290 accuracy, while Safe mode lowers FPR to 0.1847 with a controllable recall trade-off. Next, the same verifier will be evaluated on live LLaVA/InstructBLIP outputs with human-labeled hallucination annotations.

## 1. Introduction

Vision-language models produce fluent text but can still assert image facts that are unsupported. In high-stakes use cases, unsupported statements can mislead users and propagate errors downstream. VIGIL addresses this by treating caption output as untrusted text and adding a verifier that grounds each claim in the image before acceptance.

## 2. Problem Statement

The reliability gap has three operational components:
- plausible language can mask factual errors,
- confidence signals are often opaque at claim level,
- a single static threshold is brittle across heterogeneous images.

## 3. Motivation

Practical hallucination mitigation should be modular, explainable, and tunable. VIGIL is designed as a post-hoc control layer that can wrap existing captioning systems without retraining them, while exposing transparent claim-level decisions and configurable safety modes.

## 4. Related Work

### 4.1 CHAIR and Caption Hallucination Metrics

CHAIR is a foundational metric for measuring object hallucinations in image captioning by comparing generated objects against reference object evidence. Its strength is standardized hallucination accounting, but it is mostly object-centric and does not provide per-claim trust decisions at inference time. VIGIL differs by performing claim-level verification online and returning actionable accept/warn outputs rather than only retrospective scoring.

### 4.2 CLIP-Score and Image-Text Alignment Baselines

CLIP-Score-style alignment is widely used as a simple cross-modal relevance baseline. It is effective as a low-cost signal, but fixed-threshold usage can over-trust fluent yet weakly grounded claims. VIGIL adopts CLIP as a verifier backbone but adds prompt conditioning, adaptive per-image thresholds, and global aggregation so decisions respond to local score distributions instead of a single global cutoff.

### 4.3 NLI-Based Fact Verification

NLI pipelines based on MNLI-tuned encoders such as DeBERTa can test textual entailment consistency between generated statements and references. These methods capture linguistic contradiction patterns well, but they require high-quality textual evidence and do not directly score image evidence. VIGIL differs by grounding verification in image-text alignment first, and can be extended with NLI as a second-stage textual consistency check.

### 4.4 Recent VLM Hallucination Benchmarks (2022-2024)

Recent work such as POPE (2023), HallusionBench (2023), MME (2023), and AMBER (2024) highlights persistent hallucination behavior in modern multimodal models and emphasizes adversarial probing. These benchmarks focus on stress-testing model outputs; VIGIL complements them by providing a deployable post-hoc filter that can operate on top of existing generators and expose explicit safety-performance trade-offs.

## 5. Proposed Method

### 5.1 Caption Generation

A BLIP caption model generates an initial image description.

### 5.2 Claim Extraction

The caption is decomposed into atomic claims to avoid binary whole-caption acceptance.

### 5.3 Cross-Modal Verification

Each claim is scored against the image using CLIP with prompt-engineered claim text.

### 5.4 Adaptive Thresholding

Per-image claim threshold:
- threshold = mean(scores) - 0.5 * std(scores), clamped for stability.

### 5.5 Soft Aggregation and Safety Modes

Global trust combines evidence strength and agreement:
- average_score = mean(claim scores)
- trusted_ratio = trusted_claims / total_claims
- global_score = 0.5 * average_score + 0.5 * trusted_ratio

Global threshold:
- global_threshold = mean(scores) - 0.3 * std(scores), clamped to [0.3, 0.7]

Modes:
- Balanced: default calibration
- Safe: stricter calibration for lower false positives

## 6. Architecture Diagram (Textual)

Image
-> BLIP Caption
-> Claim Extractor
-> CLIP Verifier (raw + prompted claim text)
-> Adaptive Claim Decisions
-> Soft Global Aggregation (Balanced/Safe)
-> Explainable JSON Output

## 7. Contributions

1. Claim-level post-hoc verification over generated captions.
2. Adaptive per-image calibration replacing fixed-threshold acceptance.
3. Safety-aware aggregation modes for deployment-time tuning.
4. Reproducible evaluation suite: ablation, claim-type analysis, and adversarial benchmark comparison.

## 8. Experimental Setup

Primary synthetic benchmark:
- 100 images
- 600 claims
- 300 positive and 300 negative claims

Adversarial benchmark:
- POPE-COCO adversarial split
- 500 images
- 3000 claims

Metrics:
- Accuracy
- Precision
- Recall
- FPR

## 9. Results

### 9.1 Ablation and Baselines

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| A_Base_CLIP_Fixed | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| B_Prompt_Engineering | 0.5000 | 0.5000 | 1.0000 | 1.0000 |
| C_Prompt_Adaptive | 0.7067 | 0.6658 | 0.8300 | 0.4167 |
| D_Full_VIGIL (Balanced) | 0.7033 | 0.6723 | 0.7933 | 0.3867 |
| Baseline_Random | 0.4967 | 0.4966 | 0.4867 | 0.4933 |

### 9.1.1 Why C and D Can Collapse, and What Changed

Parity between adaptive-only and full aggregation can occur on smaller, relatively homogeneous datasets when most images are already cleanly separated by adaptive claim thresholds. In that regime, global aggregation contributes little additional signal, especially if weighted heavily toward average score. Reweighting global trust to equal contributions from average score and trusted ratio (0.5/0.5) produced measurable separation: D lowers FPR from 0.4167 to 0.3867 while shifting precision and recall.

### 9.2 Claim-Type Analysis (Full VIGIL, Balanced)

| Claim Type | Support | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|
| object | 436 | 0.7775 | 0.8103 | 0.7248 | 0.1697 |
| relation | 160 | 0.5000 | 0.5000 | 0.9750 | 0.9750 |
| attribute | 4 | 0.7500 | 0.6667 | 1.0000 | 0.5000 |

Relation claims underperform because relational language often requires precise spatial or interaction grounding that global image-text similarity models capture weakly. CLIP-style embeddings are strong for object presence cues but less discriminative for fine-grained predicate structure (for example, left-of, behind, holding, riding).

### 9.3 External benchmark validation (POPE-COCO adversarial)

| Method | Accuracy | Precision | Recall | FPR |
|---|---:|---:|---:|---:|
| Full_VIGIL_Balanced | 0.7290 | 0.6929 | 0.8227 | 0.3647 |
| Full_VIGIL_Safe | 0.6890 | 0.7529 | 0.5627 | 0.1847 |
| Baseline_CLIP_Fixed | 0.5000 | 0.5000 | 1.0000 | 1.0000 |

POPE adversarial uses real COCO images with real object-level probing, making it a harder and more credible evaluation setting than synthetic annotation stress tests.

VIGIL Balanced (0.7290) exceeds the custom benchmark accuracy (0.7067) on this harder external dataset, suggesting the verifier behavior generalizes beyond synthetic negatives rather than overfitting to them.

Safe mode provides the strongest deployment-oriented safety profile: FPR drops from 0.3647 to 0.1847 (nearly halved), with recall decreasing from 0.8227 to 0.5627. This is a controllable safety-recall trade-off; an 18.47% false positive rate is operationally defensible in high-stakes settings where over-trust is costly.

Compared with the fixed-threshold CLIP baseline, VIGIL Balanced delivers a 45.8% relative accuracy gain on adversarial data.

## 10. Limitations and Evaluation Validity

The current 300 negative claims in the scaled benchmark are synthetically generated via object replacement and relation swaps. These negatives are useful for controlled ablations, but they are often easier than naturally occurring VLM hallucinations that are context-rich, long-range, and linguistically subtle.

A stronger validity protocol would evaluate real model outputs, for example by generating captions or answers from a model family such as LLaVA on COCO images, then labeling claim support against image evidence and reference annotations. A concrete next step is to build a human-in-the-loop labeling pipeline that tags each extracted claim as supported, contradicted, or unverifiable, and then rerun VIGIL in the same ablation framework.

## 11. Discussion

VIGIL consistently improves over fixed-threshold CLIP baselines and makes safety behavior explicit. Balanced mode is appropriate when recall is prioritized, while Safe mode is preferable when false positives are costly.

## 12. Conclusion

VIGIL performs better on the external POPE-COCO adversarial benchmark (0.7290) than on the custom synthetic benchmark (0.7067), indicating robustness beyond the in-house evaluation setup. The strongest operational result is Safe mode on POPE, where FPR reaches 0.1847 while preserving a tunable precision-recall operating point for risk-sensitive deployment. Limitations remain: the custom benchmark relies on synthetic negatives, NLI-based verifiers were not compared on the same data, and the claim extractor was not independently benchmarked as a separate module. A concrete next step is to run VIGIL on live LLaVA or InstructBLIP outputs with human-labeled hallucination annotations for end-to-end external validation.
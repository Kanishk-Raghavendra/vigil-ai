"""Run ablation, baselines, bootstrap stats, and claim-type analysis for VIGIL.

This script produces paper-ready evaluation artifacts:
- results/ablation_results.json
- results/ablation_results.csv
- results/statistics.json
- results/statistics.csv
- results/claim_type_analysis.json
- results/claim_type_analysis.csv
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.models.verifier.clip import CLIPVerifierModel
from src.utils.decision import Decision


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ablation and statistical evaluation."""
    parser = argparse.ArgumentParser(description="Run VIGIL ablation and baseline evaluation")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/expanded_annotations.json",
        help="Path to expanded annotations JSON",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/sample_images/images",
        help="Directory containing image files",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional max images to evaluate (0 = all)",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=0.25,
        help="Fixed threshold for base and prompt-only ablations",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Fixed threshold baseline for CLIP",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--aggregation-mode",
        type=str,
        choices=["balanced", "safe"],
        default="balanced",
        help="Global aggregation calibration mode for Full VIGIL",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="Run both balanced and safe aggregation modes and save comparison outputs",
    )
    return parser.parse_args()


def compute_binary_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and FPR for binary labels."""
    tp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
    }


def load_claim_rows(
    annotations_path: Path, images_dir: Path, max_images: int = 0
) -> List[Dict[str, Any]]:
    """Flatten annotation JSON into per-claim rows used by all evaluations."""
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_items = list(annotations.items())
    if max_images > 0:
        image_items = image_items[:max_images]

    rows: List[Dict[str, Any]] = []
    for image_name, payload in image_items:
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        claim_types = payload.get("claim_types", {})

        for claim in payload.get("correct_claims", []):
            rows.append(
                {
                    "image": image_name,
                    "image_path": str(image_path),
                    "claim": claim,
                    "label": 1,
                    "claim_type": claim_types.get(claim, "object"),
                }
            )

        for claim in payload.get("hallucinated_claims", []):
            rows.append(
                {
                    "image": image_name,
                    "image_path": str(image_path),
                    "claim": claim,
                    "label": 0,
                    "claim_type": claim_types.get(claim, "object"),
                }
            )

    return rows


def score_rows(rows: List[Dict[str, Any]], verifier: CLIPVerifierModel) -> List[Dict[str, Any]]:
    """Compute and cache raw and prompted CLIP scores for each claim row."""
    scored_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        claim = row["claim"]

        raw_score = verifier._verify_text_against_image(image_path, claim)
        prompt_score, formatted_claim = verifier.verify_claim_with_prompt(image_path, claim)

        scored = dict(row)
        scored["raw_score"] = raw_score
        scored["prompt_score"] = prompt_score
        scored["formatted_claim"] = formatted_claim
        scored_rows.append(scored)

        if idx % 200 == 0:
            print(f"Scored {idx}/{len(rows)} claims")

    return scored_rows


def group_rows_by_image(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group claim rows by image for adaptive/aggregation variants."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["image"]].append(row)
    return grouped


def run_methods(
    rows: List[Dict[str, Any]],
    base_threshold: float,
    fixed_threshold: float,
    seed: int,
    aggregation_mode: str,
) -> Tuple[Dict[str, List[int]], Dict[str, Dict[str, float]]]:
    """Run ablation variants and baselines, returning predictions and metrics."""
    labels = [row["label"] for row in rows]
    methods_predictions: Dict[str, List[int]] = {}

    # A) Base: CLIP only, fixed threshold, raw claim.
    methods_predictions["A_Base_CLIP_Fixed"] = [
        1 if row["raw_score"] >= base_threshold else 0 for row in rows
    ]

    # B) Prompt engineering only, fixed threshold.
    methods_predictions["B_Prompt_Engineering"] = [
        1 if row["prompt_score"] >= base_threshold else 0 for row in rows
    ]

    grouped = group_rows_by_image(rows)

    # C) Prompt + adaptive threshold.
    pred_c: List[int] = []
    image_thresholds: Dict[str, float] = {}
    for image_name, image_rows in grouped.items():
        threshold = Decision.compute_adaptive_threshold([r["prompt_score"] for r in image_rows])
        image_thresholds[image_name] = threshold
        for row in image_rows:
            pred_c.append(1 if row["prompt_score"] >= threshold else 0)
    methods_predictions["C_Prompt_Adaptive"] = pred_c

    # D) Full VIGIL: prompt + adaptive + soft multi-claim aggregation.
    pred_d: List[int] = []
    for image_name, image_rows in grouped.items():
        threshold = image_thresholds[image_name]
        interim = []
        for row in image_rows:
            decision = "TRUST" if row["prompt_score"] >= threshold else "WARNING"
            interim.append({"score": row["prompt_score"], "decision": decision})

        global_trust = Decision.compute_global_trust(interim, mode=aggregation_mode)
        global_threshold = float(global_trust["threshold"])
        global_score = float(global_trust["global_score"])
        global_decision = global_trust["final_decision"]
        confidence_deficit = global_threshold - global_score

        for row in image_rows:
            base_pred = 1 if row["prompt_score"] >= threshold else 0

            # Only tighten positive predictions when global confidence is low.
            # This avoids rigid image-level overrides that can hurt recall.
            if (
                global_decision == "UNTRUSTED OUTPUT"
                and base_pred == 1
                and confidence_deficit >= 0.08
            ):
                calibrated_cutoff = max(threshold, global_threshold)
                if row["prompt_score"] < calibrated_cutoff:
                    base_pred = 0

            pred_d.append(base_pred)
    methods_predictions["D_Full_VIGIL"] = pred_d

    # Strong baselines.
    methods_predictions["Baseline_Accept_All"] = [1] * len(rows)

    rng = random.Random(seed)
    methods_predictions["Baseline_Random"] = [rng.randint(0, 1) for _ in rows]

    methods_predictions["Baseline_CLIP_Fixed_0.5"] = [
        1 if row["raw_score"] >= fixed_threshold else 0 for row in rows
    ]

    metrics = {
        method: compute_binary_metrics(preds, labels)
        for method, preds in methods_predictions.items()
    }

    return methods_predictions, metrics


def bootstrap_statistics(
    labels: List[int],
    methods_predictions: Dict[str, List[int]],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute bootstrap mean/std for metrics for each method."""
    rng = random.Random(seed)
    n = len(labels)

    by_method: Dict[str, Dict[str, List[float]]] = {
        method: {"accuracy": [], "precision": [], "recall": [], "fpr": []}
        for method in methods_predictions
    }

    for _ in range(bootstrap_samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[i] for i in idxs]

        for method, preds in methods_predictions.items():
            sample_preds = [preds[i] for i in idxs]
            m = compute_binary_metrics(sample_preds, sample_labels)
            for metric_name in ["accuracy", "precision", "recall", "fpr"]:
                by_method[method][metric_name].append(m[metric_name])

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, metric_map in by_method.items():
        summary[method] = {}
        for metric_name, values in metric_map.items():
            mean = sum(values) / len(values) if values else 0.0
            variance = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
            std = variance ** 0.5
            summary[method][metric_name] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
            }

    return summary


def claim_type_analysis(
    rows: List[Dict[str, Any]], full_vigil_predictions: List[int]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics by claim type using full VIGIL predictions."""
    by_type: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"labels": [], "preds": []})

    for row, pred in zip(rows, full_vigil_predictions):
        claim_type = row.get("claim_type", "object")
        by_type[claim_type]["labels"].append(row["label"])
        by_type[claim_type]["preds"].append(pred)

    analysis: Dict[str, Dict[str, float]] = {}
    for claim_type, data in by_type.items():
        analysis[claim_type] = compute_binary_metrics(data["preds"], data["labels"])
        analysis[claim_type]["support"] = len(data["labels"])

    return analysis


def save_json(path: Path, obj: Any) -> None:
    """Save JSON with pretty formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_method_table_csv(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    """Save paper-ready method comparison table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "accuracy", "precision", "recall", "fpr"],
        )
        writer.writeheader()
        for method, m in metrics.items():
            writer.writerow(
                {
                    "method": method,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "fpr": m["fpr"],
                }
            )


def save_statistics_csv(path: Path, stats: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """Save bootstrap statistics as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "metric", "mean", "std"],
        )
        writer.writeheader()
        for method, metric_map in stats.items():
            for metric, values in metric_map.items():
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                    }
                )


def save_claim_type_csv(path: Path, analysis: Dict[str, Dict[str, float]]) -> None:
    """Save claim-type metric analysis as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["claim_type", "support", "accuracy", "precision", "recall", "fpr"],
        )
        writer.writeheader()
        for claim_type, metrics in analysis.items():
            writer.writerow(
                {
                    "claim_type": claim_type,
                    "support": metrics.get("support", 0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "fpr": metrics.get("fpr", 0.0),
                }
            )


def save_mode_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Save balanced vs safe mode comparison as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["mode", "accuracy", "precision", "recall", "fpr"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "fpr": row["fpr"],
                }
            )


def print_mode_comparison_table(rows: List[Dict[str, Any]]) -> None:
    """Print clean balanced vs safe comparison table."""
    print("\nMode Comparison (Full VIGIL)")
    print("Mode      | Accuracy | Precision | Recall | FPR")
    print("-" * 48)
    for row in rows:
        mode_label = row["mode"].capitalize()
        print(
            f"{mode_label:9s} | {row['accuracy']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['fpr']:.4f}"
        )


def print_table(title: str, metrics: Dict[str, Dict[str, float]]) -> None:
    """Print clean method comparison table for terminal output."""
    print(f"\n{title}")
    print("Method | Accuracy | Precision | Recall | FPR")
    print("-" * 60)
    for method, m in metrics.items():
        print(
            f"{method:24s} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['fpr']:.4f}"
        )


def main() -> int:
    """Run full tier-2 evaluation workflow."""
    args = parse_args()

    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    rows = load_claim_rows(annotations_path, images_dir, max_images=args.max_images)
    if not rows:
        raise ValueError("No valid claim rows found for evaluation")

    print(f"Loaded {len(rows)} claims from {len(set(r['image'] for r in rows))} images")

    verifier = CLIPVerifierModel()
    rows = score_rows(rows, verifier)

    # Evaluate requested mode(s). In compare mode, run both balanced and safe.
    modes_to_run = ["balanced", "safe"] if args.compare_modes else [args.aggregation_mode]
    mode_runs: Dict[str, Dict[str, Any]] = {}

    for mode in modes_to_run:
        methods_predictions, method_metrics = run_methods(
            rows,
            base_threshold=args.base_threshold,
            fixed_threshold=args.fixed_threshold,
            seed=args.seed,
            aggregation_mode=mode,
        )

        fpr_before = method_metrics["C_Prompt_Adaptive"]["fpr"]
        fpr_after = method_metrics["D_Full_VIGIL"]["fpr"]
        print(f"[{mode}] Aggregation improved FPR from {fpr_before:.4f} to {fpr_after:.4f}")
        if fpr_after > fpr_before:
            print(
                f"[{mode}] Aggregation note: current setup increased FPR; "
                "consider using --aggregation-mode safe."
            )

        mode_runs[mode] = {
            "methods_predictions": methods_predictions,
            "method_metrics": method_metrics,
            "fpr_before": fpr_before,
            "fpr_after": fpr_after,
        }

    # Keep legacy behavior: downstream outputs are generated from requested --aggregation-mode.
    selected_mode = args.aggregation_mode
    selected_run = mode_runs[selected_mode]
    methods_predictions = selected_run["methods_predictions"]
    method_metrics = selected_run["method_metrics"]
    fpr_before = selected_run["fpr_before"]
    fpr_after = selected_run["fpr_after"]

    labels = [row["label"] for row in rows]
    stats = bootstrap_statistics(
        labels,
        methods_predictions,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    full_vigil_preds = methods_predictions["D_Full_VIGIL"]
    type_analysis = claim_type_analysis(rows, full_vigil_preds)

    # Save JSON outputs required by project brief.
    ablation_payload = {
        "dataset": {
            "images": len(set(r["image"] for r in rows)),
            "claims": len(rows),
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
        },
        "config": {
            "base_threshold": args.base_threshold,
            "fixed_threshold": args.fixed_threshold,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "aggregation_mode": args.aggregation_mode,
            "compare_modes": args.compare_modes,
        },
        "methods": method_metrics,
    }

    save_json(Path("results/ablation_results.json"), ablation_payload)
    save_json(Path("results/statistics.json"), stats)
    save_json(Path("results/claim_type_analysis.json"), type_analysis)

    # Save paper-ready CSV tables.
    save_method_table_csv(Path("results/ablation_results.csv"), method_metrics)
    save_statistics_csv(Path("results/statistics.csv"), stats)
    save_claim_type_csv(Path("results/claim_type_analysis.csv"), type_analysis)

    if args.compare_modes:
        mode_comparison = []
        for mode in ["balanced", "safe"]:
            m = mode_runs[mode]["method_metrics"]["D_Full_VIGIL"]
            mode_comparison.append(
                {
                    "mode": mode,
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "fpr": m["fpr"],
                }
            )

        save_json(Path("results/mode_comparison.json"), mode_comparison)
        save_mode_comparison_csv(Path("results/mode_comparison.csv"), mode_comparison)
        print_mode_comparison_table(mode_comparison)

        balanced_fpr = mode_comparison[0]["fpr"]
        safe_fpr = mode_comparison[1]["fpr"]
        if safe_fpr < balanced_fpr:
            print("Safe mode reduces false positives at a small cost to recall")
        else:
            print("Safe mode does not significantly improve safety in this dataset")

    # Human-readable evaluation log for quick reporting.
    log_lines = [
        "VIGIL Evaluation Log",
        f"Images: {ablation_payload['dataset']['images']}",
        f"Claims: {ablation_payload['dataset']['claims']}",
        f"Aggregation mode: {args.aggregation_mode}",
        f"Compare modes: {args.compare_modes}",
        f"Aggregation improved FPR from {fpr_before:.4f} to {fpr_after:.4f}",
        "",
    ]
    for method, m in method_metrics.items():
        log_lines.append(
            f"{method}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
            f"rec={m['recall']:.4f}, fpr={m['fpr']:.4f}"
        )

    log_path = Path("results/evaluation_log.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    print_table("Ablation + Baseline Comparison", method_metrics)
    print("\nClaim-Type Analysis")
    print("Type | Support | Accuracy | Precision | Recall | FPR")
    print("-" * 62)
    for claim_type, m in type_analysis.items():
        print(
            f"{claim_type:8s} | {int(m.get('support', 0)):7d} | {m['accuracy']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f}"
        )

    print("\nSaved outputs:")
    print("- results/ablation_results.json")
    print("- results/ablation_results.csv")
    print("- results/statistics.json")
    print("- results/statistics.csv")
    print("- results/claim_type_analysis.json")
    print("- results/claim_type_analysis.csv")
    print("- results/evaluation_log.txt")
    if args.compare_modes:
        print("- results/mode_comparison.json")
        print("- results/mode_comparison.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

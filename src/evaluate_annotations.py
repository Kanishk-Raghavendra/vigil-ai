"""Evaluate VIGIL verifier using annotated claims.

This script reads annotations in the format:
{
  "image.jpg": {
    "correct_claims": [...],
    "hallucinated_claims": [...]
  }
}

For each correct claim, ground truth is TRUST.
For each hallucinated claim, ground truth is WARNING.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.verifier.clip import CLIPVerifierModel
from src.utils.metrics import MetricsComputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VIGIL on annotated claims")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/sample_images/annotations.json",
        help="Path to annotations JSON",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/sample_images/images",
        help="Directory containing image files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="Decision threshold for TRUST/WARNING",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of images (0 = all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/metrics/benchmark_annotations.json",
        help="Path to save benchmark JSON",
    )
    parser.add_argument(
        "--sweep-json",
        type=str,
        default="results/threshold_sweep.json",
        help="Path to save threshold sweep JSON",
    )
    parser.add_argument(
        "--sweep-csv",
        type=str,
        default="results/threshold_sweep.csv",
        help="Path to save threshold sweep CSV",
    )
    parser.add_argument(
        "--failure-output",
        type=str,
        default="results/failure_cases.json",
        help="Path to save incorrect prediction analysis",
    )
    return parser.parse_args()


def threshold_grid(start: float = 0.1, stop: float = 0.9, step: float = 0.05) -> List[float]:
    """Build an inclusive threshold grid with stable floating-point rounding."""
    values: List[float] = []
    t = start
    while t <= stop + 1e-9:
        values.append(round(t, 2))
        t += step
    return values


def compute_binary_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute confusion-matrix-based metrics for binary labels.

    Label convention:
    - 1: correct claim (positive class)
    - 0: hallucinated claim (negative class)
    """
    tp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    tn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 0)
    fp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    hallucination_detection_rate = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "hallucination_detection_rate": round(hallucination_detection_rate, 4),
    }


def run_threshold_sweep(scores: List[float], labels: List[int]) -> List[Dict[str, float]]:
    """Evaluate metrics over thresholds from 0.1 to 0.9 with step 0.05."""
    results: List[Dict[str, float]] = []

    for threshold in threshold_grid(0.1, 0.9, 0.05):
        predictions = [1 if score >= threshold else 0 for score in scores]
        metrics = compute_binary_metrics(predictions, labels)
        results.append(
            {
                "threshold": threshold,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "fpr": metrics["fpr"],
                "hallucination_detection_rate": metrics["hallucination_detection_rate"],
            }
        )

    return results


def select_recommended_threshold(
    sweep_results: List[Dict[str, float]], min_recall: float = 0.2
) -> Optional[Dict[str, float]]:
    """Pick threshold with lowest FPR while enforcing recall > min_recall.

    Tie-breakers: higher accuracy, then higher precision.
    """
    eligible = [row for row in sweep_results if row["recall"] > min_recall]
    if not eligible:
        return None

    return min(
        eligible,
        key=lambda row: (row["fpr"], -row["accuracy"], -row["precision"]),
    )


def print_threshold_table(sweep_results: List[Dict[str, float]]) -> None:
    """Print clean terminal table for sweep metrics."""
    print("\nThreshold | Accuracy | Precision | Recall | FPR")
    print("-" * 49)
    for row in sweep_results:
        print(
            f"{row['threshold']:>8.2f} | "
            f"{row['accuracy']:>8.4f} | "
            f"{row['precision']:>9.4f} | "
            f"{row['recall']:>6.4f} | "
            f"{row['fpr']:>5.4f}"
        )


def save_threshold_sweep_json(path: Path, sweep_results: List[Dict[str, float]]) -> None:
    """Save threshold sweep results to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)


def save_threshold_sweep_csv(path: Path, sweep_results: List[Dict[str, float]]) -> None:
    """Save threshold sweep results to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "hallucination_detection_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sweep_results)


def evaluate_claims(
    items: List[Tuple[str, Dict[str, Any]]], images_dir: Path, threshold: float
) -> Tuple[
    List[Dict[str, Any]],
    List[str],
    List[str],
    List[float],
    List[int],
    List[Dict[str, Any]],
]:
    """Run verifier over annotated claims and collect rich outputs.

    Returns per-claim rows and both string/binary labels for downstream analysis.
    """
    verifier = CLIPVerifierModel()

    predictions: List[str] = []
    ground_truth: List[str] = []
    per_claim: List[Dict[str, Any]] = []
    scores: List[float] = []
    binary_labels: List[int] = []
    failure_cases: List[Dict[str, Any]] = []

    for image_name, payload in items:
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        for claim in payload.get("correct_claims", []):
            if hasattr(verifier, "verify_claim_with_prompt"):
                score, formatted_claim = verifier.verify_claim_with_prompt(image_path, claim)
            else:
                score = verifier.verify_claim(image_path, claim)
                formatted_claim = claim
            pred = "TRUST" if score >= threshold else "WARNING"

            predictions.append(pred)
            ground_truth.append("TRUST")
            scores.append(score)
            binary_labels.append(1)

            per_claim.append(
                {
                    "image": image_name,
                    "claim": claim,
                    "formatted_claim": formatted_claim,
                    "type": "correct_claim",
                    "score": round(score, 4),
                    "prediction": pred,
                    "ground_truth": "TRUST",
                }
            )

            if pred != "TRUST":
                failure_cases.append(
                    {
                        "image": image_name,
                        "claim": claim,
                        "score": round(score, 4),
                        "predicted_label": pred,
                        "actual_label": "TRUST",
                    }
                )

        for claim in payload.get("hallucinated_claims", []):
            if hasattr(verifier, "verify_claim_with_prompt"):
                score, formatted_claim = verifier.verify_claim_with_prompt(image_path, claim)
            else:
                score = verifier.verify_claim(image_path, claim)
                formatted_claim = claim
            pred = "TRUST" if score >= threshold else "WARNING"

            predictions.append(pred)
            ground_truth.append("WARNING")
            scores.append(score)
            binary_labels.append(0)

            per_claim.append(
                {
                    "image": image_name,
                    "claim": claim,
                    "formatted_claim": formatted_claim,
                    "type": "hallucinated_claim",
                    "score": round(score, 4),
                    "prediction": pred,
                    "ground_truth": "WARNING",
                }
            )

            if pred != "WARNING":
                failure_cases.append(
                    {
                        "image": image_name,
                        "claim": claim,
                        "score": round(score, 4),
                        "predicted_label": pred,
                        "actual_label": "WARNING",
                    }
                )

    return per_claim, predictions, ground_truth, scores, binary_labels, failure_cases


def main() -> int:
    args = parse_args()

    annotations_path = Path(args.annotations)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    sweep_json_path = Path(args.sweep_json)
    sweep_csv_path = Path(args.sweep_csv)
    failure_output_path = Path(args.failure_output)

    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    items = list(annotations.items())
    if args.limit > 0:
        items = items[: args.limit]

    (
        per_claim,
        predictions,
        ground_truth,
        scores,
        binary_labels,
        failure_cases,
    ) = evaluate_claims(
        items=items,
        images_dir=images_dir,
        threshold=args.threshold,
    )

    metrics = MetricsComputer.compute_all_metrics(predictions, ground_truth)

    summary = {
        "n_images_used": len({row["image"] for row in per_claim}),
        "n_claims": len(per_claim),
        "threshold": args.threshold,
        "metrics": metrics,
        "predictions": {
            "trust": sum(1 for p in predictions if p == "TRUST"),
            "warning": sum(1 for p in predictions if p == "WARNING"),
        },
        "ground_truth": {
            "trust": sum(1 for g in ground_truth if g == "TRUST"),
            "warning": sum(1 for g in ground_truth if g == "WARNING"),
        },
        "per_claim": per_claim,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sweep_results = run_threshold_sweep(scores=scores, labels=binary_labels)
    save_threshold_sweep_json(sweep_json_path, sweep_results)
    save_threshold_sweep_csv(sweep_csv_path, sweep_results)

    failure_output_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_output_path.open("w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2)

    recommended = select_recommended_threshold(sweep_results, min_recall=0.2)

    print("Benchmark complete")
    print(f"Images used: {summary['n_images_used']}")
    print(f"Claims evaluated: {summary['n_claims']}")
    print(f"Metrics: {json.dumps(metrics)}")
    print(f"Saved: {output_path}")
    print_threshold_table(sweep_results)
    print(f"Saved threshold sweep JSON: {sweep_json_path}")
    print(f"Saved threshold sweep CSV: {sweep_csv_path}")
    print(f"Saved failure cases: {failure_output_path}")
    print(f"Failure count: {len(failure_cases)}")

    if recommended is not None:
        print(f"Recommended threshold: {recommended['threshold']:.2f}")
    else:
        print("Recommended threshold: none (no threshold satisfied recall > 0.2)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

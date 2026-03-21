"""Run POPE-COCO evaluation with ablation-style method outputs.

Compares:
- Full_VIGIL (Balanced)
- Full_VIGIL (Safe)
- Baseline_CLIP_Fixed_0.5
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.models.verifier.clip import CLIPVerifierModel
from src.utils.decision import Decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VIGIL evaluation on POPE")
    parser.add_argument(
        "--pope-annotations",
        type=str,
        required=True,
        help="Path to POPE annotation file (json or jsonl)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory containing MS-COCO images",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/pope_results.json",
        help="Output JSON path (ablation-like methods format)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/pope_results.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def normalize_label(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "supported", "support"}:
        return 1
    if text in {"no", "n", "false", "0", "unsupported", "hallucinated"}:
        return 0
    raise ValueError(f"Unsupported POPE label: {value}")


def extract_claim_text(sample: Dict[str, Any]) -> str:
    for key in ["claim", "text", "question", "prompt", "query"]:
        if key in sample and str(sample[key]).strip():
            value = str(sample[key]).strip()
            if value.endswith("?"):
                value = value[:-1].strip()
            # Normalize common POPE question templates into claim text.
            value = value.replace("Is there ", "There is ")
            value = value.replace("is there ", "there is ")
            return value
    raise ValueError(f"Could not find claim text field in sample keys={list(sample.keys())}")


def extract_image_name(sample: Dict[str, Any]) -> str:
    for key in ["image", "image_id", "image_path", "filename", "img"]:
        if key in sample and str(sample[key]).strip():
            value = str(sample[key]).strip()
            if value.isdigit():
                return f"{int(value):012d}.jpg"
            return Path(value).name
    raise ValueError(f"Could not find image field in sample keys={list(sample.keys())}")


def image_name_candidates(image_name: str) -> List[str]:
    """Return possible local filename variants for a POPE image entry."""
    candidates = [Path(image_name).name]
    match = re.search(r"(\d{12})\.jpg$", image_name)
    if match:
        candidates.append(f"{match.group(1)}.jpg")
    return list(dict.fromkeys(candidates))


def load_pope_rows(path: Path, images_dir: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        try:
            obj = json.loads(text)
            rows = obj if isinstance(obj, list) else obj.get("data", [])
        except json.JSONDecodeError:
            # Some POPE files are JSONL but use .json suffix.
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    parsed: List[Dict[str, Any]] = []
    for row in rows:
        label_value = row.get("label", row.get("answer", row.get("gt_label")))
        if label_value is None:
            continue

        image_name = extract_image_name(row)
        image_path = None
        resolved_image_name = None
        for candidate in image_name_candidates(image_name):
            candidate_path = images_dir / candidate
            if candidate_path.exists():
                image_path = candidate_path
                resolved_image_name = candidate
                break

        if image_path is None:
            continue

        parsed.append(
            {
                "image": image_name,
                "image_path": str(image_path),
                "resolved_image": resolved_image_name,
                "claim": extract_claim_text(row),
                "label": normalize_label(label_value),
            }
        )

    return parsed


def compute_binary_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
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


def group_by_image(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["resolved_image"]].append(row)
    return grouped


def score_rows(rows: List[Dict[str, Any]], verifier: CLIPVerifierModel) -> List[Dict[str, Any]]:
    """Attach raw and prompt scores for each POPE probe row."""
    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        prompt_score, _formatted = verifier.verify_claim_with_prompt(row["image_path"], row["claim"])
        raw_score = verifier.verify_claim(row["image_path"], row["claim"])
        scored = dict(row)
        scored["prompt_score"] = float(prompt_score)
        scored["raw_score"] = float(raw_score)
        scored_rows.append(scored)
    return scored_rows


def full_vigil_predictions(rows: List[Dict[str, Any]], aggregation_mode: str) -> List[int]:
    """Run D_Full_VIGIL logic consistent with run_ablation.py."""
    grouped = group_by_image(rows)
    predictions: List[int] = []

    for _image_name, image_rows in grouped.items():
        adaptive_threshold = Decision.compute_adaptive_threshold(
            [row["prompt_score"] for row in image_rows]
        )
        interim = []
        for row in image_rows:
            decision = "TRUST" if row["prompt_score"] >= adaptive_threshold else "WARNING"
            interim.append({"score": row["prompt_score"], "decision": decision})

        global_trust = Decision.compute_global_trust(interim, mode=aggregation_mode)
        global_threshold = float(global_trust["threshold"])
        global_score = float(global_trust["global_score"])
        trusted_ratio = float(global_trust["trusted_ratio"])

        confidence_deficit = global_threshold - global_score

        mixedness = 1.0 - abs((2.0 * trusted_ratio) - 1.0)
        mode_strength = 1.0 if aggregation_mode == "balanced" else 1.35

        adjustment = 0.0
        if confidence_deficit > 0:
            bounded_deficit = min(0.2, confidence_deficit)
            adjustment += (
                0.22 * bounded_deficit * (0.4 + 0.6 * mixedness) * mode_strength
            )
            if trusted_ratio < 0.5:
                adjustment += 0.04 * (0.5 - trusted_ratio) * mode_strength
        elif confidence_deficit < -0.08 and trusted_ratio > 0.7:
            adjustment -= 0.03 * (trusted_ratio - 0.7)

        for row in image_rows:
            calibrated_threshold = adaptive_threshold + adjustment
            calibrated_threshold = max(0.2, min(0.9, calibrated_threshold))
            predictions.append(1 if row["prompt_score"] >= calibrated_threshold else 0)

    return predictions


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "accuracy", "precision", "recall", "fpr"],
        )
        writer.writeheader()
        for method, metrics in payload["methods"].items():
            writer.writerow(
                {
                    "method": method,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "fpr": metrics["fpr"],
                }
            )


def main() -> int:
    args = parse_args()

    annotations_path = Path(args.pope_annotations)
    images_dir = Path(args.images_dir)

    if not annotations_path.exists():
        raise FileNotFoundError(f"POPE annotations not found: {annotations_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    rows = load_pope_rows(annotations_path, images_dir)
    if not rows:
        raise ValueError("No valid POPE rows found after parsing and image filtering")

    print(f"Loaded {len(rows)} POPE probes from {len(set(r['resolved_image'] for r in rows))} images")
    verifier = CLIPVerifierModel()
    scored_rows = score_rows(rows, verifier)

    labels = [row["label"] for row in scored_rows]
    balanced_preds = full_vigil_predictions(scored_rows, aggregation_mode="balanced")
    safe_preds = full_vigil_predictions(scored_rows, aggregation_mode="safe")
    baseline_preds = [1 if row["raw_score"] >= 0.5 else 0 for row in scored_rows]

    methods = {
        "Full_VIGIL_Balanced": compute_binary_metrics(balanced_preds, labels),
        "Full_VIGIL_Safe": compute_binary_metrics(safe_preds, labels),
        "Baseline_CLIP_Fixed_0.5": compute_binary_metrics(baseline_preds, labels),
    }

    payload = {
        "dataset": {
            "split": "POPE-COCO adversarial",
            "images": len(set(r["resolved_image"] for r in rows)),
            "claims": len(rows),
            "positives": sum(labels),
            "negatives": len(labels) - sum(labels),
        },
        "config": {
            "fixed_threshold": 0.5,
        },
        "methods": methods,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    save_json(output_json, payload)
    save_csv(output_csv, payload)

    print("\nPOPE Adversarial Comparison")
    print("Method | Accuracy | Precision | Recall | FPR")
    print("-" * 62)
    for method, metrics in methods.items():
        print(
            f"{method:24} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['fpr']:.4f}"
        )
    print(f"\nSaved: {output_json}")
    print(f"Saved: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

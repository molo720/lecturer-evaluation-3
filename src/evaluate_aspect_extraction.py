#!/usr/bin/env python3
"""Evaluate aspect extraction and joint aspect-sentiment pairs against human labels.

Input must be independently annotated. This script intentionally does not
invent labels for the existing synthetic training corpus. See
docs/ASPECT_ANNOTATION_GUIDE.md and Data/aspect_annotation_template.csv.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from core.lexicons import ASPECTS
from core.ml_engine import load_all_models, run_analysis

SENTIMENTS = ("positive", "neutral", "negative")
REQUIRED_COLUMNS = {"comment_id", "comment", "gold_aspects", "gold_pairs"}


def parse_json_list(value, field: str, row_id: str):
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} for comment_id={row_id} must be valid JSON: {exc}") from exc
    else:
        raise ValueError(f"{field} is empty for comment_id={row_id}")
    if not isinstance(parsed, list):
        raise ValueError(f"{field} for comment_id={row_id} must be a JSON list")
    return parsed


def safe_prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def evaluate_annotations(
    records: Iterable[dict],
    analyser: Callable[[str, int], dict] = run_analysis,
) -> dict:
    """Return metrics for aspect extraction, conditioned polarity, and joint pairs."""
    rows = list(records)
    if not rows:
        raise ValueError("No annotated rows supplied. Annotate the template before evaluation.")

    overall_aspect = Counter(tp=0, fp=0, fn=0)
    overall_pair = Counter(tp=0, fp=0, fn=0)
    per_aspect = {aspect: Counter(tp=0, fp=0, fn=0, gold_support=0, predicted_support=0)
                  for aspect in ASPECTS}
    conditioned_correct = 0
    conditioned_total = 0
    sentiment_cm = [[0 for _ in SENTIMENTS] for _ in SENTIMENTS]
    unknown_aspects = Counter()

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("comment_id") or index)
        comment = str(row.get("comment") or "").strip()
        if not comment:
            raise ValueError(f"comment is empty for comment_id={row_id}")

        gold_aspect_list = parse_json_list(row.get("gold_aspects"), "gold_aspects", row_id)
        gold_pair_list = parse_json_list(row.get("gold_pairs"), "gold_pairs", row_id)

        gold_aspect_set = set()
        for name in gold_aspect_list:
            if not isinstance(name, str) or name not in ASPECTS:
                unknown_aspects[str(name)] += 1
                continue
            gold_aspect_set.add(name)

        gold_pairs = set()
        gold_sentiment_by_aspect = {}
        for pair in gold_pair_list:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    f"Each gold_pairs item must be [aspect, sentiment] for comment_id={row_id}"
                )
            aspect, sentiment = pair
            if aspect not in ASPECTS:
                unknown_aspects[str(aspect)] += 1
                continue
            if sentiment not in SENTIMENTS:
                raise ValueError(
                    f"Invalid sentiment {sentiment!r} for comment_id={row_id}; "
                    f"expected one of {SENTIMENTS}"
                )
            gold_aspect_set.add(aspect)
            gold_pairs.add((aspect, sentiment))
            gold_sentiment_by_aspect[aspect] = sentiment

        result = analyser(comment, 3)
        detected_items = result.get("aspects") or []
        predicted_by_aspect = {}
        for item in detected_items:
            aspect = item.get("aspect")
            sentiment = item.get("sentiment")
            if aspect not in ASPECTS:
                unknown_aspects[str(aspect)] += 1
                continue
            if sentiment not in SENTIMENTS:
                continue
            predicted_by_aspect[aspect] = sentiment
        predicted_aspect_set = set(predicted_by_aspect)
        predicted_pairs = {(aspect, sentiment) for aspect, sentiment in predicted_by_aspect.items()}

        aspect_tp = gold_aspect_set & predicted_aspect_set
        aspect_fp = predicted_aspect_set - gold_aspect_set
        aspect_fn = gold_aspect_set - predicted_aspect_set
        overall_aspect["tp"] += len(aspect_tp)
        overall_aspect["fp"] += len(aspect_fp)
        overall_aspect["fn"] += len(aspect_fn)
        for aspect in ASPECTS:
            if aspect in gold_aspect_set:
                per_aspect[aspect]["gold_support"] += 1
            if aspect in predicted_aspect_set:
                per_aspect[aspect]["predicted_support"] += 1
            if aspect in aspect_tp:
                per_aspect[aspect]["tp"] += 1
            if aspect in aspect_fp:
                per_aspect[aspect]["fp"] += 1
            if aspect in aspect_fn:
                per_aspect[aspect]["fn"] += 1

        pair_tp = gold_pairs & predicted_pairs
        pair_fp = predicted_pairs - gold_pairs
        pair_fn = gold_pairs - predicted_pairs
        overall_pair["tp"] += len(pair_tp)
        overall_pair["fp"] += len(pair_fp)
        overall_pair["fn"] += len(pair_fn)

        # Conditional polarity metrics are computed only when the aspect was
        # correctly extracted. This is conditional, not end-to-end performance.
        for aspect in gold_aspect_set & predicted_aspect_set:
            if aspect not in gold_sentiment_by_aspect:
                continue
            gold_sentiment = gold_sentiment_by_aspect[aspect]
            predicted_sentiment = predicted_by_aspect[aspect]
            gold_i = SENTIMENTS.index(gold_sentiment)
            pred_i = SENTIMENTS.index(predicted_sentiment)
            sentiment_cm[gold_i][pred_i] += 1
            conditioned_total += 1
            conditioned_correct += int(gold_sentiment == predicted_sentiment)

    if overall_aspect["tp"] + overall_aspect["fn"] == 0:
        raise ValueError("No valid gold aspects found; extraction recall/F1 cannot be estimated.")

    per_aspect_results = {}
    for aspect, counts in per_aspect.items():
        per_aspect_results[aspect] = {
            **safe_prf(counts["tp"], counts["fp"], counts["fn"]),
            "gold_support": int(counts["gold_support"]),
            "predicted_support": int(counts["predicted_support"]),
        }

    return {
        "n_annotated_comments": len(rows),
        "aspect_identification_micro": safe_prf(
            overall_aspect["tp"], overall_aspect["fp"], overall_aspect["fn"]
        ),
        "aspect_identification_by_aspect": per_aspect_results,
        "end_to_end_aspect_sentiment_pair_micro": safe_prf(
            overall_pair["tp"], overall_pair["fp"], overall_pair["fn"]
        ),
        "sentiment_given_correctly_extracted_aspect": {
            "accuracy": round(conditioned_correct / conditioned_total, 4) if conditioned_total else None,
            "support": int(conditioned_total),
            "confusion_matrix_actual_rows_predicted_columns": sentiment_cm,
            "class_order": list(SENTIMENTS),
            "interpretation": (
                "Conditional polarity accuracy only for aspects that the system extracted correctly. "
                "It must be read alongside the end-to-end pair score, because missed aspects are excluded."
            ),
        },
        "canonical_aspect_names": list(ASPECTS),
        "unknown_aspect_labels": dict(unknown_aspects),
        "evidence_boundary": (
            "Valid only for the supplied independently annotated comments. Do not apply these values "
            "to the synthetic training set unless the annotations are independent of the generator."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="Data/aspect_annotation_template.csv",
        help="CSV with independent gold aspect and aspect-sentiment annotations",
    )
    parser.add_argument(
        "--output",
        default="Data/aspect_extraction_metrics.json",
        help="Path for the JSON metrics output",
    )
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {source}. Copy Data/aspect_annotation_template.csv, "
            "annotate comments independently, and pass the completed file with --input."
        )
    frame = pd.read_csv(source, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing required columns: {sorted(missing)}")
    frame = frame[frame["comment"].astype(str).str.strip().ne("")]
    if frame.empty:
        raise ValueError(
            "The annotation CSV contains no completed comments. No extraction metrics were calculated."
        )

    # Match application inference as closely as possible by loading the committed
    # model artefacts, if present. Keep the benchmark independent of database data.
    load_all_models()
    results = evaluate_annotations(frame.to_dict(orient="records"))
    results["input_file"] = str(source)
    results["annotation_status"] = "human annotations supplied by the researcher; verify provenance before reporting"
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved evaluation metrics to {destination}")


if __name__ == "__main__":
    main()

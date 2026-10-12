#!/usr/bin/env python3
"""Reproducible, phrase-template-family holdout benchmark.

This is a controlled synthetic benchmark. Each authored phrase template is
assigned wholly to the train or test bank before comments are generated.
Results must not be described as validation on authentic institutional text.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_models import aspect_templates  # noqa: E402
from core.lexicons import ASPECTS  # noqa: E402
from core.ml_engine import run_analysis  # noqa: E402
from src.preprocessing import preprocess_for_vectorizer  # noqa: E402

SEED = 42
TRAIN_DOCUMENTS = 8000
TEST_DOCUMENTS = 2000
LABELS = ["positive", "neutral", "negative"]
CONNECTORS = [
    " but ",
    "; ",
    ". Also, ",
    ", while ",
    ". However, ",
    " and ",
]
CLASS_WEIGHTS = [0.42, 0.20, 0.38]  # order positive, neutral, negative


def split_template_banks(seed: int = SEED, test_fraction: float = 0.25):
    """Hold out whole exact authored templates within every aspect/polarity bank."""
    rng = random.Random(seed)
    train_bank, test_bank = {}, {}
    all_train, all_test = [], []
    for aspect in ASPECTS:
        if aspect not in aspect_templates:
            raise ValueError(f"Aspect absent from generator banks: {aspect}")
        for polarity in LABELS:
            phrases = list(dict.fromkeys(aspect_templates[aspect][polarity]))
            if len(phrases) < 2:
                raise ValueError(f"Need >=2 templates for {aspect}/{polarity}")
            rng.shuffle(phrases)
            n_test = max(1, min(len(phrases) - 1, round(len(phrases) * test_fraction)))
            test_phrases = phrases[:n_test]
            train_phrases = phrases[n_test:]
            train_bank[(aspect, polarity)] = train_phrases
            test_bank[(aspect, polarity)] = test_phrases
            all_train.extend(train_phrases)
            all_test.extend(test_phrases)

    train_set = {normalise_phrase(x) for x in all_train}
    test_set = {normalise_phrase(x) for x in all_test}
    overlap = sorted(train_set & test_set)
    if overlap:
        raise AssertionError(f"Exact train/test phrase overlap detected: {overlap[:5]}")
    return train_bank, test_bank, len(train_set), len(test_set)


def normalise_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def document_label(polarities: list[str]) -> str:
    counts = Counter(polarities)
    if counts["positive"] > counts["negative"]:
        return "positive"
    if counts["negative"] > counts["positive"]:
        return "negative"
    return "neutral"


def make_documents(bank, n: int, seed: int, side: str):
    rng = random.Random(seed)
    records = []
    for idx in range(n):
        n_aspects = rng.choices([1, 2, 3], weights=[0.25, 0.50, 0.25], k=1)[0]
        aspects = rng.sample(ASPECTS, n_aspects)
        clauses, gold = [], []
        for aspect in aspects:
            polarity = rng.choices(LABELS, weights=CLASS_WEIGHTS, k=1)[0]
            choices = bank.get((aspect, polarity), [])
            if not choices:
                raise ValueError(f"No {side} template for {aspect}/{polarity}")
            phrase = rng.choice(choices)
            clauses.append(phrase)
            gold.append((aspect, polarity))
        text = clauses[0]
        for clause in clauses[1:]:
            text += rng.choice(CONNECTORS) + clause
        records.append({
            "id": f"{side}-{idx + 1}",
            "comment": text,
            "labels": gold,
            "document_sentiment": document_label([p for _, p in gold]),
        })
    return records


def class_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )[0]), 4),
        "macro_recall": round(float(precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )[1]), 4),
        "macro_f1": round(float(precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )[2]), 4),
        "weighted_f1": round(float(precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
        )[2]), 4),
        "classes": LABELS,
        "support_by_class": {label: int(report[label]["support"]) for label in LABELS},
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate_aspect_extraction(test_records):
    tp = fp = fn = 0
    per_aspect = {a: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for a in ASPECTS}
    joint_tp = joint_fp = joint_fn = 0
    per_comment_errors = []

    for rec in test_records:
        gold_aspects = {a for a, _ in rec["labels"]}
        gold_pairs = set(rec["labels"])
        output = run_analysis(rec["comment"], 3)
        predicted_aspects = {item.get("aspect") for item in output.get("aspects", [])}
        predicted_pairs = {
            (item.get("aspect"), item.get("sentiment"))
            for item in output.get("aspects", [])
        }

        row_tp = gold_aspects & predicted_aspects
        row_fp = predicted_aspects - gold_aspects
        row_fn = gold_aspects - predicted_aspects
        tp += len(row_tp)
        fp += len(row_fp)
        fn += len(row_fn)
        for aspect in ASPECTS:
            gold_has, pred_has = aspect in gold_aspects, aspect in predicted_aspects
            per_aspect[aspect]["support"] += int(gold_has)
            if gold_has and pred_has:
                per_aspect[aspect]["tp"] += 1
            elif pred_has and not gold_has:
                per_aspect[aspect]["fp"] += 1
            elif gold_has and not pred_has:
                per_aspect[aspect]["fn"] += 1

        joint_tp += len(gold_pairs & predicted_pairs)
        joint_fp += len(predicted_pairs - gold_pairs)
        joint_fn += len(gold_pairs - predicted_pairs)
        if row_fp or row_fn or (gold_pairs != predicted_pairs):
            if len(per_comment_errors) < 30:
                per_comment_errors.append({
                    "input": rec["comment"],
                    "gold_aspect_sentiment_pairs": sorted([list(p) for p in gold_pairs]),
                    "predicted_aspect_sentiment_pairs": sorted([list(p) for p in predicted_pairs]),
                    "unmatched_aspects": sorted(list(row_fn)),
                    "false_aspects": sorted(list(row_fp)),
                })

    return {
        "task_definition": "Exact aspect-name identification; each comment's generator labels are treated as gold. Micro counts aggregate across comments.",
        "micro": prf(tp, fp, fn),
        "per_aspect": {
            a: {
                **prf(per_aspect[a]["tp"], per_aspect[a]["fp"], per_aspect[a]["fn"]),
                "gold_support_comments": per_aspect[a]["support"],
            }
            for a in ASPECTS
        },
        "end_to_end_aspect_sentiment_pair_micro": prf(joint_tp, joint_fp, joint_fn),
        "error_examples_first_30": per_comment_errors,
        "n_test_comments": len(test_records),
        "n_gold_aspect_mentions": sum(len(x["labels"]) for x in test_records),
        "limitation": "The gold aspects and polarities are generator-assigned, not independently human-annotated. This evaluates robustness to held-out authored phrases, not generalisation to authentic student feedback.",
    }


def evaluate_known_aspect_sentiment(train_bank, test_bank):
    results = {}
    all_y = []
    svm_all = []
    nb_all = []
    for aspect in ASPECTS:
        x_train, y_train = [], []
        x_test, y_test = [], []
        for polarity in LABELS:
            for phrase in train_bank[(aspect, polarity)]:
                x_train.append(preprocess_for_vectorizer(phrase))
                y_train.append(polarity)
            for phrase in test_bank[(aspect, polarity)]:
                x_test.append(preprocess_for_vectorizer(phrase))
                y_test.append(polarity)

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=30000
        )
        X_tr = vectorizer.fit_transform(x_train)
        X_te = vectorizer.transform(x_test)
        svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
        nb = MultinomialNB(alpha=0.1)
        svm.fit(X_tr, y_train)
        nb.fit(X_tr, y_train)
        svm_pred = svm.predict(X_te)
        nb_pred = nb.predict(X_te)
        results[aspect] = {
            "training_templates": len(x_train),
            "heldout_templates": len(x_test),
            "svm": class_metrics(y_test, svm_pred),
            "naive_bayes": class_metrics(y_test, nb_pred),
        }
        all_y.extend(y_test)
        svm_all.extend(svm_pred)
        nb_all.extend(nb_pred)
    return {
        "task_definition": "Aspect identity is supplied as known; held-out phrase templates are classified for sentiment. This is separate from discovering the aspect.",
        "per_aspect": results,
        "pooled_svm": class_metrics(all_y, svm_all),
        "pooled_naive_bayes": class_metrics(all_y, nb_all),
        "limitation": "The phrase banks are small and generator-authored. These scores are a template-family diagnostic, not evidence of real-world validity.",
    }


def main():
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    train_bank, test_bank, n_train_templates, n_test_templates = split_template_banks()
    train_records = make_documents(train_bank, TRAIN_DOCUMENTS, SEED + 1, "train")
    test_records = make_documents(test_bank, TEST_DOCUMENTS, SEED + 2, "test")

    x_train = [preprocess_for_vectorizer(r["comment"]) for r in train_records]
    y_train = [r["document_sentiment"] for r in train_records]
    x_test = [preprocess_for_vectorizer(r["comment"]) for r in test_records]
    y_test = [r["document_sentiment"] for r in test_records]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000
    )
    X_train = vectorizer.fit_transform(x_train)
    X_test = vectorizer.transform(x_test)

    doc_svm = LinearSVC(C=1.0, max_iter=5000, random_state=SEED)
    doc_nb = MultinomialNB(alpha=0.1)
    doc_svm.fit(X_train, y_train)
    doc_nb.fit(X_train, y_train)
    svm_pred, nb_pred = doc_svm.predict(X_test), doc_nb.predict(X_test)

    train_phrases = {
        normalise_phrase(p)
        for bank in train_bank.values()
        for p in bank
    }
    test_phrases = {
        normalise_phrase(p)
        for bank in test_bank.values()
        for p in bank
    }
    overlap = sorted(train_phrases & test_phrases)
    if overlap:
        raise AssertionError(f"Template leakage: {len(overlap)} exact phrase(s) overlap")

    doc_metrics = {
        "linear_svm": class_metrics(y_test, svm_pred),
        "multinomial_naive_bayes": class_metrics(y_test, nb_pred),
        "baseline_most_frequent_accuracy": round(float(max(Counter(y_train).values()) / len(y_train)), 4),
        "test_label_distribution": dict(Counter(y_test)),
    }
    results = {
        "title": "Template-family holdout benchmark",
        "status": "executed",
        "run_utc_note": "Timestamp omitted to keep a seeded run's JSON stable; CI records the run date.",
        "seed": SEED,
        "template_holdout": {
            "definition": "Each authored phrase template in every aspect/polarity bank is assigned entirely to either train or test before documents are generated; all clauses from a document are kept in that document's partition.",
            "test_fraction_per_aspect_polarity_bank": 0.25,
            "train_unique_templates": n_train_templates,
            "test_unique_templates": n_test_templates,
            "exact_template_overlap": len(overlap),
            "exact_overlap_check": "PASS" if not overlap else "FAIL",
            "train_documents": len(train_records),
            "test_documents": len(test_records),
        },
        "document_level_classification": doc_metrics,
        "aspect_extraction": evaluate_aspect_extraction(test_records),
        "aspect_sentiment_given_known_aspect": evaluate_known_aspect_sentiment(train_bank, test_bank),
        "methodology_boundaries": [
            "This benchmark is a newly executed controlled synthetic diagnostic, not a test on real institutional student feedback.",
            "Template holdout reduces exact phrase leakage but does not remove shared topic vocabulary or prove domain transfer.",
            "Aspect extraction metrics compare the deployed lexicon-guided aspect matcher against generator-assigned aspect labels.",
            "Sentiment-given-known-aspect assumes the correct aspect is already supplied and therefore does not measure aspect discovery.",
            "End-to-end aspect-sentiment pair metrics require both the correct aspect and the correct polarity.",
            "The dataset labels are synthetic/generator-assigned for this benchmark; no human annotation or inter-annotator agreement is claimed.",
        ],
        "runtime_seconds": None,
    }
    results["runtime_seconds"] = round(time.perf_counter() - started, 3)

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "template_family_holdout_metrics.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path.relative_to(ROOT)),
        "status": results["status"],
        "runtime_seconds": results["runtime_seconds"],
        "template_holdout": results["template_holdout"],
        "document_level_classification": results["document_level_classification"],
        "aspect_extraction_micro": results["aspect_extraction"]["micro"],
        "aspect_sentiment_pair_micro": results["aspect_extraction"]["end_to_end_aspect_sentiment_pair_micro"],
        "aspect_sentiment_given_known_aspect_pooled_svm": results["aspect_sentiment_given_known_aspect"]["pooled_svm"],
        "aspect_sentiment_given_known_aspect_pooled_naive_bayes": results["aspect_sentiment_given_known_aspect"]["pooled_naive_bayes"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

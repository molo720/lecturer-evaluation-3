"""Template-family holdout evaluation for the synthetic phrase-bank corpus.

This is deliberately a separate experiment from the existing random 60/20/20
benchmark. Test comments include at least one held-out phrase-bank item; every
comment containing a held-out item is excluded from train/validation. Results
measure generalisation to unseen phrase items within the synthetic generator,
not transfer to authentic student feedback or aspect-extraction accuracy.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocessing import preprocess_for_vectorizer
from src.train_models import generate_synthetic_dataset

SEED = 42
LABELS = ["positive", "neutral", "negative"]
TFIDF_ARGS = {
    "ngram_range": (1, 2),
    "sublinear_tf": True,
    "min_df": 2,
    "max_features": 50_000,
}


def parse_family_ids(value) -> list[str]:
    """Parse a generated JSON list of stable phrase-bank item IDs."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    raise ValueError(f"Invalid template_family_ids value: {value!r}")


def split_by_template_families(
    data: pd.DataFrame,
    test_family_fraction: float = 0.25,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Hold out phrase items within each aspect/polarity bank, avoiding row leakage.

    A row with even one held-out phrase-family ID is assigned to the test set;
    such rows are excluded entirely from the training pool, including their
    otherwise non-held-out clauses. The returned training pool contains no
    held-out IDs. The test set contains at least one held-out ID per row.
    """
    if not 0 < test_family_fraction < 1:
        raise ValueError("test_family_fraction must be between 0 and 1")

    frame = data.copy()
    if "template_family_ids" not in frame:
        raise ValueError("Dataset needs a template_family_ids column")
    frame["_family_ids"] = frame["template_family_ids"].apply(parse_family_ids)

    banks: dict[str, list[str]] = defaultdict(list)
    for family_id in sorted({fid for ids in frame["_family_ids"] for fid in ids}):
        parts = family_id.rsplit("::", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected family ID format: {family_id}")
        bank = "::".join(parts[:2])  # aspect + polarity
        banks[bank].append(family_id)

    rng = random.Random(seed)
    held_out: set[str] = set()
    for bank, family_ids in sorted(banks.items()):
        if len(family_ids) < 2:
            continue
        n_hold = min(len(family_ids) - 1, max(1, math.ceil(len(family_ids) * test_family_fraction)))
        held_out.update(rng.sample(family_ids, n_hold))

    if not held_out:
        raise ValueError("No phrase-bank families could be held out")

    test_mask = frame["_family_ids"].apply(lambda ids: bool(set(ids) & held_out))
    train_pool = frame.loc[~test_mask].drop(columns=["_family_ids"]).copy()
    test_set = frame.loc[test_mask].drop(columns=["_family_ids"]).copy()

    train_ids = {fid for value in train_pool["template_family_ids"] for fid in parse_family_ids(value)}
    test_ids = {fid for value in test_set["template_family_ids"] for fid in parse_family_ids(value)}
    if train_ids & held_out:
        raise AssertionError("Leakage: held-out family occurs in training pool")
    if not test_set.empty and not all(set(parse_family_ids(v)) & held_out for v in test_set["template_family_ids"]):
        raise AssertionError("A test row does not contain a held-out family")
    if not held_out <= test_ids:
        raise AssertionError("Some declared held-out families are absent from the test set")
    if train_pool.empty or test_set.empty:
        raise ValueError("Family split produced an empty train pool or test set")

    train_pool.attrs["held_out_families"] = sorted(held_out)
    test_set.attrs["held_out_families"] = sorted(held_out)
    return train_pool, test_set, held_out


def _macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def _metrics(y_true, y_pred) -> dict:
    per_class = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(_macro_f1(y_true, y_pred), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
        "classes_in_matrix_order": LABELS,
        "per_class": {
            label: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def run_evaluation(num_samples: int = 1600, test_family_fraction: float = 0.25) -> dict:
    """Generate a reproducible synthetic corpus and evaluate on held-out phrase IDs."""
    random.seed(SEED)
    np.random.seed(SEED)
    data = generate_synthetic_dataset(num_samples=num_samples)
    train_pool, test_set, held_out = split_by_template_families(
        data, test_family_fraction=test_family_fraction, seed=SEED
    )

    train_idx, val_idx = train_test_split(
        train_pool.index.to_numpy(),
        test_size=0.2,
        random_state=SEED,
        stratify=train_pool["document_sentiment"],
    )
    train = train_pool.loc[train_idx]
    validation = train_pool.loc[val_idx]
    test = test_set

    # Tuning vocabulary is fitted on the inner training fold only.
    train_text = train["comment"].map(preprocess_for_vectorizer)
    val_text = validation["comment"].map(preprocess_for_vectorizer)
    y_train = train["document_sentiment"].to_numpy()
    y_val = validation["document_sentiment"].to_numpy()

    tuning_vectorizer = TfidfVectorizer(**TFIDF_ARGS)
    x_train = tuning_vectorizer.fit_transform(train_text)
    x_val = tuning_vectorizer.transform(val_text)

    best_c, best_svm_f1 = None, -1.0
    for c_value in (0.5, 1.0, 2.0):
        model = LinearSVC(C=c_value, max_iter=5000, random_state=SEED)
        model.fit(x_train, y_train)
        score = _macro_f1(y_val, model.predict(x_val))
        if score > best_svm_f1:
            best_c, best_svm_f1 = c_value, score

    best_alpha, best_nb_f1 = None, -1.0
    for alpha in (0.1, 0.5, 1.0):
        model = MultinomialNB(alpha=alpha)
        model.fit(x_train, y_train)
        score = _macro_f1(y_val, model.predict(x_val))
        if score > best_nb_f1:
            best_alpha, best_nb_f1 = alpha, score

    # Refit vocabulary and selected models on the complete non-held-out pool.
    pool_text = train_pool["comment"].map(preprocess_for_vectorizer)
    test_text = test["comment"].map(preprocess_for_vectorizer)
    final_vectorizer = TfidfVectorizer(**TFIDF_ARGS)
    x_pool = final_vectorizer.fit_transform(pool_text)
    x_test = final_vectorizer.transform(test_text)
    y_pool = train_pool["document_sentiment"].to_numpy()
    y_test = test["document_sentiment"].to_numpy()

    svm = LinearSVC(C=best_c, max_iter=5000, random_state=SEED).fit(x_pool, y_pool)
    nb = MultinomialNB(alpha=best_alpha).fit(x_pool, y_pool)
    svm_pred = svm.predict(x_test)
    nb_pred = nb.predict(x_test)

    train_families = {
        fid for value in train_pool["template_family_ids"] for fid in parse_family_ids(value)
    }
    test_family_hits = sum(
        bool(set(parse_family_ids(value)) & held_out) for value in test["template_family_ids"]
    )
    return {
        "experiment": "synthetic_template_family_holdout",
        "seed": SEED,
        "synthetic_records_generated": int(len(data)),
        "test_family_fraction_per_aspect_polarity_bank": test_family_fraction,
        "held_out_family_count": int(len(held_out)),
        "held_out_family_ids": sorted(held_out),
        "train_pool_records": int(len(train_pool)),
        "inner_train_records": int(len(train)),
        "validation_records": int(len(validation)),
        "held_out_test_records": int(len(test)),
        "test_rows_with_at_least_one_held_out_family": int(test_family_hits),
        "training_and_test_rows_are_disjoint": bool(set(train_pool.index).isdisjoint(set(test.index))),
        "held_out_family_ids_absent_from_training": bool(not (train_families & held_out)),
        "model_selection": {
            "validation_metric": "Macro-F1",
            "selected_svm_C": best_c,
            "validation_svm_macro_f1": round(best_svm_f1, 4),
            "selected_nb_alpha": best_alpha,
            "validation_nb_macro_f1": round(best_nb_f1, 4),
            "tfidf": {
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "min_df": 2,
                "max_features": 50000,
                "vocabulary_fit": "training pool only; never fitted on held-out test comments",
            },
        },
        "svm": _metrics(y_test, svm_pred),
        "naive_bayes": _metrics(y_test, nb_pred),
        "interpretation_boundary": (
            "Synthetic phrase-family holdout only. It does not validate aspect-extraction "
            "precision/recall/F1, transfer to authentic student feedback, or user usefulness. "
            "A held-out comment may also contain non-held-out phrase items, but that entire "
            "comment is excluded from training."
        ),
    }


def main() -> None:
    results = run_evaluation()
    out = Path("Data/template_family_holdout_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

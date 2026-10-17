#!/usr/bin/env python3
"""Compare record-level and normalized-text-grouped training/evaluation splits.

Run: python src/audit_split_protocols.py
Imports data-building/preprocessing helpers from src/train_models.py, but does not
modify train_models.py or write generated data/models to disk.
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_models import (  # noqa: E402
    generate_synthetic_dataset,
    load_main_dataset,
    load_slang_dataset,
    preprocess_for_vectorizer,
    resolve_data_directory,
    resolve_dataset_path,
    resolve_main_csv_path,
    stratified_train_val_test,
)

RANDOM_STATE = 42
LABEL_ORDER = ("positive", "neutral", "negative")
SOURCE_ORDER = ("synthetic", "nigerian", "nigerian_slang")
SOURCE_DISPLAY = {"synthetic": "TEMPLATE", "nigerian": "PRIMARY", "nigerian_slang": "SLANG"}


def as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def lower_text(value):
    # Exact-text audit key requested: lowercase only.
    return as_text(value).lower()


def normalized_text(value):
    # Group key: Unicode normalization, case folding, trimmed/collapsed whitespace.
    value = unicodedata.normalize("NFKC", as_text(value))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def conflicting_keys(frame):
    labels_by_text = defaultdict(set)
    for text, label in zip(frame["comment"], frame["document_sentiment"]):
        key = lower_text(text)
        if key:
            labels_by_text[key].add(str(label).lower())
    return {key for key, labels in labels_by_text.items() if len(labels) > 1}


def summary_lines(frame, title):
    keys = frame["comment"].map(lower_text)
    nonempty = keys[keys != ""]
    counts = nonempty.value_counts()
    conflicts = conflicting_keys(frame)
    conflict_rows = int(keys.isin(conflicts).sum())
    return [
        f"{title}: records={len(frame)}",
        f"  unique_nonempty_lowercased_texts={nonempty.nunique()}",
        f"  exact_duplicate_rows_beyond_first={len(nonempty) - nonempty.nunique()}",
        f"  lowercased_texts_repeated_at_least_once={int((counts > 1).sum())}",
        f"  conflicting_label_texts={len(conflicts)}",
        f"  rows_belonging_to_conflicting_label_texts={conflict_rows}",
    ]


def random_record_split(labels):
    # Use the same two-stage helper and fallback behavior as train_models.py.
    indices = np.arange(len(labels))
    train, val, test, *_ = stratified_train_val_test(
        indices, labels, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )
    return {"TRAIN": np.asarray(train, dtype=int),
            "VALIDATION": np.asarray(val, dtype=int),
            "TEST": np.asarray(test, dtype=int)}


def grouped_split(labels, groups):
    # Split groups in two stages. GroupShuffleSplit targets 20% of unique groups
    # for test and then 25% of the remaining groups for validation (20% overall).
    indices = np.arange(len(labels))
    temp_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        .split(indices, labels, groups)
    )
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
        .split(temp_idx, labels[temp_idx], groups[temp_idx])
    )
    return {"TRAIN": temp_idx[train_rel],
            "VALIDATION": temp_idx[val_rel],
            "TEST": np.asarray(test_idx, dtype=int)}


def split_lines(frame, split, group_keys, title):
    lines = [f"SPLIT SIZES ({title})"]
    group_sets = {}
    for part in ("TRAIN", "VALIDATION", "TEST"):
        idx = split[part]
        counts = frame.iloc[idx]["document_sentiment"].astype(str).str.lower().value_counts()
        group_sets[part] = set(group_keys[idx])
        lines.append(
            f"  {part}: rows={len(idx)}, unique_normalized_text_groups={len(group_sets[part])}, "
            + ", ".join(f"{lab}={int(counts.get(lab, 0))}" for lab in LABEL_ORDER)
        )
    lines.append(
        "  normalized_group_intersections: "
        f"train-validation={len(group_sets['TRAIN'] & group_sets['VALIDATION'])}, "
        f"train-test={len(group_sets['TRAIN'] & group_sets['TEST'])}, "
        f"validation-test={len(group_sets['VALIDATION'] & group_sets['TEST'])}"
    )
    return lines


def exact_overlap_lines(frame, split, protocol):
    train_keys = frame.iloc[split["TRAIN"]]["comment"].map(lower_text)
    test_keys = frame.iloc[split["TEST"]]["comment"].map(lower_text)
    train_counts = Counter(k for k in train_keys if k)
    test_counts = Counter(k for k in test_keys if k)
    shared = set(train_counts) & set(test_counts)
    return [
        f"TRAIN-TEST EXACT-TEXT OVERLAP ({protocol})",
        "  text_key=comment.lower() only; empty comments excluded",
        f"  distinct_shared_lowercased_texts={len(shared)}",
        f"  train_rows_with_shared_text={sum(train_counts[k] for k in shared)}",
        f"  test_rows_with_shared_text={sum(test_counts[k] for k in shared)}",
        f"  cross_partition_row_pairs_with_same_text={sum(train_counts[k] * test_counts[k] for k in shared)}",
    ]


def metric_lines(y_true, y_pred, title):
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    lines = [f"  {title}: rows={len(y_true)}"]
    if len(y_true) == 0:
        return lines + [
            "    accuracy=NA; macro_f1=NA; weighted_f1=NA",
            "    confusion_matrix=true_[positive,neutral,negative]_by_predicted_[positive,neutral,negative]: NA",
            "    per_class_recall: positive=NA, neutral=NA, negative=NA",
        ]
    lines.append(
        f"    accuracy={accuracy_score(y_true, y_pred):.6f}; "
        f"macro_f1={f1_score(y_true, y_pred, labels=list(LABEL_ORDER), average='macro', zero_division=0):.6f}; "
        f"weighted_f1={f1_score(y_true, y_pred, labels=list(LABEL_ORDER), average='weighted', zero_division=0):.6f}"
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_ORDER))
    lines.append(
        "    confusion_matrix=true_[positive,neutral,negative]_by_predicted_[positive,neutral,negative]: "
        + str(cm.tolist())
    )
    recalls = recall_score(y_true, y_pred, labels=list(LABEL_ORDER), average=None, zero_division=0)
    supports = [int(np.sum(y_true == label)) for label in LABEL_ORDER]
    lines.append("    per_class_recall: " + ", ".join(
        f"{label}={recall:.6f}(support={n})" for label, recall, n in zip(LABEL_ORDER, recalls, supports)
    ))
    return lines


def evaluate(frame, split, protocol, all_conflict_keys):
    lines = ["", "=" * 88, f"PROTOCOL: {protocol}", "=" * 88]
    lines.extend(split_lines(frame, split, frame["group_key"].to_numpy(), protocol))
    lines.extend(exact_overlap_lines(frame, split, protocol))

    train = frame.iloc[split["TRAIN"]]
    validation = frame.iloc[split["VALIDATION"]]
    test = frame.iloc[split["TEST"]]
    y_train = train["document_sentiment"].to_numpy(dtype=str)
    y_validation = validation["document_sentiment"].to_numpy(dtype=str)
    y_test = test["document_sentiment"].to_numpy(dtype=str)

    # Same TF-IDF parameters and fit pattern as train_models.py: fit vocabulary on
    # train only, transform validation/test, then fit final models on train+validation.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000
    )
    x_train = vectorizer.fit_transform(train["preprocessed_comment"].astype(str))
    x_validation = vectorizer.transform(validation["preprocessed_comment"].astype(str))
    x_test = vectorizer.transform(test["preprocessed_comment"].astype(str))
    x_train_validation = vstack([x_train, x_validation], format="csr")
    y_train_validation = np.concatenate([y_train, y_validation])
    lines.append(
        "TFIDF: min_df=2, max_features=50000, ngram_range=(1, 2), sublinear_tf=True; "
        f"fitted_vocabulary_size={len(vectorizer.vocabulary_)}"
    )
    lines.append(f"FINAL_MODEL_FIT_ROWS={len(y_train_validation)} (TRAIN + VALIDATION)")

    models = [
        ("LinearSVC_C1.0", LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE)),
        ("MultinomialNB_alpha0.1", MultinomialNB(alpha=0.1)),
    ]
    test_sources = test["source"].astype(str).to_numpy()
    test_lower_keys = test["comment"].map(lower_text).to_numpy()
    conflict_mask = np.asarray([key in all_conflict_keys for key in test_lower_keys], dtype=bool)

    for model_name, model in models:
        model.fit(x_train_validation, y_train_validation)
        predictions = model.predict(x_test)
        lines.extend(["", f"MODEL: {model_name}"])
        lines.extend(metric_lines(y_test, predictions, "POOLED TEST"))
        lines.append("PER-SOURCE TEST METRICS:")
        for source in SOURCE_ORDER:
            mask = test_sources == source
            lines.extend(metric_lines(y_test[mask], predictions[mask], SOURCE_DISPLAY[source]))
        lines.append("CONFLICTING-LABEL TEXT TEST METRICS:")
        lines.extend(metric_lines(y_test[conflict_mask], predictions[conflict_mask], "CONFLICTING TEXT ROWS"))
        lines.extend(metric_lines(y_test[~conflict_mask], predictions[~conflict_mask], "NON-CONFLICTING TEXT ROWS"))
    return lines


def build_pool():
    data_dir = Path(resolve_data_directory(str(REPO_ROOT)))
    primary_path = resolve_main_csv_path(str(REPO_ROOT))
    if not primary_path:
        raise FileNotFoundError(
            "Primary CSV not found at Data/nigerian_lecturer_evaluations_10000.csv "
            "or data/nigerian_lecturer_evaluations_10000.csv."
        )
    slang_path = resolve_dataset_path(str(REPO_ROOT), "nigerian_slang_dataset.csv", str(data_dir))
    if not Path(slang_path).is_file():
        raise FileNotFoundError(f"Slang CSV not found: {slang_path}")

    # The training module sets Python and NumPy seeds to 42 on import. Generate the
    # 1,600 templates first, then load primary, then slang, matching train_models.py.
    with contextlib.redirect_stdout(io.StringIO()):
        synthetic = generate_synthetic_dataset(num_samples=1600)
        synthetic["source"] = "synthetic"
        primary = load_main_dataset(primary_path)
        slang = load_slang_dataset(slang_path)
    if not primary.empty and "source" not in primary.columns:
        primary["source"] = "nigerian"
    if not slang.empty and "source" not in slang.columns:
        slang["source"] = "nigerian_slang"

    frames = [synthetic, primary]
    if not slang.empty:
        frames.append(slang)
    pool = pd.concat(frames, ignore_index=True)
    pool["source"] = pool["source"].fillna("UNKNOWN").astype(str)
    pool["document_sentiment"] = pool["document_sentiment"].astype(str).str.lower()
    pool["comment"] = pool["comment"].map(as_text)
    pool["preprocessed_comment"] = pool["comment"].map(preprocess_for_vectorizer)
    pool["group_key"] = pool["comment"].map(normalized_text)
    return pool, data_dir, Path(primary_path), Path(slang_path), len(synthetic), len(primary), len(slang)


def main():
    pool, data_dir, primary_path, slang_path, n_templates, n_primary, n_slang = build_pool()
    primary_file_rows = len(pd.read_csv(primary_path))
    slang_file_rows = len(pd.read_csv(slang_path))
    lines = [
        "LECTURER EVALUATION: SPLIT-PROTOCOL AUDIT",
        f"repository={REPO_ROOT}",
        f"primary_file={primary_path}",
        f"slang_file={slang_path}",
        "pool_order=synthetic -> primary -> slang",
        "document_sentiment=labels returned by the existing train_models.py loaders",
        "",
        "SOURCE AND DUPLICATE SUMMARY",
        f"RAW_PRIMARY_CSV_ROWS={primary_file_rows}",
        f"RAW_SLANG_CSV_ROWS={slang_file_rows}",
        f"GENERATED_TEMPLATE_RECORDS={n_templates}",
        f"LOADED_PRIMARY_RECORDS_AFTER_LOADER={n_primary}",
        f"LOADED_SLANG_RECORDS={n_slang}",
    ]
    for source in SOURCE_ORDER:
        lines.extend(summary_lines(pool.loc[pool["source"] == source], SOURCE_DISPLAY[source]))
    lines.extend(summary_lines(pool, "POOLED"))

    lines.extend(["", "CONFLICTING-LABEL TEXT SUMMARY"])
    for source in (*SOURCE_ORDER, "POOLED"):
        subset = pool if source == "POOLED" else pool.loc[pool["source"] == source]
        name = "POOLED" if source == "POOLED" else SOURCE_DISPLAY[source]
        lines.append(f"  {name}: conflicting_unique_lowercased_texts={len(conflicting_keys(subset))}")

    labels = pool["document_sentiment"].to_numpy(dtype=str)
    random_split = random_record_split(labels)
    group_keys = pool["group_key"].to_numpy(dtype=str)
    grouped = grouped_split(labels, group_keys)
    lines.extend([
        "",
        "SPLIT PROTOCOL DEFINITIONS",
        "  RANDOM_RECORD_LEVEL: existing stratified_train_val_test helper; test_size=0.20, validation=0.20 of original, random_state=42; same stratify/fallback behavior.",
        "  NORMALIZED_TEXT_GROUPED: GroupShuffleSplit random_state=42; hold out 20% of unique text groups for test then 25% of remaining groups for validation.",
        "  Grouped split row shares can differ from 60/20/20 because text groups have unequal row counts.",
        "  Group normalization: Unicode NFKC + casefold + trim + collapse whitespace.",
        "  Conflict: a non-empty lowercased comment text has more than one document_sentiment label in the pooled corpus.",
        "  Fixed unweighted models: LinearSVC(C=1.0, max_iter=5000, random_state=42); MultinomialNB(alpha=0.1).",
        "  TF-IDF: min_df=2, max_features=50000, ngram_range=(1, 2), sublinear_tf=True.",
        "  Final model fit uses TRAIN + VALIDATION; TF-IDF vocabulary is fit on TRAIN only.",
    ])
    all_conflicts = conflicting_keys(pool)
    lines.extend(evaluate(pool, random_split, "RANDOM_RECORD_LEVEL", all_conflicts))
    lines.extend(evaluate(pool, grouped, "NORMALIZED_TEXT_GROUPED", all_conflicts))
    lines.append("")
    lines.append("END OF REPORT")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare record-level and normalized-text-grouped evaluation splits.

Run from repository root:
    python src/train_models.py
    python src/audit_split_protocols.py

Uses helpers in train_models.py without modifying it. Prints a plain-text report.
"""
from __future__ import annotations

import contextlib
import io
import json
import random
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
    generate_synthetic_dataset, load_main_dataset, load_slang_dataset,
    preprocess_for_vectorizer, resolve_data_directory, resolve_dataset_path,
    resolve_main_csv_path, stratified_train_val_test,
)

RANDOM_STATE = 42
GROUP_SEEDS = (42, 43, 44, 45, 46)
LABEL_ORDER = ("positive", "neutral", "negative")
SOURCE_ORDER = ("synthetic", "nigerian", "nigerian_slang")
SOURCE_DISPLAY = {"synthetic": "TEMPLATE", "nigerian": "PRIMARY", "nigerian_slang": "SLANG"}


def as_text(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def lower_text(value):
    return as_text(value).lower()


def normalized_text(value):
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
    return [
        f"{title}: records={len(frame)}",
        f"  unique_nonempty_lowercased_texts={nonempty.nunique()}",
        f"  exact_duplicate_rows_beyond_first={len(nonempty) - nonempty.nunique()}",
        f"  lowercased_texts_repeated_at_least_once={int((counts > 1).sum())}",
        f"  conflicting_label_texts={len(conflicts)}",
        f"  rows_belonging_to_conflicting_label_texts={int(keys.isin(conflicts).sum())}",
    ]


def random_record_split(labels):
    indices = np.arange(len(labels))
    train, val, test, *_ = stratified_train_val_test(
        indices, labels, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE
    )
    return {"TRAIN": np.asarray(train, dtype=int),
            "VALIDATION": np.asarray(val, dtype=int),
            "TEST": np.asarray(test, dtype=int)}


def grouped_split(labels, groups, seed):
    indices = np.arange(len(labels))
    temp_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(indices, labels, groups)
    )
    train_rel, val_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            temp_idx, labels[temp_idx], groups[temp_idx]
        )
    )
    return {"TRAIN": temp_idx[train_rel],
            "VALIDATION": temp_idx[val_rel],
            "TEST": np.asarray(test_idx, dtype=int)}


def load_selected_model_settings(metrics_path):
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing {metrics_path}. Run 'python src/train_models.py' first.")
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    try:
        svm = data["document_level"]["svm"]
        nb = data["document_level"]["naive_bayes"]
        svm_c = float(svm["selected_C"])
        svm_weight_raw = svm["selected_class_weight"]
        nb_alpha = float(nb["selected_alpha"])
        nb_weighted = nb["class_weighted_training"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Metrics JSON lacks selected-model metadata. Run the current "
            "src/train_models.py first. Required keys: svm.selected_C, "
            "svm.selected_class_weight, naive_bayes.selected_alpha, "
            "naive_bayes.class_weighted_training."
        ) from exc
    if isinstance(svm_weight_raw, str):
        svm_weight = None if svm_weight_raw.strip().lower() in {"", "none", "null"} else svm_weight_raw
    else:
        svm_weight = svm_weight_raw
    if isinstance(nb_weighted, str):
        nb_weighted = nb_weighted.strip().lower() in {"true", "1", "yes"}
    return {
        "svm_C": svm_c, "svm_class_weight": svm_weight,
        "nb_alpha": nb_alpha, "nb_class_weighted": bool(nb_weighted),
        "metrics_path": metrics_path,
    }


def split_lines(frame, split, group_keys, title):
    lines = [f"SPLIT SIZES ({title})"]
    part_groups = {}
    for part in ("TRAIN", "VALIDATION", "TEST"):
        idx = split[part]
        counts = frame.iloc[idx]["document_sentiment"].astype(str).str.lower().value_counts()
        part_groups[part] = set(group_keys[idx])
        lines.append(
            f"  {part}: rows={len(idx)}, unique_normalized_text_groups={len(part_groups[part])}, "
            + ", ".join(f"{lab}={int(counts.get(lab, 0))}" for lab in LABEL_ORDER)
        )
    lines.append("  normalized_group_intersections: "
                 f"train-validation={len(part_groups['TRAIN'] & part_groups['VALIDATION'])}, "
                 f"train-test={len(part_groups['TRAIN'] & part_groups['TEST'])}, "
                 f"validation-test={len(part_groups['VALIDATION'] & part_groups['TEST'])}")
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


def metric_values(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    if not len(y_true):
        return {"n": 0, "accuracy": np.nan, "macro_f1": np.nan, "weighted_f1": np.nan}
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(LABEL_ORDER), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=list(LABEL_ORDER), average="weighted", zero_division=0)),
    }


def metric_lines(y_true, y_pred, title, result_store=None, context=None):
    y_true, y_pred = np.asarray(y_true, dtype=str), np.asarray(y_pred, dtype=str)
    values = metric_values(y_true, y_pred)
    if result_store is not None and context is not None:
        result_store.append({**context, "cohort": title, **values})
    lines = [f"  {title}: rows={len(y_true)}"]
    if not len(y_true):
        return lines + [
            "    accuracy=NA; macro_f1=NA; weighted_f1=NA",
            "    confusion_matrix=true_[positive,neutral,negative]_by_predicted_[positive,neutral,negative]: NA",
            "    per_class_recall: positive=NA, neutral=NA, negative=NA",
        ]
    lines.append(f"    accuracy={values['accuracy']:.6f}; macro_f1={values['macro_f1']:.6f}; "
                 f"weighted_f1={values['weighted_f1']:.6f}")
    cm = confusion_matrix(y_true, y_pred, labels=list(LABEL_ORDER))
    lines.append("    confusion_matrix=true_[positive,neutral,negative]_by_predicted_[positive,neutral,negative]: "
                 + str(cm.tolist()))
    recalls = recall_score(y_true, y_pred, labels=list(LABEL_ORDER), average=None, zero_division=0)
    supports = [int(np.sum(y_true == label)) for label in LABEL_ORDER]
    lines.append("    per_class_recall: " + ", ".join(
        f"{label}={recall:.6f}(support={n})"
        for label, recall, n in zip(LABEL_ORDER, recalls, supports)
    ))
    return lines


def model_instances(settings):
    svm_weight_label = settings["svm_class_weight"] or "none"
    nb_weight_label = "weighted" if settings["nb_class_weighted"] else "unweighted"
    return [
        (f"LinearSVC_C{settings['svm_C']}_class_weight_{svm_weight_label}",
         LinearSVC(C=settings["svm_C"], class_weight=settings["svm_class_weight"],
                   max_iter=5000, random_state=RANDOM_STATE), "svm"),
        (f"MultinomialNB_alpha{settings['nb_alpha']}_class_weighting_{nb_weight_label}",
         MultinomialNB(alpha=settings["nb_alpha"]), "naive_bayes"),
    ]


def nb_sample_weights(labels):
    counts = pd.Series(labels).value_counts()
    return np.asarray([len(labels) / (len(counts) * counts[label]) for label in labels], dtype=float)


def evaluate(frame, split, protocol, all_conflict_keys, globally_unique_mask,
             settings, result_store, seed_label):
    lines = ["", "=" * 88, f"PROTOCOL: {protocol}", "=" * 88]
    lines.extend(split_lines(frame, split, frame["group_key"].to_numpy(), protocol))
    lines.extend(exact_overlap_lines(frame, split, protocol))
    train, validation, test = (frame.iloc[split[k]] for k in ("TRAIN", "VALIDATION", "TEST"))
    y_train = train["document_sentiment"].to_numpy(dtype=str)
    y_validation = validation["document_sentiment"].to_numpy(dtype=str)
    y_test = test["document_sentiment"].to_numpy(dtype=str)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)
    x_train = vectorizer.fit_transform(train["preprocessed_comment"].astype(str))
    x_validation = vectorizer.transform(validation["preprocessed_comment"].astype(str))
    x_test = vectorizer.transform(test["preprocessed_comment"].astype(str))
    x_train_validation = vstack([x_train, x_validation], format="csr")
    y_train_validation = np.concatenate([y_train, y_validation])
    lines.append(f"TFIDF: min_df=2, max_features=50000, ngram_range=(1, 2), sublinear_tf=True; "
                 f"fitted_vocabulary_size={len(vectorizer.vocabulary_)}")
    lines.append(f"FINAL_MODEL_FIT_ROWS={len(y_train_validation)} (TRAIN + VALIDATION)")
    lines.append("SELECTED_MODELS_FROM_METRICS_JSON: "
                 f"SVM C={settings['svm_C']}, class_weight={settings['svm_class_weight']!r}; "
                 f"NB alpha={settings['nb_alpha']}, class_weighted_training={settings['nb_class_weighted']}")

    sources = test["source"].astype(str).to_numpy()
    lower_keys = test["comment"].map(lower_text).to_numpy()
    conflict_mask = np.asarray([key in all_conflict_keys for key in lower_keys], dtype=bool)
    unique_mask = globally_unique_mask[split["TEST"]]
    for model_name, model, model_key in model_instances(settings):
        if model_key == "naive_bayes" and settings["nb_class_weighted"]:
            model.fit(x_train_validation, y_train_validation,
                      sample_weight=nb_sample_weights(y_train_validation))
        else:
            model.fit(x_train_validation, y_train_validation)
        predictions = model.predict(x_test)
        context = {"seed": seed_label, "model": model_name, "model_key": model_key}
        lines.extend(["", f"MODEL: {model_name}"])
        lines.extend(metric_lines(y_test, predictions, "POOLED TEST", result_store, context))
        lines.append("PER-SOURCE TEST METRICS:")
        for source in SOURCE_ORDER:
            mask = sources == source
            lines.extend(metric_lines(y_test[mask], predictions[mask], SOURCE_DISPLAY[source], result_store, context))
        lines.append("UNIQUE-TEXT TEST METRICS: rows whose normalized comment occurs exactly once "
                     "in the full pooled corpus (blank comment excluded)")
        lines.extend(metric_lines(y_test[unique_mask], predictions[unique_mask],
                                  "GLOBALLY UNIQUE TEXT ROWS", result_store, context))
        lines.append("CONFLICTING-LABEL TEXT TEST METRICS:")
        lines.extend(metric_lines(y_test[conflict_mask], predictions[conflict_mask],
                                  "CONFLICTING TEXT ROWS", result_store, context))
        lines.extend(metric_lines(y_test[~conflict_mask], predictions[~conflict_mask],
                                  "NON-CONFLICTING TEXT ROWS", result_store, context))
    return lines


def template_comparison_lines(generated, saved_path):
    lines = ["", "TEMPLATE-RECORD COMPARISON"]
    if not saved_path.is_file():
        return lines + [f"SAVED_TEMPLATE_FILE_MISSING={saved_path}"]
    saved = pd.read_csv(saved_path, keep_default_na=False, low_memory=False).reset_index(drop=True)
    generated = generated.reset_index(drop=True)
    columns = list(dict.fromkeys([*generated.columns, *saved.columns]))
    lines.extend([
        f"  generated_file_rows={len(generated)}", f"  saved_file={saved_path}",
        f"  saved_file_rows={len(saved)}",
        f"  columns_only_in_generated={sorted(set(generated.columns) - set(saved.columns))}",
        f"  columns_only_in_saved={sorted(set(saved.columns) - set(generated.columns))}",
    ])

    def canon(value):
        if value is None or value is pd.NA:
            return "<NA>"
        try:
            if pd.isna(value):
                return "<NA>"
        except (TypeError, ValueError):
            pass
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isfinite(value) and float(value).is_integer():
                return str(int(value))
            return repr(float(value))
        return str(value)

    def signature(frame, index):
        return tuple(canon(frame.at[index, c]) if c in frame.columns else "<COLUMN_MISSING>" for c in columns)

    compared = min(len(generated), len(saved))
    different = [i for i in range(compared) if signature(generated, i) != signature(saved, i)]
    extra_generated, extra_saved = max(0, len(generated) - len(saved)), max(0, len(saved) - len(generated))
    lines.extend([
        f"  rows_compared_at_same_position={compared}",
        f"  rows_different_at_same_position={len(different)}",
        f"  extra_generated_rows={extra_generated}",
        f"  extra_saved_rows={extra_saved}",
        f"  total_rows_different_including_row_count_mismatch={len(different) + extra_generated + extra_saved}",
    ])
    if different:
        lines.append(f"  first_20_differing_zero_based_row_positions={different[:20]}")
    return lines


def build_pool():
    data_dir = Path(resolve_data_directory(str(REPO_ROOT)))
    primary_path = resolve_main_csv_path(str(REPO_ROOT))
    if not primary_path:
        raise FileNotFoundError("Primary CSV not found in Data/ or data/.")
    slang_path = resolve_dataset_path(str(REPO_ROOT), "nigerian_slang_dataset.csv", str(data_dir))
    if not Path(slang_path).is_file():
        raise FileNotFoundError(f"Slang CSV not found: {slang_path}")

    # Reproduce the train_models.py initial generator state explicitly.
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    with contextlib.redirect_stdout(io.StringIO()):
        synthetic = generate_synthetic_dataset(num_samples=1600)
    synthetic["source"] = "synthetic"
    with contextlib.redirect_stdout(io.StringIO()):
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
    return pool, data_dir, Path(primary_path), Path(slang_path), synthetic, len(primary), len(slang)


def grouped_seed_summary(result_store):
    lines = ["", "GROUPED-PROTOCOL FIVE-SEED SUMMARY", f"seeds={list(GROUP_SEEDS)}"]
    rows = [r for r in result_store if r.get("seed") in GROUP_SEEDS]
    cohorts = ["POOLED TEST", "GLOBALLY UNIQUE TEXT ROWS", *[SOURCE_DISPLAY[s] for s in SOURCE_ORDER]]
    for model in sorted({row["model"] for row in rows}):
        lines.extend(["", f"MODEL: {model}"])
        for cohort in cohorts:
            chosen = [r for r in rows if r["model"] == model and r["cohort"] == cohort]
            acc = np.asarray([r["accuracy"] for r in chosen if np.isfinite(r["accuracy"])])
            f1 = np.asarray([r["macro_f1"] for r in chosen if np.isfinite(r["macro_f1"])])
            if not len(acc):
                lines.append(f"  {cohort}: valid_seeds=0; accuracy_mean=NA; accuracy_std=NA; macro_f1_mean=NA; macro_f1_std=NA")
                continue
            lines.append(f"  {cohort}: valid_seeds={len(acc)}; accuracy_mean={np.mean(acc):.6f}; "
                         f"accuracy_std={np.std(acc, ddof=0):.6f}; macro_f1_mean={np.mean(f1):.6f}; "
                         f"macro_f1_std={np.std(f1, ddof=0):.6f}")
    return lines


def main():
    pool, data_dir, primary_path, slang_path, synthetic, n_primary, n_slang = build_pool()
    primary_file_rows = len(pd.read_csv(primary_path, keep_default_na=False, low_memory=False))
    slang_file_rows = len(pd.read_csv(slang_path, keep_default_na=False, low_memory=False))
    metrics_path = data_dir / "model_evaluation_metrics.json"
    settings = load_selected_model_settings(metrics_path)
    lines = [
        "LECTURER EVALUATION: SPLIT-PROTOCOL AUDIT",
        f"repository={REPO_ROOT}", f"primary_file={primary_path}", f"slang_file={slang_path}",
        f"metrics_file={metrics_path}", "pool_order=synthetic -> primary -> slang",
        "document_sentiment=labels returned by existing train_models.py loaders",
        "", "SELECTED MODEL CONFIGURATION FROM FRESH METRICS JSON",
        f"  SVM: selected_C={settings['svm_C']}; selected_class_weight={settings['svm_class_weight']!r}",
        f"  Naive Bayes: selected_alpha={settings['nb_alpha']}; class_weighted_training={settings['nb_class_weighted']}",
        "  Weighted NB sample weights: N / (number_of_classes * class_count), matching fit_best_document_models.",
        "", "SOURCE AND DUPLICATE SUMMARY",
        f"RAW_PRIMARY_CSV_ROWS={primary_file_rows}", f"RAW_SLANG_CSV_ROWS={slang_file_rows}",
        f"GENERATED_TEMPLATE_RECORDS={len(synthetic)}",
        f"LOADED_PRIMARY_RECORDS_AFTER_LOADER={n_primary}", f"LOADED_SLANG_RECORDS={n_slang}",
    ]
    for source in SOURCE_ORDER:
        lines.extend(summary_lines(pool.loc[pool["source"] == source], SOURCE_DISPLAY[source]))
    lines.extend(summary_lines(pool, "POOLED"))
    lines.extend(["", "CONFLICTING-LABEL TEXT SUMMARY"])
    for source in (*SOURCE_ORDER, "POOLED"):
        subset = pool if source == "POOLED" else pool.loc[pool["source"] == source]
        name = "POOLED" if source == "POOLED" else SOURCE_DISPLAY[source]
        lines.append(f"  {name}: conflicting_unique_lowercased_texts={len(conflicting_keys(subset))}")

    lines.extend(template_comparison_lines(synthetic, data_dir / "synthetic_evaluations.csv"))
    labels = pool["document_sentiment"].to_numpy(dtype=str)
    group_keys = pool["group_key"].to_numpy(dtype=str)
    group_counts = pd.Series(group_keys).value_counts(dropna=False).to_dict()
    globally_unique_mask = np.asarray([(key != "" and group_counts.get(key, 0) == 1) for key in group_keys])
    all_conflicts = conflicting_keys(pool)
    result_store = []
    lines.extend([
        "", "SPLIT PROTOCOL DEFINITIONS",
        "  RANDOM_RECORD_LEVEL: existing stratified_train_val_test helper; test_size=0.20, validation=0.20 of original, random_state=42; same stratify/fallback behavior.",
        f"  NORMALIZED_TEXT_GROUPED: GroupShuffleSplit over seeds {list(GROUP_SEEDS)}; test=20% of text groups, then validation=25% of remaining groups.",
        "  Grouped split row shares can differ from 60/20/20 because text groups have unequal row counts.",
        "  Group normalization: Unicode NFKC + casefold + trim + collapse whitespace.",
        "  Conflict: non-empty lowercased comment text has more than one document_sentiment label.",
        "  Globally unique-text test rows: normalized comment occurs exactly once in the pooled corpus; blank comments excluded.",
        "  TF-IDF: min_df=2, max_features=50000, ngram_range=(1, 2), sublinear_tf=True.",
        "  Final model fit uses TRAIN + VALIDATION; TF-IDF vocabulary fit on TRAIN only.",
    ])
    lines.extend(evaluate(pool, random_record_split(labels), "RANDOM_RECORD_LEVEL",
                          all_conflicts, globally_unique_mask, settings, result_store, "RANDOM_42"))
    for seed in GROUP_SEEDS:
        split = grouped_split(labels, group_keys, seed)
        counts = pool.iloc[split["TEST"]]["document_sentiment"].astype(str).str.lower().value_counts()
        lines.extend(["", "#" * 88, f"GROUPED SPLIT SEED {seed}: TEST CLASS COUNTS",
                      f"  rows={len(split['TEST'])}; " + "; ".join(
                          f"{label}={int(counts.get(label, 0))}" for label in LABEL_ORDER)])
        lines.extend(evaluate(pool, split, f"NORMALIZED_TEXT_GROUPED_SEED_{seed}",
                              all_conflicts, globally_unique_mask, settings, result_store, seed))
    lines.extend(grouped_seed_summary(result_store))
    lines.extend(["", "END OF REPORT"])
    print("\n".join(lines))


if __name__ == "__main__":
    main()

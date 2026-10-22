#!/usr/bin/env python3
"""
Clean the synthetic training corpus without modifying original Data/ CSVs.

Writes:
  results/uniqueness_report.csv
  data/cleaned/dedup_exact.csv
  data/cleaned/dedup_drop_conflicts.csv
  data/cleaned/dedup_majority.csv
  data/cleaned/majority_resolution_log.csv
"""
from __future__ import annotations

import ast
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.train_models import (  # noqa: E402
    generate_synthetic_dataset,
    load_main_dataset,
    load_slang_dataset,
    resolve_data_directory,
    PRIMARY_DATASET_FILENAME,
    SLANG_DATASET_FILENAME,
)

SEED = 42
SOURCES_ORDER = ("nigerian", "synthetic", "nigerian_slang")


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def clause_tokens(text: str) -> list[str]:
    """Rough clause split for uniqueness-of-clauses statistic (not used for training labels)."""
    import re
    parts = re.split(r"[,;.]+|\b(?:but|however|although|though)\b", str(text or ""), flags=re.I)
    return [normalize_text(p) for p in parts if normalize_text(p)]


def load_pooled() -> pd.DataFrame:
    data_dir = resolve_data_directory(ROOT)
    primary_path = os.path.join(data_dir, PRIMARY_DATASET_FILENAME)
    slang_path = os.path.join(data_dir, SLANG_DATASET_FILENAME)
    synth_path = os.path.join(data_dir, "synthetic_evaluations.csv")

    frames = []
    # Prefer committed template CSV so cleaning is reproducible without re-seeding the generator.
    if os.path.isfile(synth_path):
        synth = pd.read_csv(synth_path)
    else:
        random = __import__("random")
        random.seed(SEED)
        np.random.seed(SEED)
        synth = generate_synthetic_dataset(num_samples=1600)
    if "source" not in synth.columns:
        synth["source"] = "synthetic"
    frames.append(synth)

    main = load_main_dataset(primary_path)
    frames.append(main)

    slang = load_slang_dataset(slang_path)
    if len(slang):
        if "source" not in slang.columns:
            slang["source"] = "nigerian_slang"
        frames.append(slang)

    df = pd.concat(frames, ignore_index=True)
    df["comment"] = df["comment"].astype(str)
    df["document_sentiment"] = df["document_sentiment"].astype(str).str.lower().str.strip()
    df["text_key"] = df["comment"].map(normalize_text)
    df["source"] = df["source"].astype(str)
    return df


def uniqueness_stats(df: pd.DataFrame) -> list[dict]:
    rows = []
    total = len(df)
    unique_texts = df["text_key"].nunique()

    # exact duplicate (text+label) rows after first occurrence
    pair = df["text_key"] + "||" + df["document_sentiment"]
    exact_dup_after_first = int(pair.duplicated(keep="first").sum())

    # texts with more than one label
    label_sets = df.groupby("text_key")["document_sentiment"].nunique()
    conflict_texts = label_sets[label_sets > 1]
    n_conflict_texts = int(len(conflict_texts))
    n_conflict_rows = int(df["text_key"].isin(conflict_texts.index).sum())

    rows.append(
        {
            "scope": "pooled",
            "total_rows": total,
            "unique_texts": unique_texts,
            "exact_duplicate_text_label_rows_after_first": exact_dup_after_first,
            "texts_with_multiple_labels": n_conflict_texts,
            "rows_covered_by_multi_label_texts": n_conflict_rows,
            "pct_unique_clauses": _pct_unique_clauses(df),
        }
    )

    for src in SOURCES_ORDER:
        sub = df[df["source"] == src]
        if len(sub) == 0:
            continue
        pair = sub["text_key"] + "||" + sub["document_sentiment"]
        label_sets = sub.groupby("text_key")["document_sentiment"].nunique()
        conflict_texts = label_sets[label_sets > 1]
        rows.append(
            {
                "scope": src,
                "total_rows": len(sub),
                "unique_texts": int(sub["text_key"].nunique()),
                "exact_duplicate_text_label_rows_after_first": int(pair.duplicated(keep="first").sum()),
                "texts_with_multiple_labels": int(len(conflict_texts)),
                "rows_covered_by_multi_label_texts": int(sub["text_key"].isin(conflict_texts.index).sum()),
                "pct_unique_clauses": _pct_unique_clauses(sub),
            }
        )
    return rows


def _pct_unique_clauses(df: pd.DataFrame) -> float:
    clauses = []
    for t in df["comment"].tolist():
        clauses.extend(clause_tokens(t))
    if not clauses:
        return 0.0
    return round(100.0 * len(set(clauses)) / len(clauses), 4)


def dedup_exact(df: pd.DataFrame) -> pd.DataFrame:
    """Keep first row per (text_key, document_sentiment)."""
    key = df["text_key"] + "||" + df["document_sentiment"]
    return df.loc[~key.duplicated(keep="first")].copy()


def dedup_drop_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all rows whose text_key has more than one label; then exact-dedup."""
    label_sets = df.groupby("text_key")["document_sentiment"].nunique()
    ok = label_sets[label_sets == 1].index
    cleaned = df[df["text_key"].isin(ok)].copy()
    return dedup_exact(cleaned)


def dedup_majority(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each text_key, assign majority document_sentiment.
    Ties (no strict majority) are dropped.
    Log: resolved count and dropped tie count.
    """
    log_rows = []
    keep_keys = {}
    for text_key, grp in df.groupby("text_key"):
        counts = Counter(grp["document_sentiment"].tolist())
        most = counts.most_common()
        if len(most) == 1:
            keep_keys[text_key] = most[0][0]
            log_rows.append(
                {
                    "text_key": text_key,
                    "action": "single_label",
                    "chosen_label": most[0][0],
                    "label_counts": json.dumps(dict(counts)),
                    "n_rows": len(grp),
                }
            )
        elif most[0][1] > most[1][1]:
            keep_keys[text_key] = most[0][0]
            log_rows.append(
                {
                    "text_key": text_key,
                    "action": "majority_resolved",
                    "chosen_label": most[0][0],
                    "label_counts": json.dumps(dict(counts)),
                    "n_rows": len(grp),
                }
            )
        else:
            log_rows.append(
                {
                    "text_key": text_key,
                    "action": "tie_dropped",
                    "chosen_label": "",
                    "label_counts": json.dumps(dict(counts)),
                    "n_rows": len(grp),
                }
            )

    resolved = sum(1 for r in log_rows if r["action"] == "majority_resolved")
    dropped = sum(1 for r in log_rows if r["action"] == "tie_dropped")
    single = sum(1 for r in log_rows if r["action"] == "single_label")
    summary = pd.DataFrame(log_rows)
    summary.attrs["resolved"] = resolved
    summary.attrs["dropped"] = dropped
    summary.attrs["single"] = single

    # one row per text_key with majority label; keep first full row and overwrite label
    out_rows = []
    seen = set()
    for _, row in df.iterrows():
        tk = row["text_key"]
        if tk in seen:
            continue
        if tk not in keep_keys:
            continue
        seen.add(tk)
        r = row.copy()
        r["document_sentiment"] = keep_keys[tk]
        out_rows.append(r)
    out = pd.DataFrame(out_rows)
    return out, summary


def save_corpus(df: pd.DataFrame, path: str) -> None:
    cols = [
        c
        for c in [
            "id",
            "lecturer_name",
            "course",
            "course_code",
            "rating",
            "comment",
            "document_sentiment",
            "aspect_labels",
            "source",
            "text_key",
        ]
        if c in df.columns
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[cols].to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    random = __import__("random")
    random.seed(SEED)
    np.random.seed(SEED)

    df = load_pooled()
    print(f"Pooled rows: {len(df)}")

    report = uniqueness_stats(df)
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "uniqueness_report.csv")
    pd.DataFrame(report).to_csv(report_path, index=False)
    print(f"Wrote {report_path}")
    for r in report:
        print(r)

    cleaned_dir = os.path.join(ROOT, "data", "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    exact = dedup_exact(df)
    save_corpus(exact, os.path.join(cleaned_dir, "dedup_exact.csv"))
    print(f"dedup_exact: {len(exact)} rows")

    drop_c = dedup_drop_conflicts(df)
    save_corpus(drop_c, os.path.join(cleaned_dir, "dedup_drop_conflicts.csv"))
    print(f"dedup_drop_conflicts: {len(drop_c)} rows")

    maj, log = dedup_majority(df)
    save_corpus(maj, os.path.join(cleaned_dir, "dedup_majority.csv"))
    log_path = os.path.join(cleaned_dir, "majority_resolution_log.csv")
    log.to_csv(log_path, index=False)
    n_res = int((log["action"] == "majority_resolved").sum())
    n_tie = int((log["action"] == "tie_dropped").sum())
    print(f"dedup_majority: {len(maj)} rows; majority_resolved={n_res}; ties_dropped={n_tie}")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

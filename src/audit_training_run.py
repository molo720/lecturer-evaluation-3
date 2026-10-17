#!/usr/bin/env python3
"""Plain-text audit of saved lecturer-evaluation training data and models.

Run from the repository root: python src/audit_training_run.py
Requires pandas, numpy, scikit-learn, and joblib as in the training environment.
"""
from __future__ import annotations

import ast
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

LABELS = ("positive", "neutral", "negative")
SOURCES = [("PRIMARY_CSV", "nigerian"), ("TEMPLATE", "synthetic"), ("SLANG", "nigerian_slang")]
THRESHOLD = 0.9


def parse_aspects(value):
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        values = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        try:
            values = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(values, (list, tuple)):
        return []
    return [(str(x[0]).strip(), str(x[1]).strip().lower())
            for x in values if isinstance(x, (list, tuple)) and len(x) >= 2]


def record_key(row):
    try:
        rating = str(int(float(row.get("rating", ""))))
    except (TypeError, ValueError, OverflowError):
        rating = str(row.get("rating", "")).strip().lower()
    return (str(row.get("comment", "")).lower(), rating,
            str(row.get("document_sentiment", "")).strip().lower(),
            json.dumps(sorted(parse_aspects(row.get("aspect_labels", ""))),
                       ensure_ascii=False, separators=(",", ":")))


def class_report(frame, label_column="document_sentiment"):
    counts = frame[label_column].astype(str).str.lower().value_counts()
    lines = [f"  ROWS={len(frame)}"]
    for label in LABELS:
        n = int(counts.get(label, 0))
        lines.append(f"  {label}: {n} ({100*n/len(frame):.2f}%)" if len(frame)
                     else f"  {label}: 0 (0.00%)")
    for label in sorted(set(counts.index) - set(LABELS)):
        n = int(counts[label])
        lines.append(f"  {label}: {n} ({100*n/len(frame):.2f}%)")
    return lines


def code_line(lines, fragment):
    return next((i for i, line in enumerate(lines, 1) if fragment in line), None)


def overlap(train, test):
    a = Counter(str(x).lower() for x in train["comment"] if str(x))
    b = Counter(str(x).lower() for x in test["comment"] if str(x))
    shared = set(a).intersection(b)
    return (len(shared), sum(a[x] for x in shared), sum(b[x] for x in shared),
            sum(a[x] * b[x] for x in shared))


def cosine_counts(Xtr, Xte):
    pairs, matched_tr, matched_te = 0, set(), set()
    for start in range(0, Xte.shape[0], 32):
        sims = cosine_similarity(Xte[start:start + 32], Xtr, dense_output=False).tocsr()
        for i in range(sims.shape[0]):
            row = sims.getrow(i)
            idx = row.indices[row.data > THRESHOLD]
            if len(idx):
                pairs += len(idx)
                matched_te.add(start + i)
                matched_tr.update(map(int, idx))
    return pairs, len(matched_te), len(matched_tr)


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = next((root / name for name in ("Data", "data")
                     if (root / name / "preprocessed_evaluations.csv").is_file()), None)
    if data_dir is None:
        raise FileNotFoundError("Could not find Data/preprocessed_evaluations.csv or data/preprocessed_evaluations.csv.")
    source_files = {
        "PRIMARY_CSV": data_dir / "nigerian_lecturer_evaluations_10000.csv",
        "TEMPLATE": data_dir / "synthetic_evaluations.csv",
        "SLANG": data_dir / "nigerian_slang_dataset.csv",
    }
    pool_path = data_dir / "preprocessed_evaluations.csv"
    code_path = root / "src" / "train_models.py"
    missing = [str(p) for p in [*source_files.values(), pool_path, code_path] if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required file(s) missing: " + "; ".join(missing))

    raw = {name: pd.read_csv(path, keep_default_na=False, low_memory=False)
           for name, path in source_files.items()}
    pool = pd.read_csv(pool_path, keep_default_na=False, low_memory=False)
    needed = {"comment", "rating", "document_sentiment", "aspect_labels",
              "preprocessed_comment", "source"}
    if not needed.issubset(pool.columns):
        raise ValueError(f"Pooled CSV lacks columns: {sorted(needed - set(pool.columns))}")
    pool["source"] = pool["source"].astype(str)
    pool["document_sentiment"] = pool["document_sentiment"].astype(str).str.lower()
    code = code_path.read_text(encoding="utf-8")
    code_lines = code.splitlines()
    report = [
        "TRAINING DATA AUDIT REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Repository root: {root}",
        f"Data directory: {data_dir}",
        f"Pooled training input: {pool_path}",
        f"Training code: {code_path}",
    ]

    report += ["", "1. ROW COUNTS AND EXACT-DUPLICATE REMOVAL",
               "Raw input-file row counts (CSV header excluded):"]
    for name, path in source_files.items():
        report.append(f"  {name} | {path} | rows={len(raw[name])}")
    report.append("Duplicate key used for pooled records: lowercase comment + normalized rating + lowercase document_sentiment + sorted parsed aspect_labels; source, IDs and lecturer/course fields excluded.")
    local_dedup_masks = pd.Series(False, index=pool.index)
    for display, src in SOURCES:
        idx = pool.index[pool["source"] == src]
        keys = pool.loc[idx].apply(record_key, axis=1)
        dup = keys.duplicated(keep="first")
        local_dedup_masks.loc[idx] = dup.to_numpy()
        report.append(f"  {display} | pool_rows_before={len(idx)} | source_local_exact_duplicates_removed={int(dup.sum())} | after_source_local_dedup={len(idx)-int(dup.sum())}")
    local_unique = pool.loc[~local_dedup_masks].copy()
    global_dup = local_unique.apply(record_key, axis=1).duplicated(keep="first")
    pooled_after_global = local_unique.loc[~global_dup]
    report.append(f"  POOLED | before_any_exact_duplicate_removal={len(pool)}")
    report.append(f"  POOLED | after_source_local_exact_duplicate_removal={len(local_unique)}")
    report.append(f"  POOLED | cross-source exact duplicate rows removed={int(global_dup.sum())}")
    report.append(f"  POOLED | after_global_exact_duplicate_removal={len(pooled_after_global)}")

    report += ["", "2. LABEL CREATION CODE PATHS AND LABEL COUNTS"]
    report.append(f"  PRIMARY_CSV path: src/train_models.py:{code_line(code_lines, 'def load_main_dataset(')} load_main_dataset() -> src/train_models.py:{code_line(code_lines, 'def rating_to_sentiment(')} rating_to_sentiment(). Rating <=2 negative, >=4 positive, all other or invalid ratings neutral in the loader. Aspect labels inherit document sentiment.")
    report.append(f"  TEMPLATE path: src/train_models.py:{code_line(code_lines, 'def generate_synthetic_dataset(')} generate_synthetic_dataset(). Document label is generator-assigned from target sentiment and clause polarities; ties can be neutral; rating is assigned separately.")
    report.append(f"  SLANG path: src/train_models.py:{code_line(code_lines, 'def load_slang_dataset(')} load_slang_dataset(). Existing document_sentiment and aspect_labels are read from the CSV; this loader does not map sentiment from rating.")
    report.append("Observed document_sentiment labels in the pooled input:")
    report.extend(class_report(pool))
    report.append("Observed document_sentiment labels by source:")
    for display, src in SOURCES:
        report.append(f"  {display} | source={src}")
        report.extend(class_report(pool.loc[pool["source"] == src]))

    report += ["", "3. TRAIN / VALIDATION / TEST SPLIT CLASS DISTRIBUTIONS"]
    split_line = code_line(code_lines, "def stratified_train_val_test(")
    y = pool["document_sentiment"].to_numpy(dtype=str)
    all_idx = np.arange(len(pool))
    try:
        temp_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42, stratify=y)
        test_strat = True
    except ValueError:
        temp_idx, test_idx = train_test_split(all_idx, test_size=0.2, random_state=42)
        test_strat = False
    val_ratio = 0.2 / (1 - 0.2)
    try:
        train_idx, val_idx = train_test_split(temp_idx, test_size=val_ratio, random_state=42, stratify=y[temp_idx])
        val_strat = True
    except ValueError:
        train_idx, val_idx = train_test_split(temp_idx, test_size=val_ratio, random_state=42)
        val_strat = False
    report.extend([f"Split function: src/train_models.py:{split_line} stratified_train_val_test()",
                   f"Test stratify call succeeded: {test_strat}",
                   f"Validation stratify call succeeded: {val_strat}"])
    for name, idx in [("FULL_POOL", all_idx), ("TRAIN", train_idx),
                      ("VALIDATION", val_idx), ("TEST", test_idx)]:
        report.append(name + " class distribution:")
        report.extend(class_report(pool.iloc[idx]))

    report += ["", '4. ASPECT-LEVEL "SAMPLES" AND F1 AVERAGING']
    aspect_counts = Counter()
    no_aspect = unparseable = total_aspect_entries = 0
    for value in pool["aspect_labels"]:
        text = str(value).strip()
        pairs = parse_aspects(value)
        total_aspect_entries += len(pairs)
        if not pairs:
            no_aspect += 1
            if text:
                try:
                    parsed = json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    try:
                        parsed = ast.literal_eval(text)
                    except (ValueError, SyntaxError):
                        parsed = None
                if parsed is None or not isinstance(parsed, (list, tuple)):
                    unparseable += 1
        aspect_counts.update(asp for asp, _sent in pairs)
    report.append("One aspect training row is built for each (aspect, sentiment) entry in aspect_labels; the saved 'Samples' value is len(sub_data) for that aspect.")
    report.append(f"Total aspect entries/clauses: {total_aspect_entries}")
    report.append(f"Records with no parsed aspect: {no_aspect}")
    report.append(f"Records with non-empty unparseable aspect_labels: {unparseable}")
    report.append("Aspect entries by aspect:")
    for aspect, count in sorted(aspect_counts.items(), key=lambda x: x[0].casefold()):
        report.append(f"  {aspect}: {count}")
    report.append("Validation/model selection uses average='macro' via score_macro_f1(). Aspect test exports both macro F1 and weighted F1.")

    report += ["", "5. TF-IDF VECTORISER SETTINGS"]
    tree = ast.parse(code)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if name == "TfidfVectorizer":
                params = {kw.arg: ast.literal_eval(kw.value) if isinstance(kw.value, (ast.Constant, ast.Tuple, ast.List)) else "<dynamic>"
                          for kw in node.keywords if kw.arg}
                calls.append((node.lineno, params))
    for line, p in sorted(calls):
        report.append(f"  code line {line}: min_df={p.get('min_df','<default>')}; max_features={p.get('max_features','<default>')}; ngram_range={p.get('ngram_range','<default>')}; sublinear_tf={p.get('sublinear_tf','<default>')}")
    models = root / "models"
    doc_vec_path = models / "vectorizer_document.pkl"
    if not doc_vec_path.is_file():
        raise FileNotFoundError(f"Missing saved document vectorizer: {doc_vec_path}")
    doc_vec = joblib.load(doc_vec_path)
    def params(v):
        p = v.get_params(deep=False)
        return f"min_df={p.get('min_df')}; max_features={p.get('max_features')}; ngram_range={p.get('ngram_range')}; sublinear_tf={p.get('sublinear_tf')}; fitted_vocabulary_size={len(getattr(v,'vocabulary_',{}))}"
    report.append("Saved document vectorizer: " + params(doc_vec))
    for aspect in sorted(aspect_counts, key=str.casefold):
        path = models / ("vectorizer_" + aspect.replace(" ", "_").lower() + ".pkl")
        report.append(f"Saved aspect vectorizer {aspect}: " + (params(joblib.load(path)) if path.is_file() else "MISSING"))

    report += ["", "6. TRAIN-TEST NEAR-DUPLICATE OVERLAP",
               "Exact text comparison is str(comment).lower() only; no punctuation or whitespace normalization.",
               f"TF-IDF cosine similarity threshold: strictly greater than {THRESHOLD}."]
    tr = pool.iloc[train_idx].reset_index(drop=True)
    te = pool.iloc[test_idx].reset_index(drop=True)
    def write_overlap(label, a, b):
        shared, train_rows, test_rows, pairs = overlap(a, b)
        report.append(f"  {label} exact: distinct_shared_texts={shared}; train_rows_with_match={train_rows}; test_rows_with_match={test_rows}; train-test row pairs={pairs}")
    write_overlap("POOLED", tr, te)
    for display, src in SOURCES:
        write_overlap(display, tr[tr["source"] == src], te[te["source"] == src])
    Xtr = doc_vec.transform(tr["preprocessed_comment"].astype(str))
    Xte = doc_vec.transform(te["preprocessed_comment"].astype(str))
    def write_cosine(label, tx, vx):
        pairs, test_rows, train_rows = cosine_counts(tx, vx)
        report.append(f"  {label} cosine: pairs_above_threshold={pairs}; test_rows_with_match={test_rows}; train_rows_with_match={train_rows}")
    write_cosine("POOLED", Xtr, Xte)
    tr_src = tr["source"].to_numpy()
    te_src = te["source"].to_numpy()
    for display, src in SOURCES:
        write_cosine(display, Xtr[np.flatnonzero(tr_src == src)], Xte[np.flatnonzero(te_src == src)])

    report += ["", "7. DOCUMENT-LEVEL TEST METRICS PER SOURCE"]
    svm_path, nb_path = models / "svm_document.pkl", models / "nb_document.pkl"
    if not svm_path.is_file() or not nb_path.is_file():
        raise FileNotFoundError("Missing saved document models; run src/train_models.py first.")
    svm, nb = joblib.load(svm_path), joblib.load(nb_path)
    pred_svm, pred_nb = svm.predict(Xte), nb.predict(Xte)
    yt = te["document_sentiment"].to_numpy(dtype=str)
    for display, src in SOURCES:
        mask = te_src == src
        if not mask.any():
            report.append(f"  {display}: test_rows=0; SVM accuracy=NA Macro-F1=NA; NB accuracy=NA Macro-F1=NA")
            continue
        report.append(f"  {display}: test_rows={int(mask.sum())}")
        report.append(f"    SVM accuracy={accuracy_score(yt[mask], pred_svm[mask]):.6f}; Macro-F1={f1_score(yt[mask], pred_svm[mask], average='macro', zero_division=0):.6f}")
        report.append(f"    Naive Bayes accuracy={accuracy_score(yt[mask], pred_nb[mask]):.6f}; Macro-F1={f1_score(yt[mask], pred_nb[mask], average='macro', zero_division=0):.6f}")
    report += ["", "END OF REPORT"]
    output = "\n".join(report) + "\n"
    print(output, end="")
    target = data_dir / "training_audit_report.txt"
    target.write_text(output, encoding="utf-8")
    print(f"REPORT_FILE={target}")


if __name__ == "__main__":
    main()

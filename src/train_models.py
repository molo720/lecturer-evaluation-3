import os
import random
import json
import shutil
import ast
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Import preprocessing
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text, preprocess_for_vectorizer

random.seed(42)
np.random.seed(42)


aspect_templates = {
    "Teaching Clarity": {
        "positive": [
            "Explains complex concepts with clarity and precision.",
            "Breaks down difficult topics into digestible steps.",
            "Makes abstract ideas very easy to understand.",
            "Every lecture explanation is logical and clear.",
            "Uses practical analogies that simplify technical concepts.",
            "Delivers structured and comprehensible lectures every time.",
            "Clarifies every question thoroughly until we grasp the topic.",
            "The explanations are articulate and straightforward to follow.",
            "Brilliant teaching clarity; makes challenging algorithms look simple."
        ],
        "negative": [
            "Explanations are confusing and lack structure.",
            "Rushes through difficult slides without proper explanation.",
            "Complicates straightforward concepts unnecessarily.",
            "Speaks too fast and jumps between topics erratically.",
            "Difficult to follow what the lecturer is trying to convey.",
            "Lacks pedagogical clarity; students leave the hall confused.",
            "Uses dense jargon without breaking down fundamentals.",
            "Explanations are vague and leave too many unanswered gaps.",
            "Fails to explain core principles clearly."
        ],
        "neutral": [
            "Lectures follow the textbook explanations adequately.",
            "The explanations are standard and neither exciting nor unclear.",
            "Coverage of the syllabus was delivered as outlined.",
            "Explanations are moderate and aligned with the course guide.",
            "Lecture delivery is acceptable for introductory topics."
        ]
    },
    "Course Organisation": {
        "positive": [
            "Course is exceptionally organized and follows the syllabus perfectly.",
            "The lecture schedule and weekly milestones are well planned.",
            "Maintains a clear course outline from week one to final exams.",
            "Course structure is cohesive and modules transition smoothly.",
            "Outstanding planning; topics progress logically and systematically.",
            "The timetable and course pace are managed impeccably."
        ],
        "negative": [
            "The course is poorly organized with no clear syllabus.",
            "Course outline was never provided and topics were disjointed.",
            "Disorganized scheduling; classes were shifted erratically.",
            "The syllabus pace was rushed heavily in the final weeks.",
            "Lacks structure; we jumped from chapter five back to chapter two.",
            "Total lack of course organization throughout the entire semester."
        ],
        "neutral": [
            "The course follows the institutional standard syllabus.",
            "Course topics were addressed according to the departmental calendar.",
            "The syllabus pacing was fairly typical of university modules.",
            "Module sequence followed the assigned curriculum."
        ]
    },
    "Assessment and Grading": {
        "positive": [
            "Grading is fair, objective, and transparently communicated.",
            "Continuous assessments directly test what was taught in class.",
            "Provides detailed rubrics and constructive feedback on scripts.",
            "Exams were well-balanced and marked with complete fairness.",
            "Returns test scores quickly with helpful correction notes.",
            "Grading criteria are explicit and applied equitably to all students."
        ],
        "negative": [
            "Grading is arbitrary, inconsistent, and unfairly penalized.",
            "Test questions covered material that was never taught in class.",
            "Continuous assessment scripts were never returned all semester.",
            "Grading rubrics are completely absent and marks appear random.",
            "Exams were unreasonably difficult and far beyond the syllabus.",
            "Vague feedback on assignments; nobody knows why they lost marks."
        ],
        "neutral": [
            "Assessments adhered to standard departmental grading scales.",
            "Two tests and one exam were administered as scheduled.",
            "Exam difficulty reflected the average level of past papers.",
            "Marks were computed according to normal university regulations."
        ]
    },
    "Lecturer Punctuality": {
        "positive": [
            "Always punctual and starts every class right on time.",
            "Consistently arrives early and utilizes the full lecture period.",
            "Exemplary punctuality and regular attendance all semester.",
            "Never misses a lecture and values students' scheduled time.",
            "Punctual arrival to 8 AM lectures without exception."
        ],
        "negative": [
            "Habitually late to lectures and wastes valuable class time.",
            "Arrives thirty to forty minutes behind schedule repeatedly.",
            "Frequently cancels lectures without giving prior notice.",
            "Missed multiple lecture sessions without making up the hours.",
            "Very poor punctuality; classes start whenever he feels like it."
        ],
        "neutral": [
            "Lecturer arrives roughly on time for most scheduled classes.",
            "Attendance was generally consistent with occasional slight delays.",
            "Lectures started within reasonable university grace periods.",
            "Session timing was generally maintained."
        ]
    },
    "Availability": {
        "positive": [
            "Always available during office hours for student consultation.",
            "Very approachable and willing to assist students after hours.",
            "Welcoming to students seeking help with difficult course projects.",
            "Office doors are always open for academic guidance.",
            "Readily accessible and makes time to mentor struggling learners."
        ],
        "negative": [
            "Virtually impossible to locate outside scheduled lecture hours.",
            "Never available during posted office hours; doors always locked.",
            "Dismissive and impatient when students approach for assistance.",
            "Unavailable to provide guidance on term papers and projects.",
            "Students cannot reach the lecturer for urgent academic inquiries."
        ],
        "neutral": [
            "Available primarily during designated departmental consultation hours.",
            "Office hours are kept when requested in advance.",
            "Consultation is available within standard university hours.",
            "Meeting the lecturer requires scheduling an appointment."
        ]
    },
    "Communication": {
        "positive": [
            "Communicates announcements clearly, proactively, and timely.",
            "Responds promptly to student emails with helpful guidance.",
            "Keeps students well informed about assignment changes and events.",
            "Professional and courteous communication across all channels.",
            "Clear verbal and written communication throughout the semester."
        ],
        "negative": [
            "Fails to respond to student emails and academic inquiries.",
            "Announcements are confusing, contradictory, and last-minute.",
            "Poor communication; students were uninformed about test venues.",
            "Ignores emails regarding coursework clarifications.",
            "Communication channel is disorganized and unreliable."
        ],
        "neutral": [
            "Class announcements were relayed through the class representative.",
            "Communication was limited to official notice boards and portals.",
            "Email responses take two to three business days on average.",
            "Standard course notices were shared periodically."
        ]
    },
    "Student Interaction": {
        "positive": [
            "Encourages active classroom participation and welcoming debates.",
            "Treats every student with dignity, patience, and respect.",
            "Creates an engaging, dynamic, and interactive learning environment.",
            "Listens attentively to student viewpoints and questions.",
            "Fosters classroom interaction where everyone feels safe to speak."
        ],
        "negative": [
            "Intimidates students and ridicules genuine questions in class.",
            "Class is a dull one-way monologue with zero student interaction.",
            "Discourages questions and dismisses student contributions rudely.",
            "Hostile classroom environment where students fear participating.",
            "Completely unengaging; ignores students raising hands."
        ],
        "neutral": [
            "Class interaction was typical with occasional question sessions.",
            "Questions are entertained towards the final ten minutes of class.",
            "Interaction level was standard for a large lecture hall.",
            "Lectures are predominantly instructional with some Q&A."
        ]
    },
    "Use of Teaching Materials": {
        "positive": [
            "Provides comprehensive slides, reference materials, and handouts.",
            "Uses modern multimedia, projectors, and code examples effectively.",
            "Shares lecture notes well ahead of time on the student portal.",
            "Teaching materials are rich, up-to-date, and beautifully designed.",
            "Curated readings and textbook recommendations are extremely helpful."
        ],
        "negative": [
            "Does not share lecture slides or study materials with the class.",
            "Uses outdated, blurry, and handwritten materials from a decade ago.",
            "Refuses to upload lecture notes or reference documents.",
            "Teaching materials are riddled with typos, errors, and missing pages.",
            "Zero supplementary learning materials provided throughout the course."
        ],
        "neutral": [
            "Standard slide presentations were projected during lecture.",
            "Recommended textbooks are available in the university library.",
            "Course materials were shared according to standard faculty policy.",
            "Lecture slides cover only basic bullet points."
        ]
    }
}

CONNECTORS_CONTRAST = [
    ", but ", ", however, ", ", though ", ", yet ", ", on the other hand, ", " although "
]
CONNECTORS_AND = [
    " and ", ". Furthermore, ", ". Also, ", ". Additionally, ", "; moreover, "
]

def generate_synthetic_dataset(num_samples=1800):
    """
    Generates a rich, domain-specific evaluation dataset covering:
    - All 8 teaching aspects
    - Document-level sentiment (positive, negative, neutral)
    - Multi-clause aspect combinations with realistic contrastive and additive connectors
    - Nigerian university courses and lecturers
    """
    courses = [
        ("CSC410", "Special Computing"),
        ("CSC412", "Data Science & Big Data"),
        ("CSC406", "Cloud Computing Architectures"),
        ("CSC101", "Introduction to Computer Science"),
        ("CSC408", "Machine Learning & Neural Nets"),
        ("CSC302", "Database Design & Management"),
        ("CSC304", "Operating Systems Principles"),
        ("CSC201", "Data Structures and Algorithms")
    ]
    lecturers = [
        "Dr. Okafor", "Dr. Adeyemi", "Prof. Martins",
        "Dr. Faith", "Dr. Pomele", "Prof. Balogun",
        "Dr. Chukwu", "Dr. (Mrs) Adeleke"
    ]
    
    aspect_list = list(aspect_templates.keys())
    records = []

    for i in range(num_samples):
        ccode, cname = random.choice(courses)
        lecturer = random.choice(lecturers)
        
        # Decide number of aspects mentioned in this comment (1, 2, or 3)
        k = random.choices([1, 2, 3], weights=[0.25, 0.50, 0.25])[0]
        selected_aspects = random.sample(aspect_list, k)
        
        # Decide document sentiment bias
        target_doc_sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.42, 0.38, 0.20])[0]
        
        clauses = []
        clause_labels = []
        
        if target_doc_sentiment == "neutral":
            # Mostly neutral statements
            for asp in selected_aspects:
                phrase = random.choice(aspect_templates[asp]["neutral"])
                clauses.append(phrase)
                clause_labels.append((asp, "neutral"))
            comment_text = " ".join(clauses)
            doc_sentiment = "neutral"
            rating = random.choice([3, 3, 3, 2, 4])
        elif target_doc_sentiment == "positive":
            # Mostly positive, may have 1 neutral or 1 contrastive negative
            for idx, asp in enumerate(selected_aspects):
                if idx > 0 and random.random() < 0.25:
                    sent = random.choice(["neutral", "negative"])
                else:
                    sent = "positive"
                phrase = random.choice(aspect_templates[asp][sent])
                clauses.append(phrase)
                clause_labels.append((asp, sent))
            
            # Combine clauses
            if len(clauses) == 1:
                comment_text = clauses[0]
            elif len(clauses) == 2:
                if clause_labels[0][1] != clause_labels[1][1]:
                    connector = random.choice(CONNECTORS_CONTRAST)
                else:
                    connector = random.choice(CONNECTORS_AND)
                comment_text = clauses[0] + connector + clauses[1]
            else:
                comment_text = clauses[0] + random.choice(CONNECTORS_AND) + clauses[1] + random.choice(CONNECTORS_CONTRAST) + clauses[2]
            
            pos_c = sum(1 for _, s in clause_labels if s == "positive")
            neg_c = sum(1 for _, s in clause_labels if s == "negative")
            if pos_c > neg_c:
                doc_sentiment = "positive"
                rating = random.choice([4, 5, 5, 4])
            elif neg_c > pos_c:
                doc_sentiment = "negative"
                rating = random.choice([1, 2])
            else:
                doc_sentiment = "neutral"
                rating = 3
        else: # negative
            for idx, asp in enumerate(selected_aspects):
                if idx > 0 and random.random() < 0.25:
                    sent = random.choice(["neutral", "positive"])
                else:
                    sent = "negative"
                phrase = random.choice(aspect_templates[asp][sent])
                clauses.append(phrase)
                clause_labels.append((asp, sent))
            
            if len(clauses) == 1:
                comment_text = clauses[0]
            elif len(clauses) == 2:
                if clause_labels[0][1] != clause_labels[1][1]:
                    connector = random.choice(CONNECTORS_CONTRAST)
                else:
                    connector = random.choice(CONNECTORS_AND)
                comment_text = clauses[0] + connector + clauses[1]
            else:
                comment_text = clauses[0] + random.choice(CONNECTORS_AND) + clauses[1] + random.choice(CONNECTORS_CONTRAST) + clauses[2]
            
            pos_c = sum(1 for _, s in clause_labels if s == "positive")
            neg_c = sum(1 for _, s in clause_labels if s == "negative")
            if neg_c > pos_c:
                doc_sentiment = "negative"
                rating = random.choice([1, 2, 1, 2])
            elif pos_c > neg_c:
                doc_sentiment = "positive"
                rating = random.choice([4, 5])
            else:
                doc_sentiment = "neutral"
                rating = 3

        records.append({
            "id": i + 1,
            "lecturer_name": lecturer,
            "course": cname,
            "course_code": ccode,
            "rating": rating,
            "comment": comment_text,
            "document_sentiment": doc_sentiment,
            "aspect_labels": json.dumps(clause_labels)
        })

    df = pd.DataFrame(records)
    return df


NIGERIAN_CSV_CANDIDATES = [
    os.path.join("data", "nigerian_lecturer_evaluations_10000.csv"),
    r"c:\Users\USER\Downloads\nigerian_lecturer_evaluations_10000.csv",
]

SLANG_DATASET_PATH = os.path.join("data", "nigerian_slang_dataset.csv")

ASPECT_NAME_MAP = {
    "teaching clarity": "Teaching Clarity",
    "course organisation": "Course Organisation",
    "course organization": "Course Organisation",
    "assessment and grading": "Assessment and Grading",
    "lecturer punctuality": "Lecturer Punctuality",
    "availability": "Availability",
    "communication": "Communication",
    "student interaction": "Student Interaction",
    "class rep & student relations / drama": "Student Interaction",
    "use of teaching materials": "Use of Teaching Materials",
    "handouts & textbook sales": "Use of Teaching Materials",
}


def rating_to_sentiment(rating):
    try:
        r = int(rating)
    except (TypeError, ValueError):
        return "neutral"
    if r <= 2:
        return "negative"
    if r >= 4:
        return "positive"
    return "neutral"


def resolve_nigerian_csv_path():
    for path in NIGERIAN_CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def parse_primary_aspects(raw):
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    return [part.strip().strip('"') for part in str(raw).split(",") if part.strip()]


def map_aspect_name(name):
    return ASPECT_NAME_MAP.get((name or "").strip().lower())


def load_nigerian_dataset(csv_path):
    """Maps the Nigerian SET CSV onto the training schema used by this project."""
    raw = pd.read_csv(csv_path)
    records = []
    for i, row in raw.iterrows():
        comment = str(row.get("Comment") or "").strip()
        if not comment:
            continue
        rating = row.get("Rating", 3)
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            rating = 3
        doc_sentiment = rating_to_sentiment(rating)
        mapped = []
        for aspect_name in parse_primary_aspects(row.get("Primary Aspects")):
            canon = map_aspect_name(aspect_name)
            if canon:
                mapped.append((canon, doc_sentiment))
        records.append({
            "id": f"ng-{i + 1}",
            "lecturer_name": str(row.get("Lecturer") or "").strip(),
            "course": str(row.get("Course") or "").strip(),
            "course_code": str(row.get("Course Code") or "").strip(),
            "rating": rating,
            "comment": comment,
            "document_sentiment": doc_sentiment,
            "aspect_labels": json.dumps(mapped),
            "source": "nigerian",
        })
    return pd.DataFrame(records)


def load_slang_dataset(csv_path):
    """Loads the Nigerian slang and sarcasm dataset."""
    if not os.path.exists(csv_path):
        print(f"[WARNING] Slang dataset not found at {csv_path}, skipping...")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required_cols = ["id", "lecturer_name", "course", "course_code", "rating", "comment", "document_sentiment", "aspect_labels"]
    for col in required_cols:
        if col not in df.columns:
            print(f"[ERROR] Slang dataset missing required column: {col}")
            return pd.DataFrame()

    print(f"Loaded {len(df)} slang/sarcasm records from {csv_path}")
    return df


def stratified_train_val_test(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Hold out test_size, then hold out val_size of the original data from the remainder."""
    def _split(features, labels, size):
        try:
            return train_test_split(
                features, labels, test_size=size, random_state=random_state, stratify=labels
            )
        except ValueError:
            return train_test_split(features, labels, test_size=size, random_state=random_state)

    X_temp, X_test, y_temp, y_test = _split(X, y, test_size)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = _split(X_temp, y_temp, val_ratio)
    return X_train, X_val, X_test, y_train, y_val, y_test


def score_macro_f1(y_true, y_pred):
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return f1


def classify_metrics(y_true, y_pred, labels=("positive", "neutral", "negative")):
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(p_macro, 4),
        "macro_recall": round(r_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weight, 4),
        "confusion_matrix": cm.tolist(),
        "classes": list(labels),
    }


def fit_best_document_models(X_train, X_val, X_test, y_train, y_val, y_test):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=50000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    X_trainval_vec = vectorizer.transform(np.concatenate([X_train, X_val]))
    y_trainval = np.concatenate([y_train, y_val])

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_vec, y_train)
    dummy_acc = accuracy_score(y_test, dummy.predict(X_test_vec))

    best_svm_c, best_svm_f1 = 1.0, -1.0
    for C in (0.5, 1.0, 2.0):
        cand = LinearSVC(C=C, max_iter=5000, random_state=42)
        cand.fit(X_train_vec, y_train)
        f1 = score_macro_f1(y_val, cand.predict(X_val_vec))
        print(f"  SVM validation Macro-F1 (C={C}): {f1:.4f}", flush=True)
        if f1 > best_svm_f1:
            best_svm_c, best_svm_f1 = C, f1

    best_nb_alpha, best_nb_f1 = 0.5, -1.0
    for alpha in (0.1, 0.5, 1.0):
        cand = MultinomialNB(alpha=alpha)
        cand.fit(X_train_vec, y_train)
        f1 = score_macro_f1(y_val, cand.predict(X_val_vec))
        print(f"  Naive Bayes validation Macro-F1 (alpha={alpha}): {f1:.4f}", flush=True)
        if f1 > best_nb_f1:
            best_nb_alpha, best_nb_f1 = alpha, f1

    svm = LinearSVC(C=best_svm_c, max_iter=5000, random_state=42)
    svm.fit(X_trainval_vec, y_trainval)
    nb = MultinomialNB(alpha=best_nb_alpha)
    nb.fit(X_trainval_vec, y_trainval)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        svm_cv = cross_val_score(
            LinearSVC(C=best_svm_c, max_iter=5000, random_state=42),
            X_train_vec, y_train, cv=cv, scoring="f1_macro"
        )
        nb_cv = cross_val_score(
            MultinomialNB(alpha=best_nb_alpha),
            X_train_vec, y_train, cv=cv, scoring="f1_macro"
        )
        svm_cv_mean, svm_cv_std = float(svm_cv.mean()), float(svm_cv.std())
        nb_cv_mean, nb_cv_std = float(nb_cv.mean()), float(nb_cv.std())
    except ValueError:
        svm_cv_mean = svm_cv_std = nb_cv_mean = nb_cv_std = 0.0

    svm_metrics = classify_metrics(y_test, svm.predict(X_test_vec))
    nb_metrics = classify_metrics(y_test, nb.predict(X_test_vec))
    svm_metrics["selected_C"] = best_svm_c
    svm_metrics["validation_macro_f1"] = round(best_svm_f1, 4)
    svm_metrics["cv_macro_f1_mean"] = round(svm_cv_mean, 4)
    svm_metrics["cv_macro_f1_std"] = round(svm_cv_std, 4)
    nb_metrics["selected_alpha"] = best_nb_alpha
    nb_metrics["validation_macro_f1"] = round(best_nb_f1, 4)
    nb_metrics["cv_macro_f1_mean"] = round(nb_cv_mean, 4)
    nb_metrics["cv_macro_f1_std"] = round(nb_cv_std, 4)

    return vectorizer, svm, nb, dummy_acc, svm_metrics, nb_metrics


def main():
    print("========================================================================", flush=True)
    print("    LECTURER EVALUATION SYSTEM:  MODEL TRAINING", flush=True)
    print("========================================================================", flush=True)

    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n[STEP 1/5] Building combined training corpus...", flush=True)
    synthetic_df = generate_synthetic_dataset(num_samples=1600)
    synthetic_df["source"] = "synthetic"
    synthetic_df.to_csv("data/synthetic_evaluations.csv", index=False)
    print(f"Generated {len(synthetic_df)} synthetic evaluation records.", flush=True)

    nigerian_path = resolve_nigerian_csv_path()
    if not nigerian_path:
        raise FileNotFoundError(
            "Nigerian evaluation CSV not found. Place it at data/nigerian_lecturer_evaluations_10000.csv "
            "or c:\\Users\\USER\\Downloads\\nigerian_lecturer_evaluations_10000.csv"
        )
    local_copy = os.path.join("data", "nigerian_lecturer_evaluations_10000.csv")
    if os.path.abspath(nigerian_path) != os.path.abspath(local_copy):
        shutil.copy2(nigerian_path, local_copy)
        print(f"Copied Nigerian dataset to {local_copy}", flush=True)
    nigerian_df = load_nigerian_dataset(local_copy)
    print(f"Loaded {len(nigerian_df)} Nigerian evaluation records from {local_copy}.", flush=True)

    # Load slang/sarcasm dataset if it exists
    slang_df = load_slang_dataset(SLANG_DATASET_PATH)

    # Combine all datasets
    datasets_to_combine = [synthetic_df, nigerian_df]
    if not slang_df.empty:
        datasets_to_combine.append(slang_df)

    df = pd.concat(datasets_to_combine, ignore_index=True)
    print("Combined document sentiment distribution:\n", df["document_sentiment"].value_counts(), flush=True)
    print("Source counts:\n", df["source"].value_counts(), flush=True)

    print("\n[STEP 2/5] Running preprocessing pipeline (cleaning, contractions, stemming)...", flush=True)
    df["preprocessed_comment"] = df["comment"].apply(preprocess_for_vectorizer)
    df.to_csv("data/preprocessed_evaluations.csv", index=False)
    print("Preprocessing completed and saved to data/preprocessed_evaluations.csv.", flush=True)

    print("\n[STEP 3/5] Training document-level classifiers (train / validate / test)...", flush=True)
    X_doc = df["preprocessed_comment"].values
    y_doc = df["document_sentiment"].values
    X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d = stratified_train_val_test(X_doc, y_doc)
    print(
        f"  Split sizes — train: {len(X_train_d)}, validation: {len(X_val_d)}, test: {len(X_test_d)}",
        flush=True,
    )

    doc_vectorizer, svm_doc, nb_doc, dummy_acc, svm_doc_metrics, nb_doc_metrics = fit_best_document_models(
        X_train_d, X_val_d, X_test_d, y_train_d, y_val_d, y_test_d
    )
    print(f"  [Baseline] Majority-class test accuracy: {dummy_acc:.4f}", flush=True)
    print("\n  --- Document-level SVM (held-out test) ---", flush=True)
    print(f"  Accuracy       : {svm_doc_metrics['accuracy']:.4f}", flush=True)
    print(f"  Macro-F1       : {svm_doc_metrics['macro_f1']:.4f}", flush=True)
    print(f"  Weighted-F1    : {svm_doc_metrics['weighted_f1']:.4f}", flush=True)
    print(f"  Macro-Precision: {svm_doc_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro-Recall   : {svm_doc_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Confusion Matrix (pos, neu, neg):\n{np.array(svm_doc_metrics['confusion_matrix'])}", flush=True)

    print("\n  --- Document-level Naive Bayes (held-out test) ---", flush=True)
    print(f"  Accuracy       : {nb_doc_metrics['accuracy']:.4f}", flush=True)
    print(f"  Macro-F1       : {nb_doc_metrics['macro_f1']:.4f}", flush=True)
    print(f"  Weighted-F1    : {nb_doc_metrics['weighted_f1']:.4f}", flush=True)
    print(f"  Macro-Precision: {nb_doc_metrics['macro_precision']:.4f}", flush=True)
    print(f"  Macro-Recall   : {nb_doc_metrics['macro_recall']:.4f}", flush=True)
    print(f"  Confusion Matrix (pos, neu, neg):\n{np.array(nb_doc_metrics['confusion_matrix'])}", flush=True)

    joblib.dump(svm_doc, "models/svm_document.pkl")
    joblib.dump(nb_doc, "models/nb_document.pkl")
    joblib.dump(doc_vectorizer, "models/vectorizer_document.pkl")
    print("\nSaved document-level models to models/", flush=True)

    print("\n[STEP 4/5] Training aspect-level classifiers for all aspects...", flush=True)
    aspect_records = []
    for _, row in df.iterrows():
        labels = json.loads(row["aspect_labels"] or "[]")
        for asp, sent in labels:
            aspect_records.append({
                "aspect": asp,
                "text": row["preprocessed_comment"],
                "sentiment": sent,
            })

    aspect_df = pd.DataFrame(aspect_records)
    aspect_metrics = []

    for aspect in aspect_templates.keys():
        sub_data = aspect_df[aspect_df["aspect"] == aspect]
        if len(sub_data) < 20:
            continue

        X_asp = sub_data["text"].values
        y_asp = sub_data["sentiment"].values
        X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_train_val_test(X_asp, y_asp)

        asp_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=30000)
        X_tr_v = asp_vec.fit_transform(X_tr)
        X_va_v = asp_vec.transform(X_va)
        X_te_v = asp_vec.transform(X_te)
        X_trva_v = asp_vec.transform(np.concatenate([X_tr, X_va]))
        y_trva = np.concatenate([y_tr, y_va])

        best_c, best_svm_f1 = 1.0, -1.0
        for C in (0.5, 1.0, 2.0):
            cand = LinearSVC(C=C, max_iter=5000, random_state=42)
            cand.fit(X_tr_v, y_tr)
            f1 = score_macro_f1(y_va, cand.predict(X_va_v))
            if f1 > best_svm_f1:
                best_c, best_svm_f1 = C, f1

        best_alpha, best_nb_f1 = 0.5, -1.0
        for alpha in (0.1, 0.5, 1.0):
            cand = MultinomialNB(alpha=alpha)
            cand.fit(X_tr_v, y_tr)
            f1 = score_macro_f1(y_va, cand.predict(X_va_v))
            if f1 > best_nb_f1:
                best_alpha, best_nb_f1 = alpha, f1

        svm_asp = LinearSVC(C=best_c, max_iter=5000, random_state=42)
        svm_asp.fit(X_trva_v, y_trva)
        nb_asp = MultinomialNB(alpha=best_alpha)
        nb_asp.fit(X_trva_v, y_trva)
        svm_asp_pred = svm_asp.predict(X_te_v)
        nb_asp_pred = nb_asp.predict(X_te_v)

        svm_acc = accuracy_score(y_te, svm_asp_pred)
        _, _, svm_macro_f1, _ = precision_recall_fscore_support(y_te, svm_asp_pred, average="macro", zero_division=0)
        _, _, svm_weight_f1, _ = precision_recall_fscore_support(y_te, svm_asp_pred, average="weighted", zero_division=0)
        nb_acc = accuracy_score(y_te, nb_asp_pred)
        _, _, nb_macro_f1, _ = precision_recall_fscore_support(y_te, nb_asp_pred, average="macro", zero_division=0)
        _, _, nb_weight_f1, _ = precision_recall_fscore_support(y_te, nb_asp_pred, average="weighted", zero_division=0)

        safe_name = aspect.replace(" ", "_").lower()
        joblib.dump(svm_asp, f"models/svm_{safe_name}.pkl")
        joblib.dump(nb_asp, f"models/nb_{safe_name}.pkl")
        joblib.dump(asp_vec, f"models/vectorizer_{safe_name}.pkl")

        aspect_metrics.append({
            "Aspect": aspect,
            "Samples": len(sub_data),
            "SVM_Acc": round(svm_acc, 4),
            "SVM_Macro_F1": round(svm_macro_f1, 4),
            "SVM_Weighted_F1": round(svm_weight_f1, 4),
            "SVM_Val_Macro_F1": round(best_svm_f1, 4),
            "SVM_Confusion_Matrix": confusion_matrix(y_te, svm_asp_pred).tolist(),
            "NB_Acc": round(nb_acc, 4),
            "NB_Macro_F1": round(nb_macro_f1, 4),
            "NB_Weighted_F1": round(nb_weight_f1, 4),
            "NB_Val_Macro_F1": round(best_nb_f1, 4),
            "NB_Confusion_Matrix": confusion_matrix(y_te, nb_asp_pred).tolist(),
            "Best_Model": "SVM" if svm_macro_f1 >= nb_macro_f1 else "Naive Bayes",
        })

    asp_results_df = pd.DataFrame(aspect_metrics)
    print("\nAspect-level test evaluation summary:", flush=True)
    print(asp_results_df.to_string(index=False), flush=True)
    asp_results_df.to_csv("data/classification_results_summary.csv", index=False)

    print("\n[STEP 5/5] Saving detailed metrics for the evaluation dashboard...", flush=True)

    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_minutes = runtime_seconds / 60

    metrics_data = {
        "dataset": {
            "nigerian_records": int(len(nigerian_df)),
            "synthetic_records": int(len(synthetic_df)),
            "slang_records": int(len(slang_df)) if not slang_df.empty else 0,
            "total_records": int(len(df)),
            "train_size": int(len(X_train_d)),
            "validation_size": int(len(X_val_d)),
            "test_size": int(len(X_test_d)),
            "split": "60% train / 20% validation / 20% test",
            "document_label_source": "Nigerian ratings 1-2 negative, 3 neutral, 4-5 positive; synthetic gold labels; slang/sarcasm gold labels",
        },
        "document_level": {
            "majority_baseline_accuracy": round(dummy_acc, 4),
            "svm": svm_doc_metrics,
            "naive_bayes": nb_doc_metrics,
        },
        "aspect_level": aspect_metrics,
        "runtime": {
            "total_seconds": round(runtime_seconds, 2),
            "total_minutes": round(runtime_minutes, 2),
            "hardware": "Local training on development machine"
        }
    }

    with open("data/model_evaluation_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("Model metrics saved to data/model_evaluation_metrics.json.", flush=True)
    print("=" * 70, flush=True)
    print("   ALL MODELS TRAINED AND SAVED SUCCESSFULLY!", flush=True)
    print(f"   Total training time: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)", flush=True)
    print("========================================================================", flush=True)

if __name__ == "__main__":
    main()


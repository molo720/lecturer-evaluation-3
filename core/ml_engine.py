import os
import re
import string
import joblib
from functools import lru_cache
from core.lexicons import ASPECTS, ASPECT_KEYWORDS, NEGATION_MARKERS
from src.preprocessing import clean_text, preprocess_for_vectorizer

def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def models_dir():
    """Get the models directory, creating it if it doesn't exist."""
    root = project_root()
    path = os.path.join(root, "models")
    os.makedirs(path, exist_ok=True)
    return path

# In-memory model cache
svm_doc_model = None
nb_doc_model = None
doc_vectorizer = None
aspect_models = {}
aspect_vectorizers = {}
evaluation_metrics = {}

def load_all_models():
    """Loads document-level and aspect-level trained models into memory cache."""
    global svm_doc_model, nb_doc_model, doc_vectorizer, aspect_models, aspect_vectorizers

    models_path = models_dir()
    print(f"[INFO] Looking for models in: {models_path}")

    # 1. Document-level models
    svm_path = os.path.join(models_path, "svm_document.pkl")
    nb_path = os.path.join(models_path, "nb_document.pkl")
    vec_path = os.path.join(models_path, "vectorizer_document.pkl")

    if os.path.exists(svm_path):
        svm_doc_model = joblib.load(svm_path)
        print("[INFO] Loaded document-level SVM model.")
    else:
        print(f"[WARNING] Document-level SVM model not found at {svm_path}")

    if os.path.exists(nb_path):
        nb_doc_model = joblib.load(nb_path)
        print("[INFO] Loaded document-level Naive Bayes model.")
    else:
        print(f"[WARNING] Document-level Naive Bayes model not found at {nb_path}")

    if os.path.exists(vec_path):
        doc_vectorizer = joblib.load(vec_path)
        print("[INFO] Loaded document-level vectorizer.")
    else:
        print(f"[WARNING] Document-level vectorizer not found at {vec_path}")

    # 2. Aspect-level models
    aspect_models = {}
    aspect_vectorizers = {}
    for aspect in ASPECTS:
        clean_name = aspect.lower().replace(" ", "_")
        svm_path = os.path.join(models_path, f"svm_{clean_name}.pkl")
        vec_path = os.path.join(models_path, f"vectorizer_{clean_name}.pkl")
        if os.path.exists(svm_path) and os.path.exists(vec_path):
            aspect_models[aspect] = joblib.load(svm_path)
            aspect_vectorizers[aspect] = joblib.load(vec_path)
        else:
            print(f"[WARNING] Aspect model files not found for {aspect}: {svm_path}, {vec_path}")
    print(f"[INFO] Loaded {len(aspect_models)} aspect-specific models out of {len(ASPECTS)} total.")

def predict_document_sentiment(cleaned_text):
    """Predicts overall document sentiment using the trained Linear SVM."""
    if svm_doc_model and doc_vectorizer:
        try:
            vec = doc_vectorizer.transform([cleaned_text])
            return svm_doc_model.predict(vec)[0]
        except Exception:
            pass
    return "neutral"

def predict_aspect_sentiment(aspect_name, clause_text):
    """Predicts aspect-level sentiment for a specific clause using aspect classifier."""
    if aspect_name in aspect_models and aspect_name in aspect_vectorizers:
        try:
            vec = aspect_vectorizers[aspect_name].transform([clause_text])
            return aspect_models[aspect_name].predict(vec)[0]
        except Exception:
            pass
    return "neutral"

def split_into_clauses(text):
    """Splits a comment into grammatical sub-clauses for fine-grained aspect analysis."""
    clauses = re.split(r'[,;.\n]+|\b(?:but|however|although|though|except|whereas|and yet|on the other hand)\b', text, flags=re.IGNORECASE)
    return [c.strip() for c in clauses if c.strip()]

def has_negation(text):
    """Checks if a clause contains a negation marker."""
    words = re.findall(r'\b\w+\b', text.lower())
    return any(w in NEGATION_MARKERS for w in words)

@lru_cache(maxsize=0)  # Disabled cache to force re-analysis with updated lexicons
def run_analysis_cached(comment_text, rating_val):
    """
    LRU-cached aspect extraction and sentiment polarity inference.
    Caches parsed results so dashboard aggregations and repeated comments execute in microseconds.
    """
    preprocessed_full = clean_text(comment_text)
    clauses = split_into_clauses(comment_text)
    clauses_clean = [clean_text(c) for c in clauses]

    aspect_results = []
    aspect_scores = {}

    for aspect in ASPECTS:
        keywords = ASPECT_KEYWORDS.get(aspect, {})
        matched_clause = None
        keyword_sentiment = None
        matched_keyword = None

        # 1. Match clauses against aspect keywords (check negative first for insults)
        for orig_c, clean_c in zip(clauses, clauses_clean):
            orig_lower = orig_c.lower()
            # Sort keywords by length (longest first) to match phrases before words
            # Check negative keywords first (for insults and derogatory terms)
            negative_keywords = sorted(keywords.get("negative", []), key=len, reverse=True)
            for kw in negative_keywords:
                if kw in orig_lower:
                    matched_clause = orig_c
                    keyword_sentiment = "negative"
                    matched_keyword = kw
                    break
            if matched_clause:
                break
            # Then check positive keywords (sorted by length)
            positive_keywords = sorted(keywords.get("positive", []), key=len, reverse=True)
            for kw in positive_keywords:
                if kw in orig_lower:
                    matched_clause = orig_c
                    keyword_sentiment = "positive"
                    matched_keyword = kw
                    break
            if matched_clause:
                break
            # Finally check neutral keywords (sorted by length)
            neutral_keywords = sorted(keywords.get("neutral", []), key=len, reverse=True)
            for kw in neutral_keywords:
                if kw in orig_lower:
                    matched_clause = orig_c
                    keyword_sentiment = "neutral"
                    matched_keyword = kw
                    break
            if matched_clause:
                break

        # 2. Polarity determination
        if matched_clause:
            clause_clean = clean_text(matched_clause)
            if keyword_sentiment in ("positive", "negative"):
                final_sentiment = keyword_sentiment
            else:
                final_sentiment = predict_aspect_sentiment(aspect, clause_clean)

            # 3. Negation handling
            if has_negation(matched_clause):
                if "not bad" in matched_clause.lower():
                    final_sentiment = "positive"
                elif final_sentiment == "positive":
                    final_sentiment = "negative"
                elif final_sentiment == "negative":
                    final_sentiment = "positive"

            score_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            numeric_score = score_map.get(final_sentiment, 0.0)

            # Convert sentiment to score for display
            sentiment_score_map = {"positive": 5, "neutral": 3, "negative": 1}
            sentiment_score = sentiment_score_map.get(final_sentiment, 3)

            aspect_results.append({
                "aspect": aspect,
                "sentiment": final_sentiment,
                "score": numeric_score,
                "span": matched_clause.strip(),  # Changed from "evidence" to "span" for template
                "keyword": matched_keyword or "classifier-inferred",
                "negated": has_negation(matched_clause),
                "sentiment_score": sentiment_score,  # Added for template
                "composite_score": numeric_score  # Added for template
            })
            aspect_scores[aspect] = numeric_score
        else:
            aspect_scores[aspect] = 0.0

    # Overall Composite Score
    active_scores = [a['score'] for a in aspect_results]
    sentiment_cs = (sum(active_scores) / len(active_scores)) if active_scores else 0.0
    normalized_rating = (rating_val - 3.0) / 2.0
    overall_cs = round(0.4 * normalized_rating + 0.6 * sentiment_cs, 2)

    doc_sentiment = predict_document_sentiment(preprocessed_full)
    if doc_sentiment == "neutral" and active_scores:
        if sentiment_cs > 0.1:
            doc_sentiment = "positive"
        elif sentiment_cs < -0.1:
            doc_sentiment = "negative"

    return {
        "comment": comment_text,
        "rating": rating_val,
        "preprocessed": preprocessed_full,
        "aspects": aspect_results,
        "overall_cs": overall_cs,
        "document_sentiment": doc_sentiment
    }

def run_analysis(comment, rating=3):
    """Wrapper that ensures inputs are normalized before hitting the LRU cache."""
    c_str = str(comment or "").strip()
    try:
        r_int = int(rating)
    except Exception:
        r_int = 3
    res = run_analysis_cached(c_str, r_int)
    # Return a shallow copy so callers don't mutate cached results
    return dict(res)

def generate_textual_summary(aspect_counts, overall_sentiment):
    """Generates automated qualitative synthesis and actionable recommendations (Section 3.6.2)."""
    strengths = []
    weaknesses = []

    for aspect, counts in aspect_counts.items():
        total = sum(counts.values())
        if total > 0:
            pos_ratio = counts["positive"] / total
            neg_ratio = counts["negative"] / total
            if pos_ratio >= 0.5:
                strengths.append(aspect)
            elif neg_ratio >= 0.35:
                weaknesses.append(aspect)

    paragraphs = []
    if overall_sentiment == "positive":
        paragraphs.append("Overall sentiment towards this lecturer's pedagogical engagement is predominantly positive.")
    elif overall_sentiment == "negative":
        paragraphs.append("Student sentiment indicates targeted areas requiring institutional pedagogy and engagement review.")
    else:
        paragraphs.append("Student evaluation comments reflect balanced feedback with moderate ratings across core teaching aspects.")

    if strengths:
        paragraphs.append(f"Core teaching strengths commended by students include: {', '.join(strengths)}.")
    if weaknesses:
        paragraphs.append(f"Identified developmental priorities for departmental improvement include: {', '.join(weaknesses)}.")
    else:
        paragraphs.append("No critical aspect-level deficiencies were highlighted across student submissions.")

    recommendations = []
    if "Assessment and Grading" in weaknesses:
        recommendations.append("Establish transparent grading rubrics and provide prompt script feedback to students.")
    if "Lecturer Punctuality" in weaknesses:
        recommendations.append("Prioritize timely lecture commencement and minimize rescheduled periods.")
    if "Communication" in weaknesses:
        recommendations.append("Enhance digital communication channels and ensure swift responses to academic student inquiries.")
    if "Teaching Clarity" in weaknesses:
        recommendations.append("Adopt practical examples and interactive explanations to clarify complex curriculum concepts.")
    if "Use of Teaching Materials" in weaknesses:
        recommendations.append("Upload modern lecture slides, readings, and supplementary materials in advance.")

    if not recommendations:
        recommendations.append("Continue maintaining high standards of instructional quality and responsive student mentoring.")

    paragraphs.append("Actionable Next Steps: " + " ".join(recommendations))
    return " ".join(paragraphs)


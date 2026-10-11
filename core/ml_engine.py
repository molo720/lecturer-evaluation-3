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

POLARITY_TO_SIGNED = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
POLARITY_TO_SET = {"positive": 5, "neutral": 3, "negative": 1}
NEGATION_WINDOW = 5
UNCLASSIFIED = "unclassified"
INSUFFICIENT = "insufficient_evidence"

ASPECT_STATUS_LABELS = {
    "strength": "Strength Area",
    "improvement": "Improvement Area",
    "balanced": "Balanced / Neutral",
    "insufficient": "Insufficient evidence",
}

# Contractions and n't forms that word-tokenizers otherwise split.
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|neither|nowhere|hardly|scarcely|barely|"
    r"without|rarely|cannot|doesn't|doesnt|don't|dont|won't|wont|can't|cant|"
    r"couldn't|couldnt|wouldn't|wouldnt|shouldn't|shouldnt|isn't|isnt|aren't|arent|"
    r"wasn't|wasnt|weren't|werent|hasn't|hasnt|haven't|havent|hadn't|hadnt|"
    r"didn't|didnt|no be|no get|no dey|no sabi)\b|n['’]t\b",
    re.IGNORECASE,
)


def split_into_clauses(text):
    """Splits a comment into grammatical sub-clauses for fine-grained aspect analysis."""
    clauses = re.split(
        r'[,;.\n]+|\b(?:but|however|although|though|except|whereas|and yet|on the other hand)\b',
        text,
        flags=re.IGNORECASE,
    )
    return [c.strip() for c in clauses if c.strip()]


def normalize_match_text(text):
    """Lowercase + contraction expansion so doesn't/isn't match the same as does not / is not."""
    return clean_text(text or "")


def has_negation(text):
    """True if the text contains a negation marker, including contracted n't forms."""
    if not text:
        return False
    normalized = normalize_match_text(text)
    if _NEGATION_PATTERN.search(normalized):
        return True
    words = re.findall(r"\b\w+\b", normalized)
    return any(w in NEGATION_MARKERS for w in words)


def _keyword_encodes_negation(keyword):
    """Polar keywords such as 'not punctual' already include their sign."""
    return has_negation(keyword)


def _tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


def negation_in_scope(clause, keyword):
    """
    Apply negation only in the same clause, to the left of the matched keyword,
    within NEGATION_WINDOW tokens. Do not re-flip keywords that already contain negation.
    """
    if not clause or not keyword:
        return False
    haystack = normalize_match_text(clause)
    kw = normalize_match_text(keyword)
    if not kw or _keyword_encodes_negation(keyword):
        return False
    idx = haystack.find(kw)
    if idx < 0:
        return False
    prefix_tokens = _tokens(haystack[:idx])
    window = prefix_tokens[-NEGATION_WINDOW:]
    if not window:
        return False
    return any(_NEGATION_PATTERN.search(tok) or tok in NEGATION_MARKERS for tok in window)


def _flip_polarity(sentiment):
    if sentiment == "positive":
        return "negative"
    if sentiment == "negative":
        return "positive"
    return sentiment


def assign_clause_polarity(clause, keyword, keyword_sentiment, aspect_name):
    """Resolve aspect polarity for one clause: keyword sign, then scoped negation."""
    haystack = normalize_match_text(clause)
    if re.search(r"\bnot\s+bad\b", haystack):
        return "positive", False

    if keyword_sentiment in ("positive", "negative"):
        sentiment = keyword_sentiment
    else:
        sentiment = predict_aspect_sentiment(aspect_name, clean_text(clause))

    applied = False
    if negation_in_scope(clause, keyword):
        sentiment = _flip_polarity(sentiment)
        applied = True
    return sentiment, applied


def find_best_keyword(clause, keywords):
    """Longest keyword match in the clause; ties prefer negative polarity."""
    haystack = normalize_match_text(clause)
    best = None
    for polarity in ("negative", "positive", "neutral"):
        for kw in keywords.get(polarity, []):
            kw_n = normalize_match_text(kw)
            if not kw_n or kw_n not in haystack:
                continue
            candidate = (len(kw_n), 1 if polarity == "negative" else 0, polarity, kw)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None
    return {"polarity": best[2], "keyword": best[3]}


def _signed_to_label(mean_score):
    if mean_score > 0.1:
        return "positive"
    if mean_score < -0.1:
        return "negative"
    return "neutral"


def classify_aspect_status(counts):
    """Same decision rule as the lecturer-report aspect table."""
    pos = int((counts or {}).get("positive", 0) or 0)
    neu = int((counts or {}).get("neutral", 0) or 0)
    neg = int((counts or {}).get("negative", 0) or 0)
    if pos + neu + neg == 0:
        return "insufficient"
    if pos > neg:
        return "strength"
    if neg > pos:
        return "improvement"
    return "balanced"


def aspect_status_map(aspect_counts):
    return {aspect: classify_aspect_status(counts) for aspect, counts in (aspect_counts or {}).items()}


def compute_composite(rating_val, aspect_signed_scores):
    """
    overall_cs = 0.4 * normalized_rating + 0.6 * mean(aspect signed scores)
    Range is [-1, 1]. Unmatched aspects are omitted (not treated as 0.0 evidence).
    """
    if not aspect_signed_scores:
        return None
    sentiment_cs = sum(aspect_signed_scores) / len(aspect_signed_scores)
    try:
        rating = float(rating_val)
    except (TypeError, ValueError):
        rating = 3.0
    normalized_rating = (rating - 3.0) / 2.0
    raw = 0.4 * normalized_rating + 0.6 * sentiment_cs
    return round(max(-1.0, min(1.0, raw)), 2)


@lru_cache(maxsize=0)  # Disabled cache to force re-analysis with updated lexicons
def run_analysis_cached(comment_text, rating_val):
    """
    Aspect extraction and sentiment polarity inference.
    Unmatched / factual text is unclassified — not validated neutral (0.0) evidence.
    """
    preprocessed_full = clean_text(comment_text)
    clauses = split_into_clauses(comment_text)

    aspect_results = []

    for aspect in ASPECTS:
        keywords = ASPECT_KEYWORDS.get(aspect, {})
        clause_hits = []
        for orig_c in clauses:
            hit = find_best_keyword(orig_c, keywords)
            if not hit:
                continue
            sentiment, applied_neg = assign_clause_polarity(
                orig_c, hit["keyword"], hit["polarity"], aspect
            )
            clause_hits.append({
                "clause": orig_c.strip(),
                "keyword": hit["keyword"],
                "sentiment": sentiment,
                "score": POLARITY_TO_SIGNED.get(sentiment, 0.0),
                "negated": applied_neg,
            })

        if not clause_hits:
            continue

        mean_score = sum(h["score"] for h in clause_hits) / len(clause_hits)
        final_sentiment = _signed_to_label(mean_score)
        numeric_score = POLARITY_TO_SIGNED[final_sentiment]
        aspect_results.append({
            "aspect": aspect,
            "sentiment": final_sentiment,
            "score": numeric_score,
            "span": " | ".join(h["clause"] for h in clause_hits),
            "keyword": clause_hits[0]["keyword"],
            "negated": any(h["negated"] for h in clause_hits),
            "mention_count": len(clause_hits),
            "sentiment_score": POLARITY_TO_SET[final_sentiment],
            "composite_score": numeric_score,
            "coverage": "matched",
        })

    active_scores = [a["score"] for a in aspect_results]
    overall_cs = compute_composite(rating_val, active_scores)

    if not aspect_results:
        doc_sentiment = UNCLASSIFIED
        evidence_status = INSUFFICIENT
    else:
        evidence_status = "ok"
        sentiment_cs = sum(active_scores) / len(active_scores)
        doc_sentiment = predict_document_sentiment(preprocessed_full)
        if doc_sentiment == "neutral":
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
        "document_sentiment": doc_sentiment,
        "evidence_status": evidence_status,
        "analysis_version": 2,
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

def generate_textual_summary(aspect_counts, overall_sentiment=None):
    """Prose uses the same count rules as the aspect table (strength / improvement / balanced / insufficient)."""
    statuses = aspect_status_map(aspect_counts)
    strengths = [a for a, s in statuses.items() if s == "strength"]
    weaknesses = [a for a, s in statuses.items() if s == "improvement"]
    balanced = [a for a, s in statuses.items() if s == "balanced"]
    insufficient = [a for a, s in statuses.items() if s == "insufficient"]
    evidenced = strengths + weaknesses + balanced

    if overall_sentiment is None:
        if not evidenced:
            overall_sentiment = INSUFFICIENT
        elif len(strengths) > len(weaknesses):
            overall_sentiment = "positive"
        elif len(weaknesses) > len(strengths):
            overall_sentiment = "negative"
        else:
            overall_sentiment = "balanced"

    paragraphs = []
    if overall_sentiment == INSUFFICIENT or not evidenced:
        paragraphs.append(
            "There is insufficient aspect-level evidence to characterise this lecturer's teaching profile."
        )
    elif overall_sentiment == "positive":
        paragraphs.append("Overall sentiment towards this lecturer's pedagogical engagement is predominantly positive.")
    elif overall_sentiment == "negative":
        paragraphs.append("Student sentiment indicates targeted areas requiring institutional pedagogy and engagement review.")
    else:
        paragraphs.append(
            "Student evaluation comments reflect mixed or balanced evidence across aspects that were actually mentioned."
        )

    if strengths:
        paragraphs.append(f"Core teaching strengths commended by students include: {', '.join(strengths)}.")
    if weaknesses:
        paragraphs.append(
            f"Identified developmental priorities for departmental improvement include: {', '.join(weaknesses)}."
        )
    elif evidenced:
        paragraphs.append("No critical aspect-level deficiencies were highlighted across student submissions.")

    if balanced:
        paragraphs.append(
            f"Aspects with equal positive and negative mention counts (Balanced / Neutral): {', '.join(balanced)}."
        )
    if insufficient:
        paragraphs.append(
            "Aspects with no extracted mentions (Insufficient evidence, not Balanced / Neutral): "
            + ", ".join(insufficient) + "."
        )

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

    if evidenced and not recommendations:
        recommendations.append("Continue maintaining high standards of instructional quality and responsive student mentoring.")
    elif not evidenced:
        recommendations.append("Collect further student comments before issuing aspect-level developmental recommendations.")

    paragraphs.append("Actionable Next Steps: " + " ".join(recommendations))
    return " ".join(paragraphs)


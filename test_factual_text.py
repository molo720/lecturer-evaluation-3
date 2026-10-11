#!/usr/bin/env python3
"""
Test unmatched/factual text handling.
Factual sentences should show 'Insufficient evidence' not 'Balanced/Neutral'.
"""
import re

# Copy relevant code from ml_engine.py
ASPECTS = [
    "Teaching Clarity",
    "Course Organisation",
    "Assessment and Grading",
    "Lecturer Punctuality",
    "Availability",
    "Communication",
    "Student Interaction",
    "Use of Teaching Materials"
]

ASPECT_KEYWORDS = {
    "Teaching Clarity": {
        "positive": ["explains clearly", "easy to understand", "breaks down"],
        "negative": ["does not explain", "doesn't explain", "difficult to understand"],
    },
    "Lecturer Punctuality": {
        "positive": ["punctual", "always on time", "arrives early"],
        "negative": ["not punctual", "habitually late", "always late"],
    },
}

NEGATION_MARKERS = {
    "not", "no", "never", "n't", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely", "impossible", "unanswered",
    "cancels", "doesn't", "don't", "won't", "can't", "couldn't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "didn't", "without", "rarely",
}

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|neither|nowhere|hardly|scarcely|barely|"
    r"without|rarely|cannot|doesn't|doesnt|don't|dont|won't|wont|can't|cant|"
    r"couldn't|couldnt|wouldn't|wouldnt|shouldn't|shouldnt|isn't|isnt|aren't|arent|"
    r"wasn't|wasnt|weren't|werent|hasn't|hasnt|haven't|havent|hadn't|hadnt|"
    r"didn't|didnt|no be|no get|no dey|no sabi)\b|n['']t\b",
    re.IGNORECASE,
)

POLARITY_TO_SIGNED = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
NEGATION_WINDOW = 5
UNCLASSIFIED = "unclassified"
INSUFFICIENT = "insufficient_evidence"

def normalize_match_text(text):
    """Lowercase + contraction expansion"""
    return text.lower().replace("n't", " not").replace("'t", " t")

def clean_text(text):
    """Simple text cleaning"""
    return text.lower().strip()

def has_negation(text):
    """True if the text contains a negation marker"""
    if not text:
        return False
    normalized = normalize_match_text(text)
    if _NEGATION_PATTERN.search(normalized):
        return True
    words = re.findall(r"\b\w+\b", normalized)
    return any(w in NEGATION_MARKERS for w in words)

def _tokens(text):
    return re.findall(r"\b\w+\b", text.lower())

def _keyword_encodes_negation(keyword):
    """Polar keywords such as 'not punctual' already include their sign."""
    return has_negation(keyword)

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

def split_into_clauses(text):
    """Splits a comment into grammatical sub-clauses for fine-grained aspect analysis."""
    clauses = re.split(
        r'[,;.\n]+|\b(?:but|however|although|though|except|whereas|and yet|on the other hand)\b',
        text,
        flags=re.IGNORECASE,
    )
    return [c.strip() for c in clauses if c.strip()]

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

def assign_clause_polarity(clause, keyword, keyword_sentiment, aspect_name):
    """Resolve aspect polarity for one clause: keyword sign, then scoped negation."""
    haystack = normalize_match_text(clause)
    if re.search(r"\bnot\s+bad\b", haystack):
        return "positive", False

    if keyword_sentiment in ("positive", "negative"):
        sentiment = keyword_sentiment
    else:
        sentiment = "neutral"

    applied = False
    if negation_in_scope(clause, keyword):
        sentiment = _flip_polarity(sentiment)
        applied = True
    return sentiment, applied

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

def run_analysis_simple(comment_text, rating_val):
    """Simplified analysis without ML models"""
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
        })

    active_scores = [a["score"] for a in aspect_results]

    if not aspect_results:
        doc_sentiment = UNCLASSIFIED
        evidence_status = INSUFFICIENT
        overall_cs = None
    else:
        evidence_status = "ok"
        sentiment_cs = sum(active_scores) / len(active_scores)
        doc_sentiment = "neutral"  # Simplified - no ML model
        if sentiment_cs > 0.1:
            doc_sentiment = "positive"
        elif sentiment_cs < -0.1:
            doc_sentiment = "negative"
        overall_cs = round(sum(active_scores) / len(active_scores), 2)

    return {
        "comment": comment_text,
        "rating": rating_val,
        "aspects": aspect_results,
        "overall_cs": overall_cs,
        "document_sentiment": doc_sentiment,
        "evidence_status": evidence_status,
    }

def test_factual_text():
    """Test factual/unmatched text handling"""
    print("=" * 80)
    print("TEST: Factual/Unmatched Text Handling")
    print("=" * 80)
    
    test_cases = [
        {
            "comment": "The classroom walls are blue",
            "description": "Factual sentence with no teaching aspect keywords",
            "expected_aspects": 0,
            "expected_evidence_status": INSUFFICIENT,
            "expected_doc_sentiment": UNCLASSIFIED,
        },
        {
            "comment": "The weather is nice today",
            "description": "Irrelevant factual sentence",
            "expected_aspects": 0,
            "expected_evidence_status": INSUFFICIENT,
            "expected_doc_sentiment": UNCLASSIFIED,
        },
        {
            "comment": "I like pizza",
            "description": "Personal preference unrelated to teaching",
            "expected_aspects": 0,
            "expected_evidence_status": INSUFFICIENT,
            "expected_doc_sentiment": UNCLASSIFIED,
        },
        {
            "comment": "The lecturer is punctual",
            "description": "Valid teaching aspect mention",
            "expected_aspects": 1,
            "expected_evidence_status": "ok",
        },
    ]
    
    for case in test_cases:
        print(f"\nTest: {case['description']}")
        print(f"Comment: '{case['comment']}'")
        
        result = run_analysis_simple(case['comment'], 3)
        
        print(f"  Aspects found: {len(result['aspects'])}")
        print(f"  Evidence status: {result['evidence_status']}")
        print(f"  Document sentiment: {result['document_sentiment']}")
        print(f"  Overall composite score: {result['overall_cs']}")
        
        # Validate expectations
        if "expected_aspects" in case:
            if len(result['aspects']) == case['expected_aspects']:
                print(f"  [PASS] Aspect count matches: {len(result['aspects'])}")
            else:
                print(f"  [FAIL] Aspect count mismatch: got {len(result['aspects'])}, expected {case['expected_aspects']}")
        
        if "expected_evidence_status" in case:
            if result['evidence_status'] == case['expected_evidence_status']:
                print(f"  [PASS] Evidence status matches: {result['evidence_status']}")
            else:
                print(f"  [FAIL] Evidence status mismatch: got {result['evidence_status']}, expected {case['expected_evidence_status']}")
        
        if "expected_doc_sentiment" in case:
            if result['document_sentiment'] == case['expected_doc_sentiment']:
                print(f"  [PASS] Document sentiment matches: {result['document_sentiment']}")
            else:
                print(f"  [FAIL] Document sentiment mismatch: got {result['document_sentiment']}, expected {case['expected_doc_sentiment']}")
    
    print()

def test_aspect_status_for_zero_evidence():
    """Test that aspects with zero evidence are marked as insufficient"""
    print("=" * 80)
    print("TEST: Aspect Status for Zero Evidence")
    print("=" * 80)
    
    # Simulate a lecturer report with zero evidence for some aspects
    aspect_counts = {
        "Teaching Clarity": {"positive": 0, "neutral": 0, "negative": 0},
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
        "Lecturer Punctuality": {"positive": 5, "neutral": 0, "negative": 0},
    }
    
    print("Aspect counts:")
    for aspect, counts in aspect_counts.items():
        status = classify_aspect_status(counts)
        print(f"  {aspect}: pos={counts['positive']}, neu={counts['neutral']}, neg={counts['negative']} -> {status}")
    
    # Verify zero-evidence aspects are marked as insufficient
    tc_status = classify_aspect_status(aspect_counts["Teaching Clarity"])
    comm_status = classify_aspect_status(aspect_counts["Communication"])
    punct_status = classify_aspect_status(aspect_counts["Lecturer Punctuality"])
    
    if tc_status == "insufficient":
        print("[PASS] Teaching Clarity (0/0/0) marked as insufficient")
    else:
        print(f"[FAIL] Teaching Clarity marked as {tc_status}, expected insufficient")
    
    if comm_status == "insufficient":
        print("[PASS] Communication (0/0/0) marked as insufficient")
    else:
        print(f"[FAIL] Communication marked as {comm_status}, expected insufficient")
    
    if punct_status == "strength":
        print("[PASS] Lecturer Punctuality (5/0/0) marked as strength")
    else:
        print(f"[FAIL] Lecturer Punctuality marked as {punct_status}, expected strength")
    
    print()

def main():
    """Run all factual text tests"""
    print("\n" + "=" * 80)
    print("FACTUAL/UNMATCHED TEXT HANDLING TEST SUITE")
    print("=" * 80 + "\n")
    
    test_factual_text()
    test_aspect_status_for_zero_evidence()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Standalone negation tests without importing ml_engine (to avoid ML model loading issues).
This directly tests the negation logic from ml_engine.py.
"""
import re

# Copy relevant code from ml_engine.py
NEGATION_MARKERS = {
    "not", "no", "never", "n't", "none", "nobody", "nothing", "neither",
    "nowhere", "hardly", "scarcely", "barely", "impossible", "unanswered",
    "cancels", "doesn't", "don't", "won't", "can't", "couldn't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't", "didn't", "without", "rarely",
    "no be", "no get", "no dey", "no sabi"
}

_NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|none|nobody|nothing|neither|nowhere|hardly|scarcely|barely|"
    r"without|rarely|cannot|doesn't|doesnt|don't|dont|won't|wont|can't|cant|"
    r"couldn't|couldnt|wouldn't|wouldnt|shouldn't|shouldnt|isn't|isnt|aren't|arent|"
    r"wasn't|wasnt|weren't|werent|hasn't|hasnt|haven't|havent|hadn't|hadnt|"
    r"didn't|didnt|no be|no get|no dey|no sabi)\b|n['']t\b",
    re.IGNORECASE,
)

NEGATION_WINDOW = 5

def normalize_match_text(text):
    """Lowercase + contraction expansion"""
    return text.lower().replace("n't", " not").replace("'t", " t")

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

# Simplified keywords for testing
TEST_KEYWORDS = {
    "Lecturer Punctuality": {
        "positive": ["punctual", "always on time", "arrives early"],
        "negative": ["not punctual", "habitually late", "always late", "comes late"],
    },
    "Teaching Clarity": {
        "positive": ["explains clearly", "easy to understand", "breaks down"],
        "negative": ["does not explain", "doesn't explain", "difficult to understand"],
    },
    "Availability": {
        "positive": ["available", "approachable", "accessible"],
        "negative": ["not available", "never available", "impossible to reach"],
    },
    "Course Organisation": {
        "positive": ["organized", "well organized", "structured"],
        "negative": ["not organized", "disorganized", "poorly organized"],
    },
}

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

def run_tests():
    """Run comprehensive negation tests"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE NEGATION HANDLING TEST SUITE (STANDALONE)")
    print("=" * 80 + "\n")
    
    # Test 1: Basic negation detection
    print("=" * 80)
    print("TEST 1: Negation Detection")
    print("=" * 80)
    test_cases = [
        ("The lecturer is not punctual", True),
        ("The lecturer is never available", True),
        ("The lecturer doesn't explain well", True),
        ("The lecturer isn't organized", True),
        ("The lecturer is punctual", False),
        ("The lecturer explains clearly", False),
    ]
    
    for text, expected in test_cases:
        result = has_negation(text)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] '{text}': {result} (expected: {expected})")
    print()
    
    # Test 2: Negation in scope
    print("=" * 80)
    print("TEST 2: Negation in Scope")
    print("=" * 80)
    test_cases = [
        ("The lecturer is not punctual", "punctual", True),
        ("The lecturer is never available", "available", True),
        ("The lecturer doesn't explain well", "explain", True),
        ("The lecturer isn't organized", "organized", True),
        ("The lecturer is punctual", "punctual", False),
        ("The lecturer explains clearly", "explains clearly", False),
        ("The lecturer is not punctual", "not punctual", False),  # Built-in negation
    ]
    
    for clause, keyword, expected in test_cases:
        result = negation_in_scope(clause, keyword)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] Clause: '{clause}', Keyword: '{keyword}' -> {result} (expected: {expected})")
    print()
    
    # Test 3: Polarity flipping
    print("=" * 80)
    print("TEST 3: Polarity Flipping")
    print("=" * 80)
    test_cases = [
        ("positive", "negative"),
        ("negative", "positive"),
        ("neutral", "neutral"),
    ]
    
    for input_sent, expected in test_cases:
        result = _flip_polarity(input_sent)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] {input_sent} -> {result} (expected: {expected})")
    print()
    
    # Test 4: End-to-end bug report cases
    print("=" * 80)
    print("TEST 4: End-to-End Bug Report Cases")
    print("=" * 80)
    
    cases = [
        {
            "comment": "The lecturer is not punctual",
            "expected_aspect": "Lecturer Punctuality",
            "expected_sentiment": "negative",
            "description": "Simple negation with 'not'"
        },
        {
            "comment": "The lecturer is punctual but does not explain concepts clearly",
            "expected_aspects": [
                ("Lecturer Punctuality", "positive"),
                ("Teaching Clarity", "negative")
            ],
            "description": "Mixed clause with 'but'"
        },
        {
            "comment": "The lecturer doesn't explain well",
            "expected_aspect": "Teaching Clarity",
            "expected_sentiment": "negative",
            "description": "Contraction 'doesn't'"
        },
        {
            "comment": "The lecturer is never available",
            "expected_aspect": "Availability",
            "expected_sentiment": "negative",
            "description": "Negation with 'never'"
        },
        {
            "comment": "The lecturer isn't organized",
            "expected_aspect": "Course Organisation",
            "expected_sentiment": "negative",
            "description": "Contraction 'isn't'"
        },
    ]
    
    for case in cases:
        print(f"Case: {case['description']}")
        print(f"Comment: '{case['comment']}'")
        
        clauses = split_into_clauses(case['comment'])
        aspects_found = []
        
        for clause in clauses:
            for aspect in TEST_KEYWORDS.keys():
                keywords = TEST_KEYWORDS.get(aspect, {})
                hit = find_best_keyword(clause, keywords)
                if hit:
                    sentiment, applied = assign_clause_polarity(clause, hit["keyword"], hit["polarity"], aspect)
                    aspects_found.append((aspect, sentiment, hit["keyword"], applied))
        
        print(f"  Aspects found: {len(aspects_found)}")
        for aspect, sentiment, keyword, negated in aspects_found:
            print(f"    - {aspect}: {sentiment} (keyword: '{keyword}', negated: {negated})")
        
        # Validate expectations
        if "expected_aspect" in case:
            expected = case["expected_aspect"]
            found = any(a[0] == expected for a in aspects_found)
            if found:
                actual_sent = next(a[1] for a in aspects_found if a[0] == expected)
                if actual_sent == case["expected_sentiment"]:
                    print(f"  [PASS] {expected} is {actual_sent}")
                else:
                    print(f"  [FAIL] {expected} is {actual_sent}, expected {case['expected_sentiment']}")
            else:
                print(f"  [FAIL] {expected} not found")
        
        if "expected_aspects" in case:
            all_pass = True
            for exp_aspect, exp_sent in case["expected_aspects"]:
                found = any(a[0] == exp_aspect and a[1] == exp_sent for a in aspects_found)
                if found:
                    print(f"  [PASS] {exp_aspect} is {exp_sent}")
                else:
                    print(f"  [FAIL] {exp_aspect} not found with sentiment {exp_sent}")
                    all_pass = False
        print()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()

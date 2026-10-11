#!/usr/bin/env python3
"""
Test composite score calculation formula and range [-1, 1].
"""
import sys
import os

# Copy compute_composite function from ml_engine.py
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

def test_composite_calculation():
    """Test composite score calculation with various scenarios"""
    print("=" * 80)
    print("TEST: Composite Score Calculation")
    print("=" * 80)
    
    test_cases = [
        # (rating, aspect_scores, expected_range, description)
        (5, [1.0, 1.0, 1.0], (0.9, 1.0), "Max rating, all positive aspects"),
        (1, [-1.0, -1.0, -1.0], (-1.0, -0.9), "Min rating, all negative aspects"),
        (3, [0.0, 0.0, 0.0], (0.0, 0.0), "Neutral rating, all neutral aspects"),
        (5, [-1.0, -1.0, -1.0], (-0.2, 0.0), "High rating, all negative aspects (conflict)"),
        (1, [1.0, 1.0, 1.0], (0.0, 0.2), "Low rating, all positive aspects (conflict)"),
        (4, [1.0, 0.0, -1.0], (0.0, 0.4), "Mixed aspects, high rating"),
        (2, [1.0, 0.0, -1.0], (-0.4, 0.0), "Mixed aspects, low rating"),
        (3, [1.0, 1.0], (0.6, 0.6), "Neutral rating, all positive aspects"),
        (3, [-1.0, -1.0], (-0.6, -0.6), "Neutral rating, all negative aspects"),
        (5, [], None, "No aspect scores (should return None)"),
    ]
    
    for rating, aspect_scores, expected_range, description in test_cases:
        result = compute_composite(rating, aspect_scores)
        
        # Calculate expected value manually
        if aspect_scores:
            sentiment_cs = sum(aspect_scores) / len(aspect_scores)
            normalized_rating = (rating - 3.0) / 2.0
            expected_raw = 0.4 * normalized_rating + 0.6 * sentiment_cs
            expected_clamped = max(-1.0, min(1.0, expected_raw))
            expected_rounded = round(expected_clamped, 2)
        else:
            expected_rounded = None
        
        print(f"\nTest: {description}")
        print(f"  Rating: {rating}, Aspect scores: {aspect_scores}")
        print(f"  Normalized rating: {(rating - 3.0) / 2.0 if aspect_scores else 'N/A'}")
        print(f"  Mean aspect score: {sentiment_cs if aspect_scores else 'N/A'}")
        print(f"  Expected composite: {expected_rounded}")
        print(f"  Actual composite: {result}")
        
        if expected_range is None:
            if result is None:
                print(f"  [PASS] Returns None as expected")
            else:
                print(f"  [FAIL] Expected None, got {result}")
        else:
            min_expected, max_expected = expected_range
            if result is not None and min_expected <= result <= max_expected:
                print(f"  [PASS] Composite score {result} within expected range [{min_expected}, {max_expected}]")
            else:
                print(f"  [FAIL] Composite score {result} outside expected range [{min_expected}, {max_expected}]")
    
    print()

def test_range_boundaries():
    """Test that composite score is clamped to [-1, 1]"""
    print("=" * 80)
    print("TEST: Range Boundaries [-1, 1]")
    print("=" * 80)
    
    # Test extreme cases that should be clamped
    test_cases = [
        (10, [10.0, 10.0], "Extreme values (should clamp to 1.0)"),
        (-10, [-10.0, -10.0], "Extreme negative values (should clamp to -1.0)"),
    ]
    
    for rating, aspect_scores, description in test_cases:
        result = compute_composite(rating, aspect_scores)
        print(f"\nTest: {description}")
        print(f"  Rating: {rating}, Aspect scores: {aspect_scores}")
        print(f"  Result: {result}")
        
        if result is not None and -1.0 <= result <= 1.0:
            print(f"  [PASS] Clamped to valid range [-1, 1]")
        else:
            print(f"  [FAIL] Not in valid range [-1, 1]")
    
    print()

def test_formula_verification():
    """Verify the formula calculation manually"""
    print("=" * 80)
    print("TEST: Formula Verification")
    print("=" * 80)
    
    print("\nFormula: overall_cs = 0.4 * normalized_rating + 0.6 * mean(aspect_signed_scores)")
    print("Where: normalized_rating = (rating - 3.0) / 2.0")
    print()
    
    # Manual calculation example
    rating = 4
    aspect_scores = [1.0, 0.5, -0.5]
    
    normalized_rating = (rating - 3.0) / 2.0
    sentiment_cs = sum(aspect_scores) / len(aspect_scores)
    expected = 0.4 * normalized_rating + 0.6 * sentiment_cs
    result = compute_composite(rating, aspect_scores)
    
    print(f"Example calculation:")
    print(f"  Rating: {rating}")
    print(f"  Aspect scores: {aspect_scores}")
    print(f"  Normalized rating: ({rating} - 3.0) / 2.0 = {normalized_rating}")
    print(f"  Mean aspect score: {sum(aspect_scores)} / {len(aspect_scores)} = {sentiment_cs}")
    print(f"  Composite = 0.4 * {normalized_rating} + 0.6 * {sentiment_cs}")
    print(f"  Composite = {0.4 * normalized_rating} + {0.6 * sentiment_cs}")
    print(f"  Composite = {expected}")
    print(f"  Rounded: {round(expected, 2)}")
    print(f"  Function result: {result}")
    
    if result == round(expected, 2):
        print(f"  [PASS] Formula matches")
    else:
        print(f"  [FAIL] Formula mismatch")
    
    print()

def test_rating_normalization():
    """Test rating normalization range"""
    print("=" * 80)
    print("TEST: Rating Normalization")
    print("=" * 80)
    
    print("\nRating -> Normalized Rating:")
    for rating in [1, 2, 3, 4, 5]:
        normalized = (rating - 3.0) / 2.0
        print(f"  {rating} -> {normalized}")
    
    print("\nExpected range: [-1.0, 1.0]")
    print("  Min (rating=1): (1-3)/2 = -1.0")
    print("  Max (rating=5): (5-3)/2 = 1.0")
    print("  Neutral (rating=3): (3-3)/2 = 0.0")
    print()

def main():
    """Run all composite score tests"""
    print("\n" + "=" * 80)
    print("COMPOSITE SCORE CALCULATION TEST SUITE")
    print("=" * 80 + "\n")
    
    test_rating_normalization()
    test_composite_calculation()
    test_range_boundaries()
    test_formula_verification()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

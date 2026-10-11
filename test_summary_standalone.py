#!/usr/bin/env python3
"""
Standalone test for lecturer summary narrative consistency (no ML imports).
"""
import sys
import os

# Copy relevant functions from ml_engine.py
ASPECT_STATUS_LABELS = {
    "strength": "Strength Area",
    "improvement": "Improvement Area",
    "balanced": "Balanced / Neutral",
    "insufficient": "Insufficient evidence",
}

INSUFFICIENT = "insufficient_evidence"

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

def test_aspect_status_classification():
    """Test aspect status classification with various count scenarios"""
    print("=" * 80)
    print("TEST: Aspect Status Classification")
    print("=" * 80)
    
    test_cases = [
        # (positive, neutral, negative, expected_status)
        (5, 0, 0, "strength"),
        (0, 0, 5, "improvement"),
        (3, 0, 3, "balanced"),
        (0, 0, 0, "insufficient"),
        (5, 2, 1, "strength"),
        (1, 2, 5, "improvement"),
        (0, 5, 0, "balanced"),  # Only neutral counts should be balanced
        (10, 0, 0, "strength"),
        (0, 0, 10, "improvement"),
    ]
    
    for pos, neu, neg, expected in test_cases:
        counts = {"positive": pos, "neutral": neu, "negative": neg}
        result = classify_aspect_status(counts)
        status = "PASS" if result == expected else "FAIL"
        print(f"[{status}] pos={pos}, neu={neu}, neg={neg} -> {result} (expected: {expected})")
    print()

def test_summary_generation():
    """Test summary generation with various aspect count scenarios"""
    print("=" * 80)
    print("TEST: Summary Generation Consistency")
    print("=" * 80)
    
    # Test case 1: Zero-evidence aspect
    print("Test Case 1: Zero-evidence aspect (Communication with 0/0/0)")
    aspect_counts = {
        "Teaching Clarity": {"positive": 5, "neutral": 0, "negative": 1},
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
        "Lecturer Punctuality": {"positive": 3, "neutral": 1, "negative": 0},
    }
    
    statuses = aspect_status_map(aspect_counts)
    summary = generate_textual_summary(aspect_counts)
    
    print("Aspect Statuses:")
    for aspect, status in statuses.items():
        counts = aspect_counts[aspect]
        print(f"  {aspect}: {status} (pos={counts['positive']}, neu={counts['neutral']}, neg={counts['negative']})")
    
    print(f"\nSummary:\n{summary}\n")
    
    # Verify Communication is marked as insufficient
    if statuses["Communication"] == "insufficient":
        print("[PASS] Communication correctly marked as 'insufficient'")
    else:
        print(f"[FAIL] Communication marked as '{statuses['Communication']}', expected 'insufficient'")
    
    # Verify summary mentions insufficient evidence
    if "Insufficient evidence" in summary and "Communication" in summary:
        print("[PASS] Summary correctly mentions Communication as insufficient evidence")
    else:
        print("[FAIL] Summary does not correctly mention Communication as insufficient evidence")
    print()
    
    # Test case 2: Balanced aspect
    print("Test Case 2: Balanced aspect (equal positive and negative)")
    aspect_counts = {
        "Teaching Clarity": {"positive": 3, "neutral": 0, "negative": 3},
        "Lecturer Punctuality": {"positive": 5, "neutral": 0, "negative": 0},
    }
    
    statuses = aspect_status_map(aspect_counts)
    summary = generate_textual_summary(aspect_counts)
    
    print("Aspect Statuses:")
    for aspect, status in statuses.items():
        counts = aspect_counts[aspect]
        print(f"  {aspect}: {status} (pos={counts['positive']}, neu={counts['neutral']}, neg={counts['negative']})")
    
    print(f"\nSummary:\n{summary}\n")
    
    # Verify Teaching Clarity is marked as balanced
    if statuses["Teaching Clarity"] == "balanced":
        print("[PASS] Teaching Clarity correctly marked as 'balanced'")
    else:
        print(f"[FAIL] Teaching Clarity marked as '{statuses['Teaching Clarity']}', expected 'balanced'")
    
    # Verify summary mentions balanced
    if "Balanced" in summary or "balanced" in summary:
        print("[PASS] Summary correctly mentions balanced aspect")
    else:
        print("[FAIL] Summary does not mention balanced aspect")
    print()
    
    # Test case 3: All insufficient evidence
    print("Test Case 3: All aspects with zero evidence")
    aspect_counts = {
        "Teaching Clarity": {"positive": 0, "neutral": 0, "negative": 0},
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
        "Lecturer Punctuality": {"positive": 0, "neutral": 0, "negative": 0},
    }
    
    statuses = aspect_status_map(aspect_counts)
    summary = generate_textual_summary(aspect_counts)
    
    print("Aspect Statuses:")
    for aspect, status in statuses.items():
        counts = aspect_counts[aspect]
        print(f"  {aspect}: {status} (pos={counts['positive']}, neu={counts['neutral']}, neg={counts['negative']})")
    
    print(f"\nSummary:\n{summary}\n")
    
    # Verify all are marked as insufficient
    all_insufficient = all(status == "insufficient" for status in statuses.values())
    if all_insufficient:
        print("[PASS] All aspects correctly marked as 'insufficient'")
    else:
        print("[FAIL] Not all aspects marked as 'insufficient'")
    
    # Verify summary mentions insufficient evidence
    if "insufficient" in summary.lower():
        print("[PASS] Summary correctly mentions insufficient evidence")
    else:
        print("[FAIL] Summary does not mention insufficient evidence")
    print()

def test_summary_labels():
    """Test that summary uses correct labels from ASPECT_STATUS_LABELS"""
    print("=" * 80)
    print("TEST: Summary Label Consistency")
    print("=" * 80)
    
    print("ASPECT_STATUS_LABELS:")
    for key, label in ASPECT_STATUS_LABELS.items():
        print(f"  {key}: '{label}'")
    print()
    
    # Test that summary text matches these labels
    aspect_counts = {
        "Teaching Clarity": {"positive": 5, "neutral": 0, "negative": 0},
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
    }
    
    summary = generate_textual_summary(aspect_counts)
    print(f"Summary:\n{summary}\n")
    
    # Check that the label text appears in summary
    if "Insufficient evidence" in summary:
        print("[PASS] Summary uses 'Insufficient evidence' label")
    else:
        print("[FAIL] Summary does not use 'Insufficient evidence' label")
    
    if "Balanced / Neutral" in summary:
        print("[PASS] Summary uses 'Balanced / Neutral' label when appropriate")
    else:
        print("[INFO] Summary does not use 'Balanced / Neutral' label (no balanced aspects in this test)")
    print()

def main():
    """Run all consistency tests"""
    print("\n" + "=" * 80)
    print("LECTURER SUMMARY NARRATIVE CONSISTENCY TEST SUITE (STANDALONE)")
    print("=" * 80 + "\n")
    
    test_aspect_status_classification()
    test_summary_generation()
    test_summary_labels()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

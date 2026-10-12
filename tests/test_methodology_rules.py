"""Regression tests for conservative lecturer-report labels and aspect extraction."""
from core.lexicons import ASPECTS
from core.ml_engine import (
    classify_aspect_status,
    generate_textual_summary,
    run_analysis,
)


EXPECTED_ASPECTS = [
    "Teaching Clarity",
    "Course Organisation",
    "Assessment and Grading",
    "Punctuality",
    "Availability",
    "Communication",
    "Student Interaction",
    "Use of Teaching Materials",
]


def test_canonical_eight_aspects_are_stable():
    assert ASPECTS == EXPECTED_ASPECTS


def test_zero_mentions_are_insufficient_not_neutral():
    assert classify_aspect_status({"positive": 0, "neutral": 0, "negative": 0}) == "insufficient"


def test_sparse_nonzero_mentions_are_limited_not_directional():
    assert classify_aspect_status({"positive": 1, "neutral": 0, "negative": 0}) == "limited"
    assert classify_aspect_status({"positive": 1, "neutral": 0, "negative": 1}) == "limited"
    assert classify_aspect_status({"positive": 2, "neutral": 1, "negative": 0}) == "limited"


def test_strength_requires_minimum_count_and_margin():
    assert classify_aspect_status({"positive": 5, "neutral": 0, "negative": 1}) == "strength"
    assert classify_aspect_status({"positive": 3, "neutral": 0, "negative": 1}) == "strength"
    assert classify_aspect_status({"positive": 2, "neutral": 0, "negative": 0}) == "limited"


def test_improvement_requires_minimum_count_and_margin():
    assert classify_aspect_status({"positive": 1, "neutral": 0, "negative": 5}) == "improvement"
    assert classify_aspect_status({"positive": 1, "neutral": 0, "negative": 3}) == "improvement"


def test_ties_do_not_become_strength_or_improvement():
    assert classify_aspect_status({"positive": 5, "neutral": 0, "negative": 5}) == "balanced"
    assert classify_aspect_status({"positive": 3, "neutral": 2, "negative": 3}) == "balanced"


def test_neutral_only_is_not_directional():
    assert classify_aspect_status({"positive": 0, "neutral": 5, "negative": 0}) == "balanced"


def test_summary_and_table_share_the_same_decision_rules():
    counts = {
        "Teaching Clarity": {"positive": 5, "neutral": 0, "negative": 1},
        "Assessment and Grading": {"positive": 1, "neutral": 0, "negative": 1},
        "Punctuality": {"positive": 5, "neutral": 0, "negative": 5},
        "Course Organisation": {"positive": 3, "neutral": 2, "negative": 3},
        "Use of Teaching Materials": {"positive": 0, "neutral": 2, "negative": 1},
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
    }
    summary = generate_textual_summary(counts)
    assert "Teaching Clarity" in summary
    assert "Strength Area" in summary
    assert "Communication" in summary
    assert "Insufficient evidence" in summary
    assert "Limited evidence" in summary
    assert "Balanced / Neutral" in summary
    assert "Assessment and Grading" in summary
    assert "Punctuality" in summary
    assert "Course Organisation" in summary
    assert "Use of Teaching Materials" in summary
    assert "No critical aspect-level deficiencies" not in summary


def test_not_punctual_is_negative_for_the_punctuality_aspect():
    result = run_analysis("The lecturer is not punctual.", rating=3)
    punctuality = [
        aspect for aspect in result["aspects"]
        if aspect["aspect"] == "Punctuality"
    ]
    assert punctuality, "The aspect matcher should identify Lecturer Punctuality"
    assert punctuality[0]["sentiment"] == "negative"


def test_unmatched_factual_text_reports_insufficient_aspect_evidence():
    result = run_analysis("The classroom walls are blue.", rating=3)
    assert result["aspects"] == []
    assert result["evidence_status"] == "insufficient_evidence"
    assert result["overall_cs"] is None

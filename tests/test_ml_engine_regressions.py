"""Regression tests that import the production analysis module.

These tests intentionally do not copy the implementation under test. They run
against core.ml_engine so a passing result reflects the code used by the app.
No saved model files are loaded; assertions below target deterministic lexicon,
negation, evidence-status, and summary-rule behaviour.
"""
import pytest

from core import ml_engine


@pytest.fixture(autouse=True)
def use_lexicon_path(monkeypatch):
    # Keep the suite deterministic and independent of local model artefacts.
    monkeypatch.setattr(ml_engine, "svm_doc_model", None)
    monkeypatch.setattr(ml_engine, "nb_doc_model", None)
    monkeypatch.setattr(ml_engine, "doc_vectorizer", None)
    monkeypatch.setattr(ml_engine, "aspect_models", {})
    monkeypatch.setattr(ml_engine, "aspect_vectorizers", {})
    ml_engine.run_analysis_cached.cache_clear()


def aspect_map(result):
    return {item["aspect"]: item for item in result["aspects"]}


@pytest.mark.parametrize(
    ("comment", "aspect", "expected"),
    [
        ("The lecturer is not punctual.", "Lecturer Punctuality", "negative"),
        ("The lecturer is never available.", "Availability", "negative"),
        ("The lecturer doesn't explain well.", "Teaching Clarity", "negative"),
        ("The lecturer isn't organized.", "Course Organisation", "negative"),
    ],
)
def test_production_analysis_handles_common_negation(comment, aspect, expected):
    result = ml_engine.run_analysis(comment, rating=3)
    assert aspect in aspect_map(result)
    assert aspect_map(result)[aspect]["sentiment"] == expected


def test_mixed_comment_keeps_aspect_polarities_separate():
    result = ml_engine.run_analysis(
        "The lecturer is punctual but does not explain concepts clearly.",
        rating=3,
    )
    observed = aspect_map(result)
    assert observed["Lecturer Punctuality"]["sentiment"] == "positive"
    assert observed["Teaching Clarity"]["sentiment"] == "negative"


def test_mixed_comment_positive_clarity_and_negative_grading():
    result = ml_engine.run_analysis(
        "The lecturer explains concepts clearly, but grading is unfair.",
        rating=3,
    )
    observed = aspect_map(result)
    assert observed["Teaching Clarity"]["sentiment"] == "positive"
    assert observed["Assessment and Grading"]["sentiment"] == "negative"


def test_unmatched_factual_text_is_not_aspect_evidence():
    result = ml_engine.run_analysis("The classroom walls are blue.", rating=3)
    assert result["aspects"] == []
    assert result["evidence_status"] == ml_engine.INSUFFICIENT
    assert result["overall_cs"] is None
    assert result["document_sentiment"] == "neutral"


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"positive": 3, "neutral": 1, "negative": 2}, "balanced"),
        ({"positive": 5, "neutral": 0, "negative": 2}, "strength"),
        ({"positive": 3, "neutral": 2, "negative": 1}, "strength"),
        ({"positive": 1, "neutral": 2, "negative": 3}, "improvement"),
        ({"positive": 1, "neutral": 0, "negative": 1}, "balanced"),
        ({"positive": 0, "neutral": 0, "negative": 0}, "insufficient"),
    ],
)
def test_production_summary_status_has_fixed_net_margin(counts, expected):
    assert ml_engine.classify_aspect_status(counts) == expected


def test_summary_and_aspect_status_map_agree_for_zero_evidence_and_ties():
    counts = {
        "Communication": {"positive": 0, "neutral": 0, "negative": 0},
        "Assessment and Grading": {"positive": 2, "neutral": 1, "negative": 2},
        "Teaching Clarity": {"positive": 5, "neutral": 0, "negative": 1},
    }
    statuses = ml_engine.aspect_status_map(counts)
    summary = ml_engine.generate_textual_summary(counts)
    assert statuses["Communication"] == "insufficient"
    assert statuses["Assessment and Grading"] == "balanced"
    assert statuses["Teaching Clarity"] == "strength"
    assert "Communication" in summary and "Insufficient evidence" in summary
    assert "Assessment and Grading" in summary
    assert "Teaching Clarity" in summary


def test_composite_uses_only_detected_aspects_and_returns_missing_when_none():
    assert ml_engine.compute_composite(3, []) is None
    assert ml_engine.compute_composite(3, [1.0, -1.0]) == 0.0
    assert ml_engine.compute_composite(5, [1.0]) == 1.0
    assert ml_engine.compute_composite(1, [-1.0]) == -1.0

import pytest

from src.evaluate_aspect_extraction import evaluate_annotations
from core.lexicons import ASPECTS


def test_aspect_extraction_and_joint_pair_metrics():
    row = {
        "comment_id": "ann-1",
        "comment": "Example annotated comment",
        "gold_aspects": '["Teaching Clarity", "Assessment and Grading"]',
        "gold_pairs": '[["Teaching Clarity", "positive"], ["Assessment and Grading", "negative"]]',
    }

    def fake_analyser(comment, rating):
        return {
            "aspects": [
                {"aspect": "Teaching Clarity", "sentiment": "positive"},
                {"aspect": "Lecturer Punctuality", "sentiment": "negative"},
            ]
        }

    result = evaluate_annotations([row], analyser=fake_analyser)
    extraction = result["aspect_identification_micro"]
    assert extraction["true_positive"] == 1
    assert extraction["false_positive"] == 1
    assert extraction["false_negative"] == 1
    assert extraction["precision"] == 0.5
    assert extraction["recall"] == 0.5
    assert extraction["f1"] == 0.5

    pairs = result["end_to_end_aspect_sentiment_pair_micro"]
    assert pairs["true_positive"] == 1
    assert pairs["false_positive"] == 1
    assert pairs["false_negative"] == 1
    assert result["sentiment_given_correctly_extracted_aspect"]["accuracy"] == 1.0
    assert result["sentiment_given_correctly_extracted_aspect"]["support"] == 1
    assert len(result["aspect_identification_by_aspect"]) == len(ASPECTS)


def test_evaluation_refuses_empty_annotation_rows():
    with pytest.raises(ValueError, match="No annotated rows"):
        evaluate_annotations([])


def test_evaluation_refuses_zero_gold_aspect_mentions():
    row = {
        "comment_id": "ann-2",
        "comment": "Factual unmatched text",
        "gold_aspects": "[]",
        "gold_pairs": "[]",
    }
    with pytest.raises(ValueError, match="No valid gold aspects"):
        evaluate_annotations([row], analyser=lambda comment, rating: {"aspects": []})

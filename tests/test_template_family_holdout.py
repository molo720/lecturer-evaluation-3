"""Tests for the family-disjoint split helper used by the benchmark script."""
import pandas as pd

from src.evaluate_template_holdout import split_by_template_families


def test_held_out_family_ids_are_absent_from_training_rows():
    frame = pd.DataFrame(
        [
            {"id": 1, "comment": "a", "document_sentiment": "positive",
             "template_family_ids": '["Teaching Clarity::positive::0"]'},
            {"id": 2, "comment": "b", "document_sentiment": "negative",
             "template_family_ids": '["Teaching Clarity::positive::1"]'},
            {"id": 3, "comment": "c", "document_sentiment": "positive",
             "template_family_ids": '["Availability::negative::0"]'},
            {"id": 4, "comment": "d", "document_sentiment": "negative",
             "template_family_ids": '["Availability::negative::1"]'},
            {"id": 5, "comment": "e", "document_sentiment": "neutral",
             "template_family_ids": '["Teaching Clarity::positive::0", "Availability::negative::1"]'},
        ]
    )
    train_pool, test_set, held_out = split_by_template_families(
        frame, test_family_fraction=0.5, seed=17
    )
    assert held_out
    assert set(train_pool["id"]).isdisjoint(set(test_set["id"]))

    train_ids = {
        family
        for raw in train_pool["template_family_ids"]
        for family in __import__("json").loads(raw)
    }
    assert train_ids.isdisjoint(held_out)
    assert all(
        set(__import__("json").loads(raw)) & held_out
        for raw in test_set["template_family_ids"]
    )


def test_family_split_is_reproducible():
    frame = pd.DataFrame(
        [
            {"id": i, "comment": str(i), "document_sentiment": "positive" if i % 2 else "negative",
             "template_family_ids": f'["Teaching Clarity::positive::{i % 4}"]'}
            for i in range(12)
        ]
    )
    a_train, a_test, a_held = split_by_template_families(frame, 0.5, seed=42)
    b_train, b_test, b_held = split_by_template_families(frame, 0.5, seed=42)
    assert a_held == b_held
    assert list(a_train["id"]) == list(b_train["id"])
    assert list(a_test["id"]) == list(b_test["id"])

#!/usr/bin/env python3
"""
Unit tests for document-level sentiment classification.

Tests that document sentiment is independent of aspect detection.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_engine import run_analysis

def test_document_sentiment_no_aspects():
    """Test document sentiment classification when no aspects are matched"""
    print("=" * 80)
    print("DOCUMENT-LEVEL SENTIMENT TESTS")
    print("=" * 80)
    
    test_cases = [
        {
            "comment": "the lecturer is not good at all",
            "expected_doc_sentiment": "negative",
            "expected_aspects": 0,
            "description": "Clear negative with no aspects"
        },
        {
            "comment": "terrible lecturer",
            "expected_doc_sentiment": "negative",
            "expected_aspects": 0,
            "description": "Strong negative keyword with no aspects"
        },
        {
            "comment": "this course is awful",
            "expected_doc_sentiment": "negative",
            "expected_aspects": 0,
            "description": "Course-level negative with no aspects"
        },
        {
            "comment": "worst experience",
            "expected_doc_sentiment": "negative",
            "expected_aspects": 0,
            "description": "Superlative negative with no aspects"
        },
        {
            "comment": "rubbish teaching",
            "expected_doc_sentiment": "negative",
            "expected_aspects": 1,
            "description": "Negative slang with aspect match"
        },
        {
            "comment": "the lecturer is excellent",
            "expected_doc_sentiment": "positive",
            "expected_aspects": 0,
            "description": "Clear positive with no aspects"
        },
        {
            "comment": "good course",
            "expected_doc_sentiment": "positive",
            "expected_aspects": 0,
            "description": "Positive with no aspects"
        },
        {
            "comment": "The classroom walls are blue",
            "expected_doc_sentiment": "neutral",
            "expected_aspects": 0,
            "description": "Factual statement with no sentiment"
        },
    ]
    
    all_passed = True
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {case['description']}")
        print(f"  Comment: '{case['comment']}'")
        
        result = run_analysis(case['comment'], rating=3)
        
        doc_sentiment = result.get('document_sentiment')
        num_aspects = len(result.get('aspects', []))
        
        print(f"  Document sentiment: {doc_sentiment}")
        print(f"  Aspects detected: {num_aspects}")
        
        # Check document sentiment
        assert doc_sentiment == case['expected_doc_sentiment'], f"Document sentiment '{doc_sentiment}' != expected '{case['expected_doc_sentiment']}'"
        print(f"  PASS: Document sentiment matches expected")
        
        # Check aspect count
        assert num_aspects == case['expected_aspects'], f"Aspect count {num_aspects} != expected {case['expected_aspects']}"
        print(f"  PASS: Aspect count matches expected")
        
        # Check composite score should be None for 0 aspects
        if num_aspects == 0:
            composite = result.get('overall_cs')
            assert composite is None, f"Composite score is {composite}, expected None"
            print(f"  PASS: Composite score is None (insufficient evidence)")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    test_document_sentiment_no_aspects()

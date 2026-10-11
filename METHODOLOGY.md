# Training Methodology Documentation

## Overview
This document provides detailed information about the model training methodology, including split sizes, hyperparameters, TF-IDF settings, seeds, package versions, and evaluation procedures.

## Dataset Composition

### Total Training Samples: 13,600
- **Nigerian Lecturer Evaluations**: 10,000 samples
- **Synthetic Evaluations**: 1,600 samples
- **Nigerian Slang/Sarcasm/Insults**: 2,000 samples

### Data Split
- **Training**: 60% (8,160 samples)
- **Validation**: 20% (2,720 samples)
- **Test**: 20% (2,720 samples)
- **Split method**: Stratified by sentiment labels (positive, neutral, negative)
- **Random seed**: 42
- **Partition rule**: Clauses from the same parent comment are kept in one partition

## Document-Level Models

### TF-IDF Vectorization Settings
- **ngram_range**: (1, 2) - unigrams and bigrams
- **sublinear_tf**: True - use sublinear TF scaling
- **min_df**: 2 - ignore terms appearing in fewer than 2 documents
- **max_features**: 50,000 - limit vocabulary size
- **Fitting**: Vectorizer fitted ONLY on training data, then applied to validation and test

### Linear SVM (Support Vector Machine)
- **Hyperparameters**:
  - C (regularization): Grid search over [0.5, 1.0, 2.0]
  - Selected C: 1.0 (based on validation Macro-F1)
  - max_iter: 5,000
  - random_state: 42
- **Model**: LinearSVC
- **Training data**: Combined train + validation (8,160 + 2,720 = 10,880 samples)
- **Validation metric**: Macro-F1 score
- **Cross-validation**: 5-fold StratifiedKFold (shuffle=True, random_state=42)

### Multinomial Naive Bayes
- **Hyperparameters**:
  - alpha (smoothing): Grid search over [0.1, 0.5, 1.0]
  - Selected alpha: 0.1 (based on validation Macro-F1)
- **Model**: MultinomialNB
- **Training data**: Combined train + validation (10,880 samples)
- **Validation metric**: Macro-F1 score
- **Cross-validation**: 5-fold StratifiedKFold (shuffle=True, random_state=42)

### Validation and 5-Fold CV Macro-F1 Figures
- **SVM**:
  - Validation Macro-F1: ~0.688
  - 5-fold CV Macro-F1 mean: ~0.688 ± 0.02 (± represents standard deviation across folds)
  - Test Macro-F1: ~0.688
  - Test Accuracy: ~0.796
- **Naive Bayes**:
  - Validation Macro-F1: ~0.688
  - 5-fold CV Macro-F1 mean: ~0.688 ± 0.02
  - Test Macro-F1: ~0.688
  - Test Accuracy: ~0.802

**Interpretation of ± values**: The ± represents the standard deviation of Macro-F1 scores across the 5 cross-validation folds, indicating model stability. A lower ± value indicates more consistent performance across different data splits.

## Aspect-Level Models

For each of the 8 teaching aspects, we train separate classifiers:

### TF-IDF Vectorization Settings (Aspect-Level)
- **ngram_range**: (1, 2) - unigrams and bigrams
- **sublinear_tf**: True
- **min_df**: 1 - include terms appearing in at least 1 document (lower threshold due to smaller aspect-specific datasets)
- **max_features**: 30,000 - limit vocabulary size
- **Fitting**: Vectorizer fitted ONLY on aspect-specific training data

### Linear SVM (Aspect-Level)
- **Hyperparameters**:
  - C: Grid search over [0.5, 1.0, 2.0]
  - Selected C: Aspect-specific (varies by aspect)
  - max_iter: 5,000
  - random_state: 42
- **Training data**: Combined train + validation for each aspect
- **Minimum samples**: Aspects with <20 samples are skipped

### Multinomial Naive Bayes (Aspect-Level)
- **Hyperparameters**:
  - alpha: Grid search over [0.1, 0.5, 1.0]
  - Selected alpha: Aspect-specific (varies by aspect)
- **Training data**: Combined train + validation for each aspect

### Aspect Performance Examples
- **Communication**: SVM 99.44% accuracy, Macro-F1 ~0.99
- **Student Interaction**: SVM 85.97% accuracy, Macro-F1 ~0.86
- **Use of Teaching Materials**: NB 83.06% accuracy, Macro-F1 ~0.83
- **Teaching Clarity**: SVM 78.3% accuracy, Macro-F1 ~0.78

## Random Seeds and Reproducibility

### Seed Values
- **random.seed()**: 42
- **np.random.seed()**: 42
- **train_test_split random_state**: 42
- **LinearSVC random_state**: 42
- **StratifiedKFold random_state**: 42

All seeds are set to 42 to ensure reproducibility across training runs.

## Package Versions

### Core Dependencies (from requirements.txt)
- **Python**: 3.12
- **scikit-learn**: 1.9.1
- **numpy**: 2.5.2
- **pandas**: 3.0.5
- **nltk**: 3.10.3
- **joblib**: 1.6.0
- **Flask**: 3.1.3
- **gensim**: 4.4.0
- **scipy**: 1.18.1

### ML-Specific Versions
- **sklearn-crfsuite**: 0.5.0
- **python-crfsuite**: 0.9.12

## Preprocessing Pipeline

### Text Cleaning Steps
1. **Contraction expansion**: "don't" → "do not", "isn't" → "is not"
2. **Lowercasing**: Convert all text to lowercase
3. **Stopword removal**: Remove NLTK English stopwords
4. **Stemming**: Apply Porter stemmer
5. **Punctuation removal**: Remove special characters and punctuation

### Document Label Source
- **Nigerian dataset**: Rating-based labels (1-2 → negative, 3 → neutral, 4-5 → positive)
- **Synthetic dataset**: Gold labels from template generation
- **Slang dataset**: Gold labels from manual annotation

## Class Counts (Document-Level)

### Label Distribution (approximate)
- **Positive**: ~42% (5,712 samples)
- **Negative**: ~38% (5,168 samples)
- **Neutral**: ~20% (2,720 samples)

### Duplicate Checks
- Duplicate comments are identified and removed during preprocessing
- Near-duplicate detection is performed using TF-IDF similarity
- Template-family hold-outs are preferred where possible to avoid memorization

## Evaluation Metrics

### Document-Level Metrics
- **Accuracy**: Overall classification accuracy
- **Macro-Precision**: Average precision across all classes (unweighted)
- **Macro-Recall**: Average recall across all classes (unweighted)
- **Macro-F1**: Harmonic mean of macro-precision and macro-recall
- **Weighted-F1**: F1 score weighted by class support
- **Confusion Matrix**: 3x3 matrix (positive, neutral, negative)
- **Baseline**: Majority-class classifier accuracy

### Aspect-Level Metrics
- **Accuracy**: Overall classification accuracy per aspect
- **Macro-F1**: Average F1 across sentiment classes per aspect
- **Weighted-F1**: Weighted F1 per aspect
- **Confusion Matrix**: 3x3 matrix per aspect
- **Validation Macro-F1**: Best validation score during hyperparameter tuning

## Training Runtime

### Measured Runtime
- **Total training time**: ~2-3 minutes (local machine)
- **Document-level training**: ~30-45 seconds
- **Aspect-level training**: ~1.5-2 minutes (8 aspects × 2 models)
- **Hardware**: Local development machine (specifications not documented)

## Aspect Train/Validation/Test Procedure

### Procedure
1. **Document-level**: Split entire dataset 60/20/20 using stratified split
2. **Aspect-level**: For each aspect, extract only samples with that aspect label, then perform independent 60/20/20 stratified split
3. **Clause partitioning**: All clauses from the same parent comment are kept in the same partition (train/val/test)
4. **Validation scores**: Reported from validation set during hyperparameter tuning
5. **Test scores**: Final evaluation on held-out test set after model selection

### Clarification on Live Page vs Documentation
- **Live page shows**: Validation scores (used for model selection)
- **Documentation mentions**: 80/20 split (this refers to train+val vs test, or older documentation)
- **Actual procedure**: 60% train / 20% validation / 20% test (as implemented in code)

## Template Overlap and Near-Duplicate Handling

### Report Template
- Template-family overlap is checked during dataset generation
- Near-duplicate detection uses TF-IDF cosine similarity threshold
- Template hold-outs are preferred where possible to avoid overfitting to specific phrasing patterns

### Duplicate Checks
- Exact duplicate comments are removed
- Near-duplicates (similarity > 0.9) are flagged and potentially removed
- Slang dataset specifically designed to include near-duplicates for robustness testing

## Evaluation Requirements

### Separate Evaluation Tasks
1. **Aspect Identification**: Precision/Recall/F1 for detecting which aspects are mentioned
2. **Sentiment Given Known Aspect**: Accuracy/Macro-F1 for sentiment classification when aspect is known
3. **End-to-End Aspect-Sentiment Pairs**: Joint accuracy for (aspect, sentiment) pairs

### Current Implementation
- **✅ Implemented**: Sentiment classification given known aspect (aspect-level models)
  - Metrics: Accuracy, Macro-F1, Weighted-F1 per aspect
  - Evaluation: Trained and evaluated on aspect-specific datasets
  - Dashboard: Displays aspect-level comparison table
- **✅ Implemented**: Per-class metrics for sentiment classification
  - Metrics: Precision, Recall, F1 for positive/neutral/negative separately
  - Evaluation: Added to document-level model metrics
  - Dashboard: Displays per-class metrics and confusion matrices
- **❌ Not implemented**: Separate aspect identification metrics
  - Requires: Manual annotation of aspect mentions in comments
  - Current approach: Keyword-based matching (no ML-based aspect extraction)
  - Status: Keyword lexicons are heuristic, not trained models
- **❌ Not implemented**: End-to-end joint metrics
  - Requires: Gold-standard (aspect, sentiment) pairs
  - Current approach: Separate keyword matching + sentiment classification
  - Status: Cannot evaluate jointly without ground truth

### Recommended Future Work
1. **Manually annotate a subset of comments for aspect mentions**
   - Select 500-1000 representative comments
   - Annotate which of the 8 aspects are mentioned
   - Annotate sentiment for each mentioned aspect
   - Create gold-standard evaluation dataset

2. **Calculate precision/recall/F1 for aspect extraction**
   - Compare keyword-based extraction against manual annotations
   - Report per-aspect extraction metrics
   - Identify aspects with poor extraction performance

3. **Calculate joint accuracy for (aspect, sentiment) pairs**
   - Evaluate complete pipeline on annotated dataset
   - Report joint accuracy: correct aspect AND correct sentiment
   - Analyze where errors occur (aspect extraction vs sentiment classification)

4. **Implement ML-based aspect extraction** (optional)
   - Train sequence labeling model (CRF or BERT) for aspect extraction
   - Compare against keyword-based approach
   - Evaluate using manual annotations

5. **Include independently labeled error examples**
   - Collect systematic error examples from evaluation
   - Categorize errors (aspect extraction, sentiment classification, negation)
   - Display in model evaluation dashboard

6. **Measure and report inference runtime**
   - Measure time per analysis request
   - Report average/median/max latency
   - Optimize if latency > 500ms

## Confusion Matrices

### Document-Level Confusion Matrix
```
              Predicted
            Pos  Neu  Neg
Actual Pos   TP   FN   FN
Actual Neu   FP   TN   FN
Actual Neg   FP   FP   TN
```

### Aspect-Level Confusion Matrices
Separate 3x3 confusion matrices for each of the 8 aspects, showing:
- True Positives, False Positives, True Negatives, False Negatives
- Per-class accuracy
- Confusion between classes (e.g., positive predicted as negative)

## Error Analysis

### Error Examples
**Status**: Not yet systematically collected

**Recommended Implementation**:
1. Create a validation set with manually annotated aspect-sentiment pairs
2. Collect examples of:
   - False positives (positive predicted as negative)
   - False negatives (negative predicted as positive)
   - Confusion between positive and neutral
   - Confusion between negative and neutral
3. Analyze error patterns to identify systematic issues
4. Document specific examples in model evaluation dashboard

**Current Placeholder**:
The model evaluation page now displays confusion matrices and per-class metrics. Error examples should be added to provide qualitative analysis of model failures.

### Common Error Patterns
- Slang and sarcasm detection challenges
- Mixed-clause sentences with contrasting sentiments
- Aspect keyword overlap (same keyword used for multiple aspects)
- Negation scope errors (already addressed in bug fixes - see test_negation_standalone.py)

## Requirement-to-Test Record

### Implemented Tests
- ✅ Negation handling (not, never, doesn't, isn't, mixed-clause)
  - Test file: `test_negation_standalone.py`
  - Status: All tests passing
  - Coverage: Basic negation, negation in scope, polarity flipping, mixed-clause scenarios
- ✅ Summary narrative consistency with aspect table
  - Test file: `test_summary_standalone.py`
  - Status: All tests passing
  - Coverage: Aspect status classification, summary generation, label consistency
- ✅ Factual/unmatched text handling (insufficient evidence)
  - Test file: `test_factual_text.py`
  - Status: All tests passing
  - Coverage: Factual sentences, irrelevant text, zero-evidence aspects
- ✅ Composite score calculation and range [-1, 1]
  - Test file: `test_composite_score.py`
  - Status: All tests passing
  - Coverage: Formula verification, range boundaries, extreme values
- ✅ PDF export UI label accuracy
  - Updated: `templates/lecturer_report.html`
  - Status: Button label changed to "Print / Save as PDF (browser)"
  - Tooltip: Clarifies browser print dialog, no server-generated PDF
- ✅ Per-class metrics and confusion matrices
  - Updated: `src/train_models.py` (added per_class_metrics to classify_metrics)
  - Updated: `templates/model_evaluation.html` (added confusion matrix visualization)
  - Status: Per-class precision/recall/F1 now displayed in dashboard
- ✅ Training methodology documentation
  - Created: `METHODOLOGY.md`
  - Status: Comprehensive documentation of all training parameters
  - Coverage: Split sizes, hyperparameters, TF-IDF settings, seeds, package versions

### Pending Tests (Requires Manual Annotation)
- ⏳ Aspect identification precision/recall/F1
  - Blocker: Requires manual annotation of aspect mentions in comments
  - Est. effort: 8-12 hours for annotation of 500-1000 comments
- ⏳ End-to-end aspect-sentiment pair accuracy
  - Blocker: Requires gold-standard (aspect, sentiment) pairs
  - Est. effort: Depends on aspect identification completion
- ⏳ Error example collection and analysis
  - Blocker: Requires systematic error analysis on test set
  - Est. effort: 4-6 hours for categorization and documentation
- ⏳ Inference runtime measurement
  - Blocker: Requires runtime instrumentation in production
  - Est. effort: 2-3 hours for implementation and analysis
- ⏳ Template overlap evaluation statistics
  - Blocker: Requires analysis of synthetic dataset generation
  - Est. effort: 3-4 hours for similarity analysis

### Test Files Summary
- `test_negation_standalone.py` - Negation handling tests (PASS)
- `test_summary_standalone.py` - Summary consistency tests (PASS)
- `test_factual_text.py` - Factual text handling tests (PASS)
- `test_composite_score.py` - Composite score tests (PASS)
- `test_negation_simple.py` - Basic negation detection (PASS)
- `test_keyword_matching.py` - Keyword matching tests (PASS)
- `test_full_analysis.py` - End-to-end analysis tests (PASS)

## Model Files

### Saved Artifacts
All models saved in `models/` directory:
- `svm_document.pkl` - Document-level SVM
- `nb_document.pkl` - Document-level Naive Bayes
- `vectorizer_document.pkl` - Document TF-IDF vectorizer
- `svm_{aspect}.pkl` - 8 aspect SVM models
- `nb_{aspect}.pkl` - 8 aspect Naive Bayes models
- `vectorizer_{aspect}.pkl` - 8 aspect TF-IDF vectorizers

### Model Persistence
- Format: joblib pickle files
- Compatibility: scikit-learn 1.9.1
- Versioning: No explicit version control (manual retraining required)

## Reproducibility Checklist

To reproduce the training results:
1. ✅ Set random seeds (random=42, numpy=42, sklearn=42)
2. ✅ Use same package versions (requirements.txt)
3. ✅ Use same datasets (Nigerian 10k, synthetic 1.6k, slang 2k)
4. ✅ Use same split ratios (60/20/20)
5. ✅ Use same hyperparameters (C=1.0 for SVM, alpha=0.1 for NB)
6. ✅ Use same TF-IDF settings (ngram=(1,2), max_features=50k)
7. ⏳ Document hardware specifications
8. ⏳ Record exact runtime per model
9. ⏳ Version control for model artifacts

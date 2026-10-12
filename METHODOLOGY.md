# Methodology and Reproducibility Record

## 1. Purpose and scope

This file separates results reported by the existing application from the new, controlled phrase-template holdout diagnostic. The project is a research prototype, not a validated institutional decision system.

The eight teaching aspects used by the current implementation are:

1. Teaching Clarity
2. Course Organisation
3. Assessment and Grading
4. Lecturer Punctuality
5. Availability
6. Communication
7. Student Interaction
8. Use of Teaching Materials

These labels should be copied verbatim in chapter text, tables, scripts, and interfaces. The older five-aspect lists, older CRF descriptions, and the stray “Food” example are not the final implemented scope. Aspect discovery in \`core/ml_engine.py\` currently uses lexicon-guided clause matching; the separate CRF artefact in \`models/\` must not be described as part of the live pipeline unless source-code evidence shows that it is called.

## 2. Dataset provenance boundary

The checked-in metrics JSON records a 13,600-record corpus split conceptually into a 10,000-row primary CSV, 1,600 controlled-template records, and 2,000 slang/informal-text records. The repository contains a file named \`Data/nigerian_lecturer_evaluations_10000.csv\`, but the repository materials inspected do not establish the primary file's original collector, collection dates, permission, institutional origin, anonymisation history, or independent annotation procedure. The file name and row count alone are not proof of provenance.

For any submission, attach the primary source's documentation if it is available. Until then, describe that component as the “primary 10,000-row CSV (provenance not independently verified)” and do not claim that it is verified authentic institutional feedback. Where labels are obtained from ratings (1–2 negative, 3 neutral, 4–5 positive), treat them as rating-derived proxy labels rather than independent text-sentiment annotations. The generated template labels are generator-assigned. Do not claim manual annotation or inter-annotator agreement without records of that work.

The 13,600 training-corpus records are also not the same quantity as evaluations currently stored in the live application database. Report those counts separately.

## 3. Previously reported document-level results

The currently checked-in \`Data/model_evaluation_metrics.json\` records the following held-out results from an existing run. These values are presented here as recorded results, not as an independently reproduced retraining run in this documentation pass.

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.7960 | 0.7680 | 0.7101 | 0.6808 | 0.7458 |
| Multinomial Naive Bayes | 0.8015 | 0.8103 | 0.7135 | 0.6776 | 0.7449 |
| Majority-class baseline | 0.4441 | — | — | — | — |

Reported training settings are SVM C=1.0 and Naive Bayes alpha=0.1; the document vectorizer uses TF-IDF unigrams and bigrams, sublinear term frequency, min_df=2, and up to 50,000 features. The stored split is 8,160 training, 2,720 validation, and 2,720 test records. A fixed random seed of 42 is documented in the training script. Confirm these details against the exact experiment record before describing them as a fully reproducible independent run.

The confusion matrices use the class order **positive, neutral, negative**, with rows as actual labels and columns as predicted labels.

### Linear SVM confusion matrix

| Actual \ Predicted | Positive | Neutral | Negative | Support |
|---|---:|---:|---:|---:|
| Positive | 853 | 38 | 1 | 892 |
| Neutral | 256 | 112 | 252 | 620 |
| Negative | 0 | 8 | 1200 | 1208 |
| Predicted total | 1109 | 158 | 1453 | 2720 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Positive | 0.7692 | 0.9563 | 0.8526 | 892 |
| Neutral | 0.7089 | 0.1806 | 0.2879 | 620 |
| Negative | 0.8259 | 0.9934 | 0.9019 | 1208 |

### Multinomial Naive Bayes confusion matrix

| Actual \ Predicted | Positive | Neutral | Negative | Support |
|---|---:|---:|---:|---:|
| Positive | 880 | 12 | 0 | 892 |
| Neutral | 274 | 99 | 247 | 620 |
| Negative | 0 | 7 | 1201 | 1208 |
| Predicted total | 1154 | 118 | 1448 | 2720 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Positive | 0.7626 | 0.9865 | 0.8602 | 892 |
| Neutral | 0.8390 | 0.1597 | 0.2683 | 620 |
| Negative | 0.8294 | 0.9942 | 0.9044 | 1208 |

The neutral class has low recall for both classifiers. Do not describe the results as proof that either model is generally superior: Naive Bayes has slightly higher overall accuracy, while SVM has slightly higher Macro-F1. Neither metric alone validates aspect extraction, end-to-end predictions, real-world generalisation, or institutional fitness.

## 4. New phrase-template-family holdout benchmark

Run:

\`\`\`bash
python -m nltk.downloader stopwords punkt punkt_tab
python src/evaluate_template_holdout.py
\`\`\`

The script creates a separate controlled train/test corpus from the phrase banks in \`src/train_models.py\`:

- Seed: 42.
- Each exact authored phrase template is assigned to either the train bank or the held-out test bank separately within each aspect/polarity group.
- No exact normalized phrase may occur in both banks; the script asserts this condition before training.
- Train comments are generated only from train-bank phrases; test comments only from held-out phrases.
- Clauses inside a single comment stay together in the same partition.
- Document-level models are Linear SVM and Multinomial Naive Bayes. The vectorizer is fitted on training comments only.
- The aspect extractor is evaluated separately by comparing its detected aspect names with generator-assigned aspect labels.
- Sentiment given a known aspect is measured on held-out phrases with a separate aspect-specific SVM and Naive Bayes model. This assumes the aspect is already known.
- End-to-end aspect-sentiment pair precision/recall/F1 requires the correct aspect and polarity in the same prediction.

The CI benchmark completed successfully on 12 October 2026. It wrote `artifacts/template_family_holdout_metrics.json` and uploaded it as workflow artifact `template-family-holdout-metrics`. The recorded run used 95 unique training templates, 30 held-out templates, 8,000 generated training comments, and 2,000 generated test comments; the exact normalized phrase-overlap check found zero templates shared between banks.

### Executed document-level results on held-out templates

| Model | Accuracy | Macro precision | Macro recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Linear SVM | 0.2890 | 0.2928 | 0.3046 | 0.2736 | 0.2685 |
| Multinomial Naive Bayes | 0.3145 | 0.3127 | 0.3168 | 0.3129 | 0.3128 |
| Training-majority baseline on held-out test | 0.3680 | — | — | — | — |

Class order for the matrices is positive, neutral, negative; rows are actual and columns are predicted.

- Linear SVM confusion matrix: `[[193, 417, 126], [166, 299, 101], [219, 393, 86]]`.
- Multinomial Naive Bayes confusion matrix: `[[180, 291, 265], [173, 187, 206], [244, 192, 262]]`.

Per-class metrics:

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| SVM | Positive | 0.3339 | 0.2622 | 0.2938 | 736 |
| SVM | Neutral | 0.2696 | 0.5283 | 0.3570 | 566 |
| SVM | Negative | 0.2748 | 0.1232 | 0.1701 | 698 |
| Naive Bayes | Positive | 0.3015 | 0.2446 | 0.2701 | 736 |
| Naive Bayes | Neutral | 0.2791 | 0.3304 | 0.3026 | 566 |
| Naive Bayes | Negative | 0.3574 | 0.3754 | 0.3662 | 698 |

The models score below the held-out majority baseline (0.3680), showing weak generalisation to unseen authored phrase templates.

### Aspect identification (separate task)

Micro precision = **0.9061**, recall = **0.5749**, F1 = **0.7035** (TP=2,306, FP=239, FN=1,705; 4,011 generator-labelled aspect mentions).

| Aspect | Precision | Recall | F1 | Gold support |
|---|---:|---:|---:|---:|
| Teaching Clarity | 1.0000 | 0.3588 | 0.5281 | 485 |
| Course Organisation | 0.6766 | 1.0000 | 0.8071 | 500 |
| Assessment and Grading | 1.0000 | 0.2050 | 0.3402 | 522 |
| Punctuality | 1.0000 | 0.4133 | 0.5849 | 496 |
| Availability | 1.0000 | 0.1802 | 0.3053 | 494 |
| Communication | 1.0000 | 1.0000 | 1.0000 | 510 |
| Student Interaction | 1.0000 | 0.8359 | 0.9106 | 524 |
| Use of Teaching Materials | 1.0000 | 0.5896 | 0.7418 | 480 |

High precision with lower recall means the extractor misses many labelled aspects, especially Assessment and Grading, Availability, and Teaching Clarity. The perfect Communication result is specific to this small synthetic phrase bank, not evidence of real-world reliability.

### End-to-end aspect–sentiment pairs

Exact pair micro-precision = **0.8585**, recall = **0.5448**, F1 = **0.6666** (TP=2,185, FP=360, FN=1,826). A pair is correct only when both aspect and polarity match.

### Sentiment given a known aspect

On only 30 held-out phrase-template instances (11 positive, 8 neutral, 11 negative), the pooled aspect-specific SVM achieved accuracy **0.2333** and Macro-F1 **0.2267**; Naive Bayes achieved accuracy **0.2667** and Macro-F1 **0.2506**. This is a very small diagnostic, not a stable performance estimate.

The benchmark remains a controlled synthetic test: exact phrase overlap is removed, but the test phrases share the project's authored aspect vocabulary and lexicon, and labels are generator-assigned. It does not demonstrate transfer to independently written student feedback. These low scores must be reported, not suppressed, and the existing random-split scores must not be presented as proof of robustness.

## 5. Aspect extraction and sentiment are separate tasks

### Aspect identification

The lexicon-guided extractor is evaluated by comparing the set of detected aspect names for each comment with its gold set of generator-assigned aspect names. Report micro precision, recall, and F1, plus per-aspect precision, recall, F1, and gold support. A false positive occurs when the extractor reports an aspect not present in the generator labels. A false negative occurs when a labelled aspect is not extracted.

### Sentiment when aspect is known

This task supplies the correct aspect and scores sentiment polarity for held-out clauses/templates. It tests polarity classification but does not test discovery of the aspect.

### End-to-end aspect-sentiment pairs

An output pair is correct only when both the aspect name and polarity match the corresponding gold pair. Report pair-level precision, recall, and F1 separately. It is possible to classify sentiment well once an aspect is given while the complete pipeline still misses the aspect or assigns the wrong aspect.

The three tasks must not be presented as interchangeable metrics.

## 6. Summary generation rules

The report table and prose call the same function, \`classify_aspect_status\`, so that narrative labels are based on the same observed counts:

- 0 positive + 0 neutral + 0 negative mentions: **Insufficient evidence**.
- Neutral-only mentions with no positive or negative mentions: **Balanced / Neutral** (this does not mean evidence of positive or negative quality).
- Fewer than three positive/negative mentions in total: **Limited evidence**.
- Strength: at least three positive mentions and positive exceeds negative by at least two.
- Improvement area: at least three negative mentions and negative exceeds positive by at least two.
- Other non-zero results: **Balanced / Neutral**.
- The report must retain the positive, neutral, and negative counts beside the label. “Limited evidence” and “Insufficient evidence” are not strengths and must not be silently converted into a positive or negative recommendation.

These are transparent reporting thresholds chosen for this prototype, not empirically optimised thresholds for teaching effectiveness. The eight aspects have no validated institutional weighting. Composite score weights of 0.4 for normalized rating and 0.6 for mean aspect sentiment remain illustrative design choices.

## 7. Requirement-to-test evidence and live-app boundaries

A dated record of observed and unverified behavior is in \`docs/REQUIREMENT_TO_TEST_MATRIX.md\`. It distinguishes source-code checks from browser observations. A visible upload or export button is not evidence that the operation succeeded. Administrator sign-in is not evidence that lecturer access is isolated. Dashboard totals must not be treated as the model training-set size.

The repository regression tests cover summary thresholds, the eight canonical aspect names, a “not punctual” example, and unmatched factual text. The phrase-family evaluation additionally writes classification reports, confusion matrices, per-class support, aspect-identification metrics, and end-to-end pair metrics. Whether the current branch's tests and benchmark pass must be taken from its GitHub Actions run, not inferred from the presence of the scripts.

## 8. RQ4 pilot study and evidence boundary

RQ4 asks how useful intended lecturers and academic administrators find the reports. No completed pilot responses are included in this repository update. Use \`docs/PILOT_USER_FEEDBACK_PROTOCOL.md\` to run and record a small, voluntary usability pilot with two or three departmental colleagues. Until actual participants have used the app and their anonymised responses are documented, RQ4 remains untested or at most planned; it must not be marked as answered.

The pilot is formative usability feedback, not a representative user-acceptance study. It cannot establish reliability, teaching effectiveness, institutional validity, or fairness.

## 9. Ethical and methodological limits

- No real institutional feedback should be imported without permission and appropriate review.
- Do not include student names, matriculation numbers, contact details, or unredacted identities in test artefacts.
- Synthetic/template labels are not independent human annotations.
- Sarcasm, code-switching, local slang, negation scope, keyword overlap, and unseen phrasing remain error sources.
- A missing aspect is missing evidence, not neutral sentiment.
- The prototype must not be used as the sole basis for employment, promotion, discipline, or other high-stakes staffing decisions.
- A future report should include model/library versions, the exact source data manifest and hashes, split IDs, generator seed, training logs, machine specifications, measured inference latency, annotated error examples, and independent human-annotated evaluation.

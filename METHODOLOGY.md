# Methodology, Evaluation Results, and Evidence Boundaries

**Revision date:** 12 October 2026  
**Scope:** Research-methodology record for the lecturer-evaluation prototype. This file distinguishes stored experimental metrics, code-level regression tests, observed deployed behaviour, and evaluations that still require human-annotated data or participants.

## 1. Dataset provenance and scope

The experiment contains 13,600 synthetic or engineered records as described in the project manuscript:

| Corpus component | Records | Label source | Evidence boundary |
|---|---:|---|---|
| LLM-generated Nigerian-context comments | 10,000 | Rating-derived proxy labels: ratings 1-2 = negative, 3 = neutral, 4-5 = positive | These are not verified institutional SET submissions. A rating is not an independent human annotation of comment sentiment. |
| Controlled multi-aspect template comments | 1,600 | Generator-assigned labels | Labels are known from generation rules, not independent annotation. |
| Nigerian slang, sarcasm, and insult examples | 2,000 | Dataset/generator-assigned labels | Inclusion of slang examples does not itself validate sarcasm recognition. |
| **Total** | **13,600** | Mixed proxy/generator labels | **No authentic institutional student comments are included in the reported experiment.** |

The word *Nigerian* describes the context and expressions represented in the corpus; it must not be used to imply that the 10,000-row component was collected from actual students. The thesis describes these records as Gemini/LLM-generated, and the repository does not contain independent provenance or ethics documentation establishing them as authentic institutional responses. The application and thesis must therefore use the synthetic/engineered description consistently.

No claim of independent human annotation, inter-annotator agreement, Cohen's kappa, or validated ground truth should be made unless records of that work are supplied. Rating-derived labels should be called *proxy labels* throughout.

## 2. Existing random-split experiment

The stored document-classification experiment uses a stratified random 60/20/20 split with seed 42: 8,160 training records, 2,720 validation records, and 2,720 held-out test records. Hyperparameters are selected using validation Macro-F1. The saved test metrics are:

| Model | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.4441 | — | — | — | — |
| Linear SVM | 0.7960 | 0.7680 | 0.7101 | 0.6808 | 0.7458 |
| Multinomial Naive Bayes | 0.8015 | 0.8103 | 0.7135 | 0.6776 | 0.7449 |

These are **random-split results on synthetic/engineered text**. The split does not demonstrate generalisation to unseen template families. They must not be presented as accuracy on real Nigerian student feedback or as proof of institutional suitability.

### 2.1 Feature extraction and parameters

The document-level vectoriser uses TF-IDF with unigrams and bigrams, sublinear_tf=True, min_df=2, and max_features=50000. The vocabulary is fitted on the training partition and then used to transform the validation and test partitions. The final reported document models use LinearSVC with selected C=1.0 and MultinomialNB with selected alpha=0.1; the candidate values are C in [0.5, 1.0, 2.0] and alpha in [0.1, 0.5, 1.0]. The source sets Python and NumPy seeds and uses random_state=42 for stratified splits.

The saved five-fold CV scores are SVM Macro-F1 0.6907 ± 0.0066 and NB Macro-F1 0.6849 ± 0.0103. The ± values are the standard deviation across five folds. These CV figures should be treated as supplementary to the held-out test scores: the current code vectorises the training partition before cross-validation, so a stricter CV estimate should fit the vectoriser inside a scikit-learn Pipeline separately within each fold.

The document test confusion matrices use class order **positive, neutral, negative**; rows are actual classes and columns are predicted classes.

**Linear SVM**

| Actual / Predicted | Positive | Neutral | Negative | Support |
|---|---:|---:|---:|---:|
| Positive | 853 | 38 | 1 | 892 |
| Neutral | 256 | 112 | 252 | 620 |
| Negative | 0 | 8 | 1,200 | 1,208 |

**Multinomial Naive Bayes**

| Actual / Predicted | Positive | Neutral | Negative | Support |
|---|---:|---:|---:|---:|
| Positive | 880 | 12 | 0 | 892 |
| Neutral | 274 | 99 | 247 | 620 |
| Negative | 0 | 7 | 1,201 | 1,208 |

### 2.2 Per-class document-level metrics

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| SVM | Positive | 0.7692 | 0.9563 | 0.8526 | 892 |
| SVM | Neutral | 0.7089 | 0.1806 | 0.2879 | 620 |
| SVM | Negative | 0.8259 | 0.9934 | 0.9019 | 1,208 |
| Naive Bayes | Positive | 0.7626 | 0.9865 | 0.8602 | 892 |
| Naive Bayes | Neutral | 0.8390 | 0.1597 | 0.2683 | 620 |
| Naive Bayes | Negative | 0.8294 | 0.9942 | 0.9044 | 1,208 |

The neutral class has substantially lower recall and F1 than the positive and negative classes. This class-specific weakness should be discussed explicitly; overall accuracy hides it.

## 3. Aspect-level sentiment is not aspect extraction

The system's live analysis path uses lexicon-guided keyword matching to propose teaching aspects and then applies polarity rules/classifiers. The model-evaluation table reports sentiment classification on labelled examples associated with each aspect. This does **not** establish that the system discovers aspects correctly in unseen raw comments.

Three distinct evaluation tasks must be kept separate:

1. **Aspect identification:** compare the aspects detected in a comment with independently human-annotated gold aspects and report precision, recall, and F1.
2. **Sentiment given a known aspect:** assess polarity only when the aspect is already supplied or correctly identified.
3. **End-to-end aspect-sentiment pairs:** a pair is correct only when both the aspect and its sentiment match the gold annotation.

Aspect-identification and end-to-end pair metrics remain **pending** until a representative sample of comments is independently annotated. The current lexicon output alone cannot provide gold labels. An annotation file schema and evaluation script should be used for a future manually annotated sample. Do not place invented extraction scores in the thesis.

The aspect-level test F1 values currently stored in the report are:

| Aspect | Total labelled examples | Held-out test examples | SVM Macro-F1 | NB Macro-F1 | Higher stored Macro-F1 |
|---|---:|---:|---:|---:|---|
| Teaching Clarity | 2,557 | 512 | 0.6834 | 0.6796 | SVM |
| Course Organisation | 2,620 | 524 | 0.7425 | 0.7152 | SVM |
| Assessment and Grading | 2,520 | 504 | 0.7047 | 0.6855 | SVM |
| Lecturer Punctuality | 2,525 | 505 | 0.6787 | 0.6748 | SVM |
| Availability | 2,625 | 525 | 0.7410 | 0.7287 | SVM |
| Communication | 3,548 | 710 | 0.6891 | 0.6896 | Naive Bayes |
| Student Interaction | 3,514 | 703 | 0.6986 | 0.6829 | SVM |
| Use of Teaching Materials | 2,124 | 425 | 0.7394 | 0.7414 | Naive Bayes |

Total labelled examples are not the same quantity as held-out test examples. The older stored matrices were generated without explicit class labels, so their matrix order was scikit-learn's sorted order (negative, neutral, positive). The updated training code now fixes and records an explicit class order (positive, neutral, negative) and records per-class support. Metrics from different matrix orders must not be mixed.

## 4. Phrase-template leakage and the separate holdout experiment

The standard random split can place the same phrase-bank item in both training and test records. It is therefore insufficient as the only evidence of generalisation. The branch adds an independent phrase-template holdout experiment:

- The generator assigns stable IDs to each phrase item, using aspect, polarity, and phrase-bank position.
- Entire comments containing a held-out phrase item are excluded from training and are placed in the test set.
- Held-out IDs are asserted to be absent from the training pool.
- Validation is performed only within the non-held-out pool; the held-out test set is not used for hyperparameter selection.
- The experiment uses the synthetic phrase-bank corpus only. It does not test transfer to real student feedback and does not measure aspect-identification accuracy.

The experiment entry point is python -m src.evaluate_template_holdout. A successful GitHub Actions run on 12 October 2026 (run 38305693936) generated and archived the benchmark JSON. The run generated 1,600 synthetic phrase-template records, held out 43 phrase items across the aspect/polarity banks, and placed every comment containing at least one held-out phrase item in the test set. The resulting non-held-out pool contained 720 records (576 inner-training and 144 validation records); the held-out test set contained 880 records. Assertions confirmed that training and test rows were disjoint and that held-out phrase IDs were absent from the training pool. This is a held-out phrase-template test, not a test of authentic student feedback.

| Phrase-template holdout model | Test accuracy | Test Macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|
| Linear SVM (selected C=2.0; validation Macro-F1 0.9482) | 0.7261 | 0.7153 | 0.7264 |
| Multinomial Naive Bayes (selected alpha=1.0; validation Macro-F1 0.9262) | 0.7080 | 0.6948 | 0.7077 |

Per-class support in this holdout test was Positive=365, Neutral=189, Negative=326 (880 comments total). SVM per-class precision/recall/F1 were Positive 0.7500/0.7315/0.7406; Neutral 0.6410/0.6614/0.6510; Negative 0.7508/0.7577/0.7542. Naive Bayes values were Positive 0.7265/0.7205/0.7235; Neutral 0.6270/0.6138/0.6203; Negative 0.7327/0.7485/0.7405.

The benchmark JSON records the exact 43 held-out phrase IDs, both confusion matrices, model settings and split sizes, and is archived as the GitHub Actions artifact “template-family-holdout-metrics” for run 38305693936. This run provides evidence of performance on unseen phrase items within the synthetic phrase-bank generator. It does not validate aspect-extraction precision/recall/F1, transfer to authentic student feedback, or user usefulness. A held-out comment may still contain other non-held-out phrases; the entire comment is nonetheless excluded from training.

Near-duplicate detection across the complete 13,600-record corpus has not been established by the existing manuscript alone. Do not say that exact duplicates were removed or that a TF-IDF similarity threshold was applied unless a repeatable duplicate audit and its counts are supplied.

## 5. Aspect-status summaries and composite scores

The production summary rule is deterministic and shared by the report table and narrative:
- Zero total mentions -> **Insufficient evidence**.
- Positive mentions exceed negative mentions by at least two -> **Strength Area**.
- Negative mentions exceed positive mentions by at least two -> **Improvement Area**.
- Smaller differences or a tie -> **Balanced / Neutral**.

The threshold is a transparent reporting heuristic, not a statistical significance test or proof of teaching effectiveness. Neutral counts contribute to whether any evidence exists but do not offset positive-minus-negative balance.

The illustrative composite is:

**overall_cs = 0.4 × ((rating - 3) / 2) + 0.6 × mean(detected aspect scores)**

where positive, neutral and negative aspect scores are +1, 0 and -1 respectively. It is bounded to [-1, 1]. If no aspect is detected, the composite must be missing/unavailable, not 0.0, because no detected aspect is not the same as validated neutral evidence. The 0.4/0.6 weights are design choices and have not been validated against an external measure of teaching effectiveness.

## 6. Functional regression tests versus live-system verification

The revision branch adds tests that import the actual production functions in core.ml_engine; these are stronger than tests that copy the implementation into a standalone test file. GitHub Actions has reported a successful run of these code-level regression tests (17 checks) after adding coverage for:
- "not punctual", "never available", "doesn't explain well", and "isn't organized";
- mixed positive/negative aspects in the same comment;
- neutral/factual unmatched text returning insufficient aspect evidence;
- deterministic summary status, including ties and zero evidence;
- composite-score boundary cases;
- leakage assertions and reproducibility of the family-split helper.

A code-level pass is **not** proof that the deployed Railway application has been re-tested. The revision branch must be reviewed, merged, deployed, and checked again on the live site before stating that these fixes are present in production.

### Requirement-to-test matrix

| Requirement | Test input/action | Expected result | Evidence currently available | Status |
|---|---|---|---|---|
| FR-01: accept evaluation data | Open the public submission page; upload a valid CSV/XLSX as administrator | Valid rows validate and persist without collecting student identity | Public form observed; import/persistence was not exercised in the 9 Oct review | Not verified end-to-end |
| FR-02: preprocess text | Contraction and case examples such as "isn't organized" | Normalised text preserves negation and sentiment cues | Production preprocessing path is used by branch regression tests | Partial, code-level |
| FR-03: sentiment analysis | Mixed comment: "The lecturer explains concepts clearly, but grading is unfair." | Teaching Clarity positive; Assessment and Grading negative | Correct mixed-aspect output observed in the 9 Oct review; branch regression test added | Pass for that test input; not a full benchmark |
| FR-03: negation | "The lecturer is not punctual." | Lecturer Punctuality negative | The 9 Oct live inspection observed a positive aspect result; branch lexicon/polarity regression now passes | Prior live failure; redeployment verification pending |
| FR-03: unmatched input | "The classroom walls are blue." | No aspects and insufficient evidence; no neutral composite should be inferred | Branch test validates the actual analysis function | Pass in branch test; live retest pending |
| FR-04: classifier comparison | Open the Model Evaluation page | Clearly labelled SVM/NB metrics, matrices, and per-class support | Page and metrics were visible in the 9 Oct review; the revised branch adds support and label-order metadata | Display feature; reproducibility audit pending |
| FR-05: lecturer report | View a report with ties and zero counts | Table and prose agree; ties are balanced; zero counts are insufficient evidence | Branch summary logic and tests added after review identified contradictions | Pass in branch test; live retest pending |
| FR-06: dashboard | Open dashboard and reconcile totals against database rows | Counts, charts and displayed summaries agree with persisted data | Dashboard rendering observed; full database reconciliation not performed | Not verified fully |
| FR-07: role restrictions | Login as Lecturer A and request Lecturer B's report URL directly | Access to another lecturer's report is denied | Code check exists; no authorised two-account penetration test recorded | Not verified end-to-end |
| FR-08: export | Export one lecturer report to CSV and inspect file contents | Valid CSV with expected fields and no unintended identity fields | CSV control observed; export was not executed during the review | Not verified end-to-end |

The table intentionally distinguishes code-level test outcomes from actions not performed on the live deployment. A reviewer should not see "Pass" for an upload, export, persistence, or access-control test unless the corresponding test has actually been executed and recorded.

## 7. Pilot user feedback (RQ4)

No pilot user feedback is recorded in this repository. RQ4 remains unanswered until two or three lecturers, departmental colleagues, or supervisors actually use the dashboard and provide feedback. Do not write hypothetical comments as if they were observed results.

Suggested small pilot protocol:
1. Obtain permission from the participants and explain that this is a prototype demonstration using synthetic data; collect no sensitive student or staff information.
2. Ask each participant to complete the same tasks: locate a lecturer report, interpret one aspect table, identify what "Insufficient evidence" means, and locate the model-evaluation comparison.
3. Ask the same short questions: Which part was clearest? Which result was confusing? Would the report help structure a discussion about teaching? What could mislead a reader? What change is most important?
4. Record date, participant role (not name), task completion, verbatim notes only with permission, and the researcher's themes. Report the number of participants and state that this is an informal formative pilot, not a validated usability study.
5. If no session is completed before submission, retain RQ4 as unanswered and describe the protocol as planned work.

## 8. Performance, security, and generalisation boundaries

The stored metrics include a local model-training runtime of 63.12 seconds, but the available record does not document the machine configuration. This value is not production request latency and is not evidence that the deployed service meets a performance objective. Per-request inference latency, persistence across restarts, import/export correctness, backup recovery, and direct-URL access-control testing require separate measured tests.

The prototype must not be described as ready for high-stakes employment decisions or university-wide adoption. Synthetic labels, template overlap, neutral-class errors, heuristic aspect extraction, unvalidated composite weights, missing real-user evaluation, and lack of authentic-data transfer testing remain material limitations.

## 9. Versioning and reproducibility record

For every final experiment, archive together:
- dataset version and SHA-256 checksums;
- generator version and seed;
- train/validation/test row IDs and class counts;
- exact preprocessing and TF-IDF settings;
- model hyperparameters and selected validation metric;
- confusion matrices, per-class precision/recall/F1/support, and test sample counts;
- package versions, machine specification, and runtime;
- phrase-template holdout IDs and overlap assertions;
- application commit SHA used for functional tests.

A displayed metric is not a complete experiment record. The deployed app, repository code, saved models, metric JSON, and thesis tables must all refer to the same documented experiment version.

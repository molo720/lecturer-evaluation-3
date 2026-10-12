# Human Annotation Guide: Aspect Extraction and Sentiment

## Purpose

This guide prepares an independent reference set to evaluate whether the system detects the correct teaching aspects in raw comments and assigns polarity to them. This is separate from document-level sentiment classification and from model accuracy on generated labels.

Do not use the model's predictions to create the gold labels. Annotators should label each comment before seeing the system output. If feasible, have two annotators label the same subset independently and resolve disagreements by documented adjudication. Do not claim inter-annotator agreement unless it is actually calculated.

## Canonical aspects

Use these names exactly. Do not add “Food” or other unsupported categories.

1. Teaching Clarity
2. Course Organisation
3. Assessment and Grading
4. Lecturer Punctuality
5. Availability
6. Communication
7. Student Interaction
8. Use of Teaching Materials

## Annotation rules

- Mark an aspect only when the comment contains evidence about that teaching dimension. Do not infer an aspect merely from an overall rating or from the lecturer's identity.
- For each mentioned aspect, assign sentiment toward that aspect as positive, negative, or neutral.
- Positive means the comment expresses favourable evaluation of that aspect.
- Negative means the comment expresses unfavourable evaluation of that aspect.
- Neutral means the text mentions the aspect but makes no clear positive or negative judgement.
- If a comment contains conflicting opinions about one aspect, record the dominant judgement only if the annotation protocol specifies a clear rule; otherwise mark it for adjudication and document the rule. Do not silently use the overall document polarity as the aspect label.
- If no teaching aspect is mentioned, use an empty JSON list for both fields: gold_aspects = [] and gold_pairs = [].
- Preserve code-mixed/slang wording as written. Annotators may use context to interpret it, but should mark uncertain cases for adjudication instead of guessing.

## CSV schema

Use the file Data/aspect_annotation_template.csv as the starting point.

| Column | Required value |
|---|---|
| comment_id | Unique anonymised row ID; do not place a student or staff name here |
| comment | Original comment text, with identifiers removed if needed |
| gold_aspects | JSON list of canonical aspect names |
| gold_pairs | JSON list of [aspect, sentiment] pairs |
| annotator_id | Pseudonym such as A or B, not a real name |
| adjudicated | true/false; use true only after disagreements are resolved |

The JSON values must be valid JSON strings inside the CSV. For instance, a clearly labelled illustrative example might use:
- gold_aspects: "[""Teaching Clarity"", ""Assessment and Grading""]"
- gold_pairs: "[[""Teaching Clarity"", ""positive""], [""Assessment and Grading"", ""negative""]]"

Do not paste illustrative rows into the final evaluation dataset as if they were collected comments. The evaluated comments must come from the chosen corpus and be annotated independently.

## Recommended sample and audit trail

For a small project pilot, annotate at least 200 comments with variation across comment length, polarity, mixed-aspect inputs, and slang/code-mixing; 500 or more is preferable if time permits. This is a practical recommendation, not a statistical guarantee. Keep an untouched final subset if annotator resources allow.

Record:
- the corpus filename, version, checksum, and selection procedure;
- the number of comments sampled, excluded, and retained;
- annotator instructions, dates, and pseudonymous IDs;
- per-annotator labels and adjudicated labels;
- unresolved or ambiguous cases and exclusion reasons;
- whether comments are synthetic, public, or genuinely collected under approval.

Never describe generated comments as genuine student feedback. If the sample is entirely synthetic, the evaluation measures agreement with human annotation on synthetic comments only; it does not establish real-world transfer.

## Run the evaluation

After completing the CSV, run:

    python -m src.evaluate_aspect_extraction --input Data/aspect_annotations_completed.csv --output Data/aspect_extraction_metrics.json

The script reports aspect-identification micro precision/recall/F1 and per-aspect scores, conditional sentiment accuracy on correctly extracted aspects, and end-to-end aspect-sentiment pair metrics. Read conditional sentiment accuracy alongside end-to-end pair F1 because the conditional score excludes missed aspects.

If the annotation CSV is empty or incomplete, the script should stop rather than produce a misleading zero score. Archive the completed annotation protocol, the input checksum, and the JSON metrics with the thesis evidence pack.

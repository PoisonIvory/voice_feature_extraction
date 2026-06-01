# Speech Feature Extraction User Stories

## Primary User Story

As a researcher preparing to discuss Decibelle voice data with Professor Patel at Northeastern, I want a reproducible speech feature extraction pipeline that reprocesses my daily voice recordings with established acoustic methodology, so that I can clearly explain how the data was processed, present any menstrual-cycle voice patterns honestly, and identify the strongest next research questions.

## Context

I have been completing daily voice tests alongside menstrual-cycle tracking and Oura data. The motivating research question is:

> Does my voice change along my menstrual cycle?

The first presentation goal is collaboration, not overclaiming. The project should show that the analysis is methodologically credible, transparent about limitations, and ready for professor feedback.

## Wednesday Presentation Goal

By Wednesday, the project should support:

- A clear explanation of the pipeline and why it follows established methodology.
- A daily-level analysis view that connects voice features with cycle day, cycle phase, and Oura context.
- Preliminary plots if the pipeline produces enough valid data in time.
- A research plan that separates confirmed outputs, exploratory findings, limitations, and next analyses.

## Methodology Story

The analysis should avoid custom acoustic biomarker code and instead use standard tools:

- Appwrite Storage as the source of truth for saved WAV recordings.
- `voice_recordings` as preferred metadata when available.
- openSMILE `eGeMAPSv02` functionals as the primary interpretable feature set.
- openSMILE `ComParE_2016` functionals as a deferred broader exploratory ML feature set after the first canonical eGeMAPSv02 + Praat/QC pass is reproducible.
- Praat/Parselmouth for compact, established phonetic measures such as F0, jitter, shimmer, HNR, formants, duration, and pause/sounding measures where appropriate.
- Parquet as the canonical analysis output.

## Acceptance Criteria

- The pipeline processes only confidently identified `vowel` and `prosody` WAV recordings.
- Every included recording records the same `extractorVersion`, library versions, feature set names, and audio SHA256 hash.
- The output includes one canonical recording-level Parquet file and one audit Parquet file.
- A daily-level analysis table can be produced from the recording-level output for the professor presentation.
- CSV/XLSX review exports, if created, are generated from Parquet rather than directly from Appwrite.
- Missing metadata, task disagreements, failed downloads, failed extraction, and out-of-scope recordings are recorded in the audit output instead of being silently dropped.
- Findings are labeled as exploratory until the full pipeline, QC review, and analysis assumptions are complete.

## Out Of Scope For The First Pass

- Writing extracted features back into Appwrite.
- Extracting `ComParE_2016` in the canonical first-pass recording-level dataset.
- Reusing old Modal/custom biomarker outputs as analysis features.
- Building a production service or mobile-app feature.
- Making strong biological claims before the standardized reprocessing and QC review are complete.


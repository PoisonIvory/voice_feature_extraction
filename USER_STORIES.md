# Speech Feature Extraction User Stories

## Primary User Story

As a researcher, I want a reproducible speech feature extraction pipeline that reprocesses saved voice recordings with established acoustic methodology, so that I can produce trustworthy recording-level feature artifacts for downstream analysis.

## Context

The goal of this repository is strictly data preparation: transform raw Appwrite WAV recordings into standardized, auditable speech-feature outputs. Interpretation, visualization, and scientific claims are handled outside this project.

## Delivery Goal

The project should support:

- A clear explanation of the extraction pipeline and why it follows established methodology.
- Reproducible generation of daily task-separated feature and recording-level audit parquet artifacts.
- Transparent failure and skip reporting for QA and downstream consumers.

## Methodology Story

The extraction pipeline should avoid custom acoustic biomarker code and instead use standard tools:

- Appwrite Storage as the source of truth for saved WAV recordings.
- `voice_recordings` as preferred metadata when available.
- openSMILE `eGeMAPSv02` functionals as the primary interpretable feature set.
- openSMILE `ComParE_2016` functionals as a deferred optional feature set after the canonical eGeMAPSv02 extraction is stable.
- Praat/Parselmouth as a deferred optional future extractor.
- Parquet as the canonical analysis output.

## Acceptance Criteria

- The pipeline processes only confidently identified `vowel` and `prosody` WAV recordings.
- Every included recording records the same `extractorVersion`, library versions, feature set names, and audio SHA256 hash.
- The output includes one canonical daily Parquet file (one row per user/day with task-separated features) and one recording-level audit Parquet file.
- CSV/XLSX review exports, if created, are generated from Parquet rather than directly from Appwrite.
- Missing metadata, task disagreements, failed downloads, failed extraction, and out-of-scope recordings are recorded in the audit output instead of being silently dropped.
- The extraction run is reproducible and auditable without requiring any analysis or visualization step.

## Out Of Scope For The First Pass

- Writing extracted features back into Appwrite.
- Extracting `ComParE_2016` in the canonical first-pass recording-level dataset.
- Reusing old Modal/custom biomarker outputs as analysis features.
- Building a production service or mobile-app feature.
- Cycle-phase joins, Oura joins, plotting, statistical analysis, and research conclusions.


# Follow-Up Milestones

This document defines the next milestones after the extraction baseline hardening work.

## Goal

Move from a reproducible recording-level extraction pipeline to a presentation-ready analysis workflow aligned with `USER_STORIES.md`.

## Milestone 1: Daily-Level Analysis Artifact

Deliver a small module that reads the canonical recordings parquet and builds a daily table.

### Scope
- Input: `data/processed/voice_features_v3_recordings.parquet`
- Output: `data/processed/voice_features_v3_daily.parquet`
- Aggregation key: `userId`, `recordedDate`, `taskType`
- Aggregations: count, completion flags, and feature summary statistics for eGeMAPS columns

### Acceptance
- One deterministic daily row per user/date/task combination
- No Appwrite reads during this step (Parquet-only transformation)

## Milestone 2: Optional Review Exports

Deliver a thin export utility that converts Parquet artifacts into review-friendly formats.

### Scope
- Source of truth stays Parquet
- Export targets: CSV and XLSX in `exports/`

### Acceptance
- Exports are generated only from Parquet, never directly from Appwrite
- Export command can select recordings, audit, or daily artifact

## Milestone 3: Praat/Parselmouth Pass

Add a dedicated module for compact phonetic features requested in user stories.

### Scope
- Features: F0 summary, jitter, shimmer, HNR, formant summaries, duration/pause measures
- Keep module separate from openSMILE extractor to preserve single responsibility

### Acceptance
- New features are versioned and lineage-tagged
- Failures are written to audit output with stage-specific failure codes

## Milestone 4: Presentation Assets

Create researcher-facing deliverables for professor review.

### Scope
- Daily-level exploratory plots
- Short written summary with sections:
  - confirmed outputs
  - exploratory findings
  - limitations
  - next analyses

### Acceptance
- All findings clearly labeled exploratory
- Methods and limitations can be explained from artifacts alone

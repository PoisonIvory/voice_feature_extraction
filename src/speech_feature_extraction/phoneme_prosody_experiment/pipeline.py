"""Pipeline orchestration for phoneme prosody experiment.

This module coordinates the full extraction flow:
1. Alignment via MFA
2. Segment feature extraction with trim policy
3. Quality assessment
4. Output to parquet with full schema
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.alignment import (
    PROSODY_CANONICAL_TRANSCRIPTION,
    align_recording,
)
from speech_feature_extraction.phoneme_prosody_experiment.alignment_quality import (
    assess_segment_quality,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
    summarize_alignment_against_template,
)
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_FEATURES_FILENAME,
)
from speech_feature_extraction.phoneme_prosody_experiment.segment_features import (
    DEFAULT_TRIM_POLICY_MS,
    SegmentFeatureExtractor,
)
from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    classify_phoneme,
    normalize_phoneme_label,
)

LOGGER = logging.getLogger(__name__)

EXTRACTOR_VERSION = "phoneme_prosody_v2"


@dataclass
class PhonemeRowData:
    """One row of the prosody_phoneme_features output."""

    # Lineage
    recordingId: str
    userId: str
    recordedDate: str
    taskType: str
    audioHash: str
    extractorVersion: str
    alignmentEngine: str
    alignmentVersion: str
    # Alignment
    phonemeIndex: int
    phonemeLabel: str
    wordLabel: str | None
    startSec: float
    endSec: float
    durationSec: float
    alignmentScoreRaw: float | None
    alignmentQuality: str
    # Context
    prevPhonemeLabel: str | None
    nextPhonemeLabel: str | None
    coarticulationContext: str
    isAdjacentToNasal: bool
    # Boundaries
    trimPolicyMs: float
    analysisStartSec: float
    analysisEndSec: float
    analysisDurationSec: float
    # Rainbow Profile
    rainbowOccurrenceIndex: int | None
    rainbowExpectedPositionRatio: float | None
    rainbowObservedPositionRatio: float | None
    rainbowPositionDeltaRatio: float | None
    rainbowTimingConsistent: bool | None
    # Grouping
    phonemeClassPrimary: str
    phonemeClassTags: str
    # Features
    segment_mfcc2_mean: float | None
    segment_h1h2_mean: float | None
    segment_f1_bandwidth_mean: float | None
    segment_f0_mean: float | None
    # QC
    qc_segment_ok: bool
    qc_segment_reason: str
    qc_numFrames: int
    qc_minFramesRequired: int
    qc_label_canonical: bool


@dataclass
class RecordingMetadata:
    """Metadata for one recording to process."""

    recording_id: str
    user_id: str
    recorded_date: str
    task_type: str
    audio_path: Path
    transcription: str = PROSODY_CANONICAL_TRANSCRIPTION


def process_recording(
    metadata: RecordingMetadata,
    alignments_dir: Path,
    feature_extractor: SegmentFeatureExtractor | None = None,
    rainbow_template: Any | None = None,
) -> list[PhonemeRowData]:
    """Process a single recording through the full pipeline.

    Args:
        metadata: Recording metadata.
        alignments_dir: Directory for TextGrid output.
        feature_extractor: Reusable extractor instance (lazy init if None).
        rainbow_template: Optional template for position comparison.

    Returns:
        List of PhonemeRowData for each aligned segment.
    """
    audio_hash = _compute_file_hash(metadata.audio_path)

    alignment = align_recording(
        audio_path=metadata.audio_path,
        recording_id=metadata.recording_id,
        output_dir=alignments_dir,
        transcription=metadata.transcription,
    )

    if not alignment.success:
        LOGGER.warning(
            "Alignment failed for %s: %s",
            metadata.recording_id,
            alignment.error_message,
        )
        return []

    if not alignment.segments:
        LOGGER.warning("No segments from alignment for %s", metadata.recording_id)
        return []

    if feature_extractor is None:
        feature_extractor = SegmentFeatureExtractor()

    recording_frames = feature_extractor.extract_recording_frames(metadata.audio_path)

    occurrence_map = _build_occurrence_alignment_map(alignment.segments, rainbow_template)

    rows: list[PhonemeRowData] = []
    segments = list(alignment.segments)

    for idx, segment in enumerate(segments):
        prev_label = segments[idx - 1].phoneme_label if idx > 0 else None
        next_label = segments[idx + 1].phoneme_label if idx < len(segments) - 1 else None

        classification = classify_phoneme(
            segment.phoneme_label,
            prev_label=prev_label,
            next_label=next_label,
        )

        features, boundaries = feature_extractor.aggregate_window(
            recording_frames,
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            trim_policy_ms=DEFAULT_TRIM_POLICY_MS,
        )

        occ_key = _occurrence_key(classification.phoneme_label, idx, segments)
        rainbow_data = occurrence_map.get(occ_key, {})
        quality = assess_segment_quality(
            duration_sec=segment.end_sec - segment.start_sec,
            voiced_ratio=features.qc_voiced_ratio if features.qc_voiced_ratio > 0 else None,
            position_delta_ratio=rainbow_data.get("position_delta_ratio"),
        )

        word_label = _find_word_for_segment(segment, alignment.word_segments)

        row = PhonemeRowData(
            recordingId=metadata.recording_id,
            userId=metadata.user_id,
            recordedDate=metadata.recorded_date,
            taskType=metadata.task_type,
            audioHash=audio_hash,
            extractorVersion=EXTRACTOR_VERSION,
            alignmentEngine=alignment.alignment_engine,
            alignmentVersion=alignment.alignment_version,
            phonemeIndex=idx,
            phonemeLabel=classification.phoneme_label,
            wordLabel=word_label,
            startSec=segment.start_sec,
            endSec=segment.end_sec,
            durationSec=segment.end_sec - segment.start_sec,
            alignmentScoreRaw=quality.score_raw,
            alignmentQuality=quality.quality,
            prevPhonemeLabel=normalize_phoneme_label(prev_label),
            nextPhonemeLabel=normalize_phoneme_label(next_label),
            coarticulationContext=classification.coarticulation_context,
            isAdjacentToNasal=classification.is_adjacent_to_nasal,
            trimPolicyMs=boundaries.trim_policy_ms,
            analysisStartSec=boundaries.analysis_start_sec,
            analysisEndSec=boundaries.analysis_end_sec,
            analysisDurationSec=boundaries.analysis_end_sec - boundaries.analysis_start_sec,
            rainbowOccurrenceIndex=rainbow_data.get("occurrence_index"),
            rainbowExpectedPositionRatio=rainbow_data.get("expected_position_ratio"),
            rainbowObservedPositionRatio=rainbow_data.get("observed_position_ratio"),
            rainbowPositionDeltaRatio=rainbow_data.get("position_delta_ratio"),
            rainbowTimingConsistent=rainbow_data.get("timing_consistent"),
            phonemeClassPrimary=classification.phoneme_class_primary,
            phonemeClassTags=",".join(classification.phoneme_class_tags),
            segment_mfcc2_mean=features.mfcc2_mean,
            segment_h1h2_mean=features.h1h2_mean,
            segment_f1_bandwidth_mean=features.f1_bandwidth_mean,
            segment_f0_mean=features.f0_mean,
            qc_segment_ok=features.qc_segment_ok,
            qc_segment_reason=features.qc_segment_reason,
            qc_numFrames=features.qc_num_frames,
            qc_minFramesRequired=features.qc_min_frames_required,
            qc_label_canonical=classification.is_canonical,
        )
        rows.append(row)

    return rows


def process_batch(
    recordings: list[RecordingMetadata],
    output_dir: Path,
    feature_extractor: SegmentFeatureExtractor | None = None,
) -> tuple[Path, int, int]:
    """Process multiple recordings and write output parquet.

    Args:
        recordings: List of recording metadata.
        output_dir: Base output directory for experiment.
        feature_extractor: Optional shared extractor instance with custom settings.

    Returns:
        Tuple of (parquet_path, success_count, failure_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    alignments_dir = output_dir / "alignments"
    alignments_dir.mkdir(parents=True, exist_ok=True)

    if feature_extractor is None:
        feature_extractor = SegmentFeatureExtractor()

    all_rows: list[PhonemeRowData] = []
    success_count = 0
    failure_count = 0

    for metadata in recordings:
        LOGGER.info("Processing %s", metadata.recording_id)
        try:
            rows = process_recording(
                metadata=metadata,
                alignments_dir=alignments_dir,
                feature_extractor=feature_extractor,
            )
            if rows:
                all_rows.extend(rows)
                success_count += 1
                LOGGER.info("Extracted %d phoneme rows from %s", len(rows), metadata.recording_id)
            else:
                failure_count += 1
        except Exception as e:
            LOGGER.exception("Failed to process %s: %s", metadata.recording_id, e)
            failure_count += 1

    if not all_rows:
        LOGGER.warning("No phoneme rows extracted from batch")
        parquet_path = output_dir / PHONEME_PROSODY_FEATURES_FILENAME
        empty_df = pd.DataFrame(columns=list(PhonemeRowData.__dataclass_fields__.keys()))
        empty_df.to_parquet(parquet_path, index=False)
        return parquet_path, success_count, failure_count

    df = pd.DataFrame([asdict(row) for row in all_rows])
    parquet_path = output_dir / PHONEME_PROSODY_FEATURES_FILENAME
    df.to_parquet(parquet_path, index=False)
    LOGGER.info("Wrote %d rows to %s", len(df), parquet_path)

    return parquet_path, success_count, failure_count


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _find_word_for_segment(
    segment: AlignedPhonemeSegment,
    word_segments: tuple,
) -> str | None:
    """Find the word containing this phoneme segment."""
    seg_mid = (segment.start_sec + segment.end_sec) / 2
    for word in word_segments:
        if word.start_sec <= seg_mid <= word.end_sec:
            return word.word
    return None


def _build_occurrence_alignment_map(
    segments: tuple[AlignedPhonemeSegment, ...],
    template: Any | None,
) -> dict[str, dict[str, Any]]:
    """Build a map of occurrence keys to rainbow alignment data."""
    if template is None:
        return {}

    try:
        summary = summarize_alignment_against_template(segments, template)
    except Exception:
        return {}

    result: dict[str, dict[str, Any]] = {}
    for occ in summary.occurrence_alignments:
        key = f"{occ.phoneme_label}#{occ.occurrence_index}"
        result[key] = {
            "occurrence_index": occ.occurrence_index,
            "expected_position_ratio": occ.expected_position_ratio,
            "observed_position_ratio": occ.observed_position_ratio,
            "position_delta_ratio": occ.position_delta_ratio,
            "timing_consistent": occ.timing_consistent,
        }
    return result


def _occurrence_key(
    phoneme_label: str,
    index: int,
    all_segments: list[AlignedPhonemeSegment],
) -> str:
    """Build occurrence key for a segment.

    Deferred-template note: this counts occurrences over the raw, unfiltered
    segment order, whereas summarize_alignment_against_template counts over
    normalized, filtered, time-sorted segments. These two orderings must be
    reconciled before the rainbow template is wired into process_batch, or
    occurrence indices will not line up.
    """
    normalized = normalize_phoneme_label(phoneme_label) or phoneme_label
    count = 1
    for i in range(index):
        seg_label = normalize_phoneme_label(all_segments[i].phoneme_label)
        if seg_label == normalized:
            count += 1
    return f"{normalized}#{count}"

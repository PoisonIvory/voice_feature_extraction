"""Alignment quality calibration for phoneme prosody experiment.

This module handles:
1. Parsing raw alignment scores from MFA output
2. Categorizing segments into quality buckets (good/marginal/poor)
3. Supporting threshold calibration via manual Praat review samples
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

QUALITY_GOOD = "good"
QUALITY_MARGINAL = "marginal"
QUALITY_POOR = "poor"
QUALITY_UNKNOWN = "unknown"


class AlignmentQuality(Enum):
    """Quality bucket for aligned phoneme segments."""

    GOOD = QUALITY_GOOD
    MARGINAL = QUALITY_MARGINAL
    POOR = QUALITY_POOR
    UNKNOWN = QUALITY_UNKNOWN


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable thresholds for quality bucket assignment.

    MFA does not provide per-phone confidence scores in standard output.
    These thresholds are based on heuristics from segment characteristics:
    - Duration relative to expected phone duration
    - Position variance from template (if available)
    - Acoustic feature stability (voiced ratio, F0 continuity)

    Default values are conservative starting points. Calibrate by sampling
    segments across the score range and reviewing boundaries in Praat.
    """

    min_duration_good_sec: float = 0.040
    min_duration_marginal_sec: float = 0.025
    min_voiced_ratio_good: float = 0.3
    min_voiced_ratio_marginal: float = 0.1
    max_position_delta_good: float = 0.05
    max_position_delta_marginal: float = 0.15


DEFAULT_THRESHOLDS = QualityThresholds()


@dataclass(frozen=True)
class SegmentQualityAssessment:
    """Quality assessment result for one aligned segment."""

    quality: str
    score_raw: float | None
    duration_sec: float
    voiced_ratio: float | None
    position_delta_ratio: float | None
    reasons: tuple[str, ...]


def assess_segment_quality(
    duration_sec: float,
    voiced_ratio: float | None = None,
    position_delta_ratio: float | None = None,
    thresholds: QualityThresholds | None = None,
) -> SegmentQualityAssessment:
    """Assess quality bucket for a phoneme segment.

    Args:
        duration_sec: Segment duration in seconds.
        voiced_ratio: Proportion of voiced frames (0.0-1.0), if available.
        position_delta_ratio: Deviation from expected position in template.
        thresholds: Custom thresholds, or defaults if None.

    Returns:
        SegmentQualityAssessment with bucket and diagnostic reasons.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    reasons: list[str] = []
    score_components: list[float] = []

    duration_score = _score_duration(duration_sec, thresholds)
    score_components.append(duration_score)
    if duration_score < 0.5:
        reasons.append(f"short_duration:{duration_sec:.3f}s")

    if voiced_ratio is not None:
        voiced_score = _score_voiced_ratio(voiced_ratio, thresholds)
        score_components.append(voiced_score)
        if voiced_score < 0.5:
            reasons.append(f"low_voiced:{voiced_ratio:.2f}")

    if position_delta_ratio is not None:
        position_score = _score_position_delta(position_delta_ratio, thresholds)
        score_components.append(position_score)
        if position_score < 0.5:
            reasons.append(f"position_drift:{position_delta_ratio:.2f}")

    if not score_components:
        return SegmentQualityAssessment(
            quality=QUALITY_UNKNOWN,
            score_raw=None,
            duration_sec=duration_sec,
            voiced_ratio=voiced_ratio,
            position_delta_ratio=position_delta_ratio,
            reasons=("no_metrics",),
        )

    combined_score = sum(score_components) / len(score_components)
    quality = _bucket_from_score(combined_score)

    return SegmentQualityAssessment(
        quality=quality,
        score_raw=combined_score,
        duration_sec=duration_sec,
        voiced_ratio=voiced_ratio,
        position_delta_ratio=position_delta_ratio,
        reasons=tuple(reasons) if reasons else ("ok",),
    )


def _score_duration(duration_sec: float, thresholds: QualityThresholds) -> float:
    """Score duration on 0-1 scale."""
    if duration_sec >= thresholds.min_duration_good_sec:
        return 1.0
    if duration_sec >= thresholds.min_duration_marginal_sec:
        return 0.5
    return 0.0


def _score_voiced_ratio(voiced_ratio: float, thresholds: QualityThresholds) -> float:
    """Score voiced ratio on 0-1 scale."""
    if voiced_ratio >= thresholds.min_voiced_ratio_good:
        return 1.0
    if voiced_ratio >= thresholds.min_voiced_ratio_marginal:
        return 0.5
    return 0.0


def _score_position_delta(delta: float, thresholds: QualityThresholds) -> float:
    """Score position delta on 0-1 scale (lower delta is better)."""
    abs_delta = abs(delta)
    if abs_delta <= thresholds.max_position_delta_good:
        return 1.0
    if abs_delta <= thresholds.max_position_delta_marginal:
        return 0.5
    return 0.0


def _bucket_from_score(score: float) -> str:
    """Map combined score to quality bucket."""
    if score >= 0.75:
        return QUALITY_GOOD
    if score >= 0.4:
        return QUALITY_MARGINAL
    return QUALITY_POOR


@dataclass(frozen=True)
class CalibrationSample:
    """Sample for manual Praat review to calibrate thresholds."""

    recording_id: str
    phoneme_index: int
    phoneme_label: str
    start_sec: float
    end_sec: float
    duration_sec: float
    computed_quality: str
    computed_score: float | None


def select_calibration_samples(
    assessments: list[tuple[str, int, str, float, float, SegmentQualityAssessment]],
    samples_per_bucket: int = 5,
) -> list[CalibrationSample]:
    """Select stratified samples across quality buckets for manual review.

    Args:
        assessments: List of (recording_id, phoneme_index, phoneme_label,
                     start_sec, end_sec, assessment) tuples.
        samples_per_bucket: Number of samples to select per quality bucket.

    Returns:
        List of CalibrationSample objects for Praat review.
    """
    by_bucket: dict[str, list[CalibrationSample]] = {
        QUALITY_GOOD: [],
        QUALITY_MARGINAL: [],
        QUALITY_POOR: [],
    }

    for rec_id, idx, label, start, end, assessment in assessments:
        sample = CalibrationSample(
            recording_id=rec_id,
            phoneme_index=idx,
            phoneme_label=label,
            start_sec=start,
            end_sec=end,
            duration_sec=assessment.duration_sec,
            computed_quality=assessment.quality,
            computed_score=assessment.score_raw,
        )
        if assessment.quality in by_bucket:
            by_bucket[assessment.quality].append(sample)

    selected: list[CalibrationSample] = []
    for bucket, samples in by_bucket.items():
        sorted_samples = sorted(samples, key=lambda s: (s.computed_score or 0.5, s.duration_sec))
        if bucket == QUALITY_GOOD:
            selected.extend(sorted_samples[:samples_per_bucket])
        elif bucket == QUALITY_MARGINAL:
            mid = len(sorted_samples) // 2
            start_idx = max(0, mid - samples_per_bucket // 2)
            selected.extend(sorted_samples[start_idx : start_idx + samples_per_bucket])
        else:
            selected.extend(sorted_samples[-samples_per_bucket:])

    return selected


def export_calibration_samples_csv(
    samples: list[CalibrationSample],
    output_path: Path,
) -> Path:
    """Export calibration samples to CSV for manual review workflow.

    The CSV includes columns for manual annotation:
    - manual_quality: Reviewer's assessed quality
    - boundary_accurate: Whether boundaries look correct in Praat
    - notes: Free-form reviewer notes

    Args:
        samples: List of calibration samples.
        output_path: Path to write CSV file.

    Returns:
        Path to the written CSV file.
    """
    import csv

    fieldnames = [
        "recording_id",
        "phoneme_index",
        "phoneme_label",
        "start_sec",
        "end_sec",
        "duration_sec",
        "computed_quality",
        "computed_score",
        "manual_quality",
        "boundary_accurate",
        "notes",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "recording_id": sample.recording_id,
                "phoneme_index": sample.phoneme_index,
                "phoneme_label": sample.phoneme_label,
                "start_sec": f"{sample.start_sec:.4f}",
                "end_sec": f"{sample.end_sec:.4f}",
                "duration_sec": f"{sample.duration_sec:.4f}",
                "computed_quality": sample.computed_quality,
                "computed_score": f"{sample.computed_score:.3f}" if sample.computed_score else "",
                "manual_quality": "",
                "boundary_accurate": "",
                "notes": "",
            })

    return output_path

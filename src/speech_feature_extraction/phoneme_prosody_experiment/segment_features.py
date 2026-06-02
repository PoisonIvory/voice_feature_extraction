"""Segment-level feature extraction for phoneme prosody experiment.

This module handles:
1. Extracting frame-level LLDs via openSMILE once per recording
2. Applying transition trim policy to each phoneme window
3. Assigning frames to phonemes by timestamp
4. Computing robust segment aggregates with frame-count QC
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    AGGREGATE_STATS,
    EGEMAPS_LLD_NAMES,
    PHONEME_PROSODY_FEATURE_VALUE_FIELDS,
    lld_value_field,
)

if TYPE_CHECKING:
    import pandas as pd


# No boundary trim by default. Averaging LLDs across the whole phoneme (the
# "Full" method) is the most reliable approach in the vowel-formant literature,
# and it keeps the qc frame count equal to the phoneme's true duration so the
# 4-frame (40 ms) threshold acts as a clean ">=40 ms" inclusion criterion.
# A positive value re-enables steady-state trimming for callers that want it.
DEFAULT_TRIM_POLICY_MS = 0.0
MIN_ANALYSIS_DURATION_SEC = 0.030
# 4 frames at a 10 ms hop = 40 ms, the minimum phoneme duration for reliable
# formant/spectral measurement (Eyben et al. eGeMAPS; vowel-formant studies).
MIN_FRAMES_FOR_VARIANCE = 4
LLD_HOP_SIZE_SEC = 0.010

F0_LLD_NAME = "F0semitoneFrom27.5Hz_sma3nz"


def _empty_feature_values() -> dict[str, float | None]:
    return {name: None for name in PHONEME_PROSODY_FEATURE_VALUE_FIELDS}


@dataclass(frozen=True)
class SegmentBoundaries:
    """Analysis boundaries after trim policy application."""

    original_start_sec: float
    original_end_sec: float
    analysis_start_sec: float
    analysis_end_sec: float
    trim_policy_ms: float
    trim_applied: bool


@dataclass(frozen=True)
class SegmentFeatures:
    """Extracted features and QC metrics for one phoneme segment."""

    feature_values: dict[str, float | None] = field(default_factory=_empty_feature_values)
    qc_segment_ok: bool = False
    qc_segment_reason: str = "segment_too_short"
    qc_num_frames: int = 0
    qc_min_frames_required: int = MIN_FRAMES_FOR_VARIANCE
    qc_voiced_frames: int = 0
    qc_voiced_ratio: float = 0.0


def compute_segment_boundaries(
    start_sec: float,
    end_sec: float,
    trim_policy_ms: float = DEFAULT_TRIM_POLICY_MS,
    min_analysis_duration_sec: float = MIN_ANALYSIS_DURATION_SEC,
) -> SegmentBoundaries:
    """Apply trim policy to segment boundaries.

    The trim removes transition effects at phoneme boundaries. The default trim
    is 0 ms (no trim; whole-phoneme averaging). When a positive trim is
    requested, it is skipped if the resulting segment would be too short for
    stable analysis.

    Args:
        start_sec: Original segment start time.
        end_sec: Original segment end time.
        trim_policy_ms: Milliseconds to trim from each side.
        min_analysis_duration_sec: Minimum duration after trimming.

    Returns:
        SegmentBoundaries with analysis window and trim metadata.
    """
    original_duration = end_sec - start_sec
    trim_sec = trim_policy_ms / 1000.0

    proposed_start = start_sec + trim_sec
    proposed_end = end_sec - trim_sec
    proposed_duration = proposed_end - proposed_start

    if proposed_duration >= min_analysis_duration_sec:
        return SegmentBoundaries(
            original_start_sec=start_sec,
            original_end_sec=end_sec,
            analysis_start_sec=proposed_start,
            analysis_end_sec=proposed_end,
            trim_policy_ms=trim_policy_ms,
            trim_applied=True,
        )
    return SegmentBoundaries(
        original_start_sec=start_sec,
        original_end_sec=end_sec,
        analysis_start_sec=start_sec,
        analysis_end_sec=end_sec,
        trim_policy_ms=trim_policy_ms,
        trim_applied=False,
    )


class SegmentFeatureExtractor:
    """Extract LLD-based features from short phoneme segments."""

    def __init__(self, min_frames_for_variance: int = MIN_FRAMES_FOR_VARIANCE) -> None:
        try:
            opensmile = importlib.import_module("opensmile")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "openSMILE Python package is not installed. Install with: pip install opensmile"
            ) from error

        self._min_frames_for_variance = max(int(min_frames_for_variance), 1)
        self._opensmile = opensmile
        self._smile_lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            sampling_rate=16000,
            resample=True,
            channels=0,
            mixdown=False,
        )
        runtime_names = list(self._smile_lld.feature_names)
        if runtime_names != list(EGEMAPS_LLD_NAMES):
            raise ValueError(
                f"openSMILE LLD schema mismatch: runtime={runtime_names!r} "
                f"expected={list(EGEMAPS_LLD_NAMES)!r}"
            )

    @property
    def lld_feature_names(self) -> list[str]:
        return list(self._smile_lld.feature_names)

    def extract_recording_frames(self, audio_path: Path) -> "pd.DataFrame":
        """Extract frame-level LLDs once for an entire recording.

        openSMILE needs surrounding context for its analysis windows, so it is
        run a single time over the full audio rather than on per-phoneme slices
        (which would be too short and return all-NaN placeholder frames). Each
        frame is annotated with its center time in seconds (``_center_sec``) so
        phoneme windows can later select their frames by timestamp.
        """
        frames = self._smile_lld.process_file(str(audio_path)).reset_index()
        start_sec = frames["start"].dt.total_seconds()
        end_sec = frames["end"].dt.total_seconds()
        frames["_center_sec"] = (start_sec + end_sec) / 2.0
        return frames

    def aggregate_window(
        self,
        recording_frames: "pd.DataFrame",
        start_sec: float,
        end_sec: float,
        trim_policy_ms: float = DEFAULT_TRIM_POLICY_MS,
    ) -> tuple[SegmentFeatures, SegmentBoundaries]:
        """Aggregate the recording's LLD frames that fall inside one phoneme.

        A frame is assigned to the phoneme whose trimmed analysis window
        contains the frame center, so each frame belongs to exactly one phoneme
        and no per-segment audio slice is created.

        Args:
            recording_frames: Frame table from ``extract_recording_frames``.
            start_sec: Segment start time from alignment.
            end_sec: Segment end time from alignment.
            trim_policy_ms: Milliseconds to trim from boundaries.

        Returns:
            Tuple of (SegmentFeatures, SegmentBoundaries).
        """
        boundaries = compute_segment_boundaries(start_sec, end_sec, trim_policy_ms)

        center = recording_frames["_center_sec"].to_numpy()
        in_window = (center >= boundaries.analysis_start_sec) & (
            center < boundaries.analysis_end_sec
        )
        window_frames = recording_frames[in_window]

        if len(window_frames) == 0:
            return (
                _empty_features("segment_too_short", self._min_frames_for_variance),
                boundaries,
            )

        features = _compute_aggregates(
            window_frames,
            min_frames_for_variance=self._min_frames_for_variance,
        )
        return features, boundaries


def _compute_aggregates(
    lld_frame: "pd.DataFrame",
    min_frames_for_variance: int = MIN_FRAMES_FOR_VARIANCE,
) -> SegmentFeatures:
    """Compute robust aggregates from LLD frame data."""
    total_frames = len(lld_frame)

    f0_values = _get_column(lld_frame, F0_LLD_NAME)
    voiced_frames = 0
    voiced_ratio = 0.0
    voiced_mask: np.ndarray | None = None
    if f0_values is not None:
        voiced_mask = f0_values > 0
        voiced_frames = int(np.sum(voiced_mask))
        voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0

    qc_ok = total_frames >= min_frames_for_variance
    qc_reason = "ok" if qc_ok else "insufficient_frames"

    feature_values: dict[str, float | None] = {}
    for lld_name in EGEMAPS_LLD_NAMES:
        values = _get_column(lld_frame, lld_name)
        use_voiced_mask = lld_name.endswith("_sma3nz")
        for stat in AGGREGATE_STATS:
            field_name = lld_value_field(lld_name, stat)
            if stat == "mean":
                feature_values[field_name] = (
                    _safe_mean_masked(values, voiced_mask)
                    if use_voiced_mask
                    else _safe_mean(values)
                )
            elif stat == "median":
                feature_values[field_name] = (
                    _safe_median_masked(values, voiced_mask)
                    if use_voiced_mask
                    else _safe_median(values)
                )
            else:
                feature_values[field_name] = (
                    _safe_std_masked(values, voiced_mask)
                    if use_voiced_mask
                    else _safe_std(values)
                )

    return SegmentFeatures(
        feature_values=feature_values,
        qc_segment_ok=qc_ok,
        qc_segment_reason=qc_reason,
        qc_num_frames=total_frames,
        qc_min_frames_required=min_frames_for_variance,
        qc_voiced_frames=voiced_frames,
        qc_voiced_ratio=voiced_ratio,
    )


def _get_column(frame: "pd.DataFrame", col_name: str) -> np.ndarray | None:
    """Safely get a column as numpy array."""
    if col_name in frame.columns:
        return frame[col_name].to_numpy()
    return None


def _safe_mean(values: np.ndarray | None) -> float | None:
    """Compute mean, returning None if invalid."""
    if values is None or len(values) == 0:
        return None
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return None
    result = float(np.mean(finite_values))
    return result if np.isfinite(result) else None


def _safe_median(values: np.ndarray | None) -> float | None:
    """Compute median, returning None if invalid."""
    if values is None or len(values) == 0:
        return None
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return None
    result = float(np.median(finite_values))
    return result if np.isfinite(result) else None


def _safe_std(values: np.ndarray | None) -> float | None:
    """Compute sample std (ddof=1), returning None if fewer than 2 finite frames."""
    if values is None or len(values) == 0:
        return None
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 2:
        return None
    result = float(np.std(finite_values, ddof=1))
    return result if np.isfinite(result) else None


def _apply_voiced_mask(
    values: np.ndarray | None, voiced_mask: np.ndarray | None
) -> np.ndarray | None:
    """Restrict values to voiced (F0>0) frames, then to finite entries.

    Falls back to all finite frames when no voiced mask is available so the
    descriptor still resolves rather than going None.
    """
    if values is None or len(values) == 0:
        return None
    if voiced_mask is not None and len(voiced_mask) == len(values):
        values = values[voiced_mask]
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return None
    return finite_values


def _safe_mean_masked(
    values: np.ndarray | None, voiced_mask: np.ndarray | None
) -> float | None:
    """Compute mean over voiced frames only, returning None if invalid."""
    masked = _apply_voiced_mask(values, voiced_mask)
    if masked is None:
        return None
    result = float(np.mean(masked))
    return result if np.isfinite(result) else None


def _safe_median_masked(
    values: np.ndarray | None, voiced_mask: np.ndarray | None
) -> float | None:
    """Compute median over voiced frames only, returning None if invalid."""
    masked = _apply_voiced_mask(values, voiced_mask)
    if masked is None:
        return None
    result = float(np.median(masked))
    return result if np.isfinite(result) else None


def _safe_std_masked(
    values: np.ndarray | None, voiced_mask: np.ndarray | None
) -> float | None:
    """Compute sample std over voiced frames only, returning None if invalid."""
    masked = _apply_voiced_mask(values, voiced_mask)
    if masked is None or len(masked) < 2:
        return None
    result = float(np.std(masked, ddof=1))
    return result if np.isfinite(result) else None


def _empty_features(reason: str, min_frames_for_variance: int) -> SegmentFeatures:
    """Return empty features with failure reason."""
    return SegmentFeatures(
        feature_values=_empty_feature_values(),
        qc_segment_ok=False,
        qc_segment_reason=reason,
        qc_num_frames=0,
        qc_min_frames_required=min_frames_for_variance,
        qc_voiced_frames=0,
        qc_voiced_ratio=0.0,
    )

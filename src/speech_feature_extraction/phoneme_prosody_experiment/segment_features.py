"""Segment-level feature extraction for phoneme prosody experiment.

This module handles:
1. Extracting frame-level LLDs via openSMILE once per recording
2. Applying transition trim policy to each phoneme window
3. Assigning frames to phonemes by timestamp
4. Computing robust segment aggregates with frame-count QC
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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

    mfcc2_mean: float | None
    mfcc2_median: float | None
    h1h2_mean: float | None
    h1h2_median: float | None
    f1_bandwidth_mean: float | None
    f1_bandwidth_median: float | None
    f0_mean: float | None
    f0_median: float | None
    qc_segment_ok: bool
    qc_segment_reason: str
    qc_num_frames: int
    qc_min_frames_required: int
    qc_voiced_frames: int
    qc_voiced_ratio: float


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

    f0_col = "F0semitoneFrom27.5Hz_sma3nz"
    mfcc2_col = "mfcc2_sma3"
    f1_bw_col = "F1bandwidth_sma3nz"
    h1h2_col = "H1-H2_sma3nz"
    h1h2_legacy_col = "logRelF0-H1-H2_sma3nz"

    f0_values = _get_column(lld_frame, f0_col)
    mfcc2_values = _get_column(lld_frame, mfcc2_col)
    f1_bw_values = _get_column(lld_frame, f1_bw_col)
    h1h2_values = _get_column(lld_frame, h1h2_col)
    if h1h2_values is None:
        # openSMILE commonly exposes the eGeMAPSv02 H1-H2 proxy as logRelF0-H1-H2.
        h1h2_values = _get_column(lld_frame, h1h2_legacy_col)

    voiced_frames = 0
    voiced_ratio = 0.0
    if f0_values is not None:
        voiced_mask = f0_values > 0
        voiced_frames = int(np.sum(voiced_mask))
        voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0

    qc_ok = total_frames >= min_frames_for_variance
    qc_reason = "ok" if qc_ok else "insufficient_frames"

    return SegmentFeatures(
        mfcc2_mean=_safe_mean(mfcc2_values),
        mfcc2_median=_safe_median(mfcc2_values),
        h1h2_mean=_safe_mean(h1h2_values),
        h1h2_median=_safe_median(h1h2_values),
        f1_bandwidth_mean=_safe_mean(f1_bw_values),
        f1_bandwidth_median=_safe_median(f1_bw_values),
        f0_mean=_safe_mean_voiced(f0_values),
        f0_median=_safe_median_voiced(f0_values),
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


def _safe_mean_voiced(values: np.ndarray | None) -> float | None:
    """Compute mean of voiced (non-zero) values only."""
    if values is None or len(values) == 0:
        return None
    voiced = values[values > 0]
    if len(voiced) == 0:
        return None
    result = float(np.mean(voiced))
    return result if np.isfinite(result) else None


def _safe_median_voiced(values: np.ndarray | None) -> float | None:
    """Compute median of voiced (non-zero) values only."""
    if values is None or len(values) == 0:
        return None
    voiced = values[values > 0]
    if len(voiced) == 0:
        return None
    result = float(np.median(voiced))
    return result if np.isfinite(result) else None


def _empty_features(reason: str, min_frames_for_variance: int) -> SegmentFeatures:
    """Return empty features with failure reason."""
    return SegmentFeatures(
        mfcc2_mean=None,
        mfcc2_median=None,
        h1h2_mean=None,
        h1h2_median=None,
        f1_bandwidth_mean=None,
        f1_bandwidth_median=None,
        f0_mean=None,
        f0_median=None,
        qc_segment_ok=False,
        qc_segment_reason=reason,
        qc_num_frames=0,
        qc_min_frames_required=min_frames_for_variance,
        qc_voiced_frames=0,
        qc_voiced_ratio=0.0,
    )

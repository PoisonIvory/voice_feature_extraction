"""Segment-level feature extraction for phoneme prosody experiment.

This module handles:
1. Slicing audio segments from alignment boundaries
2. Applying transition trim policy
3. Extracting frame-level LLDs via openSMILE
4. Computing robust segment aggregates with frame-count QC
"""

from __future__ import annotations

import importlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


DEFAULT_TRIM_POLICY_MS = 20.0
MIN_ANALYSIS_DURATION_SEC = 0.030
MIN_FRAMES_FOR_VARIANCE = 5
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

    The trim removes transition effects at phoneme boundaries. Default is 20ms
    from each side, but if the resulting segment would be too short for stable
    analysis, the trim is skipped.

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


def slice_audio_segment(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    output_path: Path | None = None,
) -> Path:
    """Extract audio segment to a temporary or specified path.

    Args:
        audio_path: Source audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_path: Optional output path. If None, creates temp file.

    Returns:
        Path to the sliced segment WAV file.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "pydub is required for audio slicing. Install with: pip install pydub"
        ) from e

    audio = AudioSegment.from_wav(str(audio_path))
    start_ms = int(start_sec * 1000)
    end_ms = int(end_sec * 1000)
    segment = audio[start_ms:end_ms]

    if output_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(temp_file.name)
        temp_file.close()

    segment.export(str(output_path), format="wav")
    return output_path


class SegmentFeatureExtractor:
    """Extract LLD-based features from short phoneme segments."""

    def __init__(self) -> None:
        try:
            opensmile = importlib.import_module("opensmile")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "openSMILE Python package is not installed. Install with: pip install opensmile"
            ) from error

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

    def extract_segment_llds(self, audio_path: Path) -> "pd.DataFrame":
        """Extract frame-level LLDs from a segment audio file."""
        return self._smile_lld.process_file(str(audio_path))

    def extract_segment_features(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
        trim_policy_ms: float = DEFAULT_TRIM_POLICY_MS,
    ) -> tuple[SegmentFeatures, SegmentBoundaries]:
        """Extract features from a phoneme segment with trim policy.

        Args:
            audio_path: Path to the full recording audio.
            start_sec: Segment start time from alignment.
            end_sec: Segment end time from alignment.
            trim_policy_ms: Milliseconds to trim from boundaries.

        Returns:
            Tuple of (SegmentFeatures, SegmentBoundaries).
        """
        boundaries = compute_segment_boundaries(start_sec, end_sec, trim_policy_ms)

        analysis_duration = boundaries.analysis_end_sec - boundaries.analysis_start_sec
        if analysis_duration < MIN_ANALYSIS_DURATION_SEC:
            return (
                _empty_features("segment_too_short"),
                boundaries,
            )

        try:
            segment_path = slice_audio_segment(
                audio_path,
                boundaries.analysis_start_sec,
                boundaries.analysis_end_sec,
            )
        except Exception as e:
            return (
                _empty_features(f"slice_failed: {str(e)[:50]}"),
                boundaries,
            )

        try:
            lld_frame = self.extract_segment_llds(segment_path)
        except Exception as e:
            _cleanup_temp(segment_path)
            return (
                _empty_features(f"lld_failed: {str(e)[:50]}"),
                boundaries,
            )
        finally:
            _cleanup_temp(segment_path)

        if lld_frame.empty:
            return (
                _empty_features("no_lld_frames"),
                boundaries,
            )

        features = _compute_aggregates(lld_frame)
        return features, boundaries


def _compute_aggregates(lld_frame: "pd.DataFrame") -> SegmentFeatures:
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

    qc_ok = total_frames >= MIN_FRAMES_FOR_VARIANCE
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
        qc_min_frames_required=MIN_FRAMES_FOR_VARIANCE,
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
    result = float(np.nanmean(values))
    return result if np.isfinite(result) else None


def _safe_median(values: np.ndarray | None) -> float | None:
    """Compute median, returning None if invalid."""
    if values is None or len(values) == 0:
        return None
    result = float(np.nanmedian(values))
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


def _empty_features(reason: str) -> SegmentFeatures:
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
        qc_min_frames_required=MIN_FRAMES_FOR_VARIANCE,
        qc_voiced_frames=0,
        qc_voiced_ratio=0.0,
    )


def _cleanup_temp(path: Path) -> None:
    """Remove temporary file if it exists."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass

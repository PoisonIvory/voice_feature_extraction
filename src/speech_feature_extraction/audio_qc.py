"""Basic WAV lineage and quality checks.

This module provides low-level audio quality assessment including:
- File integrity (SHA256 hashing)
- Format validation (sample rate, channels, bit depth)
- Signal quality (clipping detection with ratio computation)
"""

from __future__ import annotations

import hashlib
import wave
from array import array
from pathlib import Path
from typing import Any

from speech_feature_extraction.constants import AUDIO_QC_THRESHOLDS


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file for lineage tracking."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> dict[str, Any]:
    """Inspect a WAV file and return quality metrics.

    Returns a dictionary with:
    - Format info: sample rate, channels, bit depth, duration
    - Clipping detection: boolean flag and ratio of clipped samples
    - Warning codes for any detected issues
    """
    warnings: list[str] = []
    try:
        with wave.open(str(path), "rb") as wav:
            sample_rate = wav.getframerate()
            channel_count = wav.getnchannels()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            duration_sec = frame_count / sample_rate if sample_rate else None
            frames = wav.readframes(frame_count)
    except wave.Error as error:
        return {
            "qc_audio_readable": False,
            "qc_failure_reason": f"wav_read_error: {error}",
            "qc_warning_codes": ["audio_unreadable"],
        }

    min_sample_rate_hz = AUDIO_QC_THRESHOLDS["min_sample_rate_hz"]
    if sample_rate < min_sample_rate_hz:
        return {
            "qc_audio_readable": False,
            "qc_sample_rate_hz": sample_rate,
            "qc_channel_count": channel_count,
            "qc_sample_width_bytes": sample_width,
            "qc_duration_sec": duration_sec,
            "qc_failure_reason": f"sample_rate_too_low:{sample_rate}<{min_sample_rate_hz}",
            "qc_warning_codes": ["sample_rate_too_low"],
        }

    clipping_detected, clipping_ratio = _detect_clipping_with_ratio(frames, sample_width)
    if clipping_detected:
        warnings.append("clipping")
    if duration_sec is not None and duration_sec < 1:
        warnings.append("short_duration")
    if channel_count != 1:
        warnings.append("non_mono_audio")

    return {
        "qc_audio_readable": True,
        "qc_sample_rate_hz": sample_rate,
        "qc_channel_count": channel_count,
        "qc_sample_width_bytes": sample_width,
        "qc_duration_sec": duration_sec,
        "qc_clipping_detected": clipping_detected,
        "qc_clipping_ratio": clipping_ratio,
        "qc_warning_codes": warnings,
        "qc_failure_reason": None,
    }


def _detect_clipping_with_ratio(frames: bytes, sample_width: int) -> tuple[bool, float]:
    """Detect clipping and compute ratio of clipped samples.

    For 16-bit audio, samples at or above 32760 (close to max 32767) are
    considered clipped. This threshold accounts for potential rounding
    while still catching true clipping.

    Args:
        frames: Raw audio bytes
        sample_width: Bytes per sample (2 for 16-bit)

    Returns:
        Tuple of (clipping_detected, clipping_ratio)
    """
    if sample_width != 2 or not frames:
        return False, 0.0

    samples = array("h")
    samples.frombytes(frames)

    total_samples = len(samples)
    if total_samples == 0:
        return False, 0.0

    clipped_count = sum(1 for sample in samples if abs(sample) >= 32760)
    clipping_ratio = clipped_count / total_samples

    return clipped_count > 0, clipping_ratio


def _detect_clipping(frames: bytes, sample_width: int) -> bool:
    """Detect clipping (legacy interface, returns boolean only)."""
    detected, _ = _detect_clipping_with_ratio(frames, sample_width)
    return detected

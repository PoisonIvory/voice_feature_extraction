"""Basic WAV lineage and quality checks."""

from __future__ import annotations

import hashlib
import wave
from array import array
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> dict[str, Any]:
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

    clipping_detected = _detect_clipping(frames, sample_width)
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
        "qc_warning_codes": warnings,
        "qc_failure_reason": None,
    }


def _detect_clipping(frames: bytes, sample_width: int) -> bool:
    if sample_width != 2 or not frames:
        return False

    samples = array("h")
    samples.frombytes(frames)
    return any(abs(sample) >= 32760 for sample in samples)

"""Task-specific quality control for speech feature extraction.

This module implements differentiated quality gating for vowel and prosody tasks,
following best practices from:
- ASHA Expert Panel 2018 protocols for voice assessment
- eGeMAPS (Eyben et al. 2015) documentation on voiced/unvoiced segmentation
- MDVP clinical thresholds for jitter/shimmer pathology detection

Key insight: Sustained vowels and connected speech (prosody) have fundamentally
different expected acoustic characteristics, so quality criteria must differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from speech_feature_extraction.constants import (
    AUDIO_QC_THRESHOLDS,
    PROSODY_QC_THRESHOLDS,
    VOWEL_QC_THRESHOLDS,
)


@dataclass
class TaskQcResult:
    """Result of task-specific quality control evaluation."""

    task_type: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qc_task_type": self.task_type,
            "qc_task_qc_passed": self.passed,
            "qc_task_warnings": self.warnings,
            "qc_task_failures": self.failures,
            **{f"qc_{k}": v for k, v in self.metrics.items()},
        }


def compute_voiced_ratio_from_llds(lld_frame: Any) -> float:
    """Compute ratio of voiced frames from openSMILE LLD output.

    The eGeMAPS LLD feature 'F0semitoneFrom27.5Hz_sma3nz' uses the 'nz' suffix
    (non-zero) convention: frames where F0 could not be detected have value 0.

    Args:
        lld_frame: pandas DataFrame from openSMILE LLD extraction

    Returns:
        Ratio of voiced frames (0.0 to 1.0)
    """
    if lld_frame.empty:
        return 0.0

    f0_column = "F0semitoneFrom27.5Hz_sma3nz"
    if f0_column not in lld_frame.columns:
        return 0.0

    f0_values = lld_frame[f0_column]
    total_frames = len(f0_values)
    if total_frames == 0:
        return 0.0

    voiced_frames = (f0_values > 0).sum()
    return float(voiced_frames / total_frames)


def compute_f0_coefficient_of_variation(lld_frame: Any) -> float | None:
    """Compute coefficient of variation of F0 from voiced frames only.

    For sustained vowels, low CoV indicates stable pitch (good task compliance).
    For prosody, higher CoV is expected and meaningful.

    Args:
        lld_frame: pandas DataFrame from openSMILE LLD extraction

    Returns:
        Coefficient of variation (std/mean), or None if insufficient data
    """
    if lld_frame.empty:
        return None

    f0_column = "F0semitoneFrom27.5Hz_sma3nz"
    if f0_column not in lld_frame.columns:
        return None

    f0_values = lld_frame[f0_column]
    voiced_f0 = f0_values[f0_values > 0]

    if len(voiced_f0) < 10:
        return None

    mean_f0 = voiced_f0.mean()
    if mean_f0 == 0:
        return None

    return float(voiced_f0.std() / mean_f0)


def evaluate_vowel_qc(
    duration_sec: float,
    voiced_ratio: float,
    f0_cov: float | None,
    jitter_percent: float | None,
    shimmer_percent: float | None,
    shimmer_db: float | None,
    clipping_ratio: float,
) -> TaskQcResult:
    """Evaluate quality control for sustained vowel task.

    Sustained vowels should be:
    - Predominantly voiced (>90% voiced frames)
    - Stable in pitch (low F0 coefficient of variation)
    - Within clinical jitter/shimmer thresholds
    - Free from clipping artifacts

    Args:
        duration_sec: Recording duration in seconds
        voiced_ratio: Ratio of voiced frames (0.0-1.0)
        f0_cov: F0 coefficient of variation, or None
        jitter_percent: Local jitter percentage, or None
        shimmer_percent: Local shimmer percentage, or None
        shimmer_db: Shimmer in dB, or None
        clipping_ratio: Ratio of clipped samples (0.0-1.0)

    Returns:
        TaskQcResult with pass/fail status and detailed metrics
    """
    thresholds = VOWEL_QC_THRESHOLDS
    audio_thresholds = AUDIO_QC_THRESHOLDS

    warnings: list[str] = []
    failures: list[str] = []
    metrics: dict[str, Any] = {
        "duration_sec": duration_sec,
        "voiced_ratio": voiced_ratio,
        "f0_cov": f0_cov,
        "jitter_percent": jitter_percent,
        "shimmer_percent": shimmer_percent,
        "shimmer_db": shimmer_db,
        "clipping_ratio": clipping_ratio,
    }

    if duration_sec < thresholds["min_duration_sec"]:
        failures.append(f"duration_too_short:{duration_sec:.2f}s<{thresholds['min_duration_sec']}s")

    if duration_sec > thresholds["max_duration_sec"]:
        warnings.append(f"duration_unusually_long:{duration_sec:.2f}s")

    if voiced_ratio < thresholds["min_voiced_ratio"]:
        failures.append(f"insufficient_voicing:{voiced_ratio:.2%}<{thresholds['min_voiced_ratio']:.0%}")

    if f0_cov is not None and f0_cov > thresholds["max_f0_coefficient_of_variation"]:
        warnings.append(f"unstable_pitch:cov={f0_cov:.3f}>{thresholds['max_f0_coefficient_of_variation']}")

    if jitter_percent is not None and jitter_percent > thresholds["max_jitter_percent"]:
        warnings.append(f"high_jitter:{jitter_percent:.2f}%>{thresholds['max_jitter_percent']}%")

    if shimmer_percent is not None and shimmer_percent > thresholds["max_shimmer_percent"]:
        warnings.append(f"high_shimmer:{shimmer_percent:.2f}%>{thresholds['max_shimmer_percent']}%")

    if shimmer_db is not None and shimmer_db > thresholds["max_shimmer_db"]:
        warnings.append(f"high_shimmer_db:{shimmer_db:.3f}dB>{thresholds['max_shimmer_db']}dB")

    if clipping_ratio > audio_thresholds["max_clipping_ratio"]:
        failures.append(f"clipping_detected:{clipping_ratio:.4%}>{audio_thresholds['max_clipping_ratio']:.2%}")

    passed = len(failures) == 0
    return TaskQcResult(
        task_type="vowel",
        passed=passed,
        warnings=warnings,
        failures=failures,
        metrics=metrics,
    )


def evaluate_prosody_qc(
    duration_sec: float,
    voiced_ratio: float,
    clipping_ratio: float,
) -> TaskQcResult:
    """Evaluate quality control for prosody/connected speech task.

    Prosody tasks should have:
    - Both voiced and unvoiced segments (natural pauses)
    - Sufficient duration for temporal pattern analysis
    - No clipping artifacts

    Note: Jitter/shimmer thresholds are NOT applied to prosody because
    these metrics are only reliable for sustained phonation.

    Args:
        duration_sec: Recording duration in seconds
        voiced_ratio: Ratio of voiced frames (0.0-1.0)
        clipping_ratio: Ratio of clipped samples (0.0-1.0)

    Returns:
        TaskQcResult with pass/fail status and detailed metrics
    """
    thresholds = PROSODY_QC_THRESHOLDS
    audio_thresholds = AUDIO_QC_THRESHOLDS

    warnings: list[str] = []
    failures: list[str] = []
    metrics: dict[str, Any] = {
        "duration_sec": duration_sec,
        "voiced_ratio": voiced_ratio,
        "clipping_ratio": clipping_ratio,
    }

    if duration_sec < thresholds["min_duration_sec"]:
        failures.append(f"duration_too_short:{duration_sec:.2f}s<{thresholds['min_duration_sec']}s")

    if duration_sec > thresholds["max_duration_sec"]:
        warnings.append(f"duration_unusually_long:{duration_sec:.2f}s")

    if voiced_ratio < thresholds["min_voiced_ratio"]:
        failures.append(f"insufficient_voicing:{voiced_ratio:.2%}<{thresholds['min_voiced_ratio']:.0%}")

    if voiced_ratio > thresholds["max_voiced_ratio"]:
        warnings.append(f"no_pauses_detected:{voiced_ratio:.2%}>{thresholds['max_voiced_ratio']:.0%}")

    if clipping_ratio > audio_thresholds["max_clipping_ratio"]:
        failures.append(f"clipping_detected:{clipping_ratio:.4%}>{audio_thresholds['max_clipping_ratio']:.2%}")

    passed = len(failures) == 0
    return TaskQcResult(
        task_type="prosody",
        passed=passed,
        warnings=warnings,
        failures=failures,
        metrics=metrics,
    )


def evaluate_task_qc(
    task_type: str,
    duration_sec: float,
    voiced_ratio: float,
    clipping_ratio: float,
    f0_cov: float | None = None,
    jitter_percent: float | None = None,
    shimmer_percent: float | None = None,
    shimmer_db: float | None = None,
) -> TaskQcResult:
    """Dispatch to task-specific QC evaluation.

    Args:
        task_type: Either 'vowel' or 'prosody'
        duration_sec: Recording duration in seconds
        voiced_ratio: Ratio of voiced frames (0.0-1.0)
        clipping_ratio: Ratio of clipped samples (0.0-1.0)
        f0_cov: F0 coefficient of variation (vowel only)
        jitter_percent: Local jitter percentage (vowel only)
        shimmer_percent: Local shimmer percentage (vowel only)
        shimmer_db: Shimmer in dB (vowel only)

    Returns:
        TaskQcResult with pass/fail status and detailed metrics
    """
    task_type_lower = task_type.lower() if task_type else ""

    if task_type_lower == "vowel":
        return evaluate_vowel_qc(
            duration_sec=duration_sec,
            voiced_ratio=voiced_ratio,
            f0_cov=f0_cov,
            jitter_percent=jitter_percent,
            shimmer_percent=shimmer_percent,
            shimmer_db=shimmer_db,
            clipping_ratio=clipping_ratio,
        )
    elif task_type_lower == "prosody":
        return evaluate_prosody_qc(
            duration_sec=duration_sec,
            voiced_ratio=voiced_ratio,
            clipping_ratio=clipping_ratio,
        )
    else:
        return TaskQcResult(
            task_type=task_type_lower,
            passed=False,
            failures=[f"unknown_task_type:{task_type}"],
            metrics={"duration_sec": duration_sec},
        )

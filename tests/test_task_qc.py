"""Tests for task-specific quality control logic."""

from __future__ import annotations

import pytest

from speech_feature_extraction.task_qc import (
    TaskQcResult,
    evaluate_prosody_qc,
    evaluate_task_qc,
    evaluate_vowel_qc,
)


class TestVowelQc:
    """Tests for sustained vowel quality control."""

    def test_vowel_passes_with_good_metrics(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.5,
            voiced_ratio=0.95,
            f0_cov=0.08,
            jitter_percent=0.8,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert result.task_type == "vowel"
        assert len(result.failures) == 0

    def test_vowel_fails_insufficient_voicing(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.70,
            f0_cov=0.10,
            jitter_percent=0.8,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is False
        assert any("insufficient_voicing" in f for f in result.failures)

    def test_vowel_fails_too_short(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=1.0,
            voiced_ratio=0.95,
            f0_cov=0.10,
            jitter_percent=0.8,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is False
        assert any("duration_too_short" in f for f in result.failures)

    def test_vowel_fails_clipping(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.95,
            f0_cov=0.10,
            jitter_percent=0.8,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.05,
        )
        assert result.passed is False
        assert any("clipping_detected" in f for f in result.failures)

    def test_vowel_warns_unstable_pitch(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.95,
            f0_cov=0.30,
            jitter_percent=0.8,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert any("unstable_pitch" in w for w in result.warnings)

    def test_vowel_warns_high_jitter(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.95,
            f0_cov=0.10,
            jitter_percent=2.5,
            shimmer_percent=3.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert any("high_jitter" in w for w in result.warnings)

    def test_vowel_warns_high_shimmer(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.95,
            f0_cov=0.10,
            jitter_percent=0.8,
            shimmer_percent=6.0,
            shimmer_db=0.25,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert any("high_shimmer" in w for w in result.warnings)

    def test_vowel_handles_none_values(self) -> None:
        result = evaluate_vowel_qc(
            duration_sec=3.0,
            voiced_ratio=0.95,
            f0_cov=None,
            jitter_percent=None,
            shimmer_percent=None,
            shimmer_db=None,
            clipping_ratio=0.0,
        )
        assert result.passed is True


class TestProsodyQc:
    """Tests for prosody/connected speech quality control."""

    def test_prosody_passes_with_good_metrics(self) -> None:
        result = evaluate_prosody_qc(
            duration_sec=5.0,
            voiced_ratio=0.65,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert result.task_type == "prosody"
        assert len(result.failures) == 0

    def test_prosody_fails_insufficient_voicing(self) -> None:
        result = evaluate_prosody_qc(
            duration_sec=5.0,
            voiced_ratio=0.15,
            clipping_ratio=0.0,
        )
        assert result.passed is False
        assert any("insufficient_voicing" in f for f in result.failures)

    def test_prosody_fails_too_short(self) -> None:
        result = evaluate_prosody_qc(
            duration_sec=1.5,
            voiced_ratio=0.65,
            clipping_ratio=0.0,
        )
        assert result.passed is False
        assert any("duration_too_short" in f for f in result.failures)

    def test_prosody_warns_no_pauses(self) -> None:
        result = evaluate_prosody_qc(
            duration_sec=5.0,
            voiced_ratio=0.98,
            clipping_ratio=0.0,
        )
        assert result.passed is True
        assert any("no_pauses_detected" in w for w in result.warnings)

    def test_prosody_fails_clipping(self) -> None:
        result = evaluate_prosody_qc(
            duration_sec=5.0,
            voiced_ratio=0.65,
            clipping_ratio=0.02,
        )
        assert result.passed is False
        assert any("clipping_detected" in f for f in result.failures)


class TestTaskQcDispatch:
    """Tests for task type dispatch logic."""

    def test_dispatch_vowel(self) -> None:
        result = evaluate_task_qc(
            task_type="vowel",
            duration_sec=3.0,
            voiced_ratio=0.95,
            clipping_ratio=0.0,
            f0_cov=0.10,
            jitter_percent=0.8,
        )
        assert result.task_type == "vowel"

    def test_dispatch_prosody(self) -> None:
        result = evaluate_task_qc(
            task_type="prosody",
            duration_sec=5.0,
            voiced_ratio=0.65,
            clipping_ratio=0.0,
        )
        assert result.task_type == "prosody"

    def test_dispatch_case_insensitive(self) -> None:
        result = evaluate_task_qc(
            task_type="VOWEL",
            duration_sec=3.0,
            voiced_ratio=0.95,
            clipping_ratio=0.0,
        )
        assert result.task_type == "vowel"

    def test_dispatch_unknown_task_fails(self) -> None:
        result = evaluate_task_qc(
            task_type="reading",
            duration_sec=5.0,
            voiced_ratio=0.65,
            clipping_ratio=0.0,
        )
        assert result.passed is False
        assert any("unknown_task_type" in f for f in result.failures)


class TestTaskQcResult:
    """Tests for TaskQcResult dataclass."""

    def test_to_dict_includes_all_fields(self) -> None:
        result = TaskQcResult(
            task_type="vowel",
            passed=True,
            warnings=["high_jitter"],
            failures=[],
            metrics={"duration_sec": 3.0, "voiced_ratio": 0.95},
        )
        result_dict = result.to_dict()

        assert result_dict["qc_task_type"] == "vowel"
        assert result_dict["qc_task_qc_passed"] is True
        assert result_dict["qc_task_warnings"] == ["high_jitter"]
        assert result_dict["qc_task_failures"] == []
        assert result_dict["qc_duration_sec"] == 3.0
        assert result_dict["qc_voiced_ratio"] == 0.95

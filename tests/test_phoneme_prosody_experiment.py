from pathlib import Path

from speech_feature_extraction.phoneme_prosody_experiment.alignment import (
    PROSODY_CANONICAL_TRANSCRIPTION,
    RAINBOW_PASSAGE_MEDIUM_TEXT,
    RAINBOW_PASSAGE_SHORT_TEXT,
    RAINBOW_PASSAGE_TEXT,
    WordSegment,
    _build_transcription_candidates,
    _candidate_transcriptions_for_duration,
    check_mfa_available,
)
from speech_feature_extraction.phoneme_prosody_experiment.biomarkers import (
    summarize_segment_qc_stats,
)
from speech_feature_extraction.phoneme_prosody_experiment.alignment_quality import (
    QUALITY_GOOD,
    QUALITY_MARGINAL,
    QUALITY_POOR,
    QualityThresholds,
    assess_segment_quality,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_inventory import (
    RAINBOW_PASSAGE_ARPABET_SEQUENCE,
    RAINBOW_PASSAGE_EXPECTED_INVENTORY,
    RAINBOW_PASSAGE_NASAL_COUNT,
    RAINBOW_PASSAGE_PHONE_COUNTS,
    RAINBOW_PASSAGE_TOTAL_PHONES,
    get_expected_phone_count,
    validate_phone_coverage,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
    build_rainbow_template,
    summarize_alignment_against_template,
)
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_EXPERIMENT_DATA_ROOT,
    PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS,
    PHONEME_PROSODY_REQUIRED_FIELDS,
)
from speech_feature_extraction.phoneme_prosody_experiment.segment_features import (
    MIN_ANALYSIS_DURATION_SEC,
    SegmentBoundaries,
    _compute_aggregates,
    compute_segment_boundaries,
)
from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    COARTICULATION_NASAL_BOTH,
    COARTICULATION_NASAL_LEFT,
    COARTICULATION_NASAL_RIGHT,
    PHONEME_CLASS_NASAL_COUPLED,
    PHONEME_CLASS_ORAL_ANTERIOR,
    PHONEME_CLASS_VOICELESS_FRICATION,
    classify_phoneme,
    derive_coarticulation_context,
    normalize_phoneme_label,
)


def test_normalize_phoneme_label_strips_stress_suffix() -> None:
    assert normalize_phoneme_label("ae1") == "AE"
    assert normalize_phoneme_label(" ng ") == "NG"


def test_normalize_phoneme_label_maps_common_mfa_ipa_symbols() -> None:
    assert normalize_phoneme_label("ɪ") == "IH"
    assert normalize_phoneme_label("ə") == "AH"
    assert normalize_phoneme_label("aj") == "AY"
    assert normalize_phoneme_label("tʰ") == "T"
    assert normalize_phoneme_label("ɫ̩") == "L"
    assert normalize_phoneme_label("ɲ") == "N"
    assert normalize_phoneme_label("ʉː") == "UW"


def test_derive_coarticulation_context_handles_directionality() -> None:
    assert derive_coarticulation_context("N", "M") == COARTICULATION_NASAL_BOTH
    assert derive_coarticulation_context("N", "T") == COARTICULATION_NASAL_LEFT
    assert derive_coarticulation_context("T", "NG") == COARTICULATION_NASAL_RIGHT


def test_classify_phoneme_assigns_nasal_class_to_nasal_adjacent_vowel() -> None:
    classification = classify_phoneme("AE1", prev_label="B", next_label="N")

    assert classification.phoneme_label == "AE"
    assert classification.phoneme_class_primary == "AE"
    assert PHONEME_CLASS_NASAL_COUPLED in classification.phoneme_class_tags
    assert classification.is_adjacent_to_nasal is True
    assert classification.coarticulation_context == COARTICULATION_NASAL_RIGHT


def test_classify_phoneme_uses_ipa_mapping_for_nasal_adjacency() -> None:
    classification = classify_phoneme("æ", prev_label="b", next_label="ŋ")

    assert classification.phoneme_label == "AE"
    assert PHONEME_CLASS_NASAL_COUPLED in classification.phoneme_class_tags
    assert classification.coarticulation_context == COARTICULATION_NASAL_RIGHT


def test_classify_phoneme_keeps_overlap_tags_for_voiceless_oral_phone() -> None:
    classification = classify_phoneme("S", prev_label="AA", next_label="T")

    assert classification.phoneme_class_primary == "S"
    assert PHONEME_CLASS_VOICELESS_FRICATION in classification.phoneme_class_tags
    assert PHONEME_CLASS_ORAL_ANTERIOR in classification.phoneme_class_tags


def test_classify_phoneme_keeps_non_target_phone_as_granular_primary() -> None:
    classification = classify_phoneme("R")

    assert classification.phoneme_class_primary == "R"
    assert classification.phoneme_class_tags == ()


def test_schema_fields_include_experiment_isolation_root_and_no_duplicates() -> None:
    assert PHONEME_PROSODY_EXPERIMENT_DATA_ROOT == "data/experimental/phoneme_prosody"
    assert len(PHONEME_PROSODY_REQUIRED_FIELDS) == len(set(PHONEME_PROSODY_REQUIRED_FIELDS))
    assert "coarticulationContext" in PHONEME_PROSODY_REQUIRED_FIELDS
    assert "phonemeClassTags" in PHONEME_PROSODY_REQUIRED_FIELDS
    assert "rainbowExpectedPositionRatio" in PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS


def test_build_rainbow_template_tracks_occurrence_counts_and_inventory() -> None:
    reference_segments = [
        AlignedPhonemeSegment("R", 0.00, 0.08),
        AlignedPhonemeSegment("AE1", 0.08, 0.16),
        AlignedPhonemeSegment("N", 0.16, 0.22),
        AlignedPhonemeSegment("B", 0.22, 0.30),
        AlignedPhonemeSegment("OW0", 0.30, 0.38),
        AlignedPhonemeSegment("N", 0.38, 0.46),
    ]
    template = build_rainbow_template(reference_segments)

    assert template.expected_sequence == ("R", "AE", "N", "B", "OW", "N")
    assert template.expected_counts["N"] == 2
    assert template.expected_inventory == ("AE", "B", "N", "OW", "R")


def test_summarize_alignment_against_template_reports_coverage_and_timing() -> None:
    template = build_rainbow_template(
        [
            AlignedPhonemeSegment("R", 0.00, 0.08),
            AlignedPhonemeSegment("AE1", 0.08, 0.16),
            AlignedPhonemeSegment("N", 0.16, 0.24),
            AlignedPhonemeSegment("B", 0.24, 0.32),
        ]
    )
    observed = [
        AlignedPhonemeSegment("R", 0.00, 0.07),
        AlignedPhonemeSegment("AE0", 0.07, 0.15),
        AlignedPhonemeSegment("N", 0.15, 0.22),
    ]
    summary = summarize_alignment_against_template(observed, template, timing_tolerance_ratio=0.12)

    assert summary.coverage_ratio == 0.75
    assert "B#1" in summary.missing_occurrence_keys
    assert summary.sequence_match_ratio >= 0.5
    assert len(summary.occurrence_alignments) == 3


def test_summarize_alignment_against_template_marks_unexpected_occurrence() -> None:
    template = build_rainbow_template(
        [
            AlignedPhonemeSegment("R", 0.00, 0.08),
            AlignedPhonemeSegment("EY1", 0.08, 0.16),
        ]
    )
    observed = [
        AlignedPhonemeSegment("R", 0.00, 0.08),
        AlignedPhonemeSegment("EY1", 0.08, 0.16),
        AlignedPhonemeSegment("Z", 0.16, 0.22),
    ]
    summary = summarize_alignment_against_template(observed, template)

    assert summary.coverage_ratio == 1.0
    assert "Z#1" in summary.unexpected_occurrence_keys


def test_rainbow_passage_text_is_not_empty() -> None:
    assert len(RAINBOW_PASSAGE_TEXT) > 100
    assert "rainbow" in RAINBOW_PASSAGE_TEXT.lower()
    assert "sunlight" in RAINBOW_PASSAGE_TEXT.lower()


def test_prosody_canonical_transcription_is_sentence_like() -> None:
    normalized = PROSODY_CANONICAL_TRANSCRIPTION.lower()
    assert len(normalized) > 80
    assert "division of white light into many beautiful colors" in normalized
    assert "two ends apparently beyond the horizon" in normalized


def test_rainbow_inventory_has_expected_phone_count() -> None:
    assert RAINBOW_PASSAGE_TOTAL_PHONES > 200
    assert len(RAINBOW_PASSAGE_ARPABET_SEQUENCE) == RAINBOW_PASSAGE_TOTAL_PHONES


def test_rainbow_inventory_contains_expected_phones() -> None:
    assert "N" in RAINBOW_PASSAGE_EXPECTED_INVENTORY
    assert "M" in RAINBOW_PASSAGE_EXPECTED_INVENTORY
    assert "AH" in RAINBOW_PASSAGE_EXPECTED_INVENTORY
    assert "EY" in RAINBOW_PASSAGE_EXPECTED_INVENTORY


def test_rainbow_nasal_count_is_reasonable() -> None:
    assert RAINBOW_PASSAGE_NASAL_COUNT > 20


def test_get_expected_phone_count_returns_correct_value() -> None:
    assert get_expected_phone_count("AH") == RAINBOW_PASSAGE_PHONE_COUNTS["AH"]
    assert get_expected_phone_count("NONEXISTENT") == 0


def test_validate_phone_coverage_detects_missing_and_unexpected() -> None:
    observed = {"N", "M", "ZZ"}
    missing, unexpected = validate_phone_coverage(observed)

    assert "ZZ" in unexpected
    assert "AH" in missing
    assert "N" not in missing


def test_check_mfa_available_returns_tuple() -> None:
    available, message = check_mfa_available()
    assert isinstance(available, bool)
    assert isinstance(message, str)


def test_word_segment_dataclass_is_frozen() -> None:
    segment = WordSegment(word="rainbow", start_sec=0.5, end_sec=1.0)
    assert segment.word == "rainbow"
    assert segment.start_sec == 0.5
    assert segment.end_sec == 1.0


def test_candidate_transcriptions_short_duration_prefers_short_text() -> None:
    candidates = _candidate_transcriptions_for_duration(6.0)
    assert candidates[0] == RAINBOW_PASSAGE_SHORT_TEXT
    assert candidates[1] == RAINBOW_PASSAGE_MEDIUM_TEXT


def test_candidate_transcriptions_medium_duration_prefers_medium_text() -> None:
    candidates = _candidate_transcriptions_for_duration(12.0)
    assert candidates[0] == RAINBOW_PASSAGE_MEDIUM_TEXT
    assert candidates[1] == RAINBOW_PASSAGE_SHORT_TEXT


def test_candidate_transcriptions_unknown_duration_prefers_full_text() -> None:
    candidates = _candidate_transcriptions_for_duration(None)
    assert candidates[0] == RAINBOW_PASSAGE_TEXT


def test_build_transcription_candidates_prioritizes_explicit_text() -> None:
    candidates = _build_transcription_candidates("Custom transcript", audio_path=Path(__file__))
    assert candidates[0] == "Custom transcript"
    assert RAINBOW_PASSAGE_TEXT in candidates


def test_compute_segment_boundaries_applies_trim_when_duration_allows() -> None:
    boundaries = compute_segment_boundaries(
        start_sec=0.0,
        end_sec=0.200,
        trim_policy_ms=20.0,
    )
    assert boundaries.trim_applied is True
    assert abs(boundaries.analysis_start_sec - 0.020) < 1e-9
    assert abs(boundaries.analysis_end_sec - 0.180) < 1e-9


def test_compute_segment_boundaries_skips_trim_for_short_segment() -> None:
    boundaries = compute_segment_boundaries(
        start_sec=0.0,
        end_sec=0.050,
        trim_policy_ms=20.0,
    )
    assert boundaries.trim_applied is False
    assert boundaries.analysis_start_sec == 0.0
    assert boundaries.analysis_end_sec == 0.050


def test_assess_segment_quality_returns_good_for_long_segment() -> None:
    assessment = assess_segment_quality(
        duration_sec=0.100,
        voiced_ratio=0.8,
    )
    assert assessment.quality == QUALITY_GOOD
    assert assessment.score_raw is not None
    assert assessment.score_raw >= 0.75


def test_assess_segment_quality_returns_marginal_for_short_segment() -> None:
    assessment = assess_segment_quality(
        duration_sec=0.030,
        voiced_ratio=0.2,
    )
    assert assessment.quality == QUALITY_MARGINAL


def test_assess_segment_quality_returns_poor_for_very_short_segment() -> None:
    assessment = assess_segment_quality(
        duration_sec=0.015,
        voiced_ratio=0.05,
    )
    assert assessment.quality == QUALITY_POOR
    assert "short_duration" in assessment.reasons[0] or "low_voiced" in str(assessment.reasons)


def test_assess_segment_quality_uses_custom_thresholds() -> None:
    strict_thresholds = QualityThresholds(
        min_duration_good_sec=0.200,
        min_duration_marginal_sec=0.100,
    )
    assessment = assess_segment_quality(
        duration_sec=0.080,
        thresholds=strict_thresholds,
    )
    assert assessment.quality == QUALITY_POOR


def test_compute_aggregates_uses_logrel_h1h2_fallback_column() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0, 110.0, 120.0, 130.0, 140.0],
            "mfcc2_sma3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "F1bandwidth_sma3nz": [50.0, 51.0, 52.0, 53.0, 54.0],
            "logRelF0-H1-H2_sma3nz": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    features = _compute_aggregates(lld_frame)
    assert features.h1h2_mean is not None
    assert abs(features.h1h2_mean - 0.3) < 1e-9


def test_compute_aggregates_honors_custom_min_frames_threshold() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0, 110.0, 120.0],
            "mfcc2_sma3": [1.0, 2.0, 3.0],
            "F1bandwidth_sma3nz": [50.0, 51.0, 52.0],
        }
    )

    features = _compute_aggregates(lld_frame, min_frames_for_variance=4)
    assert features.qc_segment_ok is False
    assert features.qc_min_frames_required == 4
    assert features.qc_segment_reason == "insufficient_frames"


def test_summarize_segment_qc_stats_counts_common_failure_reasons() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "qc_segment_ok": [True, False, False],
            "qc_segment_reason": ["ok", "segment_too_short", "insufficient_frames"],
            "qc_numFrames": [8, 0, 3],
            "qc_minFramesRequired": [4, 4, 4],
        }
    )
    summary = summarize_segment_qc_stats(frame)

    assert summary["total_rows"] == 3
    assert summary["qc_ok_rows"] == 1
    assert summary["segment_too_short_rows"] == 1
    assert summary["insufficient_frames_rows"] == 1



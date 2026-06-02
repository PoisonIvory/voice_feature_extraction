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
    PROSODY_CANONICAL_ARPABET_SEQUENCE,
    PROSODY_CANONICAL_EXPECTED_INVENTORY,
    PROSODY_CANONICAL_NASAL_COUNT,
    PROSODY_CANONICAL_PHONE_COUNTS,
    PROSODY_CANONICAL_TOTAL_PHONES,
    RAINBOW_PASSAGE_ARPABET_SEQUENCE,
    RAINBOW_PASSAGE_EXPECTED_INVENTORY,
    RAINBOW_PASSAGE_NASAL_COUNT,
    RAINBOW_PASSAGE_TOTAL_PHONES,
    get_expected_phone_count,
    validate_phone_coverage,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
    build_rainbow_template,
    summarize_alignment_against_template,
)
from speech_feature_extraction.phoneme_prosody_experiment.pipeline import (
    assess_recording_alignment,
)
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_EXPERIMENT_DATA_ROOT,
    PHONEME_PROSODY_FEATURE_VALUE_FIELDS,
    PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS,
    PHONEME_PROSODY_REQUIRED_FIELDS,
    lld_value_field,
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
    ARPABET_PHONEMES,
    BROAD_OBSTRUENT,
    BROAD_SONORANT,
    GROUPING_UNKNOWN,
    PHONEME_GROUPINGS,
    PhonemeGrouping,
    classify_phoneme,
    group_phoneme,
    derive_coarticulation_context,
    is_canonical_phoneme,
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


def test_normalize_phoneme_label_covers_previously_leaking_ipa_symbols() -> None:
    assert normalize_phoneme_label("ɒ") == "AO"
    assert normalize_phoneme_label("ʎ") == "L"
    assert normalize_phoneme_label("ɡ") == "G"  # script g U+0261
    assert normalize_phoneme_label("ɔj") == "OY"


def test_normalize_phoneme_label_strips_arpa_stress_digits() -> None:
    assert normalize_phoneme_label("AH0") == "AH"
    assert normalize_phoneme_label("EY1") == "EY"
    assert normalize_phoneme_label("ZH") == "ZH"


def test_is_canonical_phoneme_flags_canonical_and_noncanonical() -> None:
    assert is_canonical_phoneme("AH0") is True
    assert is_canonical_phoneme("ɒ") is True  # mapped to AO
    assert is_canonical_phoneme("QQ") is False
    assert is_canonical_phoneme(None) is False


def test_classify_phoneme_marks_canonical_and_noncanonical() -> None:
    assert classify_phoneme("AH0").is_canonical is True
    assert classify_phoneme("QQ").is_canonical is False


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


def test_group_phoneme_tags_consonant_dimensions() -> None:
    grouping = group_phoneme("S0")

    assert grouping == PhonemeGrouping(
        phoneme_label="S",
        manner="fricative",
        place="alveolar",
        voicing="voiceless",
        height=None,
        broad_class=BROAD_OBSTRUENT,
    )


def test_group_phoneme_tags_vowel_height_and_sonorant_broad_class() -> None:
    grouping = group_phoneme("IY1")

    assert grouping.manner == "vowel"
    assert grouping.place == "front"
    assert grouping.height == "high"
    assert grouping.broad_class == BROAD_SONORANT


def test_group_phoneme_returns_unknown_for_noncanonical_label() -> None:
    grouping = group_phoneme("QQ")

    assert grouping.manner == GROUPING_UNKNOWN
    assert grouping.broad_class == GROUPING_UNKNOWN
    assert grouping.height is None


def test_phoneme_groupings_cover_full_arpabet_inventory() -> None:
    assert set(PHONEME_GROUPINGS) == set(ARPABET_PHONEMES)


def test_grouping_fields_present_in_required_schema() -> None:
    for field in ("phonemeManner", "phonemePlace", "phonemeVoicing", "phonemeHeight", "phonemeBroadClass"):
        assert field in PHONEME_PROSODY_REQUIRED_FIELDS


def test_schema_fields_include_experiment_isolation_root_and_no_duplicates() -> None:
    assert PHONEME_PROSODY_EXPERIMENT_DATA_ROOT == "data/experimental/phoneme_prosody"
    assert len(PHONEME_PROSODY_REQUIRED_FIELDS) == len(set(PHONEME_PROSODY_REQUIRED_FIELDS))
    assert len(PHONEME_PROSODY_FEATURE_VALUE_FIELDS) == 75
    assert "coarticulationContext" in PHONEME_PROSODY_REQUIRED_FIELDS
    assert "phonemeClassTags" in PHONEME_PROSODY_REQUIRED_FIELDS
    assert "segment_mfcc2_mean" in PHONEME_PROSODY_FEATURE_VALUE_FIELDS
    assert "segment_F0semitoneFrom27_5Hz_mean" in PHONEME_PROSODY_FEATURE_VALUE_FIELDS
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


def test_get_expected_phone_count_uses_canonical_subset() -> None:
    assert get_expected_phone_count("AH") == PROSODY_CANONICAL_PHONE_COUNTS["AH"]
    assert get_expected_phone_count("NONEXISTENT") == 0


def test_prosody_canonical_inventory_covers_sentences_two_and_three() -> None:
    assert PROSODY_CANONICAL_TOTAL_PHONES == len(PROSODY_CANONICAL_ARPABET_SEQUENCE)
    assert PROSODY_CANONICAL_TOTAL_PHONES < RAINBOW_PASSAGE_TOTAL_PHONES
    # Sentence 2 begins "The rainbow is..." -> DH AH R EY N B OW.
    assert PROSODY_CANONICAL_ARPABET_SEQUENCE[:7] == ("DH", "AH", "R", "EY", "N", "B", "OW")
    assert PROSODY_CANONICAL_EXPECTED_INVENTORY <= RAINBOW_PASSAGE_EXPECTED_INVENTORY
    assert PROSODY_CANONICAL_NASAL_COUNT > 0


def test_prosody_canonical_inventory_excludes_unrecorded_phones() -> None:
    # "OY" only appears in the unrecorded remainder ("boiling"), not in 2-3.
    assert "OY" in RAINBOW_PASSAGE_EXPECTED_INVENTORY
    assert "OY" not in PROSODY_CANONICAL_EXPECTED_INVENTORY


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


def test_compute_aggregates_uses_logrel_h1h2_column() -> None:
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
    h1h2_mean = features.feature_values["segment_logRelF0_H1_H2_mean"]
    assert h1h2_mean is not None
    assert abs(h1h2_mean - 0.3) < 1e-9


def test_compute_aggregates_excludes_unvoiced_frames_from_voiced_source() -> None:
    import pandas as pd

    # First two frames are unvoiced (F0 == 0) with diluting H1-H2/F1bw values.
    # The voiced-only mean must ignore them and equal the mean of the voiced
    # frames (10.0 for H1-H2, 60.0 for F1 bandwidth), while MFCC2 stays
    # whole-segment.
    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [0.0, 0.0, 100.0, 110.0, 120.0],
            "mfcc2_sma3": [1.0, 1.0, 4.0, 4.0, 4.0],
            "F1bandwidth_sma3nz": [0.0, 0.0, 60.0, 60.0, 60.0],
            "logRelF0-H1-H2_sma3nz": [0.0, 0.0, 10.0, 10.0, 10.0],
        }
    )

    features = _compute_aggregates(lld_frame)

    assert abs(features.feature_values["segment_logRelF0_H1_H2_mean"] - 10.0) < 1e-9
    assert abs(features.feature_values["segment_F1bandwidth_mean"] - 60.0) < 1e-9
    assert features.qc_voiced_frames == 3
    assert abs(features.feature_values["segment_mfcc2_mean"] - (1.0 + 1.0 + 4.0 + 4.0 + 4.0) / 5.0) < 1e-9


def test_compute_aggregates_allows_negative_voiced_h1h2() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0, 110.0, 120.0, 130.0],
            "mfcc2_sma3": [1.0, 2.0, 3.0, 4.0],
            "F1bandwidth_sma3nz": [50.0, 51.0, 52.0, 53.0],
            "logRelF0-H1-H2_sma3nz": [-2.0, -1.0, 1.0, 2.0],
        }
    )

    features = _compute_aggregates(lld_frame)

    assert abs(features.feature_values["segment_logRelF0_H1_H2_mean"] - 0.0) < 1e-9


def test_compute_aggregates_emits_all_feature_fields() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0, 110.0, 120.0, 130.0],
            "mfcc2_sma3": [1.0, 2.0, 3.0, 4.0],
            "F1bandwidth_sma3nz": [50.0, 51.0, 52.0, 53.0],
            "logRelF0-H1-H2_sma3nz": [0.1, 0.2, 0.3, 0.4],
        }
    )

    features = _compute_aggregates(lld_frame)

    assert len(features.feature_values) == len(PHONEME_PROSODY_FEATURE_VALUE_FIELDS)
    assert set(features.feature_values) == set(PHONEME_PROSODY_FEATURE_VALUE_FIELDS)


def test_compute_aggregates_std_requires_two_frames() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0],
            "mfcc2_sma3": [1.0],
        }
    )

    features = _compute_aggregates(lld_frame, min_frames_for_variance=1)

    assert features.feature_values["segment_mfcc2_std"] is None
    assert features.feature_values["segment_mfcc2_mean"] == 1.0


def test_compute_aggregates_std_computed_for_multiple_frames() -> None:
    import pandas as pd

    lld_frame = pd.DataFrame(
        {
            "F0semitoneFrom27.5Hz_sma3nz": [100.0, 110.0, 120.0],
            "mfcc2_sma3": [1.0, 2.0, 3.0],
        }
    )

    features = _compute_aggregates(lld_frame)

    assert abs(features.feature_values["segment_mfcc2_std"] - 1.0) < 1e-9


def test_lld_value_field_naming() -> None:
    assert lld_value_field("mfcc2_sma3", "mean") == "segment_mfcc2_mean"
    assert lld_value_field("F0semitoneFrom27.5Hz_sma3nz", "mean") == "segment_F0semitoneFrom27_5Hz_mean"
    assert lld_value_field("logRelF0-H1-H2_sma3nz", "median") == "segment_logRelF0_H1_H2_median"
    assert lld_value_field("F1bandwidth_sma3nz", "std") == "segment_F1bandwidth_std"


def test_assess_recording_alignment_passes_canonical_inventory() -> None:
    segments = [
        AlignedPhonemeSegment(phoneme_label=phone, start_sec=i * 0.1, end_sec=i * 0.1 + 0.05)
        for i, phone in enumerate(PROSODY_CANONICAL_ARPABET_SEQUENCE)
    ]

    qc = assess_recording_alignment(segments)

    assert qc.ok is True
    assert qc.coverage_ratio == 1.0
    assert qc.unexpected_phones == 0


def test_assess_recording_alignment_flags_unexpected_phones() -> None:
    # "OY" is not in the sentences-2-3 inventory; an alignment that introduces
    # it indicates wrong-content force-alignment and must fail QC.
    segments = [
        AlignedPhonemeSegment(phoneme_label="DH", start_sec=0.0, end_sec=0.05),
        AlignedPhonemeSegment(phoneme_label="AH", start_sec=0.05, end_sec=0.10),
        AlignedPhonemeSegment(phoneme_label="OY", start_sec=0.10, end_sec=0.15),
    ]

    qc = assess_recording_alignment(segments)

    assert qc.ok is False
    assert qc.unexpected_phones >= 1


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


def test_summarize_segment_qc_stats_counts_non_canonical_labels() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "qc_segment_ok": [True, True, True],
            "qc_segment_reason": ["ok", "ok", "ok"],
            "qc_numFrames": [8, 8, 8],
            "qc_minFramesRequired": [4, 4, 4],
            "qc_label_canonical": [True, False, False],
        }
    )
    summary = summarize_segment_qc_stats(frame)

    assert summary["non_canonical_label_rows"] == 2



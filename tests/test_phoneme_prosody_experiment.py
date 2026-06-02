from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_EXPERIMENT_DATA_ROOT,
    PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS,
    PHONEME_PROSODY_REQUIRED_FIELDS,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
    build_rainbow_template,
    summarize_alignment_against_template,
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


def test_derive_coarticulation_context_handles_directionality() -> None:
    assert derive_coarticulation_context("N", "M") == COARTICULATION_NASAL_BOTH
    assert derive_coarticulation_context("N", "T") == COARTICULATION_NASAL_LEFT
    assert derive_coarticulation_context("T", "NG") == COARTICULATION_NASAL_RIGHT


def test_classify_phoneme_assigns_nasal_class_to_nasal_adjacent_vowel() -> None:
    classification = classify_phoneme("AE1", prev_label="B", next_label="N")

    assert classification.phoneme_label == "AE"
    assert classification.phoneme_class_primary == PHONEME_CLASS_NASAL_COUPLED
    assert classification.is_adjacent_to_nasal is True
    assert classification.coarticulation_context == COARTICULATION_NASAL_RIGHT


def test_classify_phoneme_keeps_overlap_tags_for_voiceless_oral_phone() -> None:
    classification = classify_phoneme("S", prev_label="AA", next_label="T")

    assert classification.phoneme_class_primary == PHONEME_CLASS_VOICELESS_FRICATION
    assert PHONEME_CLASS_VOICELESS_FRICATION in classification.phoneme_class_tags
    assert PHONEME_CLASS_ORAL_ANTERIOR in classification.phoneme_class_tags


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

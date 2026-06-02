"""Isolated prosody phoneme experiment primitives."""

from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_ALIGNMENT_FIELDS,
    PHONEME_PROSODY_BOUNDARY_FIELDS,
    PHONEME_PROSODY_CONTEXT_FIELDS,
    PHONEME_PROSODY_FEATURE_QC_FIELDS,
    PHONEME_PROSODY_FEATURE_VALUE_FIELDS,
    PHONEME_PROSODY_GROUPING_FIELDS,
    PHONEME_PROSODY_LINEAGE_FIELDS,
    PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS,
    PHONEME_PROSODY_REQUIRED_FIELDS,
)
from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
    RainbowAlignmentSummary,
    RainbowOccurrenceAlignment,
    RainbowPassageTemplate,
    RainbowPhonemeOccurrence,
    build_rainbow_template,
    summarize_alignment_against_template,
)
from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    NASAL_PHONEMES,
    PHARYNGEAL_ENGAGED_PHONEMES,
    PRIMARY_CLASS_TO_PHONEMES,
    VOICELESS_FRICATION_PHONEMES,
    classify_phoneme,
    derive_coarticulation_context,
)

__all__ = [
    "PHONEME_PROSODY_ALIGNMENT_FIELDS",
    "PHONEME_PROSODY_BOUNDARY_FIELDS",
    "PHONEME_PROSODY_CONTEXT_FIELDS",
    "PHONEME_PROSODY_FEATURE_QC_FIELDS",
    "PHONEME_PROSODY_FEATURE_VALUE_FIELDS",
    "PHONEME_PROSODY_GROUPING_FIELDS",
    "PHONEME_PROSODY_LINEAGE_FIELDS",
    "PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS",
    "PHONEME_PROSODY_REQUIRED_FIELDS",
    "AlignedPhonemeSegment",
    "NASAL_PHONEMES",
    "PHARYNGEAL_ENGAGED_PHONEMES",
    "PRIMARY_CLASS_TO_PHONEMES",
    "RainbowAlignmentSummary",
    "RainbowOccurrenceAlignment",
    "RainbowPassageTemplate",
    "RainbowPhonemeOccurrence",
    "VOICELESS_FRICATION_PHONEMES",
    "build_rainbow_template",
    "classify_phoneme",
    "derive_coarticulation_context",
    "summarize_alignment_against_template",
]

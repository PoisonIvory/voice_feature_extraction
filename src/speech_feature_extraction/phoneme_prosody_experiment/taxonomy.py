"""Prosody phoneme taxonomy and coarticulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re

STRESS_SUFFIX_PATTERN = re.compile(r"\d+$")

COARTICULATION_NONE = "none"
COARTICULATION_NASAL_LEFT = "nasal_left"
COARTICULATION_NASAL_RIGHT = "nasal_right"
COARTICULATION_NASAL_BOTH = "nasal_both"

PHONEME_CLASS_NASAL_COUPLED = "nasal_coupled"
PHONEME_CLASS_PHARYNGEAL_ENGAGED = "pharyngeal_engaged"
PHONEME_CLASS_ORAL_ANTERIOR = "oral_anterior"
PHONEME_CLASS_VOICELESS_FRICATION = "voiceless_frication"
PHONEME_CLASS_UNKNOWN = "unknown"

NASAL_PHONEMES = frozenset({"M", "N", "NG"})
PHARYNGEAL_ENGAGED_PHONEMES = frozenset({"AA", "AO", "AE"})
ORAL_ANTERIOR_PHONEMES = frozenset({"T", "D", "P", "B", "S", "F"})
VOICELESS_FRICATION_PHONEMES = frozenset({"S", "F", "TH", "SH"})
VOWEL_PHONEMES = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)

PRIMARY_CLASS_TO_PHONEMES = {
    PHONEME_CLASS_NASAL_COUPLED: NASAL_PHONEMES,
    PHONEME_CLASS_PHARYNGEAL_ENGAGED: PHARYNGEAL_ENGAGED_PHONEMES,
    PHONEME_CLASS_ORAL_ANTERIOR: ORAL_ANTERIOR_PHONEMES,
    PHONEME_CLASS_VOICELESS_FRICATION: VOICELESS_FRICATION_PHONEMES,
}


@dataclass(frozen=True)
class PhonemeClassification:
    """Taxonomy assignment for one aligned segment."""

    phoneme_label: str
    phoneme_class_primary: str
    phoneme_class_tags: tuple[str, ...]
    coarticulation_context: str
    is_adjacent_to_nasal: bool


def normalize_phoneme_label(label: str | None) -> str | None:
    """Normalize aligner phone labels to stress-free uppercase ARPAbet."""
    if label is None:
        return None
    normalized = STRESS_SUFFIX_PATTERN.sub("", label.strip().upper())
    return normalized or None


def derive_coarticulation_context(prev_label: str | None, next_label: str | None) -> str:
    """Classify nasal coarticulation context from neighboring phones."""
    prev_norm = normalize_phoneme_label(prev_label)
    next_norm = normalize_phoneme_label(next_label)
    prev_is_nasal = prev_norm in NASAL_PHONEMES
    next_is_nasal = next_norm in NASAL_PHONEMES
    if prev_is_nasal and next_is_nasal:
        return COARTICULATION_NASAL_BOTH
    if prev_is_nasal:
        return COARTICULATION_NASAL_LEFT
    if next_is_nasal:
        return COARTICULATION_NASAL_RIGHT
    return COARTICULATION_NONE


def classify_phoneme(
    phoneme_label: str,
    prev_label: str | None = None,
    next_label: str | None = None,
) -> PhonemeClassification:
    """Assign granular primary phone plus overlap class tags."""
    phone = normalize_phoneme_label(phoneme_label)
    if phone is None:
        return PhonemeClassification(
            phoneme_label="",
            phoneme_class_primary=PHONEME_CLASS_UNKNOWN,
            phoneme_class_tags=(PHONEME_CLASS_UNKNOWN,),
            coarticulation_context=COARTICULATION_NONE,
            is_adjacent_to_nasal=False,
        )

    coarticulation_context = derive_coarticulation_context(prev_label, next_label)
    is_adjacent_to_nasal = coarticulation_context != COARTICULATION_NONE
    is_vowel = phone in VOWEL_PHONEMES

    tags: list[str] = []
    if phone in NASAL_PHONEMES or (is_vowel and is_adjacent_to_nasal):
        tags.append(PHONEME_CLASS_NASAL_COUPLED)
    if phone in PHARYNGEAL_ENGAGED_PHONEMES:
        tags.append(PHONEME_CLASS_PHARYNGEAL_ENGAGED)
    if phone in ORAL_ANTERIOR_PHONEMES:
        tags.append(PHONEME_CLASS_ORAL_ANTERIOR)
    if phone in VOICELESS_FRICATION_PHONEMES:
        tags.append(PHONEME_CLASS_VOICELESS_FRICATION)

    return PhonemeClassification(
        phoneme_label=phone,
        # Granular classification follows aligner/dictionary ARPAbet phones.
        phoneme_class_primary=phone,
        phoneme_class_tags=tuple(dict.fromkeys(tags)),
        coarticulation_context=coarticulation_context,
        is_adjacent_to_nasal=is_adjacent_to_nasal,
    )

"""Rainbow Passage canonical phoneme inventory.

This module locks the expected phoneme sequence for the standard Rainbow Passage
so that alignment validation can detect missing or unexpected phones. The inventory
is derived from ARPAbet transcription of the text.

The Rainbow Passage:
"When the sunlight strikes raindrops in the air, they act as a prism and form a
rainbow. The rainbow is a division of white light into many beautiful colors.
These take the shape of a long round arch, with its path high above, and its two
ends apparently beyond the horizon. There is, according to legend, a boiling pot
of gold at one end. People look, but no one ever finds it. When a man looks for
something beyond his reach, his friends say he is looking for the pot of gold at
the end of the rainbow."
"""

from __future__ import annotations

from speech_feature_extraction.phoneme_prosody_experiment.alignment import RAINBOW_PASSAGE_TEXT

RAINBOW_PASSAGE_ARPABET_SEQUENCE = (
    # "When the sunlight strikes raindrops in the air,"
    "W", "EH", "N",
    "DH", "AH",
    "S", "AH", "N", "L", "AY", "T",
    "S", "T", "R", "AY", "K", "S",
    "R", "EY", "N", "D", "R", "AA", "P", "S",
    "IH", "N",
    "DH", "AH",
    "EH", "R",
    # "they act as a prism and form a rainbow."
    "DH", "EY",
    "AE", "K", "T",
    "AE", "Z",
    "AH",
    "P", "R", "IH", "Z", "AH", "M",
    "AH", "N", "D",
    "F", "AO", "R", "M",
    "AH",
    "R", "EY", "N", "B", "OW",
    # "The rainbow is a division of white light into many beautiful colors."
    "DH", "AH",
    "R", "EY", "N", "B", "OW",
    "IH", "Z",
    "AH",
    "D", "IH", "V", "IH", "ZH", "AH", "N",
    "AH", "V",
    "W", "AY", "T",
    "L", "AY", "T",
    "IH", "N", "T", "UW",
    "M", "EH", "N", "IY",
    "B", "Y", "UW", "T", "AH", "F", "AH", "L",
    "K", "AH", "L", "ER", "Z",
    # "These take the shape of a long round arch,"
    "DH", "IY", "Z",
    "T", "EY", "K",
    "DH", "AH",
    "SH", "EY", "P",
    "AH", "V",
    "AH",
    "L", "AO", "NG",
    "R", "AW", "N", "D",
    "AA", "R", "CH",
    # "with its path high above, and its two ends apparently beyond the horizon."
    "W", "IH", "TH",
    "IH", "T", "S",
    "P", "AE", "TH",
    "HH", "AY",
    "AH", "B", "AH", "V",
    "AH", "N", "D",
    "IH", "T", "S",
    "T", "UW",
    "EH", "N", "D", "Z",
    "AH", "P", "EH", "R", "AH", "N", "T", "L", "IY",
    "B", "IH", "AA", "N", "D",
    "DH", "AH",
    "HH", "ER", "AY", "Z", "AH", "N",
    # "There is, according to legend, a boiling pot of gold at one end."
    "DH", "EH", "R",
    "IH", "Z",
    "AH", "K", "AO", "R", "D", "IH", "NG",
    "T", "UW",
    "L", "EH", "JH", "AH", "N", "D",
    "AH",
    "B", "OY", "L", "IH", "NG",
    "P", "AA", "T",
    "AH", "V",
    "G", "OW", "L", "D",
    "AE", "T",
    "W", "AH", "N",
    "EH", "N", "D",
    # "People look, but no one ever finds it."
    "P", "IY", "P", "AH", "L",
    "L", "UH", "K",
    "B", "AH", "T",
    "N", "OW",
    "W", "AH", "N",
    "EH", "V", "ER",
    "F", "AY", "N", "D", "Z",
    "IH", "T",
    # "When a man looks for something beyond his reach,"
    "W", "EH", "N",
    "AH",
    "M", "AE", "N",
    "L", "UH", "K", "S",
    "F", "AO", "R",
    "S", "AH", "M", "TH", "IH", "NG",
    "B", "IH", "AA", "N", "D",
    "HH", "IH", "Z",
    "R", "IY", "CH",
    # "his friends say he is looking for the pot of gold at the end of the rainbow."
    "HH", "IH", "Z",
    "F", "R", "EH", "N", "D", "Z",
    "S", "EY",
    "HH", "IY",
    "IH", "Z",
    "L", "UH", "K", "IH", "NG",
    "F", "AO", "R",
    "DH", "AH",
    "P", "AA", "T",
    "AH", "V",
    "G", "OW", "L", "D",
    "AE", "T",
    "DH", "AH",
    "EH", "N", "D",
    "AH", "V",
    "DH", "AH",
    "R", "EY", "N", "B", "OW",
)

RAINBOW_PASSAGE_EXPECTED_INVENTORY = frozenset(RAINBOW_PASSAGE_ARPABET_SEQUENCE)

RAINBOW_PASSAGE_PHONE_COUNTS: dict[str, int] = {}
for phone in RAINBOW_PASSAGE_ARPABET_SEQUENCE:
    RAINBOW_PASSAGE_PHONE_COUNTS[phone] = RAINBOW_PASSAGE_PHONE_COUNTS.get(phone, 0) + 1

RAINBOW_PASSAGE_TOTAL_PHONES = len(RAINBOW_PASSAGE_ARPABET_SEQUENCE)

RAINBOW_PASSAGE_NASAL_COUNT = sum(
    RAINBOW_PASSAGE_PHONE_COUNTS.get(phone, 0) for phone in ("M", "N", "NG")
)

RAINBOW_PASSAGE_PHARYNGEAL_COUNT = sum(
    RAINBOW_PASSAGE_PHONE_COUNTS.get(phone, 0) for phone in ("AA", "AO", "AE")
)


def get_expected_phone_count(phone: str) -> int:
    """Return expected occurrence count for a phone in the Rainbow Passage."""
    return RAINBOW_PASSAGE_PHONE_COUNTS.get(phone.upper(), 0)


def validate_phone_coverage(observed_phones: set[str]) -> tuple[set[str], set[str]]:
    """Check observed phones against expected inventory.

    Returns:
        Tuple of (missing_phones, unexpected_phones).
    """
    observed_normalized = {p.upper() for p in observed_phones}
    expected = RAINBOW_PASSAGE_EXPECTED_INVENTORY

    missing = expected - observed_normalized
    unexpected = observed_normalized - expected

    return missing, unexpected

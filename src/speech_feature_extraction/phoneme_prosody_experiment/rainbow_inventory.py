"""Rainbow Passage canonical phoneme inventory.

This module locks the expected phoneme sequence so alignment validation can
detect missing or unexpected phones. The inventory is derived from ARPAbet
transcription of the text.

The prosody task records only sentences 2-3 of the Rainbow Passage
(``PROSODY_CANONICAL_TRANSCRIPTION``):

    "The rainbow is a division of white light into many beautiful colors.
    These take the shape of a long round arch, with its path high above, and
    its two ends apparently beyond the horizon."

``PROSODY_CANONICAL_*`` constants describe that recorded subset and are what
coverage validation uses. ``RAINBOW_PASSAGE_*`` constants describe the full
passage and are retained for reference only.
"""

from __future__ import annotations

# Sentence 1: "When the sunlight strikes raindrops in the air, they act as a
# prism and form a rainbow." (full-passage reference only; not recorded)
_SENTENCE_ONE_ARPABET = (
    "W", "EH", "N",
    "DH", "AH",
    "S", "AH", "N", "L", "AY", "T",
    "S", "T", "R", "AY", "K", "S",
    "R", "EY", "N", "D", "R", "AA", "P", "S",
    "IH", "N",
    "DH", "AH",
    "EH", "R",
    "DH", "EY",
    "AE", "K", "T",
    "AE", "Z",
    "AH",
    "P", "R", "IH", "Z", "AH", "M",
    "AH", "N", "D",
    "F", "AO", "R", "M",
    "AH",
    "R", "EY", "N", "B", "OW",
)

# Sentence 2: "The rainbow is a division of white light into many beautiful
# colors." (recorded by the prosody task)
_SENTENCE_TWO_ARPABET = (
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
)

# Sentence 3: "These take the shape of a long round arch, with its path high
# above, and its two ends apparently beyond the horizon." (recorded)
_SENTENCE_THREE_ARPABET = (
    "DH", "IY", "Z",
    "T", "EY", "K",
    "DH", "AH",
    "SH", "EY", "P",
    "AH", "V",
    "AH",
    "L", "AO", "NG",
    "R", "AW", "N", "D",
    "AA", "R", "CH",
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
)

# Remaining passage sentences (full-passage reference only; not recorded).
_PASSAGE_REMAINDER_ARPABET = (
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

# Canonical recorded subset: sentences 2-3.
PROSODY_CANONICAL_ARPABET_SEQUENCE = _SENTENCE_TWO_ARPABET + _SENTENCE_THREE_ARPABET

# Full passage, retained for reference only.
RAINBOW_PASSAGE_ARPABET_SEQUENCE = (
    _SENTENCE_ONE_ARPABET
    + _SENTENCE_TWO_ARPABET
    + _SENTENCE_THREE_ARPABET
    + _PASSAGE_REMAINDER_ARPABET
)


def _count_phones(sequence: tuple[str, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phone in sequence:
        counts[phone] = counts.get(phone, 0) + 1
    return counts


# Canonical (prosody, sentences 2-3) derived constants.
PROSODY_CANONICAL_EXPECTED_INVENTORY = frozenset(PROSODY_CANONICAL_ARPABET_SEQUENCE)
PROSODY_CANONICAL_PHONE_COUNTS = _count_phones(PROSODY_CANONICAL_ARPABET_SEQUENCE)
PROSODY_CANONICAL_TOTAL_PHONES = len(PROSODY_CANONICAL_ARPABET_SEQUENCE)
PROSODY_CANONICAL_NASAL_COUNT = sum(
    PROSODY_CANONICAL_PHONE_COUNTS.get(phone, 0) for phone in ("M", "N", "NG")
)
PROSODY_CANONICAL_PHARYNGEAL_COUNT = sum(
    PROSODY_CANONICAL_PHONE_COUNTS.get(phone, 0) for phone in ("AA", "AO", "AE")
)

# Full-passage (reference only) derived constants.
RAINBOW_PASSAGE_EXPECTED_INVENTORY = frozenset(RAINBOW_PASSAGE_ARPABET_SEQUENCE)
RAINBOW_PASSAGE_PHONE_COUNTS = _count_phones(RAINBOW_PASSAGE_ARPABET_SEQUENCE)
RAINBOW_PASSAGE_TOTAL_PHONES = len(RAINBOW_PASSAGE_ARPABET_SEQUENCE)
RAINBOW_PASSAGE_NASAL_COUNT = sum(
    RAINBOW_PASSAGE_PHONE_COUNTS.get(phone, 0) for phone in ("M", "N", "NG")
)
RAINBOW_PASSAGE_PHARYNGEAL_COUNT = sum(
    RAINBOW_PASSAGE_PHONE_COUNTS.get(phone, 0) for phone in ("AA", "AO", "AE")
)


def get_expected_phone_count(phone: str) -> int:
    """Return expected occurrence count for a phone in the recorded subset."""
    return PROSODY_CANONICAL_PHONE_COUNTS.get(phone.upper(), 0)


def validate_phone_coverage(observed_phones: set[str]) -> tuple[set[str], set[str]]:
    """Check observed phones against the recorded sentences-2-3 inventory.

    Returns:
        Tuple of (missing_phones, unexpected_phones).
    """
    observed_normalized = {p.upper() for p in observed_phones}
    expected = PROSODY_CANONICAL_EXPECTED_INVENTORY

    missing = expected - observed_normalized
    unexpected = observed_normalized - expected

    return missing, unexpected

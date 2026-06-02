"""Prosody phoneme taxonomy and coarticulation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

STRESS_SUFFIX_PATTERN = re.compile(r"\d+$")

# Canonical ARPAbet inventory used for downstream taxonomy and grouping.
ARPABET_PHONEMES = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "B",
        "CH",
        "D",
        "DH",
        "EH",
        "ER",
        "EY",
        "F",
        "G",
        "HH",
        "IH",
        "IY",
        "JH",
        "K",
        "L",
        "M",
        "N",
        "NG",
        "OW",
        "OY",
        "P",
        "R",
        "S",
        "SH",
        "T",
        "TH",
        "UH",
        "UW",
        "V",
        "W",
        "Y",
        "Z",
        "ZH",
    }
)

# Common MFA IPA-style symbols mapped to canonical ARPAbet.
IPA_TO_ARPABET = {
    "a": "AA",
    "aː": "AA",
    "aj": "AY",
    "aw": "AW",
    "b": "B",
    "c": "K",
    "d": "D",
    "dʒ": "JH",
    "d͡ʒ": "JH",
    "e": "EY",
    "ej": "EY",
    "f": "F",
    "g": "G",
    "h": "HH",
    "i": "IY",
    "j": "Y",
    "k": "K",
    "l": "L",
    "m": "M",
    "n": "N",
    "o": "OW",
    "oj": "OY",
    "ow": "OW",
    "p": "P",
    "r": "R",
    "s": "S",
    "t": "T",
    "tʃ": "CH",
    "t͡ʃ": "CH",
    "u": "UW",
    "v": "V",
    "w": "W",
    "z": "Z",
    "æ": "AE",
    "ð": "DH",
    "ŋ": "NG",
    "ɑ": "AA",
    "ɒ": "AO",
    "ɔ": "AO",
    "ɔj": "OY",
    "ə": "AH",
    "ɚ": "ER",
    "ɝ": "ER",
    "ɐ": "AH",
    "ɟ": "G",
    "ɡ": "G",
    "ɫ": "L",
    "ɲ": "N",
    "ɾ": "D",
    "ɛ": "EH",
    "ɪ": "IH",
    "ɹ": "R",
    "ʉ": "UW",
    "ʉː": "UW",
    "ʃ": "SH",
    "ʊ": "UH",
    "ʎ": "L",
    "ʒ": "ZH",
    "θ": "TH",
}

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
    is_canonical: bool


def normalize_phoneme_label(label: str | None) -> str | None:
    """Normalize aligner phone labels to stress-free uppercase ARPAbet."""
    if label is None:
        return None
    stripped = STRESS_SUFFIX_PATTERN.sub("", label.strip())
    if not stripped:
        return None
    canonical = stripped.upper()
    if canonical in ARPABET_PHONEMES:
        return canonical

    # Defensive fallback: the ARPAbet-native english_us_arpa model emits
    # ARPAbet directly, but if an IPA-style symbol ever appears we map it.
    ipa_key = _strip_diacritics(stripped).lower()
    mapped = IPA_TO_ARPABET.get(ipa_key)
    if mapped:
        return mapped

    fallback = stripped.upper()
    return fallback or None


def is_canonical_phoneme(label: str | None) -> bool:
    """Return True if the normalized label is a canonical ARPAbet phone."""
    return normalize_phoneme_label(label) in ARPABET_PHONEMES


def _strip_diacritics(value: str) -> str:
    """Remove combining IPA marks to stabilize dictionary lookup."""
    normalized = unicodedata.normalize("NFD", value)
    cleaned = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    for marker in ("ʰ", "ʲ", "ʷ", "ː", "ˑ"):
        cleaned = cleaned.replace(marker, "")
    return cleaned


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
            is_canonical=False,
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
        is_canonical=phone in ARPABET_PHONEMES,
    )


# ── Articulatory grouping ───────────────────────────────────────────────
# Middle grouping between the recording (wav) and the individual phoneme
# unit. Each phoneme is tagged on the standard phonological feature
# dimensions so downstream analysis can roll phonemes up by whichever
# grouping turns out to be most informative.
#
# Dimensions (General American English):
#   manner      type of airflow modulation
#   place       constriction location (consonants) or backness (vowels)
#   voicing     voiced / voiceless
#   height      vowel height (vowels and diphthongs only; None otherwise)
#   broad_class obstruent vs sonorant (coarsest standard grouping; derived)

GROUPING_UNKNOWN = "unknown"

MANNER_STOP = "stop"
MANNER_AFFRICATE = "affricate"
MANNER_FRICATIVE = "fricative"
MANNER_NASAL = "nasal"
MANNER_APPROXIMANT = "approximant"
MANNER_VOWEL = "vowel"
MANNER_DIPHTHONG = "diphthong"

BROAD_OBSTRUENT = "obstruent"
BROAD_SONORANT = "sonorant"

OBSTRUENT_MANNERS = frozenset({MANNER_STOP, MANNER_AFFRICATE, MANNER_FRICATIVE})

# ARPAbet phone → articulatory features. Keys match the normalized,
# stress-free ARPAbet inventory above.
PHONEME_GROUPINGS: dict[str, dict[str, str | None]] = {
    # Stops
    "P":  {"manner": MANNER_STOP,        "place": "bilabial",     "voicing": "voiceless", "height": None},
    "B":  {"manner": MANNER_STOP,        "place": "bilabial",     "voicing": "voiced",    "height": None},
    "T":  {"manner": MANNER_STOP,        "place": "alveolar",     "voicing": "voiceless", "height": None},
    "D":  {"manner": MANNER_STOP,        "place": "alveolar",     "voicing": "voiced",    "height": None},
    "K":  {"manner": MANNER_STOP,        "place": "velar",        "voicing": "voiceless", "height": None},
    "G":  {"manner": MANNER_STOP,        "place": "velar",        "voicing": "voiced",    "height": None},
    # Affricates
    "CH": {"manner": MANNER_AFFRICATE,   "place": "postalveolar", "voicing": "voiceless", "height": None},
    "JH": {"manner": MANNER_AFFRICATE,   "place": "postalveolar", "voicing": "voiced",    "height": None},
    # Fricatives
    "F":  {"manner": MANNER_FRICATIVE,   "place": "labiodental",  "voicing": "voiceless", "height": None},
    "V":  {"manner": MANNER_FRICATIVE,   "place": "labiodental",  "voicing": "voiced",    "height": None},
    "TH": {"manner": MANNER_FRICATIVE,   "place": "dental",       "voicing": "voiceless", "height": None},
    "DH": {"manner": MANNER_FRICATIVE,   "place": "dental",       "voicing": "voiced",    "height": None},
    "S":  {"manner": MANNER_FRICATIVE,   "place": "alveolar",     "voicing": "voiceless", "height": None},
    "Z":  {"manner": MANNER_FRICATIVE,   "place": "alveolar",     "voicing": "voiced",    "height": None},
    "SH": {"manner": MANNER_FRICATIVE,   "place": "postalveolar", "voicing": "voiceless", "height": None},
    "ZH": {"manner": MANNER_FRICATIVE,   "place": "postalveolar", "voicing": "voiced",    "height": None},
    "HH": {"manner": MANNER_FRICATIVE,   "place": "glottal",      "voicing": "voiceless", "height": None},
    # Nasals
    "M":  {"manner": MANNER_NASAL,       "place": "bilabial",     "voicing": "voiced",    "height": None},
    "N":  {"manner": MANNER_NASAL,       "place": "alveolar",     "voicing": "voiced",    "height": None},
    "NG": {"manner": MANNER_NASAL,       "place": "velar",        "voicing": "voiced",    "height": None},
    # Approximants and liquids
    "L":  {"manner": MANNER_APPROXIMANT, "place": "alveolar",     "voicing": "voiced",    "height": None},
    "R":  {"manner": MANNER_APPROXIMANT, "place": "postalveolar", "voicing": "voiced",    "height": None},
    "W":  {"manner": MANNER_APPROXIMANT, "place": "labial-velar", "voicing": "voiced",    "height": None},
    "Y":  {"manner": MANNER_APPROXIMANT, "place": "palatal",      "voicing": "voiced",    "height": None},
    # Monophthong vowels
    "IY": {"manner": MANNER_VOWEL,       "place": "front",        "voicing": "voiced",    "height": "high"},
    "IH": {"manner": MANNER_VOWEL,       "place": "front",        "voicing": "voiced",    "height": "high"},
    "EH": {"manner": MANNER_VOWEL,       "place": "front",        "voicing": "voiced",    "height": "mid"},
    "AE": {"manner": MANNER_VOWEL,       "place": "front",        "voicing": "voiced",    "height": "low"},
    "AH": {"manner": MANNER_VOWEL,       "place": "central",      "voicing": "voiced",    "height": "mid"},
    "ER": {"manner": MANNER_VOWEL,       "place": "central",      "voicing": "voiced",    "height": "mid"},
    "AA": {"manner": MANNER_VOWEL,       "place": "back",         "voicing": "voiced",    "height": "low"},
    "AO": {"manner": MANNER_VOWEL,       "place": "back",         "voicing": "voiced",    "height": "mid"},
    "UH": {"manner": MANNER_VOWEL,       "place": "back",         "voicing": "voiced",    "height": "high"},
    "UW": {"manner": MANNER_VOWEL,       "place": "back",         "voicing": "voiced",    "height": "high"},
    # Diphthongs
    "EY": {"manner": MANNER_DIPHTHONG,   "place": "front",        "voicing": "voiced",    "height": "mid"},
    "AY": {"manner": MANNER_DIPHTHONG,   "place": "central",      "voicing": "voiced",    "height": "low"},
    "OY": {"manner": MANNER_DIPHTHONG,   "place": "back",         "voicing": "voiced",    "height": "mid"},
    "OW": {"manner": MANNER_DIPHTHONG,   "place": "back",         "voicing": "voiced",    "height": "mid"},
    "AW": {"manner": MANNER_DIPHTHONG,   "place": "central",      "voicing": "voiced",    "height": "low"},
}


@dataclass(frozen=True)
class PhonemeGrouping:
    """Articulatory grouping for one phoneme across standard dimensions."""

    phoneme_label: str
    manner: str
    place: str
    voicing: str
    height: str | None
    broad_class: str


def _broad_class_for_manner(manner: str) -> str:
    if manner in OBSTRUENT_MANNERS:
        return BROAD_OBSTRUENT
    if manner == GROUPING_UNKNOWN:
        return GROUPING_UNKNOWN
    return BROAD_SONORANT


def group_phoneme(label: str | None) -> PhonemeGrouping:
    """Assign articulatory groupings (manner/place/voicing/height/broad)."""
    phone = normalize_phoneme_label(label)
    features = PHONEME_GROUPINGS.get(phone) if phone is not None else None
    if phone is None or features is None:
        return PhonemeGrouping(
            phoneme_label=phone or "",
            manner=GROUPING_UNKNOWN,
            place=GROUPING_UNKNOWN,
            voicing=GROUPING_UNKNOWN,
            height=None,
            broad_class=GROUPING_UNKNOWN,
        )
    manner = features["manner"] or GROUPING_UNKNOWN
    return PhonemeGrouping(
        phoneme_label=phone,
        manner=manner,
        place=features["place"] or GROUPING_UNKNOWN,
        voicing=features["voicing"] or GROUPING_UNKNOWN,
        height=features["height"],
        broad_class=_broad_class_for_manner(manner),
    )

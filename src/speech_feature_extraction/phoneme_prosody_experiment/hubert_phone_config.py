"""ARPAbet phonological contrast definitions for the HuBERT d-prime experiment.

Single responsibility: declare the binary phonological contrasts whose
separability is measured in frozen HuBERT embedding space, following the
phonological-subspace method of Muller et al. 2026 (arXiv:2604.21706 and
arXiv:2604.10123).

Each contrast is a pair of disjoint ARPAbet phone sets ([+feature] vs
[-feature]). A feature *direction* in embedding space is the difference between
the mean embeddings of the two sets; d-prime then quantifies how separable the
two sets are along that direction within a single recording.

Design notes
------------
- The phone sets are expressed in the same canonical, stress-free ARPAbet
  inventory as ``taxonomy.py`` (ubiquitous language across the experiment), and
  the vowel height / backness assignments are kept consistent with
  ``taxonomy.py``'s ``PHONEME_GROUPINGS`` (diphthongs assigned by nucleus).
- Contrasts are deliberately *minimal* where possible: nasality contrasts nasal
  stops against oral stops (manner held constant), and voicing/manner/stridency
  are restricted to obstruents so the target feature is the only thing that
  varies. This isolates the intended phonological dimension rather than a
  confound (e.g. sonorance leaking into a voicing contrast).
- The nine contrasts mirror the paper's nine segmental d-prime features:
  five consonant contrasts (nasality, voicing, sonorance, stridency, manner)
  and four vowel contrasts (height, lowness, backness, rounding).
"""

from __future__ import annotations

from dataclasses import dataclass

from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    ARPABET_PHONEMES,
)

# Minimum tokens required in EACH pole of a contrast for a valid per-recording
# d-prime estimate (Muller et al. 2026). Recordings below this for a contrast
# receive a missing value for that contrast.
MIN_TOKENS_PER_CLASS = 5


@dataclass(frozen=True)
class PhonologicalContrast:
    """One binary phonological contrast: [+feature] vs [-feature]."""

    key: str
    label: str
    feature_family: str  # "consonant" or "vowel"
    distinctive_feature: str
    positive_phones: frozenset[str]
    negative_phones: frozenset[str]
    rationale: str

    @property
    def dprime_field(self) -> str:
        """Output column name for this contrast's per-recording d-prime."""
        return f"dprime_{self.key}"

    @property
    def n_positive_field(self) -> str:
        return f"n_{self.key}_pos"

    @property
    def n_negative_field(self) -> str:
        return f"n_{self.key}_neg"


CONSONANT_FAMILY = "consonant"
VOWEL_FAMILY = "vowel"

# Reusable obstruent voicing pairs (the cleanest [+/-voice] minimal contrast).
_VOICED_OBSTRUENTS = frozenset({"B", "D", "G", "V", "DH", "Z", "ZH", "JH"})
_VOICELESS_OBSTRUENTS = frozenset({"P", "T", "K", "F", "TH", "S", "SH", "CH"})
_OBSTRUENTS = _VOICED_OBSTRUENTS | _VOICELESS_OBSTRUENTS | frozenset({"HH"})
_SONORANT_CONSONANTS = frozenset({"M", "N", "NG", "L", "R", "W", "Y"})
_ORAL_STOPS = frozenset({"P", "B", "T", "D", "K", "G"})
_FRICATIVES = frozenset({"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"})

# Vowel feature classes. Height and backness reuse taxonomy.py's assignments
# (diphthongs by nucleus); lowness and rounding are added here. [+high]/[+low]
# are independent axes (mid vowels are [-high, -low] in both), matching the
# SPE-style treatment the paper's four vowel features imply.
_HIGH_VOWELS = frozenset({"IY", "IH", "UW", "UH"})
_NONHIGH_VOWELS = frozenset({"EH", "AH", "ER", "AO", "EY", "OY", "OW", "AE", "AA", "AY", "AW"})
_LOW_VOWELS = frozenset({"AE", "AA", "AY", "AW"})
_NONLOW_VOWELS = frozenset({"IY", "IH", "UW", "UH", "EH", "AH", "ER", "AO", "EY", "OY", "OW"})
_BACK_VOWELS = frozenset({"AA", "AO", "UH", "UW", "OY", "OW"})
_FRONT_VOWELS = frozenset({"IY", "IH", "EH", "AE", "EY", "AY"})
_ROUND_VOWELS = frozenset({"UW", "UH", "AO", "OW", "OY"})
_UNROUND_VOWELS = frozenset({"IY", "IH", "EH", "AE", "AA", "AH", "ER", "EY", "AY", "AW"})


PHONOLOGICAL_CONTRASTS: tuple[PhonologicalContrast, ...] = (
    PhonologicalContrast(
        key="nasality",
        label="Nasality (nasal vs oral stop)",
        feature_family=CONSONANT_FAMILY,
        distinctive_feature="[+nasal] vs [-nasal]",
        positive_phones=frozenset({"M", "N", "NG"}),
        negative_phones=_ORAL_STOPS,
        rationale="Nasal stops vs oral stops holds manner constant and isolates "
        "velopharyngeal coupling; the paper's primary contrast.",
    ),
    PhonologicalContrast(
        key="voicing",
        label="Voicing (obstruents)",
        feature_family=CONSONANT_FAMILY,
        distinctive_feature="[+voice] vs [-voice]",
        positive_phones=_VOICED_OBSTRUENTS,
        negative_phones=_VOICELESS_OBSTRUENTS,
        rationale="Obstruent voicing pairs (b/p, d/t, z/s, ...); restricted to "
        "obstruents so sonorant voicing does not leak into the direction.",
    ),
    PhonologicalContrast(
        key="sonorance",
        label="Sonorance (sonorant consonant vs obstruent)",
        feature_family=CONSONANT_FAMILY,
        distinctive_feature="[+sonorant] vs [-sonorant]",
        positive_phones=_SONORANT_CONSONANTS,
        negative_phones=_OBSTRUENTS,
        rationale="Sonorant consonants vs obstruents; a consonantal sonorance "
        "axis (vowels excluded to keep it consonant-internal).",
    ),
    PhonologicalContrast(
        key="stridency",
        label="Stridency (sibilant vs non-sibilant fricative)",
        feature_family=CONSONANT_FAMILY,
        distinctive_feature="[+strident] vs [-strident]",
        positive_phones=frozenset({"S", "Z", "SH", "ZH", "CH", "JH"}),
        negative_phones=frozenset({"F", "V", "TH", "DH"}),
        rationale="Sibilants/affricates vs non-sibilant fricatives; isolates "
        "high-frequency frication energy.",
    ),
    PhonologicalContrast(
        key="manner",
        label="Manner (continuant: fricative vs stop)",
        feature_family=CONSONANT_FAMILY,
        distinctive_feature="[+continuant] vs [-continuant]",
        positive_phones=_FRICATIVES,
        negative_phones=_ORAL_STOPS,
        rationale="Fricatives vs stops within obstruents; the [+/-continuant] "
        "manner axis.",
    ),
    PhonologicalContrast(
        key="vowel_height",
        label="Vowel height (high vs non-high)",
        feature_family=VOWEL_FAMILY,
        distinctive_feature="[+high] vs [-high]",
        positive_phones=_HIGH_VOWELS,
        negative_phones=_NONHIGH_VOWELS,
        rationale="High vowels vs the rest; tongue-body height axis.",
    ),
    PhonologicalContrast(
        key="vowel_lowness",
        label="Vowel lowness (low vs non-low)",
        feature_family=VOWEL_FAMILY,
        distinctive_feature="[+low] vs [-low]",
        positive_phones=_LOW_VOWELS,
        negative_phones=_NONLOW_VOWELS,
        rationale="Low vowels vs the rest; independent of [+high] so mid vowels "
        "are negative in both axes.",
    ),
    PhonologicalContrast(
        key="vowel_backness",
        label="Vowel backness (back vs front)",
        feature_family=VOWEL_FAMILY,
        distinctive_feature="[+back] vs [-back]",
        positive_phones=_BACK_VOWELS,
        negative_phones=_FRONT_VOWELS,
        rationale="Back vs front vowels (central vowels excluded); tongue-body "
        "frontness axis.",
    ),
    PhonologicalContrast(
        key="vowel_rounding",
        label="Vowel rounding (rounded vs unrounded)",
        feature_family=VOWEL_FAMILY,
        distinctive_feature="[+round] vs [-round]",
        positive_phones=_ROUND_VOWELS,
        negative_phones=_UNROUND_VOWELS,
        rationale="Rounded vs unrounded vowels; lip-rounding axis. Sparse in the "
        "Rainbow sentences 2-3 (~4 rounded tokens), so often missing per the "
        "minimum-token rule.",
    ),
)


def contrast_by_key(key: str) -> PhonologicalContrast:
    """Return the contrast with the given key, raising KeyError if absent."""
    for contrast in PHONOLOGICAL_CONTRASTS:
        if contrast.key == key:
            return contrast
    raise KeyError(f"Unknown phonological contrast: {key!r}")


def _validate_contrasts() -> None:
    """Fail fast on malformed contrast definitions (import-time invariant)."""
    seen_keys: set[str] = set()
    for contrast in PHONOLOGICAL_CONTRASTS:
        if contrast.key in seen_keys:
            raise ValueError(f"Duplicate contrast key: {contrast.key!r}")
        seen_keys.add(contrast.key)

        overlap = contrast.positive_phones & contrast.negative_phones
        if overlap:
            raise ValueError(
                f"Contrast {contrast.key!r} has phones in both poles: {sorted(overlap)}"
            )
        if not contrast.positive_phones or not contrast.negative_phones:
            raise ValueError(f"Contrast {contrast.key!r} has an empty pole")

        unknown = (contrast.positive_phones | contrast.negative_phones) - ARPABET_PHONEMES
        if unknown:
            raise ValueError(
                f"Contrast {contrast.key!r} references non-ARPAbet phones: {sorted(unknown)}"
            )


_validate_contrasts()

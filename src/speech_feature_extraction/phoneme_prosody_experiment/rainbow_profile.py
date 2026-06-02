"""Rainbow Passage phoneme profile helpers.

These utilities let us build one canonical phoneme sequence/timing profile from
an aligned reference recording and compare subsequent recordings against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import normalize_phoneme_label


@dataclass(frozen=True)
class AlignedPhonemeSegment:
    """One aligned phoneme segment emitted by an aligner."""

    phoneme_label: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class RainbowPhonemeOccurrence:
    """Expected timing for one phoneme occurrence in sequence order."""

    phoneme_label: str
    occurrence_index: int
    expected_position_ratio: float


@dataclass(frozen=True)
class RainbowPassageTemplate:
    """Canonical sequence and timing profile for one Rainbow passage style read."""

    reference_duration_sec: float
    expected_sequence: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    occurrences: tuple[RainbowPhonemeOccurrence, ...]
    expected_counts: dict[str, int]


@dataclass(frozen=True)
class RainbowOccurrenceAlignment:
    """Observed timing consistency for one phoneme occurrence."""

    phoneme_label: str
    occurrence_index: int
    expected_position_ratio: float
    observed_position_ratio: float
    position_delta_ratio: float
    timing_consistent: bool


@dataclass(frozen=True)
class RainbowAlignmentSummary:
    """Recording-level summary against a canonical Rainbow template."""

    coverage_ratio: float
    sequence_match_ratio: float
    missing_occurrence_keys: tuple[str, ...]
    unexpected_occurrence_keys: tuple[str, ...]
    occurrence_alignments: tuple[RainbowOccurrenceAlignment, ...]


def build_rainbow_template(segments: Iterable[AlignedPhonemeSegment]) -> RainbowPassageTemplate:
    """Build a canonical expected phoneme profile from one aligned reference."""
    normalized = _normalize_segments(segments)
    if not normalized:
        raise ValueError("Cannot build Rainbow template from empty segment list.")

    first_start = normalized[0].start_sec
    last_end = normalized[-1].end_sec
    duration = max(last_end - first_start, 1e-9)

    counts: dict[str, int] = {}
    sequence: list[str] = []
    occurrences: list[RainbowPhonemeOccurrence] = []
    for segment in normalized:
        label = segment.phoneme_label
        sequence.append(label)
        counts[label] = counts.get(label, 0) + 1
        midpoint = (segment.start_sec + segment.end_sec) / 2.0
        position_ratio = (midpoint - first_start) / duration
        occurrences.append(
            RainbowPhonemeOccurrence(
                phoneme_label=label,
                occurrence_index=counts[label],
                expected_position_ratio=_clamp_ratio(position_ratio),
            )
        )

    inventory = tuple(sorted(counts))
    return RainbowPassageTemplate(
        reference_duration_sec=last_end - first_start,
        expected_sequence=tuple(sequence),
        expected_inventory=inventory,
        occurrences=tuple(occurrences),
        expected_counts=counts,
    )


def summarize_alignment_against_template(
    segments: Iterable[AlignedPhonemeSegment],
    template: RainbowPassageTemplate,
    timing_tolerance_ratio: float = 0.10,
) -> RainbowAlignmentSummary:
    """Compare one aligned recording to the template for coverage and timing."""
    normalized = _normalize_segments(segments)
    if not normalized:
        raise ValueError("Cannot summarize Rainbow alignment from empty segment list.")

    first_start = normalized[0].start_sec
    last_end = normalized[-1].end_sec
    duration = max(last_end - first_start, 1e-9)

    observed_occurrence_map: dict[tuple[str, int], float] = {}
    observed_sequence: list[str] = []
    observed_counts: dict[str, int] = {}
    for segment in normalized:
        label = segment.phoneme_label
        observed_sequence.append(label)
        observed_counts[label] = observed_counts.get(label, 0) + 1
        key = (label, observed_counts[label])
        midpoint = (segment.start_sec + segment.end_sec) / 2.0
        observed_occurrence_map[key] = _clamp_ratio((midpoint - first_start) / duration)

    expected_map = {
        (occ.phoneme_label, occ.occurrence_index): occ.expected_position_ratio
        for occ in template.occurrences
    }

    alignments: list[RainbowOccurrenceAlignment] = []
    matched_expected_keys: set[tuple[str, int]] = set()
    for key, observed_position_ratio in observed_occurrence_map.items():
        if key not in expected_map:
            continue
        expected_position_ratio = expected_map[key]
        delta = observed_position_ratio - expected_position_ratio
        alignments.append(
            RainbowOccurrenceAlignment(
                phoneme_label=key[0],
                occurrence_index=key[1],
                expected_position_ratio=expected_position_ratio,
                observed_position_ratio=observed_position_ratio,
                position_delta_ratio=delta,
                timing_consistent=abs(delta) <= timing_tolerance_ratio,
            )
        )
        matched_expected_keys.add(key)

    expected_keys = set(expected_map)
    observed_keys = set(observed_occurrence_map)
    missing_keys = tuple(sorted(_format_occurrence_key(key) for key in expected_keys - observed_keys))
    unexpected_keys = tuple(sorted(_format_occurrence_key(key) for key in observed_keys - expected_keys))

    coverage_ratio = len(matched_expected_keys) / len(expected_keys) if expected_keys else 0.0
    sequence_match_ratio = _compute_sequence_match_ratio(template.expected_sequence, tuple(observed_sequence))
    return RainbowAlignmentSummary(
        coverage_ratio=coverage_ratio,
        sequence_match_ratio=sequence_match_ratio,
        missing_occurrence_keys=missing_keys,
        unexpected_occurrence_keys=unexpected_keys,
        occurrence_alignments=tuple(sorted(alignments, key=lambda item: (item.occurrence_index, item.phoneme_label))),
    )


def _normalize_segments(segments: Iterable[AlignedPhonemeSegment]) -> list[AlignedPhonemeSegment]:
    normalized: list[AlignedPhonemeSegment] = []
    for segment in segments:
        label = normalize_phoneme_label(segment.phoneme_label)
        if label is None:
            continue
        if segment.end_sec <= segment.start_sec:
            continue
        normalized.append(
            AlignedPhonemeSegment(
                phoneme_label=label,
                start_sec=float(segment.start_sec),
                end_sec=float(segment.end_sec),
            )
        )
    normalized.sort(key=lambda item: (item.start_sec, item.end_sec))
    return normalized


def _compute_sequence_match_ratio(expected: tuple[str, ...], observed: tuple[str, ...]) -> float:
    if not expected:
        return 0.0
    limit = min(len(expected), len(observed))
    if limit == 0:
        return 0.0
    exact_matches = sum(1 for index in range(limit) if expected[index] == observed[index])
    return exact_matches / len(expected)


def _format_occurrence_key(key: tuple[str, int]) -> str:
    return f"{key[0]}#{key[1]}"


def _clamp_ratio(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value

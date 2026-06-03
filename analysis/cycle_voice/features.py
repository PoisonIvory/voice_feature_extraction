"""Feature taxonomy: group eGeMAPS + d-prime columns into families and axes.

The three-axis framework is used here only as an organizing scaffold and an
interpretation lens (structural / peripheral-mucosal / central-neuroaffective),
not as a hypothesis filter. Every numeric voice feature is assigned a family and
an axis so the discovery sweep can be summarized by group.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TASKS = ("prosody", "vowel")

AXIS_STRUCTURAL = "structural_source"
AXIS_PERIPHERAL = "peripheral_mucosal"
AXIS_CENTRAL = "central_neuroaffective"
AXIS_UNASSIGNED = "unassigned"

# d-prime articulatory-contrast columns from the HuBERT phone-embedding pipeline.
DPRIME_COLUMNS = (
    "dprime_nasality",
    "dprime_voicing",
    "dprime_sonorance",
    "dprime_stridency",
    "dprime_manner",
    "dprime_vowel_height",
    "dprime_vowel_lowness",
    "dprime_vowel_backness",
    "dprime_vowel_rounding",
)


@dataclass(frozen=True)
class FeatureMeta:
    column: str
    task: str
    base: str
    family: str
    axis: str


def _family_and_axis(base: str) -> tuple[str, str]:
    """Map an eGeMAPS base feature name to (family, axis)."""
    b = base
    if "F0semitone" in b:
        if any(k in b for k in ("amean", "percentile20", "percentile50", "percentile80")):
            return "pitch_level", AXIS_STRUCTURAL
        return "pitch_dynamics", AXIS_CENTRAL  # stddevNorm, pctlrange, slopes
    if b.startswith(("F1", "F2", "F3")):
        return "formants_resonance", AXIS_PERIPHERAL
    if "jitter" in b or "shimmer" in b:
        return "perturbation", AXIS_PERIPHERAL
    if "HNR" in b:
        return "harmonicity", AXIS_PERIPHERAL
    if any(k in b for k in ("alphaRatio", "hammarbergIndex", "H1-H2", "H1-A3", "slopeV", "slopeUV")):
        return "spectral_balance", AXIS_PERIPHERAL
    if "spectralFlux" in b:
        return "spectral_flux", AXIS_PERIPHERAL
    if "mfcc" in b.lower():
        return "mfcc_timbre", AXIS_PERIPHERAL
    if "loudness" in b.lower() or "equivalentSoundLevel" in b:
        return "loudness_energy", AXIS_CENTRAL
    if "VoicedSegment" in b or "UnvoicedSegment" in b or "VoicedSegmentsPerSec" in b:
        return "temporal_rhythm", AXIS_CENTRAL
    return "other", AXIS_UNASSIGNED


def classify_feature(column: str) -> FeatureMeta | None:
    """Return metadata for an eGeMAPS daily column, or None if not a feature."""
    for task in TASKS:
        prefix = f"{task}_egemaps_"
        if column.startswith(prefix):
            base = column[len(prefix) :]
            family, axis = _family_and_axis(base)
            return FeatureMeta(column=column, task=task, base=base, family=family, axis=axis)
    return None


def egemaps_feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric eGeMAPS feature columns present in a daily voice frame."""
    cols = []
    for c in df.columns:
        meta = classify_feature(c)
        if meta is not None and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def feature_table(columns: list[str]) -> pd.DataFrame:
    """A tidy metadata table for a list of eGeMAPS columns."""
    rows = [classify_feature(c).__dict__ for c in columns if classify_feature(c)]
    return pd.DataFrame(rows)


def dprime_meta() -> pd.DataFrame:
    """Metadata rows for the articulatory d-prime contrasts (all central axis)."""
    return pd.DataFrame(
        {
            "column": list(DPRIME_COLUMNS),
            "task": "prosody",
            "base": [c.replace("dprime_", "") for c in DPRIME_COLUMNS],
            "family": "articulatory_precision",
            "axis": AXIS_CENTRAL,
        }
    )

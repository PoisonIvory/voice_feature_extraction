"""Assemble the independent joined daily table.

Single responsibility: read this project's voice features + the staged external
inputs and produce one tidy daily frame keyed by calendar date, plus a coverage
manifest and a data dictionary. The other project's derived phase label is kept
only as ``external_phase_label`` for cross-checking and is never used as input.
"""

from __future__ import annotations

import pandas as pd

from . import paths
from .features import egemaps_feature_columns

_CYCLE_SCAFFOLD = [
    "cycle_start_date",
    "next_cycle_start_date",
    "cycle_day",
    "days_to_next_start",
    "cycle_source",
    "cycle_week",
]
_CORE_OURA = ["resting_hr", "hrv", "temp_deviation"]


def load_voice_daily() -> pd.DataFrame:
    df = pd.read_parquet(paths.VOICE_DAILY)
    df = df[df["userId"] == paths.ENFORCED_USER_ID].copy()
    df["date"] = pd.to_datetime(df["dayUtc"])
    return df


def _load_external() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal = pd.read_parquet(paths.CYCLE_CALENDAR)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.rename(columns={"phase_label": "external_phase_label"})
    cal = cal[["date", "external_phase_label", *_CYCLE_SCAFFOLD]]

    horm = pd.read_parquet(paths.HORMONE_LEVELS)
    horm["date"] = pd.to_datetime(horm["date"])
    horm = horm[["date", "e3g", "pdg"]]

    oura = pd.read_parquet(paths.OURA_GONADOTROPINS)
    oura["date"] = pd.to_datetime(oura["date"])
    return cal, horm, oura


def assemble() -> pd.DataFrame:
    voice = load_voice_daily()
    cal, horm, oura = _load_external()

    merged = (
        cal.merge(oura, on="date", how="outer")
        .merge(horm, on="date", how="outer")
        .merge(voice, on="date", how="outer")
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    counts = merged.reindex(columns=["vowel_recording_count", "prosody_recording_count"]).fillna(0)
    merged["has_voice"] = counts.sum(axis=1) > 0
    merged["has_hormones"] = merged["e3g"].notna() & merged["pdg"].notna()
    merged["has_oura"] = merged[_CORE_OURA].notna().any(axis=1)
    merged["userId"] = paths.ENFORCED_USER_ID
    return merged


def coverage_manifest(df: pd.DataFrame) -> pd.DataFrame:
    def _span(mask: pd.Series) -> str:
        days = df.loc[mask, "date"]
        return f"{days.min():%Y-%m-%d} .. {days.max():%Y-%m-%d}" if mask.any() else "-"

    rows = [
        ("total_days", len(df), _span(df["date"].notna())),
        ("voice_days", int(df["has_voice"].sum()), _span(df["has_voice"])),
        ("hormone_days", int(df["has_hormones"].sum()), _span(df["has_hormones"])),
        ("oura_days", int(df["has_oura"].sum()), _span(df["has_oura"])),
        ("voice_and_hormones", int((df["has_voice"] & df["has_hormones"]).sum()), _span(df["has_voice"] & df["has_hormones"])),
        ("voice_and_oura", int((df["has_voice"] & df["has_oura"]).sum()), _span(df["has_voice"] & df["has_oura"])),
        ("voice_hormones_oura", int((df["has_voice"] & df["has_hormones"] & df["has_oura"]).sum()), _span(df["has_voice"] & df["has_hormones"] & df["has_oura"])),
        ("cycles", int(df["cycle_start_date"].nunique()), "n distinct cycle_start_date"),
    ]
    return pd.DataFrame(rows, columns=["subset", "n", "detail"])


_DERIVED = {
    "phase": "coarse cycle phase (derived)",
    "subphase": "fine cycle sub-phase (derived)",
    "premenstrual": "late-luteal / PMDD symptom window flag (derived)",
    "cycle_day_from_ovulation": "days from approximate ovulation (derived)",
    "ep_ratio": "estrogen:progesterone ratio e3g/pdg (derived)",
    "e3g_z": "standardized e3g (derived)",
    "pdg_z": "standardized pdg (derived)",
    "ep_ratio_z": "standardized E:P ratio (derived)",
    "e3g_roc": "e3g day-over-day rate of change (derived)",
    "pdg_roc": "pdg rate of change / withdrawal velocity (derived)",
}


def _describe_column(col: str) -> tuple[str, str]:
    if col in ("date", "userId"):
        return "key", "join key / subject id"
    if col in _DERIVED:
        return "derived (analysis)", _DERIVED[col]
    if col in _CYCLE_SCAFFOLD or col == "external_phase_label":
        return "cycle_calendar (external)", "Oura-anchored cycle scaffold; external_phase_label is cross-check only"
    if col in ("e3g", "pdg"):
        return "hormone_levels (external)", "Inito urinary estrogen (e3g) / progesterone (pdg) metabolite"
    if col in ("fsh", "lh"):
        return "gonadotropins (external)", "Inito FSH / LH"
    if col.startswith(("prosody_egemaps_", "vowel_egemaps_")):
        return "voice (this project)", "eGeMAPSv02 daily functional"
    if col.startswith("has_"):
        return "derived", "coverage flag"
    if col.endswith("_recording_count") or col.endswith("_recording_ids") or col in ("has_vowel", "has_prosody", "is_day_complete", "dayUtc", "extractorVersion", "featureSet", "featureLevel", "libraryVersion"):
        return "voice (this project)", "recording lineage / counts"
    return "oura (external)", "raw Oura daily biometric"


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        source, desc = _describe_column(col)
        rows.append(
            {
                "column": col,
                "source": source,
                "dtype": str(df[col].dtype),
                "n_nonnull": int(df[col].notna().sum()),
                "description": desc,
            }
        )
    return pd.DataFrame(rows)


def build_and_save() -> pd.DataFrame:
    paths.ensure_output_dirs()
    df = assemble()
    df.to_parquet(paths.ASSEMBLED_TABLE, index=False)
    coverage_manifest(df).to_csv(paths.TABLES_DIR / "coverage_manifest.csv", index=False)
    data_dictionary(df).to_csv(paths.TABLES_DIR / "data_dictionary.csv", index=False)
    n_features = len(egemaps_feature_columns(df))
    print(f"Assembled {df.shape[0]} days x {df.shape[1]} cols; {n_features} eGeMAPS features.")
    print(f"Saved: {paths.ASSEMBLED_TABLE}")
    return df


if __name__ == "__main__":
    build_and_save()

"""Independently derive cycle phases and hormone-derived features.

We do not inherit the other project's follicular/luteal label. Instead we use
the device menses anchor (cycle_day, days_to_next_start) with backward counting
(luteal length is physiologically stable at ~14 days, so counting back from the
next menses localizes ovulation and the luteal phase better than forward
counting), and refine with measured hormones + Oura temperature where present.

Outputs per day: coarse ``phase`` (menses/follicular/ovulatory/luteal), a finer
``subphase``, a ``premenstrual`` flag (the PMDD symptom window), and
``cycle_day_from_ovulation``. Hormone features: E:P ratio, standardized hormone
levels, and day-over-day rate-of-change (withdrawal velocity).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LUTEAL_LENGTH = 14  # assumed luteal length for backward counting / ovulation anchor


def _subphase(cycle_day: float, days_to_next: float) -> str | None:
    if pd.isna(cycle_day) and pd.isna(days_to_next):
        return None
    if not pd.isna(cycle_day) and cycle_day <= 5:
        return "menstrual"
    if not pd.isna(days_to_next):
        if days_to_next <= 7:
            return "late_luteal"
        if days_to_next <= 13:
            return "early_luteal"
        if days_to_next <= 16:
            return "periovulatory"
        if days_to_next <= 19:
            return "late_follicular"
        return "early_follicular"
    # Fallback when next-menses anchor is missing: forward counting only.
    if cycle_day <= 13:
        return "early_follicular"
    if cycle_day <= 16:
        return "periovulatory"
    return "early_luteal"


_PHASE_OF = {
    "menstrual": "menses",
    "early_follicular": "follicular",
    "late_follicular": "follicular",
    "periovulatory": "ovulatory",
    "early_luteal": "luteal",
    "late_luteal": "luteal",
}


def add_phases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subphase"] = [
        _subphase(cd, dn) for cd, dn in zip(out["cycle_day"], out["days_to_next_start"])
    ]
    out["phase"] = out["subphase"].map(_PHASE_OF)
    out["premenstrual"] = out["subphase"].eq("late_luteal")
    out["cycle_day_from_ovulation"] = LUTEAL_LENGTH - out["days_to_next_start"]
    return out


def _zscore(series: pd.Series) -> pd.Series:
    mask = series.notna()
    z = pd.Series(np.nan, index=series.index)
    if mask.sum() >= 3 and series[mask].std(ddof=0) > 0:
        z[mask] = (series[mask] - series[mask].mean()) / series[mask].std(ddof=0)
    return z


def add_hormone_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ep_ratio"] = out["e3g"] / out["pdg"].where(out["pdg"] > 0)
    out["e3g_z"] = _zscore(out["e3g"])
    out["pdg_z"] = _zscore(out["pdg"])
    out["ep_ratio_z"] = _zscore(out["ep_ratio"])

    # Rate-of-change on measured-hormone days only, normalized by the day gap so
    # irregular sampling does not distort the velocity. Negative pdg_roc = withdrawal.
    measured = out.loc[out["has_hormones"], ["date", "e3g", "pdg"]].sort_values("date")
    gap_days = measured["date"].diff().dt.days
    out.loc[measured.index, "e3g_roc"] = (measured["e3g"].diff() / gap_days).to_numpy()
    out.loc[measured.index, "pdg_roc"] = (measured["pdg"].diff() / gap_days).to_numpy()
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    return add_hormone_features(add_phases(df))


def phase_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Voice-day counts per phase / subphase for the coverage report."""
    voice = df[df["has_voice"]]
    rows = []
    for level in ("phase", "subphase"):
        vc = voice[level].value_counts(dropna=False)
        for name, n in vc.items():
            with_h = int(((voice[level] == name) & voice["has_hormones"]).sum())
            rows.append({"grouping": level, "value": str(name), "voice_days": int(n), "with_hormones": with_h})
    return pd.DataFrame(rows)

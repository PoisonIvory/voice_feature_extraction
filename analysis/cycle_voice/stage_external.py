"""Stage the independent external inputs into data/external (idempotent).

We copy the raw cycle scaffold and measured-hormone files verbatim, and build a
slim Oura/gonadotropin file that contains ONLY raw measured biometrics + FSH/LH,
deliberately dropping the other project's derived columns (phase labels, voice
joins) so this analysis stays independent. If the staged files already exist, it
does nothing, so re-running the pipeline is safe and offline.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from . import paths

DEFAULT_SOURCE = Path(
    os.getenv(
        "CYCLE_SOURCE_DIR",
        "/Users/ivyhamilton/Decibelle/Analysis/archive/analysis-snapshot-2026-06-02/Analysis/data/processed",
    )
)

_RAW_OURA = [
    "temp_deviation", "temp_trend_deviation", "hrv", "hrv_balance", "resting_hr",
    "average_hr", "lowest_hr", "breath_rate", "sleep_score", "readiness_score",
    "activity_score", "total_sleep_sec", "rem_sleep_sec", "deep_sleep_sec",
    "sleep_efficiency", "steps", "spo2", "breathing_disturbance",
]


def ensure(source: Path = DEFAULT_SOURCE) -> None:
    paths.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if all(p.exists() for p in (paths.CYCLE_CALENDAR, paths.HORMONE_LEVELS, paths.OURA_GONADOTROPINS)):
        return
    if not source.exists():
        raise FileNotFoundError(
            f"Staged inputs missing and source archive not found: {source}. "
            "Set CYCLE_SOURCE_DIR to the directory holding the external parquet files."
        )
    shutil.copy2(source / "cycle_calendar_daily.parquet", paths.CYCLE_CALENDAR)
    shutil.copy2(source / "hormone_daily_levels.parquet", paths.HORMONE_LEVELS)
    adf = pd.read_parquet(source / "analysis_daily.parquet")
    slim = adf[["date", *_RAW_OURA, "fsh", "lh"]].copy()
    slim.to_parquet(paths.OURA_GONADOTROPINS, index=False)

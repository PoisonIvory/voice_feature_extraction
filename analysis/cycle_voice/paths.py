"""Filesystem locations for the voice-cycle analysis.

Single responsibility: resolve and create all paths used by the analysis so no
other module hard-codes locations.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Voice artifacts produced by this project's extraction pipeline.
VOICE_DAILY = PROJECT_ROOT / "data" / "processed" / "voice_features_v4_daily.parquet"
HUBERT_DPRIME = (
    PROJECT_ROOT / "data" / "experimental" / "phoneme_prosody" / "hubert_dprime_by_recording.parquet"
)

# Independently staged external inputs (see scripts/stage_external_inputs).
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
CYCLE_CALENDAR = EXTERNAL_DIR / "cycle_calendar_daily.parquet"
HORMONE_LEVELS = EXTERNAL_DIR / "hormone_daily_levels.parquet"
OURA_GONADOTROPINS = EXTERNAL_DIR / "oura_gonadotropins_daily.parquet"

# Outputs.
ANALYSIS_DATA_DIR = PROJECT_ROOT / "data" / "analysis"
ASSEMBLED_TABLE = ANALYSIS_DATA_DIR / "cycle_voice_daily.parquet"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"

# Report + figures live together so the report is a self-contained deliverable.
REPORT_DIR = PROJECT_ROOT / "docs" / "voice_cycle_analysis"
FIGURES_DIR = REPORT_DIR / "figures"
REPORT_FILE = REPORT_DIR / "REPORT.md"

# Single enforced subject (matches the extraction pipeline scope).
ENFORCED_USER_ID = "6928d5ab0018cac7ae42"


def ensure_output_dirs() -> None:
    for directory in (ANALYSIS_DATA_DIR, OUTPUT_DIR, TABLES_DIR, REPORT_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)

"""Parquet output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def write_rows_parquet(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False)
    return path


def read_rows_parquet(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_parquet(path)
    if frame.empty:
        return []
    return frame.to_dict(orient="records")

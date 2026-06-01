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

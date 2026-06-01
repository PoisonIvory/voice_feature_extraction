"""Deterministic hashing helpers for snapshot artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 digest for file content."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def compute_column_order_hash(frame: pd.DataFrame) -> str:
    """Hash ordered column names for deterministic schema-order checks."""
    payload = json.dumps([str(column) for column in frame.columns], separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def compute_table_schema_hash(frame: pd.DataFrame) -> str:
    """Hash deterministic schema metadata from columns and dtypes."""
    schema = [
        {
            "name": str(column),
            "dtype": str(frame.dtypes[column]),
        }
        for column in frame.columns
    ]
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


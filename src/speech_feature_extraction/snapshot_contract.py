"""Snapshot manifest contract helpers."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from speech_feature_extraction.constants import (
    SNAPSHOT_DATASET_NAME,
    SNAPSHOT_DATASET_VERSION,
    SNAPSHOT_MANIFEST_VERSION,
    SNAPSHOT_REQUIRED_CORE_COLUMNS,
    SNAPSHOT_REQUIRED_FEATURE_COUNT_BY_PREFIX,
    SNAPSHOT_REQUIRED_FEATURE_PREFIXES,
)

SUPPORTED_MANIFEST_VERSIONS = {SNAPSHOT_MANIFEST_VERSION}


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 digest for a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def compute_column_order_hash(frame: pd.DataFrame) -> str:
    """Hash ordered columns for deterministic schema-order checks."""
    payload = json.dumps([str(col) for col in frame.columns], separators=(",", ":"), ensure_ascii=True)
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


def validate_required_columns(
    frame: pd.DataFrame,
    required_core_columns: Sequence[str] = SNAPSHOT_REQUIRED_CORE_COLUMNS,
    required_feature_prefixes: Sequence[str] = SNAPSHOT_REQUIRED_FEATURE_PREFIXES,
    required_feature_count_by_prefix: Mapping[str, int] = SNAPSHOT_REQUIRED_FEATURE_COUNT_BY_PREFIX,
) -> None:
    """Fail fast when required contract columns are missing."""
    missing_core_columns = [column for column in required_core_columns if column not in frame.columns]
    if missing_core_columns:
        names = ", ".join(sorted(missing_core_columns))
        raise ValueError(f"Missing required core columns: {names}")

    for prefix in required_feature_prefixes:
        if not any(str(column).startswith(prefix) for column in frame.columns):
            raise ValueError(f"Missing required feature prefix: {prefix}")

    for prefix, expected_count in required_feature_count_by_prefix.items():
        actual_count = sum(1 for column in frame.columns if str(column).startswith(prefix))
        if actual_count != expected_count:
            raise ValueError(
                f"Feature prefix count mismatch for {prefix}: expected {expected_count}, got {actual_count}"
            )


def build_file_descriptor(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    """Build a per-file manifest descriptor."""
    return {
        "path": path.name,
        "row_count": int(len(frame)),
        "schema_hash": compute_table_schema_hash(frame),
        "column_order_hash": compute_column_order_hash(frame),
        "content_sha256": compute_file_sha256(path),
        "file_size_bytes": path.stat().st_size,
    }


def build_manifest(
    *,
    snapshot: str,
    created_at: str,
    source_commit: str | None,
    pipeline_version: str,
    opensmile_version: str | None,
    files: Mapping[str, Mapping[str, Any]],
    required_core_columns: Sequence[str] = SNAPSHOT_REQUIRED_CORE_COLUMNS,
    required_feature_prefixes: Sequence[str] = SNAPSHOT_REQUIRED_FEATURE_PREFIXES,
    required_feature_count_by_prefix: Mapping[str, int] = SNAPSHOT_REQUIRED_FEATURE_COUNT_BY_PREFIX,
    python_version: str | None = None,
) -> dict[str, Any]:
    """Build the snapshot manifest payload."""
    provenance: dict[str, Any] = {
        "source_commit": source_commit,
        "pipeline_version": pipeline_version,
        "opensmile_version": opensmile_version,
    }
    if python_version:
        provenance["python_version"] = python_version
    else:
        provenance["python_version"] = platform.python_version()

    return {
        "manifest_version": SNAPSHOT_MANIFEST_VERSION,
        "dataset": SNAPSHOT_DATASET_NAME,
        "version": SNAPSHOT_DATASET_VERSION,
        "snapshot": snapshot,
        "created_at": created_at,
        "provenance": provenance,
        "files": files,
        "required_core_columns": list(required_core_columns),
        "required_feature_prefixes": list(required_feature_prefixes),
        "required_feature_count_by_prefix": dict(required_feature_count_by_prefix),
    }


def validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Validate the base manifest structure and supported versions."""
    required_keys = {
        "manifest_version",
        "dataset",
        "version",
        "snapshot",
        "created_at",
        "provenance",
        "files",
        "required_core_columns",
        "required_feature_prefixes",
        "required_feature_count_by_prefix",
    }
    missing = required_keys.difference(manifest.keys())
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Manifest missing required keys: {names}")

    manifest_version = str(manifest["manifest_version"])
    if manifest_version not in SUPPORTED_MANIFEST_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_MANIFEST_VERSIONS))
        raise ValueError(f"Unsupported manifest version {manifest_version}; supported: {supported}")

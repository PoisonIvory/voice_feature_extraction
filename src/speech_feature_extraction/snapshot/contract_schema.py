"""Versioned snapshot contract schema and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from speech_feature_extraction.snapshot.hashing import (
    compute_column_order_hash,
    compute_file_sha256,
    compute_table_schema_hash,
)

MANIFEST_VERSION = "1.0"
SUPPORTED_MANIFEST_VERSIONS = {MANIFEST_VERSION}

DATASET_NAME = "speech-features"
DATASET_VERSION = "v3"
MANIFEST_FILENAME = "manifest.json"
LATEST_POINTER_FILENAME = "latest.json"

DEFAULT_RECORDINGS_FILENAME = "voice_features_v3_recordings.parquet"
DEFAULT_AUDIT_FILENAME = "voice_features_v3_audit.parquet"

REQUIRED_CORE_COLUMNS = (
    "recordingId",
    "recordedDate",
    "taskType",
    "pipelineStatus",
    "extractorVersion",
    "featureSet",
    "featureLevel",
    "audioHash",
    "qc_task_qc_passed",
    "qc_warning_codes",
    "qc_opensmile_egemaps_success",
    "qc_feature_count_egemaps",
    "qc_feature_count_egemaps_expected",
)
REQUIRED_FEATURE_PREFIXES = ("egemaps_",)
REQUIRED_FEATURE_COUNT_BY_PREFIX = {"egemaps_": 88}


def validate_required_columns(
    frame: pd.DataFrame,
    required_core_columns: Sequence[str] = REQUIRED_CORE_COLUMNS,
    required_feature_prefixes: Sequence[str] = REQUIRED_FEATURE_PREFIXES,
    required_feature_count_by_prefix: Mapping[str, int] = REQUIRED_FEATURE_COUNT_BY_PREFIX,
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
    """Build per-file descriptor used by the manifest contract."""
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
    provenance: Mapping[str, Any],
    files: Mapping[str, Mapping[str, Any]],
    required_core_columns: Sequence[str] = REQUIRED_CORE_COLUMNS,
    required_feature_prefixes: Sequence[str] = REQUIRED_FEATURE_PREFIXES,
    required_feature_count_by_prefix: Mapping[str, int] = REQUIRED_FEATURE_COUNT_BY_PREFIX,
) -> dict[str, Any]:
    """Build snapshot manifest payload."""
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": DATASET_NAME,
        "version": DATASET_VERSION,
        "snapshot": snapshot,
        "created_at": created_at,
        "provenance": dict(provenance),
        "files": files,
        "required_core_columns": list(required_core_columns),
        "required_feature_prefixes": list(required_feature_prefixes),
        "required_feature_count_by_prefix": dict(required_feature_count_by_prefix),
    }


def validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    """Validate minimal manifest shape and supported contract version."""
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


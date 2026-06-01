import pandas as pd
import pytest

from speech_feature_extraction.snapshot.contract_schema import (
    MANIFEST_VERSION,
    REQUIRED_FEATURE_COUNT_BY_PREFIX,
    build_manifest,
    validate_manifest_contract,
    validate_required_columns,
)
from speech_feature_extraction.snapshot.hashing import compute_table_schema_hash

VOWEL_EGEMAPS_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["vowel_egemaps_"]
PROSODY_EGEMAPS_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["prosody_egemaps_"]


def _build_daily_frame() -> pd.DataFrame:
    row: dict[str, object] = {
        "userId": "u1",
        "dayUtc": "2026-06-01",
        "vowel_recording_count": 1,
        "prosody_recording_count": 1,
        "has_vowel": True,
        "has_prosody": True,
        "is_day_complete": True,
        "extractorVersion": "v4.0-daily-task-separated-median",
        "libraryVersion": "2.5.1",
    }
    for index in range(VOWEL_EGEMAPS_COUNT):
        row[f"vowel_egemaps_feature_{index}"] = float(index)
    for index in range(PROSODY_EGEMAPS_COUNT):
        row[f"prosody_egemaps_feature_{index}"] = float(index)
    return pd.DataFrame([row])


def test_compute_table_schema_hash_is_deterministic() -> None:
    frame_one = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    frame_two = pd.DataFrame({"a": [9, 8], "b": [7.0, 6.0]})
    assert compute_table_schema_hash(frame_one) == compute_table_schema_hash(frame_two)


def test_validate_required_columns_passes_with_full_contract() -> None:
    validate_required_columns(_build_daily_frame())


def test_validate_required_columns_fails_on_missing_core_column() -> None:
    frame = _build_daily_frame().drop(columns=["dayUtc"])
    with pytest.raises(ValueError, match="Missing required core columns"):
        validate_required_columns(frame)


def test_manifest_builder_and_contract_validation() -> None:
    manifest = build_manifest(
        snapshot="2026-06-01",
        created_at="2026-06-01T00:00:00+00:00",
        provenance={
            "source_commit": "abc1234",
            "pipeline_version": "v4.0-daily-task-separated-median",
            "opensmile_version": "2.5.1",
            "python_version": "3.11.9",
        },
        files={
            "daily": {
                "path": "voice_features_v4_daily.parquet",
                "row_count": 1,
                "schema_hash": "sha256:1",
                "column_order_hash": "sha256:2",
                "content_sha256": "sha256:3",
                "file_size_bytes": 42,
            },
            "audit": {
                "path": "voice_features_v4_audit.parquet",
                "row_count": 1,
                "schema_hash": "sha256:4",
                "column_order_hash": "sha256:5",
                "content_sha256": "sha256:6",
                "file_size_bytes": 43,
            },
        },
    )
    assert manifest["manifest_version"] == MANIFEST_VERSION
    validate_manifest_contract(manifest)


def test_manifest_validation_rejects_unknown_version() -> None:
    manifest = {
        "manifest_version": "9.9",
        "dataset": "speech-features",
        "version": "v4",
        "snapshot": "2026-06-01",
        "created_at": "2026-06-01T00:00:00+00:00",
        "provenance": {},
        "files": {},
        "required_core_columns": [],
        "required_feature_prefixes": [],
        "required_feature_count_by_prefix": {},
    }
    with pytest.raises(ValueError, match="Unsupported manifest version"):
        validate_manifest_contract(manifest)


def test_validate_required_columns_fails_on_incorrect_feature_count() -> None:
    frame = _build_daily_frame().drop(columns=["vowel_egemaps_feature_0"])
    with pytest.raises(ValueError, match="Feature prefix count mismatch"):
        validate_required_columns(frame)

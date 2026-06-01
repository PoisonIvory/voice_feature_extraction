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

EGEMAPS_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["egemaps_"]


def _build_recordings_frame() -> pd.DataFrame:
    row: dict[str, object] = {
        "recordingId": "r1",
        "recordedDate": "2026-06-01",
        "taskType": "vowel",
        "pipelineStatus": "completed",
        "extractorVersion": "v3.2",
        "featureSet": "opensmile.FeatureSet.eGeMAPSv02",
        "featureLevel": "opensmile.FeatureLevel.Functionals",
        "audioHash": "sha256:abc",
        "qc_task_qc_passed": True,
        "qc_warning_codes": [],
        "qc_opensmile_egemaps_success": True,
        "qc_feature_count_egemaps": EGEMAPS_COUNT,
        "qc_feature_count_egemaps_expected": EGEMAPS_COUNT,
    }
    for index in range(EGEMAPS_COUNT):
        row[f"egemaps_feature_{index}"] = float(index)
    return pd.DataFrame([row])


def test_compute_table_schema_hash_is_deterministic() -> None:
    frame_one = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    frame_two = pd.DataFrame({"a": [9, 8], "b": [7.0, 6.0]})
    assert compute_table_schema_hash(frame_one) == compute_table_schema_hash(frame_two)


def test_validate_required_columns_passes_with_full_contract() -> None:
    validate_required_columns(_build_recordings_frame())


def test_validate_required_columns_fails_on_missing_core_column() -> None:
    frame = _build_recordings_frame().drop(columns=["recordedDate"])
    with pytest.raises(ValueError, match="Missing required core columns"):
        validate_required_columns(frame)


def test_manifest_builder_and_contract_validation() -> None:
    manifest = build_manifest(
        snapshot="2026-06-01",
        created_at="2026-06-01T00:00:00+00:00",
        provenance={
            "source_commit": "abc1234",
            "pipeline_version": "v3.2-opensmile-egemaps-taskqc",
            "opensmile_version": "2.5.1",
            "python_version": "3.11.9",
        },
        files={
            "recordings": {
                "path": "voice_features_v3_recordings.parquet",
                "row_count": 1,
                "schema_hash": "sha256:1",
                "column_order_hash": "sha256:2",
                "content_sha256": "sha256:3",
                "file_size_bytes": 42,
            },
            "audit": {
                "path": "voice_features_v3_audit.parquet",
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
        "version": "v3",
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
    frame = _build_recordings_frame().drop(columns=["egemaps_feature_0"])
    with pytest.raises(ValueError, match="Feature prefix count mismatch"):
        validate_required_columns(frame)

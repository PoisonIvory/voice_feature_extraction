import json
from pathlib import Path

import pandas as pd

from speech_feature_extraction.snapshot.contract_schema import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_AUDIT_FILENAME,
    DEFAULT_RECORDINGS_FILENAME,
    LATEST_POINTER_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    REQUIRED_FEATURE_COUNT_BY_PREFIX,
)
from speech_feature_extraction.snapshot.hashing import compute_file_sha256
from speech_feature_extraction.snapshot.publisher import publish_snapshot_bundle

EGEMAPS_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["egemaps_"]


def _write_recordings_parquet(path: Path) -> None:
    row: dict[str, object] = {
        "recordingId": "r1",
        "recordedDate": "2026-06-01",
        "taskType": "vowel",
        "pipelineStatus": "completed",
        "extractorVersion": "v3.2-opensmile-egemaps-taskqc",
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
    pd.DataFrame([row]).to_parquet(path, index=False)


def _write_audit_parquet(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "recordingId": "r1",
                "pipelineStatus": "completed",
                "qc_failure_stage": None,
                "qc_failure_reason": None,
            }
        ]
    ).to_parquet(path, index=False)


def test_publish_snapshot_bundle_writes_manifest_and_latest_pointer(tmp_path: Path) -> None:
    recordings_path = tmp_path / DEFAULT_RECORDINGS_FILENAME
    audit_path = tmp_path / DEFAULT_AUDIT_FILENAME
    _write_recordings_parquet(recordings_path)
    _write_audit_parquet(audit_path)

    snapshot_root = tmp_path / "snapshots"
    manifest_path = publish_snapshot_bundle(
        recordings_path=recordings_path,
        audit_path=audit_path,
        snapshot_root=snapshot_root,
        snapshot_id="2026-06-01",
        pipeline_version="v3.2-opensmile-egemaps-taskqc",
        opensmile_version="2.5.1",
        source_commit="abc1234",
        update_latest=True,
    )

    assert manifest_path.name == MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["snapshot"] == "2026-06-01"
    assert manifest["provenance"]["pipeline_version"] == "v3.2-opensmile-egemaps-taskqc"
    assert manifest["required_feature_count_by_prefix"]["egemaps_"] == EGEMAPS_COUNT

    published_dir = snapshot_root / DATASET_NAME / DATASET_VERSION / "2026-06-01"
    published_recordings = published_dir / DEFAULT_RECORDINGS_FILENAME
    published_audit = published_dir / DEFAULT_AUDIT_FILENAME
    assert published_recordings.exists()
    assert published_audit.exists()

    recordings_descriptor = manifest["files"]["recordings"]
    assert recordings_descriptor["path"] == DEFAULT_RECORDINGS_FILENAME
    assert recordings_descriptor["content_sha256"] == compute_file_sha256(published_recordings)
    assert recordings_descriptor["file_size_bytes"] == published_recordings.stat().st_size

    latest_path = snapshot_root / DATASET_NAME / DATASET_VERSION / LATEST_POINTER_FILENAME
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    assert latest["snapshot"] == "2026-06-01"

import json
from pathlib import Path

import pandas as pd

from speech_feature_extraction.snapshot.contract_schema import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_AUDIT_FILENAME,
    DEFAULT_DAILY_FILENAME,
    LATEST_POINTER_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    REQUIRED_FEATURE_COUNT_BY_PREFIX,
)
from speech_feature_extraction.snapshot.hashing import compute_file_sha256
from speech_feature_extraction.snapshot.publisher import publish_snapshot_bundle

VOWEL_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["vowel_egemaps_"]
PROSODY_COUNT = REQUIRED_FEATURE_COUNT_BY_PREFIX["prosody_egemaps_"]


def _write_daily_parquet(path: Path) -> None:
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
    for index in range(VOWEL_COUNT):
        row[f"vowel_egemaps_feature_{index}"] = float(index)
    for index in range(PROSODY_COUNT):
        row[f"prosody_egemaps_feature_{index}"] = float(index)
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
    daily_path = tmp_path / DEFAULT_DAILY_FILENAME
    audit_path = tmp_path / DEFAULT_AUDIT_FILENAME
    _write_daily_parquet(daily_path)
    _write_audit_parquet(audit_path)

    snapshot_root = tmp_path / "snapshots"
    manifest_path = publish_snapshot_bundle(
        daily_path=daily_path,
        audit_path=audit_path,
        snapshot_root=snapshot_root,
        snapshot_id="2026-06-01",
        pipeline_version="v4.0-daily-task-separated-median",
        opensmile_version="2.5.1",
        source_commit="abc1234",
        update_latest=True,
    )

    assert manifest_path.name == MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["manifest_version"] == MANIFEST_VERSION
    assert manifest["snapshot"] == "2026-06-01"
    assert manifest["provenance"]["pipeline_version"] == "v4.0-daily-task-separated-median"
    assert manifest["required_feature_count_by_prefix"]["vowel_egemaps_"] == VOWEL_COUNT
    assert manifest["required_feature_count_by_prefix"]["prosody_egemaps_"] == PROSODY_COUNT

    published_dir = snapshot_root / DATASET_NAME / DATASET_VERSION / "2026-06-01"
    published_daily = published_dir / DEFAULT_DAILY_FILENAME
    published_audit = published_dir / DEFAULT_AUDIT_FILENAME
    assert published_daily.exists()
    assert published_audit.exists()

    daily_descriptor = manifest["files"]["daily"]
    assert daily_descriptor["path"] == DEFAULT_DAILY_FILENAME
    assert daily_descriptor["content_sha256"] == compute_file_sha256(published_daily)
    assert daily_descriptor["file_size_bytes"] == published_daily.stat().st_size

    latest_path = snapshot_root / DATASET_NAME / DATASET_VERSION / LATEST_POINTER_FILENAME
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    assert latest["snapshot"] == "2026-06-01"

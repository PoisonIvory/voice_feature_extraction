"""Snapshot publishing helpers for parquet artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from speech_feature_extraction.constants import (
    AUDIT_PARQUET,
    RECORDINGS_PARQUET,
    SNAPSHOT_DATASET_NAME,
    SNAPSHOT_DATASET_VERSION,
    SNAPSHOT_LATEST_POINTER_FILENAME,
    SNAPSHOT_MANIFEST_FILENAME,
)
from speech_feature_extraction.snapshot_contract import (
    build_file_descriptor,
    build_manifest,
    validate_manifest_contract,
    validate_required_columns,
)


def detect_source_commit(repo_dir: Path) -> str | None:
    """Return short git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    value = result.stdout.strip()
    return value or None


def publish_snapshot_bundle(
    *,
    recordings_path: Path,
    audit_path: Path,
    snapshot_root: Path,
    snapshot_id: str | None,
    pipeline_version: str,
    opensmile_version: str | None,
    source_commit: str | None,
    update_latest: bool = False,
) -> Path:
    """Publish immutable snapshot files and manifest."""
    created_at = datetime.now(tz=timezone.utc).isoformat()
    resolved_snapshot = snapshot_id or created_at[:10]
    snapshot_dir = snapshot_root / SNAPSHOT_DATASET_NAME / SNAPSHOT_DATASET_VERSION / resolved_snapshot
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    published_recordings = _copy_artifact(recordings_path, snapshot_dir / RECORDINGS_PARQUET)
    published_audit = _copy_artifact(audit_path, snapshot_dir / AUDIT_PARQUET)

    recordings_frame = pd.read_parquet(published_recordings)
    audit_frame = pd.read_parquet(published_audit)
    validate_required_columns(recordings_frame)

    file_descriptors = {
        "recordings": build_file_descriptor(published_recordings, recordings_frame),
        "audit": build_file_descriptor(published_audit, audit_frame),
    }
    manifest = build_manifest(
        snapshot=resolved_snapshot,
        created_at=created_at,
        source_commit=source_commit,
        pipeline_version=pipeline_version,
        opensmile_version=opensmile_version,
        files=file_descriptors,
    )
    validate_manifest_contract(manifest)

    manifest_path = snapshot_dir / SNAPSHOT_MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    if update_latest:
        latest_path = snapshot_root / SNAPSHOT_DATASET_NAME / SNAPSHOT_DATASET_VERSION / SNAPSHOT_LATEST_POINTER_FILENAME
        latest_payload = {"snapshot": resolved_snapshot, "updated_at": created_at}
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(latest_payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

    return manifest_path


def _copy_artifact(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path

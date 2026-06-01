"""Publishing service for immutable snapshot bundles."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from speech_feature_extraction.snapshot.contract_schema import (
    DATASET_NAME,
    DATASET_VERSION,
    DEFAULT_AUDIT_FILENAME,
    DEFAULT_RECORDINGS_FILENAME,
    LATEST_POINTER_FILENAME,
    MANIFEST_FILENAME,
    build_file_descriptor,
    build_manifest,
    validate_manifest_contract,
    validate_required_columns,
)
from speech_feature_extraction.snapshot.provenance import build_provenance


def publish_snapshot_bundle(
    *,
    recordings_path: Path,
    audit_path: Path,
    snapshot_root: Path,
    snapshot_id: str | None = None,
    pipeline_version: str | None = None,
    opensmile_version: str | None = None,
    source_commit: str | None = None,
    python_version: str | None = None,
    repo_dir: Path | None = None,
    update_latest: bool = False,
) -> Path:
    """Publish immutable snapshot files and manifest."""
    created_at = datetime.now(tz=timezone.utc).isoformat()
    resolved_snapshot = snapshot_id or created_at[:10]
    snapshot_dir = snapshot_root / DATASET_NAME / DATASET_VERSION / resolved_snapshot
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    published_recordings = _copy_artifact(recordings_path, snapshot_dir / DEFAULT_RECORDINGS_FILENAME)
    published_audit = _copy_artifact(audit_path, snapshot_dir / DEFAULT_AUDIT_FILENAME)

    recordings_frame = pd.read_parquet(published_recordings)
    audit_frame = pd.read_parquet(published_audit)
    validate_required_columns(recordings_frame)

    resolved_pipeline_version = pipeline_version or _infer_single_value(recordings_frame, "extractorVersion")
    resolved_opensmile_version = opensmile_version or _infer_single_value(recordings_frame, "libraryVersion")
    provenance = build_provenance(
        source_commit=source_commit,
        pipeline_version=resolved_pipeline_version,
        opensmile_version=resolved_opensmile_version,
        python_version=python_version,
        repo_dir=repo_dir,
    )

    file_descriptors = {
        "recordings": build_file_descriptor(published_recordings, recordings_frame),
        "audit": build_file_descriptor(published_audit, audit_frame),
    }
    manifest = build_manifest(
        snapshot=resolved_snapshot,
        created_at=created_at,
        provenance=provenance,
        files=file_descriptors,
    )
    validate_manifest_contract(manifest)

    manifest_path = snapshot_dir / MANIFEST_FILENAME
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    if update_latest:
        latest_path = snapshot_root / DATASET_NAME / DATASET_VERSION / LATEST_POINTER_FILENAME
        latest_payload = {"snapshot": resolved_snapshot, "updated_at": created_at}
        with latest_path.open("w", encoding="utf-8") as handle:
            json.dump(latest_payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")

    return manifest_path


def _copy_artifact(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def _infer_single_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    non_null_values = [str(value) for value in frame[column].dropna().unique().tolist()]
    if not non_null_values:
        return None
    return non_null_values[0]


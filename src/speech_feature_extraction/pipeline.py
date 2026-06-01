"""Batch audit and extraction orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from speech_feature_extraction.appwrite_gateway import AppwriteGateway
from speech_feature_extraction.audio_qc import inspect_wav, sha256_file
from speech_feature_extraction.config import Settings
from speech_feature_extraction.constants import AUDIT_PARQUET, EXTRACTOR_VERSION, RECORDINGS_PARQUET
from speech_feature_extraction.metadata import build_manifest_rows
from speech_feature_extraction.opensmile_egemaps import OpenSmileEgemapsExtractor
from speech_feature_extraction.parquet import write_rows_parquet


def run_audit(settings: Settings) -> Path:
    gateway = AppwriteGateway(settings)
    manifest_rows = _load_manifest_rows(gateway)
    audit_path = settings.processed_dir / AUDIT_PARQUET
    return write_rows_parquet(manifest_rows, audit_path)


def run_extract(settings: Settings, limit: int | None = None, force_download: bool = False) -> tuple[Path, Path]:
    gateway = AppwriteGateway(settings)
    manifest_rows = _load_manifest_rows(gateway)
    extractor = OpenSmileEgemapsExtractor()

    recording_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    candidates = [row for row in manifest_rows if row["pipelineStatus"] == "pending"]
    if limit is not None:
        candidates = candidates[:limit]

    for manifest_row in candidates:
        recording_id = manifest_row["recordingId"]
        audit_row = dict(manifest_row)
        try:
            wav_path = _cached_audio_path(settings.raw_audio_dir, recording_id)
            if force_download or not wav_path.exists():
                gateway.download_audio_file(recording_id, wav_path)

            audio_hash = sha256_file(wav_path)
            qc = inspect_wav(wav_path)
            if not qc["qc_audio_readable"]:
                raise ValueError(qc["qc_failure_reason"])

            features = extractor.extract_file(wav_path)
            extraction_timestamp = datetime.now(tz=timezone.utc).isoformat()
            recording_rows.append(
                {
                    **manifest_row,
                    "audioHash": audio_hash,
                    "extractorVersion": EXTRACTOR_VERSION,
                    "extractionTimestamp": extraction_timestamp,
                    "featureSet": "opensmile.FeatureSet.eGeMAPSv02",
                    "featureLevel": "opensmile.FeatureLevel.Functionals",
                    "libraryName": "opensmile",
                    "libraryVersion": extractor.library_version,
                    **_merge_qc(manifest_row, qc),
                    **features,
                    "pipelineStatus": "completed",
                    "skipReason": None,
                }
            )
            audit_row.update(
                {
                    "audioHash": audio_hash,
                    "extractorVersion": EXTRACTOR_VERSION,
                    "extractionTimestamp": extraction_timestamp,
                    "libraryName": "opensmile",
                    "libraryVersion": extractor.library_version,
                    "pipelineStatus": "completed",
                    **_merge_qc(manifest_row, qc),
                    "qc_opensmile_egemaps_success": True,
                    "qc_feature_count_egemaps": features["qc_feature_count_egemaps"],
                    "qc_failure_reason": None,
                }
            )
        except Exception as error:
            audit_row.update(
                {
                    "extractorVersion": EXTRACTOR_VERSION,
                    "extractionTimestamp": datetime.now(tz=timezone.utc).isoformat(),
                    "pipelineStatus": "failed",
                    "qc_opensmile_egemaps_success": False,
                    "qc_failure_reason": str(error),
                }
            )
        audit_rows.append(audit_row)

    skipped_rows = [row for row in manifest_rows if row["pipelineStatus"] != "pending"]
    audit_path = write_rows_parquet(skipped_rows + audit_rows, settings.processed_dir / AUDIT_PARQUET)
    recordings_path = write_rows_parquet(recording_rows, settings.processed_dir / RECORDINGS_PARQUET)
    return recordings_path, audit_path


def _load_manifest_rows(gateway: AppwriteGateway) -> list[dict[str, Any]]:
    storage_files = gateway.list_audio_files()
    voice_recordings = gateway.list_voice_recordings()
    return build_manifest_rows(storage_files, voice_recordings)


def _cached_audio_path(raw_audio_dir: Path, recording_id: str) -> Path:
    return raw_audio_dir / f"{recording_id}.wav"


def _merge_qc(manifest_row: dict[str, Any], qc: dict[str, Any]) -> dict[str, Any]:
    manifest_warnings = manifest_row.get("qc_warning_codes") or []
    qc_warnings = qc.get("qc_warning_codes") or []
    return {
        **qc,
        "qc_warning_codes": sorted(set(manifest_warnings + qc_warnings)),
    }

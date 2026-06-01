"""Batch audit and extraction orchestration."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from speech_feature_extraction.appwrite_gateway import AppwriteGateway
from speech_feature_extraction.audio_qc import inspect_wav, sha256_file
from speech_feature_extraction.config import Settings
from speech_feature_extraction.constants import AUDIT_PARQUET, EXTRACTOR_VERSION, RECORDINGS_PARQUET
from speech_feature_extraction.metadata import build_manifest_rows
from speech_feature_extraction.opensmile_egemaps import OpenSmileEgemapsExtractor
from speech_feature_extraction.parquet import read_rows_parquet, write_rows_parquet

LOGGER = logging.getLogger(__name__)


def run_audit(settings: Settings) -> Path:
    LOGGER.info("Audit run started")
    gateway = AppwriteGateway(settings)
    manifest_rows = _load_manifest_rows(gateway)
    LOGGER.info("Audit manifest rows: %d", len(manifest_rows))
    audit_path = settings.processed_dir / AUDIT_PARQUET
    output = write_rows_parquet(manifest_rows, audit_path)
    LOGGER.info("Audit parquet written: %s", output)
    return output


def run_extract(settings: Settings, limit: int | None = None, force_download: bool = False) -> tuple[Path, Path]:
    LOGGER.info("Extract run started (limit=%s force_download=%s)", limit, force_download)
    gateway = AppwriteGateway(settings)
    manifest_rows = _load_manifest_rows(gateway)
    extractor = OpenSmileEgemapsExtractor()
    recordings_existing = read_rows_parquet(settings.processed_dir / RECORDINGS_PARQUET)
    recording_rows: list[dict[str, Any]] = []
    audit_rows_by_recording_id = {row["recordingId"]: dict(row) for row in manifest_rows}
    candidates = [row for row in manifest_rows if row["pipelineStatus"] == "pending"]
    LOGGER.info(
        "Manifest loaded: total=%d pending=%d existing_recordings=%d",
        len(manifest_rows),
        len(candidates),
        len(recordings_existing),
    )
    if limit is not None:
        candidates = candidates[:limit]
        LOGGER.info("Applied extraction limit: %d candidates", len(candidates))

    for index, manifest_row in enumerate(candidates, start=1):
        recording_id = manifest_row["recordingId"]
        LOGGER.info("Processing recording %d/%d: %s", index, len(candidates), recording_id)
        extraction_timestamp = datetime.now(tz=timezone.utc).isoformat()
        audit_row = dict(manifest_row)
        audit_row.update(_build_lineage(extractor, extraction_timestamp))
        stage = "download_audio"
        try:
            wav_path = _cached_audio_path(settings.raw_audio_dir, recording_id)
            if force_download or not wav_path.exists():
                stage = "download_audio"
                LOGGER.debug("Downloading audio: %s -> %s", recording_id, wav_path)
                gateway.download_audio_file(recording_id, wav_path)
            else:
                LOGGER.debug("Using cached audio: %s", wav_path)

            stage = "audio_hash"
            LOGGER.debug("Hashing audio: %s", wav_path)
            audio_hash = sha256_file(wav_path)
            stage = "wav_qc"
            LOGGER.debug("Running WAV QC: %s", wav_path)
            qc = inspect_wav(wav_path)
            if not qc["qc_audio_readable"]:
                raise ValueError(qc["qc_failure_reason"])

            stage = "opensmile_extract"
            LOGGER.debug("Running openSMILE extraction: %s", wav_path)
            features = extractor.extract_file(wav_path)
            merged_qc = _merge_qc(manifest_row, qc)
            recording_rows.append(
                {
                    **manifest_row,
                    "audioHash": audio_hash,
                    **_build_lineage(extractor, extraction_timestamp),
                    **merged_qc,
                    **features,
                    "pipelineStatus": "completed",
                    "skipReason": None,
                }
            )
            audit_row.update(
                {
                    "audioHash": audio_hash,
                    "pipelineStatus": "completed",
                    **merged_qc,
                    "qc_opensmile_egemaps_success": True,
                    "qc_feature_count_egemaps": features["qc_feature_count_egemaps"],
                    "qc_feature_count_egemaps_expected": features["qc_feature_count_egemaps_expected"],
                    "qc_failure_stage": None,
                    "qc_failure_reason": None,
                }
            )
            LOGGER.info("Completed recording: %s", recording_id)
        except Exception as error:
            audit_row.update(
                {
                    "pipelineStatus": "failed",
                    "qc_opensmile_egemaps_success": False,
                    "qc_failure_stage": stage,
                    "qc_failure_reason": str(error),
                }
            )
            LOGGER.warning("Failed recording: %s stage=%s error=%s", recording_id, stage, error)
        audit_rows_by_recording_id[recording_id] = audit_row

    merged_recording_rows = _upsert_recording_rows(recordings_existing, recording_rows)
    audit_rows = [audit_rows_by_recording_id[key] for key in sorted(audit_rows_by_recording_id, key=str)]
    audit_path = write_rows_parquet(audit_rows, settings.processed_dir / AUDIT_PARQUET)
    recordings_path = write_rows_parquet(merged_recording_rows, settings.processed_dir / RECORDINGS_PARQUET)
    LOGGER.info(
        "Extract run finished: completed=%d failed=%d recordings_path=%s audit_path=%s",
        sum(1 for row in audit_rows if row.get("pipelineStatus") == "completed"),
        sum(1 for row in audit_rows if row.get("pipelineStatus") == "failed"),
        recordings_path,
        audit_path,
    )
    return recordings_path, audit_path


def _load_manifest_rows(gateway: AppwriteGateway) -> list[dict[str, Any]]:
    LOGGER.debug("Loading storage files and voice recordings from Appwrite")
    storage_files = gateway.list_audio_files()
    voice_recordings = gateway.list_voice_recordings()
    LOGGER.info(
        "Appwrite load complete: storage_files=%d voice_recordings=%d",
        len(storage_files),
        len(voice_recordings),
    )
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


def _build_lineage(extractor: OpenSmileEgemapsExtractor, extraction_timestamp: str) -> dict[str, Any]:
    return {
        "extractorVersion": EXTRACTOR_VERSION,
        "extractionTimestamp": extraction_timestamp,
        **extractor.extraction_metadata,
    }


def _upsert_recording_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_recording_id: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        recording_id = row.get("recordingId")
        if recording_id is None:
            continue
        rows_by_recording_id[str(recording_id)] = row

    for row in new_rows:
        recording_id = row.get("recordingId")
        if recording_id is None:
            continue
        rows_by_recording_id[str(recording_id)] = row

    return [rows_by_recording_id[key] for key in sorted(rows_by_recording_id)]

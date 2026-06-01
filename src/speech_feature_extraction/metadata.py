"""Metadata normalization for Appwrite storage-first processing."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from speech_feature_extraction.constants import IN_SCOPE_TASK_TYPES

FILENAME_PATTERN = re.compile(
    r"^(?P<user_id>[^_]+)_(?P<task_type>[^_]+)_(?P<timestamp_ms>\d+)\.wav$",
    re.IGNORECASE,
)
USER_PERMISSION_PATTERN = re.compile(r'user:([^")]+)')


def parse_filename_metadata(filename: str) -> dict[str, Any]:
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return {
            "filenameUserId": None,
            "filenameTaskType": None,
            "filenameRecordedAt": None,
            "filenameRecordedDate": None,
        }

    timestamp_ms = int(match.group("timestamp_ms"))
    recorded_at = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return {
        "filenameUserId": match.group("user_id"),
        "filenameTaskType": match.group("task_type"),
        "filenameRecordedAt": recorded_at.isoformat(),
        "filenameRecordedDate": recorded_at.date().isoformat(),
    }


def index_voice_recordings(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for document in documents:
        file_id = document.get("fileId")
        if isinstance(file_id, str) and file_id:
            indexed[file_id] = document
    return indexed


def extract_user_id_from_permissions(storage_file: dict[str, Any]) -> str | None:
    for permission in storage_file.get("$permissions", []) or []:
        if not isinstance(permission, str):
            continue
        match = USER_PERMISSION_PATTERN.search(permission)
        if match:
            return match.group(1)
    return None


def build_manifest_row(
    storage_file: dict[str, Any],
    voice_recording: dict[str, Any] | None,
) -> dict[str, Any]:
    filename = storage_file.get("name") or ""
    filename_metadata = parse_filename_metadata(filename)
    storage_user_id = extract_user_id_from_permissions(storage_file)
    warnings: list[str] = []

    metadata_task = _clean_task_type(_get(voice_recording, "taskType"))
    filename_task = _clean_task_type(filename_metadata["filenameTaskType"])
    task_type = metadata_task or filename_task

    user_id = _get(voice_recording, "userId") or storage_user_id or filename_metadata["filenameUserId"]
    recorded_at = _get(voice_recording, "recordedAt") or filename_metadata["filenameRecordedAt"]
    recorded_date = _get(voice_recording, "recordedDate") or filename_metadata["filenameRecordedDate"]

    if voice_recording is None:
        warnings.append("metadata_missing")
    if not user_id or not recorded_at:
        warnings.append("required_metadata_missing")
    if task_type is None:
        warnings.append("task_unknown")
    if metadata_task and filename_task and metadata_task != filename_task:
        warnings.append("task_disagreement")
    if task_type and task_type not in IN_SCOPE_TASK_TYPES:
        warnings.append("out_of_scope_task")

    in_scope = task_type in IN_SCOPE_TASK_TYPES
    metadata_complete = bool(user_id and recorded_at and recorded_date and task_type)

    return {
        "recordingId": storage_file.get("$id"),
        "storageFileId": storage_file.get("$id"),
        "voiceRecordingId": _get(voice_recording, "$id"),
        "userId": user_id,
        "taskType": task_type,
        "recordedAt": recorded_at,
        "recordedDate": recorded_date,
        "bucketId": _get(voice_recording, "bucketId") or "audio",
        "filename": filename,
        "storageMimeType": storage_file.get("mimeType"),
        "storageSizeBytes": storage_file.get("sizeOriginal"),
        "storageCreatedAt": storage_file.get("$createdAt"),
        "metadataPresent": voice_recording is not None,
        "metadataComplete": metadata_complete,
        "inScope": in_scope,
        "pipelineStatus": "pending" if in_scope and metadata_complete else "skipped",
        "skipReason": None if in_scope and metadata_complete else _skip_reason(task_type, metadata_complete),
        "qc_warning_codes": warnings,
    }


def build_manifest_rows(
    storage_files: list[dict[str, Any]],
    voice_recordings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recordings_by_file_id = index_voice_recordings(voice_recordings)
    rows = []
    for storage_file in storage_files:
        if not _is_wav(storage_file):
            continue
        rows.append(build_manifest_row(storage_file, recordings_by_file_id.get(storage_file.get("$id"))))
    return rows


def _get(document: dict[str, Any] | None, key: str) -> Any:
    if not document:
        return None
    value = document.get(key)
    return value if value not in ("", []) else None


def _clean_task_type(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value.lower()


def _is_wav(storage_file: dict[str, Any]) -> bool:
    filename = str(storage_file.get("name") or "").lower()
    mime_type = str(storage_file.get("mimeType") or "").lower()
    return filename.endswith(".wav") or mime_type in {"audio/wav", "audio/x-wav", "audio/wave"}


def _skip_reason(task_type: str | None, metadata_complete: bool) -> str:
    if task_type and task_type not in IN_SCOPE_TASK_TYPES:
        return "out_of_scope_task"
    if not metadata_complete:
        return "metadata_error"
    return "unknown"

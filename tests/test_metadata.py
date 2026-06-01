from speech_feature_extraction.metadata import build_manifest_row, parse_filename_metadata


def test_parse_filename_metadata_extracts_task_and_date() -> None:
    metadata = parse_filename_metadata("user123_vowel_1704067200000.wav")

    assert metadata["filenameUserId"] == "user123"
    assert metadata["filenameTaskType"] == "vowel"
    assert metadata["filenameRecordedDate"] == "2024-01-01"


def test_parse_filename_metadata_does_not_default_unknown_names() -> None:
    metadata = parse_filename_metadata("not-a-valid-name.wav")

    assert metadata["filenameUserId"] is None
    assert metadata["filenameTaskType"] is None
    assert metadata["filenameRecordedAt"] is None


def test_build_manifest_row_prefers_voice_recording_metadata() -> None:
    storage_file = {
        "$id": "file_1",
        "name": "filenameUser_vowel_1704067200000.wav",
        "mimeType": "audio/wav",
        "sizeOriginal": 123,
    }
    voice_recording = {
        "$id": "doc_1",
        "fileId": "file_1",
        "userId": "metadataUser",
        "taskType": "prosody",
        "recordedAt": "2024-01-02T00:00:00+00:00",
        "recordedDate": "2024-01-02",
        "bucketId": "audio",
    }

    row = build_manifest_row(storage_file, voice_recording)

    assert row["userId"] == "metadataUser"
    assert row["taskType"] == "prosody"
    assert row["recordedDate"] == "2024-01-02"
    assert row["pipelineStatus"] == "pending"
    assert "task_disagreement" in row["qc_warning_codes"]


def test_build_manifest_row_skips_ambiguous_metadata() -> None:
    storage_file = {
        "$id": "file_1",
        "name": "invalid.wav",
        "mimeType": "audio/wav",
        "sizeOriginal": 123,
    }

    row = build_manifest_row(storage_file, None)

    assert row["pipelineStatus"] == "skipped"
    assert row["skipReason"] == "metadata_error"
    assert "metadata_missing" in row["qc_warning_codes"]
    assert "task_unknown" in row["qc_warning_codes"]

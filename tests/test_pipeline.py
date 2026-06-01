from pathlib import Path

from speech_feature_extraction.opensmile_egemaps import LldQcMetrics
from speech_feature_extraction.pipeline import run_extract


class _Settings:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    @property
    def raw_audio_dir(self) -> Path:
        return self.data_dir / "raw_audio"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


class _FakeGateway:
    def __init__(self, settings: _Settings) -> None:
        self._settings = settings

    def download_audio_file(self, file_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file_id.encode("utf-8"))
        return destination


class _FakeExtractor:
    def __init__(self, include_geometry_derived: bool = False) -> None:
        self._include_geometry_derived = include_geometry_derived

    @property
    def extraction_metadata(self) -> dict[str, object]:
        return {
            "featureSet": "opensmile.FeatureSet.eGeMAPSv02",
            "featureLevel": "opensmile.FeatureLevel.Functionals",
            "libraryName": "opensmile",
            "libraryVersion": "2.5.0",
            "opensmileConfigName": "eGeMAPSv02",
            "opensmileConfigFile": "eGeMAPSv02.conf",
            "opensmileSamplingRateHz": 16000,
            "opensmileResampleEnabled": True,
            "opensmileChannels": 0,
            "opensmileMixdownEnabled": False,
            "opensmileGeometryDerivedEnabled": self._include_geometry_derived,
        }

    def extract_file(self, _: Path) -> dict[str, object]:
        return {
            "egemaps_feature_a": 1.23,
            "qc_opensmile_egemaps_success": True,
            "qc_feature_count_egemaps": 88,
            "qc_feature_count_egemaps_expected": 88,
        }

    def extract_file_with_qc(self, path: Path) -> tuple[dict[str, object], LldQcMetrics]:
        return (
            self.extract_file(path),
            LldQcMetrics(
                voiced_ratio=0.95,
                f0_cov=0.10,
                jitter_mean=0.8,
                shimmer_db_mean=0.2,
                total_frames=100,
                voiced_frames=95,
            ),
        )


def _settings(tmp_path: Path) -> _Settings:
    return _Settings(data_dir=tmp_path / "data")


def test_run_extract_upserts_recordings_and_keeps_unprocessed_manifest_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    existing_rows = [
        {"recordingId": "already_done", "audioHash": "existing_hash"},
        {"recordingId": "to_process", "audioHash": "old_hash"},
    ]
    parquet_store: dict[str, list[dict[str, object]]] = {
        "voice_features_v3_recordings.parquet": existing_rows
    }

    def _fake_read_rows_parquet(path: Path) -> list[dict[str, object]]:
        return list(parquet_store.get(path.name, []))

    def _fake_write_rows_parquet(rows: list[dict[str, object]], path: Path) -> Path:
        parquet_store[path.name] = list(rows)
        return path

    manifest_rows = [
        {"recordingId": "to_process", "pipelineStatus": "pending", "taskType": "vowel", "qc_warning_codes": [], "skipReason": None},
        {"recordingId": "not_processed_due_to_limit", "pipelineStatus": "pending", "taskType": "prosody", "qc_warning_codes": [], "skipReason": None},
        {"recordingId": "skipped_row", "pipelineStatus": "skipped", "taskType": None, "qc_warning_codes": ["metadata_missing"], "skipReason": "metadata_error"},
    ]

    monkeypatch.setattr("speech_feature_extraction.pipeline.AppwriteGateway", _FakeGateway)
    monkeypatch.setattr("speech_feature_extraction.pipeline.OpenSmileEgemapsExtractor", _FakeExtractor)
    monkeypatch.setattr("speech_feature_extraction.pipeline._load_manifest_rows", lambda _: manifest_rows)
    monkeypatch.setattr("speech_feature_extraction.pipeline.sha256_file", lambda _: "new_hash")
    monkeypatch.setattr("speech_feature_extraction.pipeline.read_rows_parquet", _fake_read_rows_parquet)
    monkeypatch.setattr("speech_feature_extraction.pipeline.write_rows_parquet", _fake_write_rows_parquet)
    monkeypatch.setattr(
        "speech_feature_extraction.pipeline.inspect_wav",
        lambda _: {
            "qc_audio_readable": True,
            "qc_duration_sec": 3.0,
            "qc_clipping_ratio": 0.0,
            "qc_warning_codes": [],
            "qc_failure_reason": None,
        },
    )

    run_extract(settings=settings, limit=1)

    recordings = parquet_store["voice_features_v3_recordings.parquet"]
    rows_by_id = {row["recordingId"]: row for row in recordings}
    assert set(rows_by_id) == {"already_done", "to_process"}
    assert rows_by_id["to_process"]["audioHash"] == "new_hash"

    audit = parquet_store["voice_features_v3_audit.parquet"]
    audit_rows_by_id = {row["recordingId"]: row for row in audit}
    assert set(audit_rows_by_id) == {"to_process", "not_processed_due_to_limit", "skipped_row"}
    assert audit_rows_by_id["to_process"]["pipelineStatus"] == "completed"
    assert audit_rows_by_id["not_processed_due_to_limit"]["pipelineStatus"] == "pending"


def test_run_extract_writes_structured_failure_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = _settings(tmp_path)
    parquet_store: dict[str, list[dict[str, object]]] = {}

    def _fake_read_rows_parquet(path: Path) -> list[dict[str, object]]:
        return list(parquet_store.get(path.name, []))

    def _fake_write_rows_parquet(rows: list[dict[str, object]], path: Path) -> Path:
        parquet_store[path.name] = list(rows)
        return path

    manifest_rows = [
        {"recordingId": "to_fail", "pipelineStatus": "pending", "taskType": "vowel", "qc_warning_codes": [], "skipReason": None},
    ]

    class _FailingExtractor(_FakeExtractor):
        def extract_file_with_qc(self, _: Path) -> tuple[dict[str, object], LldQcMetrics]:
            raise ValueError("boom")

    monkeypatch.setattr("speech_feature_extraction.pipeline.AppwriteGateway", _FakeGateway)
    monkeypatch.setattr("speech_feature_extraction.pipeline.OpenSmileEgemapsExtractor", _FailingExtractor)
    monkeypatch.setattr("speech_feature_extraction.pipeline._load_manifest_rows", lambda _: manifest_rows)
    monkeypatch.setattr("speech_feature_extraction.pipeline.sha256_file", lambda _: "new_hash")
    monkeypatch.setattr("speech_feature_extraction.pipeline.read_rows_parquet", _fake_read_rows_parquet)
    monkeypatch.setattr("speech_feature_extraction.pipeline.write_rows_parquet", _fake_write_rows_parquet)
    monkeypatch.setattr(
        "speech_feature_extraction.pipeline.inspect_wav",
        lambda _: {
            "qc_audio_readable": True,
            "qc_duration_sec": 3.0,
            "qc_clipping_ratio": 0.0,
            "qc_warning_codes": [],
            "qc_failure_reason": None,
        },
    )

    run_extract(settings=settings, limit=1)

    row = parquet_store["voice_features_v3_audit.parquet"][0]
    assert row["pipelineStatus"] == "failed"
    assert row["qc_failure_stage"] == "opensmile_extract"
    assert row["featureSet"] == "opensmile.FeatureSet.eGeMAPSv02"

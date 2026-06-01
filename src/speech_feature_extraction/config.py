"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from speech_feature_extraction.constants import (
    APPWRITE_AUDIO_BUCKET_ID,
    APPWRITE_DATABASE_ID,
    APPWRITE_VOICE_RECORDINGS_COLLECTION_ID,
)


@dataclass(frozen=True)
class Settings:
    appwrite_endpoint: str
    appwrite_project_id: str
    appwrite_api_key: str
    appwrite_database_id: str
    appwrite_voice_recordings_collection_id: str
    appwrite_audio_bucket_id: str
    data_dir: Path
    exports_dir: Path
    publish_snapshot: bool
    snapshot_root: Path
    snapshot_id: str | None
    snapshot_update_latest: bool

    @property
    def raw_audio_dir(self) -> Path:
        return self.data_dir / "raw_audio"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


def load_settings(env_file: str | None = None) -> Settings:
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()

    settings = Settings(
        appwrite_endpoint=os.getenv("APPWRITE_ENDPOINT", "https://sfo.cloud.appwrite.io/v1"),
        appwrite_project_id=os.getenv("APPWRITE_PROJECT_ID", ""),
        appwrite_api_key=os.getenv("APPWRITE_API_KEY", ""),
        appwrite_database_id=os.getenv("APPWRITE_DATABASE_ID", APPWRITE_DATABASE_ID),
        appwrite_voice_recordings_collection_id=os.getenv(
            "APPWRITE_VOICE_RECORDINGS_COLLECTION_ID",
            APPWRITE_VOICE_RECORDINGS_COLLECTION_ID,
        ),
        appwrite_audio_bucket_id=os.getenv("APPWRITE_AUDIO_BUCKET_ID", APPWRITE_AUDIO_BUCKET_ID),
        data_dir=Path(os.getenv("SPEECH_DATA_DIR", "data")),
        exports_dir=Path(os.getenv("SPEECH_EXPORTS_DIR", "exports")),
        publish_snapshot=_env_bool("SPEECH_PUBLISH_SNAPSHOT", False),
        snapshot_root=Path(os.getenv("SPEECH_SNAPSHOT_ROOT", "exports/snapshots")),
        snapshot_id=os.getenv("SPEECH_SNAPSHOT_ID") or None,
        snapshot_update_latest=_env_bool("SPEECH_UPDATE_LATEST", False),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    missing = []
    if not settings.appwrite_project_id:
        missing.append("APPWRITE_PROJECT_ID")
    if not settings.appwrite_api_key:
        missing.append("APPWRITE_API_KEY")
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required environment variable(s): {names}")


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

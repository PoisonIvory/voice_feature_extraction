"""Small Appwrite adapter for storage and voice metadata reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from speech_feature_extraction.config import Settings


class AppwriteGateway:
    def __init__(self, settings: Settings) -> None:
        from appwrite.client import Client
        from appwrite.services.databases import Databases
        from appwrite.services.storage import Storage

        client = Client()
        client.set_endpoint(settings.appwrite_endpoint)
        client.set_project(settings.appwrite_project_id)
        client.set_key(settings.appwrite_api_key)

        self._settings = settings
        self._databases = Databases(client)
        self._storage = Storage(client)

    def list_audio_files(self) -> list[dict[str, Any]]:
        from appwrite.query import Query

        files: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            response = self._storage.list_files(
                bucket_id=self._settings.appwrite_audio_bucket_id,
                queries=[Query.limit(limit), Query.offset(offset)],
            )
            batch = response.get("files", [])
            files.extend(batch)
            if len(batch) < limit:
                return files
            offset += limit

    def list_voice_recordings(self) -> list[dict[str, Any]]:
        from appwrite.query import Query

        documents: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            response = self._databases.list_documents(
                database_id=self._settings.appwrite_database_id,
                collection_id=self._settings.appwrite_voice_recordings_collection_id,
                queries=[Query.limit(limit), Query.offset(offset)],
            )
            batch = response.get("documents", [])
            documents.extend(batch)
            if len(batch) < limit:
                return documents
            offset += limit

    def download_audio_file(self, file_id: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = self._storage.get_file_download(
            bucket_id=self._settings.appwrite_audio_bucket_id,
            file_id=file_id,
        )

        if isinstance(content, (bytes, bytearray)):
            data = bytes(content)
        else:
            data = getattr(content, "content", b"")

        destination.write_bytes(data)
        return destination

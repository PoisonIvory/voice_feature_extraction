"""Small Appwrite adapter for storage and voice metadata reads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from speech_feature_extraction.config import Settings

LOGGER = logging.getLogger(__name__)


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
            LOGGER.debug("Listing Appwrite storage files (offset=%d limit=%d)", offset, limit)
            response = self._storage.list_files(
                bucket_id=self._settings.appwrite_audio_bucket_id,
                queries=[Query.limit(limit), Query.offset(offset)],
            )
            batch = response.get("files", [])
            files.extend(batch)
            LOGGER.debug("Fetched storage file batch size=%d total=%d", len(batch), len(files))
            if len(batch) < limit:
                LOGGER.info("Finished listing storage files: %d", len(files))
                return files
            offset += limit

    def list_voice_recordings(self) -> list[dict[str, Any]]:
        from appwrite.query import Query

        documents: list[dict[str, Any]] = []
        offset = 0
        limit = 100

        while True:
            LOGGER.debug("Listing Appwrite voice recordings (offset=%d limit=%d)", offset, limit)
            response = self._databases.list_documents(
                database_id=self._settings.appwrite_database_id,
                collection_id=self._settings.appwrite_voice_recordings_collection_id,
                queries=[Query.limit(limit), Query.offset(offset)],
            )
            batch = response.get("documents", [])
            documents.extend(batch)
            LOGGER.debug("Fetched voice recording batch size=%d total=%d", len(batch), len(documents))
            if len(batch) < limit:
                LOGGER.info("Finished listing voice recordings: %d", len(documents))
                return documents
            offset += limit

    def download_audio_file(self, file_id: str, destination: Path) -> Path:
        LOGGER.debug("Downloading Appwrite audio file_id=%s", file_id)
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
        LOGGER.debug("Downloaded %d bytes to %s", len(data), destination)
        return destination

"""Command-line interface for the speech feature pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from speech_feature_extraction.config import load_settings, load_snapshot_publish_settings
from speech_feature_extraction.pipeline import run_audit, run_extract
from speech_feature_extraction.snapshot.contract_schema import (
    DEFAULT_AUDIT_FILENAME,
    DEFAULT_DAILY_FILENAME,
)
from speech_feature_extraction.snapshot.publisher import publish_snapshot_bundle

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract-speech-features",
        description="Audit Appwrite WAV recordings and extract openSMILE eGeMAPSv02 features.",
    )
    parser.add_argument("--env-file", help="Path to a dotenv file. Defaults to .env.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level. Defaults to INFO.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="List Appwrite WAVs and write the audit parquet.")
    audit_parser.add_argument("--user-id", help="Filter to a single user ID.")

    extract_parser = subparsers.add_parser(
        "extract",
        help="Download in-scope WAVs, extract features, and build daily task-separated output.",
    )
    extract_parser.add_argument("--user-id", help="Filter to a single user ID.")
    extract_parser.add_argument("--limit", type=int, help="Maximum number of pending recordings to process.")
    extract_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload WAV files even when they already exist in the local cache.",
    )

    publish_parser = subparsers.add_parser(
        "publish-snapshot",
        help="Build snapshot manifest and publish immutable artifact bundle.",
    )
    publish_parser.add_argument(
        "--daily-path",
        help="Path to daily canonical parquet. Defaults to data/processed canonical output.",
    )
    publish_parser.add_argument(
        "--audit-path",
        help="Path to audit parquet. Defaults to data/processed canonical output.",
    )
    publish_parser.add_argument(
        "--snapshot-root",
        help="Snapshot root directory. Defaults to SPEECH_SNAPSHOT_ROOT or exports/snapshots.",
    )
    publish_parser.add_argument("--snapshot-id", help="Snapshot ID, e.g. 2026-06-01.")
    publish_parser.add_argument(
        "--update-latest",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Update latest.json pointer for this dataset version.",
    )
    publish_parser.add_argument(
        "--pipeline-version",
        help="Override manifest provenance pipeline_version. Defaults to value inferred from parquet.",
    )
    publish_parser.add_argument(
        "--opensmile-version",
        help="Override manifest provenance opensmile_version. Defaults to value inferred from parquet.",
    )
    publish_parser.add_argument(
        "--source-commit",
        help="Override manifest provenance source_commit. Defaults to SPEECH_SNAPSHOT_SOURCE_COMMIT or git.",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)
    LOGGER.info("Starting command=%s", args.command)
    if args.command == "audit":
        settings = load_settings(args.env_file)
        audit_path = run_audit(settings, user_id=args.user_id)
        LOGGER.info("Audit finished: %s", audit_path)
        print(f"Wrote audit parquet: {audit_path}")
        return

    if args.command == "extract":
        settings = load_settings(args.env_file)
        daily_path, audit_path = run_extract(
            settings,
            limit=args.limit,
            force_download=args.force_download,
            user_id=args.user_id,
        )
        LOGGER.info("Extract finished: daily=%s audit=%s", daily_path, audit_path)
        print(f"Wrote daily parquet: {daily_path}")
        print(f"Wrote audit parquet: {audit_path}")
        return

    if args.command == "publish-snapshot":
        snapshot_settings = load_snapshot_publish_settings(args.env_file)
        daily_path = (
            Path(args.daily_path)
            if args.daily_path
            else snapshot_settings.processed_dir / DEFAULT_DAILY_FILENAME
        )
        audit_path = (
            Path(args.audit_path) if args.audit_path else snapshot_settings.processed_dir / DEFAULT_AUDIT_FILENAME
        )
        snapshot_root = Path(args.snapshot_root) if args.snapshot_root else snapshot_settings.snapshot_root
        snapshot_id = args.snapshot_id or snapshot_settings.snapshot_id
        update_latest = (
            snapshot_settings.snapshot_update_latest if args.update_latest is None else args.update_latest
        )
        pipeline_version = args.pipeline_version or snapshot_settings.snapshot_pipeline_version
        opensmile_version = args.opensmile_version or snapshot_settings.snapshot_opensmile_version
        source_commit = args.source_commit or snapshot_settings.snapshot_source_commit
        manifest_path = publish_snapshot_bundle(
            daily_path=daily_path,
            audit_path=audit_path,
            snapshot_root=snapshot_root,
            snapshot_id=snapshot_id,
            update_latest=update_latest,
            pipeline_version=pipeline_version,
            opensmile_version=opensmile_version,
            source_commit=source_commit,
            repo_dir=Path(__file__).resolve().parents[2],
        )
        LOGGER.info("Snapshot publish finished: %s", manifest_path)
        print(f"Wrote snapshot manifest: {manifest_path}")
        return

    parser.error(f"Unknown command: {args.command}")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

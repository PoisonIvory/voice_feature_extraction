"""Command-line interface for the speech feature pipeline."""

from __future__ import annotations

import argparse
import logging

from speech_feature_extraction.config import load_settings
from speech_feature_extraction.pipeline import run_audit, run_extract

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

    extract_parser = subparsers.add_parser("extract", help="Download in-scope WAVs and extract features.")
    extract_parser.add_argument("--user-id", help="Filter to a single user ID.")
    extract_parser.add_argument("--limit", type=int, help="Maximum number of pending recordings to process.")
    extract_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload WAV files even when they already exist in the local cache.",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)
    LOGGER.info("Starting command=%s", args.command)
    settings = load_settings(args.env_file)

    if args.command == "audit":
        audit_path = run_audit(settings, user_id=args.user_id)
        LOGGER.info("Audit finished: %s", audit_path)
        print(f"Wrote audit parquet: {audit_path}")
        return

    if args.command == "extract":
        recordings_path, audit_path = run_extract(
            settings,
            limit=args.limit,
            force_download=args.force_download,
            user_id=args.user_id,
        )
        LOGGER.info("Extract finished: recordings=%s audit=%s", recordings_path, audit_path)
        print(f"Wrote recordings parquet: {recordings_path}")
        print(f"Wrote audit parquet: {audit_path}")
        return

    parser.error(f"Unknown command: {args.command}")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

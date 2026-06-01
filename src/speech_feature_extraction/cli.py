"""Command-line interface for the speech feature pipeline."""

from __future__ import annotations

import argparse

from speech_feature_extraction.config import load_settings
from speech_feature_extraction.pipeline import run_audit, run_extract


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract-speech-features",
        description="Audit Appwrite WAV recordings and extract openSMILE eGeMAPSv02 features.",
    )
    parser.add_argument("--env-file", help="Path to a dotenv file. Defaults to .env.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("audit", help="List Appwrite WAVs and write the audit parquet.")

    extract_parser = subparsers.add_parser("extract", help="Download in-scope WAVs and extract features.")
    extract_parser.add_argument("--limit", type=int, help="Maximum number of pending recordings to process.")
    extract_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload WAV files even when they already exist in the local cache.",
    )

    args = parser.parse_args()
    settings = load_settings(args.env_file)

    if args.command == "audit":
        audit_path = run_audit(settings)
        print(f"Wrote audit parquet: {audit_path}")
        return

    if args.command == "extract":
        recordings_path, audit_path = run_extract(
            settings,
            limit=args.limit,
            force_download=args.force_download,
        )
        print(f"Wrote recordings parquet: {recordings_path}")
        print(f"Wrote audit parquet: {audit_path}")
        return

    parser.error(f"Unknown command: {args.command}")

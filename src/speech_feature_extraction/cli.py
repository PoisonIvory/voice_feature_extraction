"""Command-line interface for the speech feature pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from speech_feature_extraction.config import load_settings, load_snapshot_publish_settings
from speech_feature_extraction.constants import ENFORCED_USER_ID
from speech_feature_extraction.pipeline import run_audit, run_extract
from speech_feature_extraction.snapshot.contract_schema import (
    DEFAULT_AUDIT_FILENAME,
    DEFAULT_DAILY_FILENAME,
)
from speech_feature_extraction.snapshot.publisher import publish_snapshot_bundle

LOGGER = logging.getLogger(__name__)

DEFAULT_RAW_AUDIO_DIR = Path("data/raw_audio")
DEFAULT_EXPERIMENT_OUTPUT_DIR = Path("data/experimental/phoneme_prosody")


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
    audit_parser.add_argument(
        "--user-id",
        help=f"Required user ID. Must be {ENFORCED_USER_ID}.",
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Download in-scope WAVs, extract features, and build daily task-separated output.",
    )
    extract_parser.add_argument(
        "--user-id",
        help=f"Required user ID. Must be {ENFORCED_USER_ID}.",
    )
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

    phoneme_parser = subparsers.add_parser(
        "extract-phoneme-prosody",
        help="[EXPERIMENTAL] Run forced alignment on prosody recordings and extract phoneme-level features.",
    )
    phoneme_parser.add_argument(
        "--audio-dir",
        help=f"Directory containing prosody WAV files. Defaults to {DEFAULT_RAW_AUDIO_DIR}.",
    )
    phoneme_parser.add_argument(
        "--output-dir",
        help=f"Output directory for alignments and features. Defaults to {DEFAULT_EXPERIMENT_OUTPUT_DIR}.",
    )
    phoneme_parser.add_argument(
        "--recording-ids",
        nargs="*",
        help="Specific recording IDs to process. Defaults to all prosody recordings.",
    )
    phoneme_parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of recordings to process.",
    )
    phoneme_parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check MFA installation and model availability, do not run alignment.",
    )
    phoneme_parser.add_argument(
        "--transcription",
        help="Explicit transcript to use as primary alignment text for all recordings.",
    )
    phoneme_parser.add_argument(
        "--min-frames-for-variance",
        type=int,
        default=4,
        help="Minimum LLD frames required for qc_segment_ok (default: 4).",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)
    LOGGER.info("Starting command=%s", args.command)
    if args.command == "audit":
        settings = load_settings(args.env_file)
        scoped_user_id = _resolve_cli_user_id(parser, args.user_id)
        audit_path = run_audit(settings, user_id=scoped_user_id)
        LOGGER.info("Audit finished: %s", audit_path)
        print(f"Wrote audit parquet: {audit_path}")
        return

    if args.command == "extract":
        settings = load_settings(args.env_file)
        scoped_user_id = _resolve_cli_user_id(parser, args.user_id)
        daily_path, audit_path = run_extract(
            settings,
            limit=args.limit,
            force_download=args.force_download,
            user_id=scoped_user_id,
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

    if args.command == "extract-phoneme-prosody":
        _run_phoneme_prosody_extraction(args)
        return

    parser.error(f"Unknown command: {args.command}")


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _resolve_cli_user_id(parser: argparse.ArgumentParser, requested_user_id: str | None) -> str:
    if requested_user_id is None:
        return ENFORCED_USER_ID
    if requested_user_id != ENFORCED_USER_ID:
        parser.error(f"--user-id must be {ENFORCED_USER_ID}")
    return requested_user_id


def _run_phoneme_prosody_extraction(args: argparse.Namespace) -> None:
    """Run experimental phoneme prosody extraction."""
    from speech_feature_extraction.phoneme_prosody_experiment.alignment import (
        PROSODY_CANONICAL_TRANSCRIPTION,
        check_mfa_available,
        check_mfa_models_available,
    )
    from speech_feature_extraction.phoneme_prosody_experiment.biomarkers import (
        summarize_segment_qc_stats,
    )
    from speech_feature_extraction.phoneme_prosody_experiment.pipeline import (
        RecordingMetadata,
        process_batch,
    )
    from speech_feature_extraction.phoneme_prosody_experiment.segment_features import (
        SegmentFeatureExtractor,
    )

    audio_dir = Path(args.audio_dir) if args.audio_dir else DEFAULT_RAW_AUDIO_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_EXPERIMENT_OUTPUT_DIR

    print("[EXPERIMENTAL] Phoneme prosody extraction")
    print(f"Audio directory: {audio_dir}")
    print(f"Output directory: {output_dir}")
    print(
        f"Primary transcription: {(args.transcription or PROSODY_CANONICAL_TRANSCRIPTION)[:80]}"
        f"{'...' if len(args.transcription or PROSODY_CANONICAL_TRANSCRIPTION) > 80 else ''}"
    )
    print(f"Min frames for qc_segment_ok: {args.min_frames_for_variance}")
    print()

    mfa_available, mfa_version = check_mfa_available()
    if not mfa_available:
        print(f"ERROR: MFA not available: {mfa_version}")
        print()
        print("Install MFA with conda:")
        print("  conda install -c conda-forge montreal-forced-aligner")
        return

    print(f"MFA version: {mfa_version}")

    models_ok, models_msg = check_mfa_models_available()
    if not models_ok:
        print(f"ERROR: {models_msg}")
        print()
        print("Download required models:")
        print("  mfa model download acoustic english_us_arpa")
        print("  mfa model download dictionary english_us_arpa")
        return

    print("MFA models: OK")

    if args.check_only:
        print()
        print("Check complete. MFA is ready for alignment.")
        return

    if not audio_dir.exists():
        print(f"ERROR: Audio directory does not exist: {audio_dir}")
        return

    audit_path = Path("data/processed/voice_features_v4_audit.parquet")
    if not audit_path.exists():
        print(f"ERROR: Audit parquet not found: {audit_path}")
        print("The audit is the source of truth for task type. Run the 'audit' command")
        print("first so only prosody recordings (not vowel) are processed.")
        return

    import pandas as pd

    audit_df = pd.read_parquet(audit_path)
    prosody_ids = set(audit_df.loc[audit_df["taskType"] == "prosody", "recordingId"].tolist())
    prosody_completed_ids = set(
        audit_df.loc[
            (audit_df["taskType"] == "prosody") & (audit_df["pipelineStatus"] == "completed"),
            "recordingId",
        ].tolist()
    )
    print(f"Audit has {len(prosody_ids)} prosody recordings ({len(prosody_completed_ids)} completed)")

    if args.recording_ids:
        requested = set(args.recording_ids)
        recording_ids = requested & prosody_ids
        dropped = requested - prosody_ids
        if dropped:
            print(f"Skipping {len(dropped)} requested non-prosody recording(s): {sorted(dropped)}")
        print(f"Filtering to {len(recording_ids)} specified prosody recordings")
    else:
        recording_ids = prosody_completed_ids

    # Task-type guard: only prosody recordings from the audit are ever processed.
    wav_files = [f for f in audio_dir.glob("*.wav") if f.stem in recording_ids]

    if args.limit:
        wav_files = wav_files[: args.limit]

    print(f"Processing {len(wav_files)} recordings")
    print()

    recordings: list[RecordingMetadata] = []
    for wav_path in wav_files:
        recording_id = wav_path.stem
        user_id = "unknown"
        recorded_date = "unknown"

        row = audit_df[audit_df["recordingId"] == recording_id]
        if not row.empty:
            user_id = str(row.iloc[0].get("userId", "unknown"))
            rd = row.iloc[0].get("recordedDate")
            if rd is not None:
                recorded_date = str(rd)[:10] if hasattr(rd, "__str__") else str(rd)

        recordings.append(
            RecordingMetadata(
                recording_id=recording_id,
                user_id=user_id,
                recorded_date=recorded_date,
                task_type="prosody",
                audio_path=wav_path,
                transcription=args.transcription or PROSODY_CANONICAL_TRANSCRIPTION,
            )
        )

    feature_extractor = SegmentFeatureExtractor(min_frames_for_variance=args.min_frames_for_variance)
    parquet_path, success_count, failure_count = process_batch(
        recordings=recordings,
        output_dir=output_dir,
        feature_extractor=feature_extractor,
    )

    print()
    print(f"Extraction complete: {success_count} succeeded, {failure_count} failed")
    print(f"Feature parquet: {parquet_path}")
    print(f"TextGrid files: {output_dir / 'alignments'}")
    try:
        import pandas as pd

        qc_summary = summarize_segment_qc_stats(pd.read_parquet(parquet_path))
        print(
            "QC summary: "
            f"ok={qc_summary['qc_ok_rows']}/{qc_summary['total_rows']} "
            f"({qc_summary['qc_ok_ratio']:.1%}), "
            f"segment_too_short={qc_summary['segment_too_short_rows']}, "
            f"insufficient_frames={qc_summary['insufficient_frames_rows']}, "
            f"non_canonical_labels={qc_summary['non_canonical_label_rows']}, "
            f"median_frames={qc_summary['median_qc_num_frames']:.1f}, "
            f"median_min_required={qc_summary['median_qc_min_frames_required']:.1f}"
        )
    except Exception as error:
        LOGGER.warning("Unable to compute segment QC summary: %s", error)


if __name__ == "__main__":
    main()

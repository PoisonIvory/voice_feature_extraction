"""One-off utility to group raw WAV cache files by in-scope task type."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from speech_feature_extraction.constants import AUDIT_PARQUET, IN_SCOPE_TASK_TYPES

RAW_AUDIO_DIR = Path("data/raw_audio")
PROCESSED_DIR = Path("data/processed")
AUDIT_PATH = PROCESSED_DIR / AUDIT_PARQUET


def _build_task_mapping(audit_frame: pd.DataFrame) -> dict[str, str]:
    task_by_recording_id: dict[str, str] = {}
    for row in audit_frame.itertuples(index=False):
        recording_id = str(getattr(row, "recordingId", ""))
        task_type = str(getattr(row, "taskType", ""))
        if not recording_id or task_type not in IN_SCOPE_TASK_TYPES:
            continue
        task_by_recording_id[recording_id] = task_type
    return task_by_recording_id


def main() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(f"Audit parquet not found: {AUDIT_PATH}")
    if not RAW_AUDIO_DIR.exists():
        raise FileNotFoundError(f"Raw audio directory not found: {RAW_AUDIO_DIR}")

    audit_frame = pd.read_parquet(AUDIT_PATH)
    task_by_recording_id = _build_task_mapping(audit_frame)

    moved_counts = {task_type: 0 for task_type in sorted(IN_SCOPE_TASK_TYPES)}
    unknown_recording_ids: list[str] = []

    for wav_path in sorted(RAW_AUDIO_DIR.glob("*.wav")):
        task_type = task_by_recording_id.get(wav_path.stem)
        if task_type is None:
            unknown_recording_ids.append(wav_path.stem)
            continue

        destination_dir = RAW_AUDIO_DIR / task_type
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / wav_path.name
        shutil.move(str(wav_path), str(destination_path))
        moved_counts[task_type] += 1

    print("Split complete.")
    print(f"Moved counts: {moved_counts}")
    print(f"Unknown/unmapped files left in root: {len(unknown_recording_ids)}")
    if unknown_recording_ids:
        print("Unknown recording IDs:")
        for recording_id in unknown_recording_ids:
            print(f"- {recording_id}")


if __name__ == "__main__":
    main()

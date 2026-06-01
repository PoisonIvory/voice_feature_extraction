"""Daily task-separated aggregation for speech features."""

from __future__ import annotations

from typing import Any

import pandas as pd

TASK_TYPES = ("vowel", "prosody")


def build_daily_feature_rows(recording_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate recording-level rows into one row per user and UTC day.

    Expected input rows include:
    - userId
    - recordedAt (ISO-8601 timestamp)
    - taskType (vowel/prosody)
    - egemaps_* columns
    """
    if not recording_rows:
        return []

    frame = pd.DataFrame(recording_rows)
    if frame.empty:
        return []

    required_columns = {"userId", "recordedAt", "taskType"}
    missing = sorted(column for column in required_columns if column not in frame.columns)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"Missing required columns for daily aggregation: {names}")

    frame = frame.copy()
    frame["recordedAt"] = pd.to_datetime(frame["recordedAt"], utc=True, errors="coerce")
    frame = frame[frame["recordedAt"].notna()]
    if frame.empty:
        return []

    frame["dayUtc"] = frame["recordedAt"].dt.strftime("%Y-%m-%d")
    frame["taskType"] = frame["taskType"].astype(str).str.lower()
    frame = frame[frame["taskType"].isin(TASK_TYPES)]
    if frame.empty:
        return []

    feature_columns = sorted(column for column in frame.columns if str(column).startswith("egemaps_"))
    if not feature_columns:
        raise ValueError("Daily aggregation requires egemaps_ feature columns")

    grouped = (
        frame.groupby(["userId", "dayUtc", "taskType"], as_index=False)[feature_columns]
        .median(numeric_only=True)
        .sort_values(["userId", "dayUtc", "taskType"])
    )

    wide_rows: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in grouped.iterrows():
        key = (str(row["userId"]), str(row["dayUtc"]))
        task_type = str(row["taskType"])
        wide = wide_rows.setdefault(
            key,
            {
                "userId": key[0],
                "dayUtc": key[1],
                "vowel_recording_count": 0,
                "prosody_recording_count": 0,
            },
        )
        for feature_name in feature_columns:
            wide[f"{task_type}_{feature_name}"] = row[feature_name]

    counts = frame.groupby(["userId", "dayUtc", "taskType"], as_index=False).size()
    for _, row in counts.iterrows():
        key = (str(row["userId"]), str(row["dayUtc"]))
        task_type = str(row["taskType"])
        wide_rows[key][f"{task_type}_recording_count"] = int(row["size"])

    ids = (
        frame.groupby(["userId", "dayUtc", "taskType"], as_index=False)["recordingId"]
        .apply(lambda values: ";".join(sorted({str(value) for value in values if pd.notna(value)})))
        .rename(columns={"recordingId": "recordingIds"})
    )
    for _, row in ids.iterrows():
        key = (str(row["userId"]), str(row["dayUtc"]))
        task_type = str(row["taskType"])
        wide_rows[key][f"{task_type}_recording_ids"] = row["recordingIds"]

    for metadata_column in ("extractorVersion", "featureSet", "featureLevel", "libraryVersion"):
        if metadata_column not in frame.columns:
            continue
        metadata_values = (
            frame.groupby(["userId", "dayUtc"], as_index=False)[metadata_column]
            .apply(lambda values: next((value for value in values if pd.notna(value)), None))
            .rename(columns={metadata_column: "metadataValue"})
        )
        for _, row in metadata_values.iterrows():
            key = (str(row["userId"]), str(row["dayUtc"]))
            wide_rows[key][metadata_column] = row["metadataValue"]

    ordered_rows = [wide_rows[key] for key in sorted(wide_rows)]
    for row in ordered_rows:
        for task_type in TASK_TYPES:
            for feature_name in feature_columns:
                prefixed_name = f"{task_type}_{feature_name}"
                if prefixed_name not in row:
                    row[prefixed_name] = None
        row["has_vowel"] = row.get("vowel_recording_count", 0) > 0
        row["has_prosody"] = row.get("prosody_recording_count", 0) > 0
        row["is_day_complete"] = row["has_vowel"] and row["has_prosody"]

    return ordered_rows

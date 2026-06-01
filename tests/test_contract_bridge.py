import pandas as pd

from speech_feature_extraction.snapshot.contract_schema import (
    REQUIRED_FEATURE_COUNT_BY_PREFIX,
    validate_required_columns,
)


def test_extraction_shaped_daily_row_satisfies_contract() -> None:
    vowel_count = REQUIRED_FEATURE_COUNT_BY_PREFIX["vowel_egemaps_"]
    prosody_count = REQUIRED_FEATURE_COUNT_BY_PREFIX["prosody_egemaps_"]
    row: dict[str, object] = {
        "userId": "u1",
        "dayUtc": "2026-06-01",
        "vowel_recording_count": 2,
        "prosody_recording_count": 1,
        "has_vowel": True,
        "has_prosody": True,
        "is_day_complete": True,
        "extractorVersion": "v4.0-daily-task-separated-median",
        "libraryVersion": "2.5.1",
    }
    for index in range(vowel_count):
        row[f"vowel_egemaps_feature_{index}"] = float(index)
    for index in range(prosody_count):
        row[f"prosody_egemaps_feature_{index}"] = float(index)

    frame = pd.DataFrame([row])
    validate_required_columns(frame)


from speech_feature_extraction.daily_aggregation import build_daily_feature_rows


def test_build_daily_feature_rows_aggregates_median_per_task() -> None:
    rows = [
        {
            "recordingId": "r1",
            "userId": "u1",
            "recordedAt": "2026-06-01T10:00:00+00:00",
            "taskType": "vowel",
            "egemaps_a": 1.0,
            "egemaps_b": 10.0,
        },
        {
            "recordingId": "r2",
            "userId": "u1",
            "recordedAt": "2026-06-01T12:00:00+00:00",
            "taskType": "vowel",
            "egemaps_a": 3.0,
            "egemaps_b": 30.0,
        },
        {
            "recordingId": "r3",
            "userId": "u1",
            "recordedAt": "2026-06-01T14:00:00+00:00",
            "taskType": "prosody",
            "egemaps_a": 8.0,
            "egemaps_b": 80.0,
        },
    ]

    daily = build_daily_feature_rows(rows)

    assert len(daily) == 1
    row = daily[0]
    assert row["userId"] == "u1"
    assert row["dayUtc"] == "2026-06-01"
    assert row["vowel_egemaps_a"] == 2.0
    assert row["vowel_egemaps_b"] == 20.0
    assert row["prosody_egemaps_a"] == 8.0
    assert row["prosody_egemaps_b"] == 80.0
    assert row["vowel_recording_count"] == 2
    assert row["prosody_recording_count"] == 1
    assert row["has_vowel"] is True
    assert row["has_prosody"] is True
    assert row["is_day_complete"] is True


def test_build_daily_feature_rows_keeps_incomplete_day_with_missing_task_nulls() -> None:
    rows = [
        {
            "recordingId": "r1",
            "userId": "u1",
            "recordedAt": "2026-06-01T23:59:59+00:00",
            "taskType": "vowel",
            "egemaps_a": 5.0,
        }
    ]

    daily = build_daily_feature_rows(rows)

    assert len(daily) == 1
    row = daily[0]
    assert row["dayUtc"] == "2026-06-01"
    assert row["vowel_egemaps_a"] == 5.0
    assert row["prosody_egemaps_a"] is None
    assert row["vowel_recording_count"] == 1
    assert row["prosody_recording_count"] == 0
    assert row["has_vowel"] is True
    assert row["has_prosody"] is False
    assert row["is_day_complete"] is False


def test_build_daily_feature_rows_uses_utc_day_from_recorded_at() -> None:
    rows = [
        {
            "recordingId": "r1",
            "userId": "u1",
            "recordedAt": "2026-06-01T23:30:00-07:00",
            "taskType": "vowel",
            "egemaps_a": 1.0,
        }
    ]

    daily = build_daily_feature_rows(rows)

    assert len(daily) == 1
    assert daily[0]["dayUtc"] == "2026-06-02"

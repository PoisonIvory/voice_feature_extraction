import pandas as pd

from speech_feature_extraction.snapshot.contract_schema import (
    REQUIRED_FEATURE_COUNT_BY_PREFIX,
    validate_required_columns,
)


def test_extraction_shaped_recordings_row_satisfies_contract() -> None:
    egemaps_count = REQUIRED_FEATURE_COUNT_BY_PREFIX["egemaps_"]
    row: dict[str, object] = {
        "recordingId": "r1",
        "recordedDate": "2026-06-01",
        "taskType": "vowel",
        "pipelineStatus": "completed",
        "extractorVersion": "v3.2-opensmile-egemaps-taskqc",
        "extractionTimestamp": "2026-06-01T15:16:12+00:00",
        "featureSet": "opensmile.FeatureSet.eGeMAPSv02",
        "featureLevel": "opensmile.FeatureLevel.Functionals",
        "libraryName": "opensmile",
        "libraryVersion": "2.5.1",
        "audioHash": "sha256:abc",
        "qc_task_qc_passed": True,
        "qc_warning_codes": [],
        "qc_opensmile_egemaps_success": True,
        "qc_feature_count_egemaps": egemaps_count,
        "qc_feature_count_egemaps_expected": egemaps_count,
    }
    for index in range(egemaps_count):
        row[f"egemaps_feature_{index}"] = float(index)

    frame = pd.DataFrame([row])
    validate_required_columns(frame)


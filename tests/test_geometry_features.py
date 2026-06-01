from speech_feature_extraction.geometry_features import compute_geometry_derived_features


def test_compute_geometry_derived_features_returns_expected_values() -> None:
    features = {
        "F1frequency_sma3nz_amean": 500.0,
        "F2frequency_sma3nz_amean": 1500.0,
        "F3frequency_sma3nz_amean": 2500.0,
    }

    derived = compute_geometry_derived_features(features)

    assert derived["egemaps_geom_f1_f2_delta_hz_amean"] == 1000.0
    assert derived["egemaps_geom_f2_f3_delta_hz_amean"] == 1000.0
    assert derived["egemaps_geom_f1_f3_delta_hz_amean"] == 2000.0
    assert derived["egemaps_geom_f2_f1_ratio_amean"] == 3.0
    assert derived["egemaps_geom_f3_f2_ratio_amean"] == 2500.0 / 1500.0
    assert derived["egemaps_geom_f3_f1_ratio_amean"] == 5.0
    assert derived["egemaps_geom_formant_spacing_hz_amean"] == 1000.0
    assert derived["egemaps_geom_apparent_vtl_cm_amean"] == 17.5


def test_compute_geometry_derived_features_handles_missing_formants() -> None:
    features = {
        "F1frequency_sma3nz_amean": 500.0,
        "F2frequency_sma3nz_amean": None,
        "F3frequency_sma3nz_amean": 2500.0,
    }

    derived = compute_geometry_derived_features(features)

    assert derived["egemaps_geom_f1_f2_delta_hz_amean"] is None
    assert derived["egemaps_geom_f2_f3_delta_hz_amean"] is None
    assert derived["egemaps_geom_formant_spacing_hz_amean"] is None
    assert derived["egemaps_geom_apparent_vtl_cm_amean"] is None

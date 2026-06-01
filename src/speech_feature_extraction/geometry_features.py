"""Derived vocal-tract geometry proxies from eGeMAPS formant outputs."""

from __future__ import annotations

from typing import Any, Mapping

from speech_feature_extraction.constants import OPENSMILE_GEOMETRY_DERIVED_PREFIX

# Speed of sound approximation used for simple VTL proxy calculation.
# Formula: VTL ~= c / (2 * dF), where dF is mean formant spacing.
_SPEED_OF_SOUND_CM_PER_SEC = 35000.0


def compute_geometry_derived_features(egemaps_features: Mapping[str, Any]) -> dict[str, float | None]:
    """Compute optional geometry-derived features from eGeMAPS formants.

    Uses only existing eGeMAPSv02 functionals (F1/F2/F3 means), so no
    additional extractor is required.
    """
    f1 = _to_positive_float(egemaps_features.get("F1frequency_sma3nz_amean"))
    f2 = _to_positive_float(egemaps_features.get("F2frequency_sma3nz_amean"))
    f3 = _to_positive_float(egemaps_features.get("F3frequency_sma3nz_amean"))

    d12 = _delta(f2, f1)
    d23 = _delta(f3, f2)
    d13 = _delta(f3, f1)
    spacing = _mean_of_positive(d12, d23)

    vtl_cm = None
    if spacing is not None and spacing > 0:
        vtl_cm = _SPEED_OF_SOUND_CM_PER_SEC / (2.0 * spacing)

    values = {
        "f1_f2_delta_hz_amean": d12,
        "f2_f3_delta_hz_amean": d23,
        "f1_f3_delta_hz_amean": d13,
        "f2_f1_ratio_amean": _ratio(f2, f1),
        "f3_f2_ratio_amean": _ratio(f3, f2),
        "f3_f1_ratio_amean": _ratio(f3, f1),
        "formant_spacing_hz_amean": spacing,
        "apparent_vtl_cm_amean": vtl_cm,
    }
    return {f"{OPENSMILE_GEOMETRY_DERIVED_PREFIX}{key}": value for key, value in values.items()}


def _to_positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    if numeric != numeric:
        return None
    return numeric


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    delta = a - b
    if delta <= 0:
        return None
    return delta


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b <= 0:
        return None
    ratio = a / b
    if ratio <= 0:
        return None
    return ratio


def _mean_of_positive(a: float | None, b: float | None) -> float | None:
    values = [value for value in (a, b) if value is not None and value > 0]
    if not values:
        return None
    return sum(values) / len(values)

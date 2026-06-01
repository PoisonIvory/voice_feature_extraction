"""openSMILE eGeMAPSv02 feature extraction."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from speech_feature_extraction.constants import (
    OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT,
    OPENSMILE_EGEMAPS_PREFIX,
)


class OpenSmileEgemapsExtractor:
    """Thin wrapper around the official opensmile Python package."""

    def __init__(self) -> None:
        import opensmile

        self._opensmile = opensmile
        self._feature_set_name = "opensmile.FeatureSet.eGeMAPSv02"
        self._feature_level_name = "opensmile.FeatureLevel.Functionals"
        self._sampling_rate_hz = 16000
        self._resample = True
        self._channels = 0
        self._mixdown = False
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            sampling_rate=self._sampling_rate_hz,
            resample=self._resample,
            channels=self._channels,
            mixdown=self._mixdown,
        )

    @property
    def library_version(self) -> str:
        try:
            return version("opensmile")
        except PackageNotFoundError:
            return "unknown"

    @property
    def feature_names(self) -> list[str]:
        return list(self._smile.feature_names)

    @property
    def extraction_metadata(self) -> dict[str, Any]:
        return {
            "featureSet": self._feature_set_name,
            "featureLevel": self._feature_level_name,
            "libraryName": "opensmile",
            "libraryVersion": self.library_version,
            "opensmileConfigName": self._smile.config_name,
            "opensmileConfigFile": Path(self._smile.config_path).name,
            "opensmileSamplingRateHz": self._sampling_rate_hz,
            "opensmileResampleEnabled": self._resample,
            "opensmileChannels": self._channels,
            "opensmileMixdownEnabled": self._mixdown,
        }

    def extract_file(self, path: Path) -> dict[str, Any]:
        frame = self._smile.process_file(str(path))
        if frame.empty:
            raise ValueError(f"openSMILE returned no rows for {path}")
        if len(frame) != 1:
            raise ValueError(f"openSMILE returned {len(frame)} rows for {path}; expected 1 row")

        row = frame.iloc[0]
        if row.isna().any():
            raise ValueError(f"openSMILE returned NaN features for {path}")

        expected_names = self.feature_names
        returned_names = list(row.index)
        if returned_names != expected_names:
            raise ValueError(f"openSMILE feature schema mismatch for {path}")
        if len(expected_names) != OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"openSMILE runtime feature count {len(expected_names)} does not match expected "
                f"{OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT}"
            )

        features = row.to_dict()
        prefixed = {
            f"{OPENSMILE_EGEMAPS_PREFIX}{name}": value
            for name, value in features.items()
        }
        prefixed["qc_opensmile_egemaps_success"] = True
        prefixed["qc_feature_count_egemaps"] = len(features)
        prefixed["qc_feature_count_egemaps_expected"] = OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT
        return prefixed

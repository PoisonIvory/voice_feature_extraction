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
        self._smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
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

    def extract_file(self, path: Path) -> dict[str, Any]:
        frame = self._smile.process_file(str(path))
        if frame.empty:
            raise ValueError(f"openSMILE returned no rows for {path}")

        features = frame.iloc[0].to_dict()
        prefixed = {
            f"{OPENSMILE_EGEMAPS_PREFIX}{name}": value
            for name, value in features.items()
        }
        prefixed["qc_opensmile_egemaps_success"] = True
        prefixed["qc_feature_count_egemaps"] = len(features)
        prefixed["qc_feature_count_egemaps_expected"] = OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT
        return prefixed

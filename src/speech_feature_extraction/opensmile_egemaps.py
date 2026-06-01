"""openSMILE eGeMAPSv02 feature extraction.

This module provides both Functionals (88 features per file) and Low-Level
Descriptors (LLDs, frame-by-frame features) for quality control purposes.

The LLD extraction enables computation of:
- Voiced frame ratio (for task-specific QC)
- F0 coefficient of variation (for vowel stability assessment)
- Frame-level jitter/shimmer (where 'nz' suffix means non-zero/voiced only)
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from speech_feature_extraction.constants import (
    OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT,
    OPENSMILE_EGEMAPS_PREFIX,
)
from speech_feature_extraction.geometry_features import compute_geometry_derived_features


@dataclass
class LldQcMetrics:
    """Quality metrics derived from Low-Level Descriptor analysis."""

    voiced_ratio: float
    f0_cov: float | None
    jitter_mean: float | None
    shimmer_db_mean: float | None
    total_frames: int
    voiced_frames: int


class OpenSmileEgemapsExtractor:
    """Wrapper around the official opensmile Python package.

    Provides both Functionals extraction (for analysis) and LLD extraction
    (for task-specific quality control).
    """

    def __init__(self, include_geometry_derived: bool = False) -> None:
        import opensmile

        self._opensmile = opensmile
        self._feature_set_name = "opensmile.FeatureSet.eGeMAPSv02"
        self._feature_level_name = "opensmile.FeatureLevel.Functionals"
        self._include_geometry_derived = include_geometry_derived
        self._sampling_rate_hz = 16000
        self._resample = True
        self._channels = 0
        self._mixdown = False

        self._smile_functionals = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
            sampling_rate=self._sampling_rate_hz,
            resample=self._resample,
            channels=self._channels,
            mixdown=self._mixdown,
        )

        self._smile_lld = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            sampling_rate=self._sampling_rate_hz,
            resample=self._resample,
            channels=self._channels,
            mixdown=self._mixdown,
        )

    @property
    def _smile(self) -> Any:
        """Backward compatibility alias for functionals extractor."""
        return self._smile_functionals

    @property
    def library_version(self) -> str:
        try:
            return version("opensmile")
        except PackageNotFoundError:
            return "unknown"

    @property
    def feature_names(self) -> list[str]:
        return list(self._smile_functionals.feature_names)

    @property
    def lld_feature_names(self) -> list[str]:
        return list(self._smile_lld.feature_names)

    @property
    def extraction_metadata(self) -> dict[str, Any]:
        return {
            "featureSet": self._feature_set_name,
            "featureLevel": self._feature_level_name,
            "libraryName": "opensmile",
            "libraryVersion": self.library_version,
            "opensmileConfigName": self._smile_functionals.config_name,
            "opensmileConfigFile": Path(self._smile_functionals.config_path).name,
            "opensmileSamplingRateHz": self._sampling_rate_hz,
            "opensmileResampleEnabled": self._resample,
            "opensmileChannels": self._channels,
            "opensmileMixdownEnabled": self._mixdown,
            "opensmileGeometryDerivedEnabled": self._include_geometry_derived,
        }

    def extract_lld_qc_metrics(self, path: Path) -> LldQcMetrics:
        """Extract Low-Level Descriptors and compute QC metrics.

        This method extracts frame-by-frame features and computes:
        - Voiced ratio: proportion of frames where F0 was detected
        - F0 CoV: coefficient of variation of F0 (std/mean) for voiced frames
        - Jitter mean: average jitter across voiced frames
        - Shimmer dB mean: average shimmer in dB across voiced frames

        The 'nz' suffix in eGeMAPS LLD names (e.g., F0semitoneFrom27.5Hz_sma3nz)
        indicates "non-zero" values are only computed for voiced frames.

        Args:
            path: Path to the audio file

        Returns:
            LldQcMetrics with voiced ratio and perturbation metrics
        """
        lld_frame = self._smile_lld.process_file(str(path))

        if lld_frame.empty:
            return LldQcMetrics(
                voiced_ratio=0.0,
                f0_cov=None,
                jitter_mean=None,
                shimmer_db_mean=None,
                total_frames=0,
                voiced_frames=0,
            )

        total_frames = len(lld_frame)

        f0_col = "F0semitoneFrom27.5Hz_sma3nz"
        jitter_col = "jitterLocal_sma3nz"
        shimmer_col = "shimmerLocaldB_sma3nz"

        f0_values = lld_frame[f0_col] if f0_col in lld_frame.columns else None
        jitter_values = lld_frame[jitter_col] if jitter_col in lld_frame.columns else None
        shimmer_values = lld_frame[shimmer_col] if shimmer_col in lld_frame.columns else None

        if f0_values is None:
            return LldQcMetrics(
                voiced_ratio=0.0,
                f0_cov=None,
                jitter_mean=None,
                shimmer_db_mean=None,
                total_frames=total_frames,
                voiced_frames=0,
            )

        voiced_mask = f0_values > 0
        voiced_frames = int(voiced_mask.sum())
        voiced_ratio = voiced_frames / total_frames if total_frames > 0 else 0.0

        f0_cov: float | None = None
        if voiced_frames >= 10:
            voiced_f0 = f0_values[voiced_mask]
            mean_f0 = voiced_f0.mean()
            if mean_f0 > 0:
                f0_cov = float(voiced_f0.std() / mean_f0)

        jitter_mean: float | None = None
        if jitter_values is not None and voiced_frames > 0:
            voiced_jitter = jitter_values[voiced_mask]
            valid_jitter = voiced_jitter[voiced_jitter > 0]
            if len(valid_jitter) > 0:
                jitter_mean = float(valid_jitter.mean() * 100)

        shimmer_db_mean: float | None = None
        if shimmer_values is not None and voiced_frames > 0:
            voiced_shimmer = shimmer_values[voiced_mask]
            valid_shimmer = voiced_shimmer[~voiced_shimmer.isna()]
            if len(valid_shimmer) > 0:
                shimmer_db_mean = float(valid_shimmer.mean())

        return LldQcMetrics(
            voiced_ratio=voiced_ratio,
            f0_cov=f0_cov,
            jitter_mean=jitter_mean,
            shimmer_db_mean=shimmer_db_mean,
            total_frames=total_frames,
            voiced_frames=voiced_frames,
        )

    def extract_file(self, path: Path) -> dict[str, Any]:
        """Extract eGeMAPSv02 Functionals features from an audio file.

        Args:
            path: Path to the audio file

        Returns:
            Dictionary with prefixed feature names and QC metadata
        """
        frame = self._smile_functionals.process_file(str(path))
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
        if getattr(self, "_include_geometry_derived", False):
            geometry_features = compute_geometry_derived_features(features)
            prefixed.update(geometry_features)
            prefixed["qc_feature_block_geometry_derived_enabled"] = True
            prefixed["qc_feature_count_geometry_derived"] = len(geometry_features)
        prefixed["qc_opensmile_egemaps_success"] = True
        prefixed["qc_feature_count_egemaps"] = len(features)
        prefixed["qc_feature_count_egemaps_expected"] = OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT
        return prefixed

    def extract_file_with_qc(self, path: Path) -> tuple[dict[str, Any], LldQcMetrics]:
        """Extract both Functionals and LLD-based QC metrics.

        This is the recommended method for production use, as it provides
        both the analysis features and the metrics needed for task-specific
        quality control in a single call.

        Args:
            path: Path to the audio file

        Returns:
            Tuple of (features_dict, lld_qc_metrics)
        """
        features = self.extract_file(path)
        lld_qc = self.extract_lld_qc_metrics(path)
        return features, lld_qc

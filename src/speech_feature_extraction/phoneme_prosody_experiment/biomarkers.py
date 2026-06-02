"""Longitudinal biomarker computation for phoneme prosody experiment.

This module handles:
1. Grouping phoneme features by user + phoneme + date for longitudinal tracking
2. Computing nasal vs oral contrast biomarkers
3. Tracking per-phoneme and per-class trajectories across days
4. Generating count-aware reliability flags for sparse data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    PHONEME_CLASS_NASAL_COUPLED,
    PHONEME_CLASS_ORAL_ANTERIOR,
    PHONEME_CLASS_PHARYNGEAL_ENGAGED,
    PHONEME_CLASS_VOICELESS_FRICATION,
)

MIN_OBSERVATIONS_FOR_CONTRAST = 3
MIN_OBSERVATIONS_FOR_RELIABLE_MEAN = 5


@dataclass(frozen=True)
class DailyPhonemeAggregate:
    """Aggregated features for one phoneme on one day."""

    user_id: str
    recorded_date: str
    phoneme_label: str
    phoneme_class_primary: str
    mfcc2_mean: float | None
    h1h2_mean: float | None
    f1_bandwidth_mean: float | None
    f0_mean: float | None
    observation_count: int
    qc_reliable: bool
    qc_reason: str


@dataclass(frozen=True)
class DailyClassAggregate:
    """Aggregated features for one phoneme class on one day."""

    user_id: str
    recorded_date: str
    phoneme_class: str
    mfcc2_mean: float | None
    h1h2_mean: float | None
    f1_bandwidth_mean: float | None
    f0_mean: float | None
    phoneme_count: int
    observation_count: int
    qc_reliable: bool


@dataclass(frozen=True)
class NasalOralContrast:
    """Daily nasal vs oral biomarker contrast."""

    user_id: str
    recorded_date: str
    nasal_mfcc2_mean: float | None
    oral_mfcc2_mean: float | None
    mfcc2_contrast: float | None
    nasal_h1h2_mean: float | None
    oral_h1h2_mean: float | None
    h1h2_contrast: float | None
    nasal_f1bw_mean: float | None
    pharyngeal_f1bw_mean: float | None
    f1bw_contrast: float | None
    nasal_observation_count: int
    oral_observation_count: int
    qc_reliable: bool
    qc_reason: str


def compute_daily_phoneme_aggregates(
    features_df: "pd.DataFrame",
) -> list[DailyPhonemeAggregate]:
    """Aggregate segment features by user + phoneme + date.

    Args:
        features_df: DataFrame with segment-level features.
            Required columns: userId, recordedDate, phonemeLabel,
            phonemeClassPrimary, segment_mfcc2_mean, segment_h1h2_mean,
            segment_f1_bandwidth_mean, qc_segment_ok

    Returns:
        List of DailyPhonemeAggregate objects.
    """
    required_cols = [
        "userId",
        "recordedDate",
        "phonemeLabel",
        "phonemeClassPrimary",
        "segment_mfcc2_mean",
        "qc_segment_ok",
    ]
    for col in required_cols:
        if col not in features_df.columns:
            raise ValueError(f"Missing required column: {col}")

    valid_df = features_df[features_df["qc_segment_ok"] == True].copy()

    aggregates: list[DailyPhonemeAggregate] = []
    groupby_cols = ["userId", "recordedDate", "phonemeLabel", "phonemeClassPrimary"]

    for (user_id, date, phoneme, pclass), group in valid_df.groupby(groupby_cols):
        count = len(group)
        reliable = count >= MIN_OBSERVATIONS_FOR_RELIABLE_MEAN
        reason = "ok" if reliable else f"low_count:{count}"

        aggregates.append(
            DailyPhonemeAggregate(
                user_id=str(user_id),
                recorded_date=str(date),
                phoneme_label=str(phoneme),
                phoneme_class_primary=str(pclass),
                mfcc2_mean=_safe_mean(group.get("segment_mfcc2_mean")),
                h1h2_mean=_safe_mean(group.get("segment_h1h2_mean")),
                f1_bandwidth_mean=_safe_mean(group.get("segment_f1_bandwidth_mean")),
                f0_mean=_safe_mean(group.get("segment_f0_mean")),
                observation_count=count,
                qc_reliable=reliable,
                qc_reason=reason,
            )
        )

    return aggregates


def compute_daily_class_aggregates(
    features_df: "pd.DataFrame",
) -> list[DailyClassAggregate]:
    """Aggregate segment features by user + class + date.

    Args:
        features_df: DataFrame with segment-level features.

    Returns:
        List of DailyClassAggregate objects for each class on each day.
    """
    valid_df = features_df[features_df["qc_segment_ok"] == True].copy()

    aggregates: list[DailyClassAggregate] = []
    groupby_cols = ["userId", "recordedDate", "phonemeClassPrimary"]

    for (user_id, date, pclass), group in valid_df.groupby(groupby_cols):
        count = len(group)
        phoneme_count = group["phonemeLabel"].nunique()
        reliable = count >= MIN_OBSERVATIONS_FOR_RELIABLE_MEAN

        aggregates.append(
            DailyClassAggregate(
                user_id=str(user_id),
                recorded_date=str(date),
                phoneme_class=str(pclass),
                mfcc2_mean=_safe_mean(group.get("segment_mfcc2_mean")),
                h1h2_mean=_safe_mean(group.get("segment_h1h2_mean")),
                f1_bandwidth_mean=_safe_mean(group.get("segment_f1_bandwidth_mean")),
                f0_mean=_safe_mean(group.get("segment_f0_mean")),
                phoneme_count=phoneme_count,
                observation_count=count,
                qc_reliable=reliable,
            )
        )

    return aggregates


def compute_daily_nasal_oral_contrasts(
    features_df: "pd.DataFrame",
) -> list[NasalOralContrast]:
    """Compute daily nasal vs oral biomarker contrasts.

    Primary biomarkers per plan:
    - nasal_mean_mfcc2 - oral_mean_mfcc2
    - nasal_mean_H1H2 - oral_mean_H1H2
    - nasal_mean_F1bw and pharyngeal_mean_F1bw trajectories

    Args:
        features_df: DataFrame with segment-level features.

    Returns:
        List of NasalOralContrast objects, one per user-day.
    """
    valid_df = features_df[features_df["qc_segment_ok"] == True].copy()

    contrasts: list[NasalOralContrast] = []

    for (user_id, date), day_group in valid_df.groupby(["userId", "recordedDate"]):
        nasal_mask = day_group["phonemeClassPrimary"] == PHONEME_CLASS_NASAL_COUPLED
        oral_mask = day_group["phonemeClassPrimary"] == PHONEME_CLASS_ORAL_ANTERIOR
        pharyngeal_mask = day_group["phonemeClassPrimary"] == PHONEME_CLASS_PHARYNGEAL_ENGAGED

        nasal_df = day_group[nasal_mask]
        oral_df = day_group[oral_mask]
        pharyngeal_df = day_group[pharyngeal_mask]

        nasal_count = len(nasal_df)
        oral_count = len(oral_df)

        nasal_mfcc2 = _safe_mean(nasal_df.get("segment_mfcc2_mean"))
        oral_mfcc2 = _safe_mean(oral_df.get("segment_mfcc2_mean"))
        mfcc2_contrast = _safe_subtract(nasal_mfcc2, oral_mfcc2)

        nasal_h1h2 = _safe_mean(nasal_df.get("segment_h1h2_mean"))
        oral_h1h2 = _safe_mean(oral_df.get("segment_h1h2_mean"))
        h1h2_contrast = _safe_subtract(nasal_h1h2, oral_h1h2)

        nasal_f1bw = _safe_mean(nasal_df.get("segment_f1_bandwidth_mean"))
        pharyngeal_f1bw = _safe_mean(pharyngeal_df.get("segment_f1_bandwidth_mean"))
        f1bw_contrast = _safe_subtract(nasal_f1bw, pharyngeal_f1bw)

        reliable = (
            nasal_count >= MIN_OBSERVATIONS_FOR_CONTRAST
            and oral_count >= MIN_OBSERVATIONS_FOR_CONTRAST
        )
        reasons: list[str] = []
        if nasal_count < MIN_OBSERVATIONS_FOR_CONTRAST:
            reasons.append(f"low_nasal:{nasal_count}")
        if oral_count < MIN_OBSERVATIONS_FOR_CONTRAST:
            reasons.append(f"low_oral:{oral_count}")
        reason = ",".join(reasons) if reasons else "ok"

        contrasts.append(
            NasalOralContrast(
                user_id=str(user_id),
                recorded_date=str(date),
                nasal_mfcc2_mean=nasal_mfcc2,
                oral_mfcc2_mean=oral_mfcc2,
                mfcc2_contrast=mfcc2_contrast,
                nasal_h1h2_mean=nasal_h1h2,
                oral_h1h2_mean=oral_h1h2,
                h1h2_contrast=h1h2_contrast,
                nasal_f1bw_mean=nasal_f1bw,
                pharyngeal_f1bw_mean=pharyngeal_f1bw,
                f1bw_contrast=f1bw_contrast,
                nasal_observation_count=nasal_count,
                oral_observation_count=oral_count,
                qc_reliable=reliable,
                qc_reason=reason,
            )
        )

    return contrasts


@dataclass(frozen=True)
class PhonemeTrajectory:
    """Trajectory of one phoneme across days for one user."""

    user_id: str
    phoneme_label: str
    phoneme_class_primary: str
    dates: tuple[str, ...]
    mfcc2_values: tuple[float | None, ...]
    h1h2_values: tuple[float | None, ...]
    f1bw_values: tuple[float | None, ...]
    observation_counts: tuple[int, ...]
    total_days: int
    reliable_days: int


def compute_phoneme_trajectories(
    aggregates: list[DailyPhonemeAggregate],
) -> list[PhonemeTrajectory]:
    """Build longitudinal trajectories for each phoneme per user.

    Args:
        aggregates: List of daily phoneme aggregates.

    Returns:
        List of PhonemeTrajectory objects, one per user-phoneme pair.
    """
    from collections import defaultdict

    grouped: dict[tuple[str, str], list[DailyPhonemeAggregate]] = defaultdict(list)
    for agg in aggregates:
        key = (agg.user_id, agg.phoneme_label)
        grouped[key].append(agg)

    trajectories: list[PhonemeTrajectory] = []
    for (user_id, phoneme), day_aggs in grouped.items():
        sorted_aggs = sorted(day_aggs, key=lambda a: a.recorded_date)

        dates = tuple(a.recorded_date for a in sorted_aggs)
        mfcc2_values = tuple(a.mfcc2_mean for a in sorted_aggs)
        h1h2_values = tuple(a.h1h2_mean for a in sorted_aggs)
        f1bw_values = tuple(a.f1_bandwidth_mean for a in sorted_aggs)
        counts = tuple(a.observation_count for a in sorted_aggs)
        reliable_days = sum(1 for a in sorted_aggs if a.qc_reliable)

        pclass = sorted_aggs[0].phoneme_class_primary if sorted_aggs else "unknown"

        trajectories.append(
            PhonemeTrajectory(
                user_id=user_id,
                phoneme_label=phoneme,
                phoneme_class_primary=pclass,
                dates=dates,
                mfcc2_values=mfcc2_values,
                h1h2_values=h1h2_values,
                f1bw_values=f1bw_values,
                observation_counts=counts,
                total_days=len(dates),
                reliable_days=reliable_days,
            )
        )

    return trajectories


def summarize_phoneme_occurrence_stats(
    features_df: "pd.DataFrame",
) -> "pd.DataFrame":
    """Generate per-phoneme occurrence summaries for reliability assessment.

    Flags rare phones that should be pooled at the class level.

    Args:
        features_df: DataFrame with segment-level features.

    Returns:
        DataFrame with phoneme occurrence statistics.
    """
    import pandas as pd

    summary = features_df.groupby(["phonemeLabel", "phonemeClassPrimary"]).agg(
        total_observations=("phonemeLabel", "count"),
        recording_count=("recordingId", "nunique"),
        day_count=("recordedDate", "nunique"),
        qc_ok_count=("qc_segment_ok", "sum"),
    ).reset_index()

    summary["qc_ok_ratio"] = summary["qc_ok_count"] / summary["total_observations"]
    summary["is_sparse"] = summary["total_observations"] < MIN_OBSERVATIONS_FOR_RELIABLE_MEAN
    summary["recommend_pool_to_class"] = summary["is_sparse"]

    return summary


def _safe_mean(series: "pd.Series | None") -> float | None:
    """Compute mean, handling None and empty series."""
    if series is None or len(series) == 0:
        return None
    values = series.dropna()
    if len(values) == 0:
        return None
    result = float(values.mean())
    return result if np.isfinite(result) else None


def _safe_subtract(a: float | None, b: float | None) -> float | None:
    """Subtract b from a, returning None if either is None."""
    if a is None or b is None:
        return None
    result = a - b
    return result if np.isfinite(result) else None

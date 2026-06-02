"""Longitudinal biomarker computation for phoneme prosody experiment.

This module handles:
1. Grouping phoneme features by user + phoneme + date for longitudinal tracking
2. Tracking per-phoneme trajectories across days
3. Generating count-aware reliability flags for sparse data
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

MIN_OBSERVATIONS_FOR_RELIABLE_MEAN = 5


@dataclass(frozen=True)
class DailyPhonemeAggregate:
    """Aggregated features for one phoneme on one day."""

    user_id: str
    recorded_date: str
    phoneme_label: str
    mfcc2_mean: float | None
    h1h2_mean: float | None
    f1_bandwidth_mean: float | None
    f0_mean: float | None
    observation_count: int
    qc_reliable: bool
    qc_reason: str


@dataclass(frozen=True)
class PhonemeTrajectory:
    """Trajectory of one phoneme across days for one user."""

    user_id: str
    phoneme_label: str
    dates: tuple[str, ...]
    mfcc2_values: tuple[float | None, ...]
    h1h2_values: tuple[float | None, ...]
    f1bw_values: tuple[float | None, ...]
    observation_counts: tuple[int, ...]
    total_days: int
    reliable_days: int


def compute_daily_phoneme_aggregates(
    features_df: "pd.DataFrame",
) -> list[DailyPhonemeAggregate]:
    """Aggregate segment features by user + phoneme + date.

    Args:
        features_df: DataFrame with segment-level features.
            Required columns: userId, recordedDate, phonemeLabel,
            segment_mfcc2_mean, qc_segment_ok

    Returns:
        List of DailyPhonemeAggregate objects.
    """
    required_cols = [
        "userId",
        "recordedDate",
        "phonemeLabel",
        "segment_mfcc2_mean",
        "qc_segment_ok",
    ]
    for col in required_cols:
        if col not in features_df.columns:
            raise ValueError(f"Missing required column: {col}")

    valid_df = features_df[features_df["qc_segment_ok"] == True].copy()

    aggregates: list[DailyPhonemeAggregate] = []
    groupby_cols = ["userId", "recordedDate", "phonemeLabel"]

    for (user_id, date, phoneme), group in valid_df.groupby(groupby_cols):
        count = len(group)
        reliable = count >= MIN_OBSERVATIONS_FOR_RELIABLE_MEAN
        reason = "ok" if reliable else f"low_count:{count}"

        aggregates.append(
            DailyPhonemeAggregate(
                user_id=str(user_id),
                recorded_date=str(date),
                phoneme_label=str(phoneme),
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

        trajectories.append(
            PhonemeTrajectory(
                user_id=user_id,
                phoneme_label=phoneme,
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

    Args:
        features_df: DataFrame with segment-level features.

    Returns:
        DataFrame with phoneme occurrence statistics.
    """
    import pandas as pd

    summary = features_df.groupby("phonemeLabel").agg(
        total_observations=("phonemeLabel", "count"),
        recording_count=("recordingId", "nunique"),
        day_count=("recordedDate", "nunique"),
        qc_ok_count=("qc_segment_ok", "sum"),
    ).reset_index()

    summary["qc_ok_ratio"] = summary["qc_ok_count"] / summary["total_observations"]
    summary["is_sparse"] = summary["total_observations"] < MIN_OBSERVATIONS_FOR_RELIABLE_MEAN

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

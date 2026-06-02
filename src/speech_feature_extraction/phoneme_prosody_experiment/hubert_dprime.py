"""Phonological-subspace d-prime from phone-level HuBERT embeddings.

Single responsibility: turn the per-phone embedding table into one d-prime per
phonological contrast per recording (stages 3-4 of Muller et al. 2026).

Stage 3 - feature directions
    For each binary contrast [+f] vs [-f], the direction is the difference of
    the mean embeddings of the two phone categories. In the paper this is
    estimated from healthy controls (a group disjoint from the speakers being
    scored). This within-speaker longitudinal design has no separate control
    group, so by default the direction for each recording is estimated
    leave-one-recording-out (LORO): the difference of category means computed
    from every *other* qc-passing recording. LORO keeps the direction
    independent of the recording it scores, which removes the in-sample
    inflation that a single pooled "grand-mean" direction would introduce
    (estimating and scoring on the same tokens gives a non-zero d-prime even for
    random embeddings). A pooled ``grand_mean`` mode is available for
    inspection; the difference between the two is itself a useful diagnostic.

Stage 4 - projection + d-prime
    Each recording's phone embeddings are projected onto the contrast direction
    and category separation is quantified with d', the signal-detection
    sensitivity index, d' = |mu_pos - mu_neg| / sqrt((var_pos + var_neg) / 2).
    A contrast needs at least ``MIN_TOKENS_PER_CLASS`` tokens in each pole for a
    valid estimate; otherwise it is missing (NaN) for that recording.

Pure NumPy so the math is unit-testable without torch or real audio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.hubert_phone_config import (
    MIN_TOKENS_PER_CLASS,
    PHONOLOGICAL_CONTRASTS,
    PhonologicalContrast,
)

EMBEDDING_COLUMN_PREFIX = "hubert_"


def embedding_columns(dim: int) -> list[str]:
    """Ordered embedding column names ``hubert_0 .. hubert_{dim-1}``."""
    return [f"{EMBEDDING_COLUMN_PREFIX}{i}" for i in range(dim)]


def detect_embedding_columns(df: pd.DataFrame) -> list[str]:
    """Return the embedding columns present, ordered by their numeric index."""
    cols = [c for c in df.columns if c.startswith(EMBEDDING_COLUMN_PREFIX)]
    return sorted(cols, key=lambda c: int(c.removeprefix(EMBEDDING_COLUMN_PREFIX)))


def feature_direction(
    positive: np.ndarray, negative: np.ndarray
) -> np.ndarray | None:
    """Unit contrast direction = mean(positive) - mean(negative).

    Returns ``None`` if either category is empty or the difference is a zero
    vector. The result is L2-normalised for numerical cleanliness; d-prime is
    invariant to the direction's scale.
    """
    if positive.shape[0] == 0 or negative.shape[0] == 0:
        return None
    direction = positive.mean(axis=0) - negative.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return direction / norm


def dprime(positive_projection: np.ndarray, negative_projection: np.ndarray) -> float:
    """Signal-detection d' between two 1-D projected score distributions."""
    if positive_projection.size < 2 or negative_projection.size < 2:
        return float("nan")
    mean_gap = abs(float(positive_projection.mean()) - float(negative_projection.mean()))
    pooled_var = 0.5 * (
        float(np.var(positive_projection, ddof=1))
        + float(np.var(negative_projection, ddof=1))
    )
    if pooled_var <= 0.0 or not np.isfinite(pooled_var):
        return float("nan")
    return mean_gap / np.sqrt(pooled_var)


def estimate_directions(
    embeddings: np.ndarray,
    phone_labels: np.ndarray,
    contrasts: tuple[PhonologicalContrast, ...] = PHONOLOGICAL_CONTRASTS,
) -> dict[str, np.ndarray]:
    """Estimate one grand-mean direction per contrast from pooled embeddings."""
    directions: dict[str, np.ndarray] = {}
    for contrast in contrasts:
        pos = embeddings[np.isin(phone_labels, list(contrast.positive_phones))]
        neg = embeddings[np.isin(phone_labels, list(contrast.negative_phones))]
        direction = feature_direction(pos, neg)
        if direction is not None:
            directions[contrast.key] = direction
    return directions


GRAND_MEAN = "grand_mean"
LEAVE_ONE_RECORDING_OUT = "leave_one_recording_out"


class _ContrastAccumulator:
    """Per-recording pole sums/counts enabling O(1) leave-one-recording-out means."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.pos_sum: dict[str, np.ndarray] = {}
        self.neg_sum: dict[str, np.ndarray] = {}
        self.pos_count: dict[str, int] = {}
        self.neg_count: dict[str, int] = {}
        self.total_pos_sum = np.zeros(dim)
        self.total_neg_sum = np.zeros(dim)
        self.total_pos_count = 0
        self.total_neg_count = 0

    def add(self, recording_id: str, pos: np.ndarray, neg: np.ndarray) -> None:
        self.pos_sum[recording_id] = pos.sum(axis=0) if pos.shape[0] else np.zeros(self.dim)
        self.neg_sum[recording_id] = neg.sum(axis=0) if neg.shape[0] else np.zeros(self.dim)
        self.pos_count[recording_id] = int(pos.shape[0])
        self.neg_count[recording_id] = int(neg.shape[0])
        self.total_pos_sum += self.pos_sum[recording_id]
        self.total_neg_sum += self.neg_sum[recording_id]
        self.total_pos_count += self.pos_count[recording_id]
        self.total_neg_count += self.neg_count[recording_id]

    def grand_direction(self) -> np.ndarray | None:
        if self.total_pos_count == 0 or self.total_neg_count == 0:
            return None
        return _normalize(
            self.total_pos_sum / self.total_pos_count
            - self.total_neg_sum / self.total_neg_count
        )

    def loro_direction(self, recording_id: str) -> np.ndarray | None:
        pos_count = self.total_pos_count - self.pos_count[recording_id]
        neg_count = self.total_neg_count - self.neg_count[recording_id]
        if pos_count == 0 or neg_count == 0:
            return None
        return _normalize(
            (self.total_pos_sum - self.pos_sum[recording_id]) / pos_count
            - (self.total_neg_sum - self.neg_sum[recording_id]) / neg_count
        )


def _normalize(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return None
    return vector / norm


def compute_dprime_table(
    embeddings_df: pd.DataFrame,
    contrasts: tuple[PhonologicalContrast, ...] = PHONOLOGICAL_CONTRASTS,
    min_tokens_per_class: int = MIN_TOKENS_PER_CLASS,
    direction_estimation: str = LEAVE_ONE_RECORDING_OUT,
    recording_meta_columns: tuple[str, ...] = ("recordedDate", "userId"),
) -> pd.DataFrame:
    """One row per recording: d-prime per contrast plus per-pole token counts.

    ``direction_estimation`` selects how each contrast direction is estimated:
    ``leave_one_recording_out`` (default, recommended) computes the direction
    from every other recording so it is independent of the scored recording;
    ``grand_mean`` uses one pooled direction for all recordings. Recordings with
    fewer than ``min_tokens_per_class`` tokens in either pole of a contrast
    receive NaN for that contrast.
    """
    if direction_estimation not in (GRAND_MEAN, LEAVE_ONE_RECORDING_OUT):
        raise ValueError(f"Unknown direction_estimation: {direction_estimation!r}")

    emb_cols = detect_embedding_columns(embeddings_df)
    if not emb_cols:
        raise ValueError("No hubert_* embedding columns found in embeddings_df")
    dim = len(emb_cols)

    grouped = {rid: g for rid, g in embeddings_df.groupby("recordingId", sort=True)}
    group_embeddings = {rid: g[emb_cols].to_numpy(dtype=np.float64) for rid, g in grouped.items()}
    group_labels = {rid: g["phonemeLabel"].astype(str).to_numpy() for rid, g in grouped.items()}
    group_masks = {
        rid: {
            contrast.key: (
                np.isin(group_labels[rid], list(contrast.positive_phones)),
                np.isin(group_labels[rid], list(contrast.negative_phones)),
            )
            for contrast in contrasts
        }
        for rid in grouped
    }

    accumulators = {contrast.key: _ContrastAccumulator(dim) for contrast in contrasts}
    for rid in grouped:
        emb = group_embeddings[rid]
        for contrast in contrasts:
            pos_mask, neg_mask = group_masks[rid][contrast.key]
            accumulators[contrast.key].add(rid, emb[pos_mask], emb[neg_mask])

    rows: list[dict[str, object]] = []
    for rid, group in grouped.items():
        emb = group_embeddings[rid]
        row: dict[str, object] = {"recordingId": rid}
        for meta_col in recording_meta_columns:
            if meta_col in group.columns:
                row[meta_col] = group[meta_col].iloc[0]
        row["n_phones_total"] = int(len(group))

        for contrast in contrasts:
            pos_mask, neg_mask = group_masks[rid][contrast.key]
            n_pos = int(np.sum(pos_mask))
            n_neg = int(np.sum(neg_mask))
            row[contrast.n_positive_field] = n_pos
            row[contrast.n_negative_field] = n_neg

            acc = accumulators[contrast.key]
            direction = (
                acc.grand_direction()
                if direction_estimation == GRAND_MEAN
                else acc.loro_direction(rid)
            )
            if (
                direction is None
                or n_pos < min_tokens_per_class
                or n_neg < min_tokens_per_class
            ):
                row[contrast.dprime_field] = float("nan")
                continue

            pos_proj = emb[pos_mask] @ direction
            neg_proj = emb[neg_mask] @ direction
            row[contrast.dprime_field] = dprime(pos_proj, neg_proj)

        rows.append(row)

    return pd.DataFrame(rows)

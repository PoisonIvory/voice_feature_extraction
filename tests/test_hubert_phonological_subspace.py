"""Tests for the HuBERT phonological-subspace d-prime experiment.

These cover the torch-free parts: contrast config invariants, the d-prime math,
feature-direction / direction-estimation logic, and the frame timing + pooling
helpers. The torch model path itself is exercised by the CLI, not unit tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.hubert_extract import (
    frame_center_times,
    pool_phone_embedding,
)
from speech_feature_extraction.phoneme_prosody_experiment.hubert_phone_config import (
    MIN_TOKENS_PER_CLASS,
    PHONOLOGICAL_CONTRASTS,
    PhonologicalContrast,
    contrast_by_key,
)
from speech_feature_extraction.phoneme_prosody_experiment.hubert_dprime import (
    GRAND_MEAN,
    LEAVE_ONE_RECORDING_OUT,
    compute_dprime_table,
    detect_embedding_columns,
    dprime,
    embedding_columns,
    feature_direction,
)
from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import (
    ARPABET_PHONEMES,
)


# --- config -------------------------------------------------------------------

def test_nine_segmental_contrasts_five_consonant_four_vowel() -> None:
    assert len(PHONOLOGICAL_CONTRASTS) == 9
    families = [c.feature_family for c in PHONOLOGICAL_CONTRASTS]
    assert families.count("consonant") == 5
    assert families.count("vowel") == 4


def test_contrast_poles_are_disjoint_and_canonical_arpabet() -> None:
    for contrast in PHONOLOGICAL_CONTRASTS:
        assert not (contrast.positive_phones & contrast.negative_phones)
        assert contrast.positive_phones and contrast.negative_phones
        assert (contrast.positive_phones | contrast.negative_phones) <= ARPABET_PHONEMES


def test_contrast_keys_unique_and_field_names() -> None:
    keys = [c.key for c in PHONOLOGICAL_CONTRASTS]
    assert len(keys) == len(set(keys))
    nasality = contrast_by_key("nasality")
    assert nasality.dprime_field == "dprime_nasality"
    assert nasality.n_positive_field == "n_nasality_pos"
    assert nasality.n_negative_field == "n_nasality_neg"


def test_nasality_is_nasal_vs_oral_stop() -> None:
    nasality = contrast_by_key("nasality")
    assert nasality.positive_phones == {"M", "N", "NG"}
    assert nasality.negative_phones == {"P", "B", "T", "D", "K", "G"}


# --- d-prime math -------------------------------------------------------------

def test_dprime_large_for_separated_distributions() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(5.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert dprime(a, b) > 4.0


def test_dprime_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 1000)
    b = rng.normal(0.0, 1.0, 1000)
    assert abs(dprime(a, b)) < 0.3


def test_dprime_is_symmetric_in_magnitude() -> None:
    a = np.array([2.0, 3.0, 4.0, 5.0])
    b = np.array([0.0, 1.0, 2.0, 3.0])
    assert dprime(a, b) == dprime(b, a)


def test_dprime_nan_when_too_few_tokens() -> None:
    assert np.isnan(dprime(np.array([1.0]), np.array([0.0, 1.0, 2.0])))


def test_dprime_nan_when_zero_variance() -> None:
    assert np.isnan(dprime(np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])))


# --- feature direction --------------------------------------------------------

def test_feature_direction_points_from_negative_to_positive_and_unit_norm() -> None:
    pos = np.array([[2.0, 0.0], [4.0, 0.0]])
    neg = np.array([[-2.0, 0.0], [-4.0, 0.0]])
    direction = feature_direction(pos, neg)
    assert direction is not None
    np.testing.assert_allclose(direction, np.array([1.0, 0.0]), atol=1e-9)


def test_feature_direction_none_for_empty_or_zero_difference() -> None:
    assert feature_direction(np.empty((0, 3)), np.ones((2, 3))) is None
    identical = np.ones((3, 4))
    assert feature_direction(identical, identical) is None


# --- direction estimation tables ----------------------------------------------

def _synthetic_embeddings(
    rng: np.random.Generator,
    dim: int = 16,
    n_recordings: int = 8,
    plant_contrast: PhonologicalContrast | None = None,
    plant_magnitude: float = 8.0,
) -> pd.DataFrame:
    """Build a phone-embedding frame with controlled phone counts per recording."""
    nasality = contrast_by_key("nasality")
    pos = sorted(nasality.positive_phones)
    neg = sorted(nasality.negative_phones)
    # 6 tokens per pole per recording -> above the 5-token minimum.
    labels = (pos * 2)[:6] + (neg + neg)[:6]
    frames = []
    for r in range(n_recordings):
        rid = f"rec_{r:02d}"
        emb = rng.normal(0.0, 1.0, (len(labels), dim))
        block = pd.DataFrame(emb, columns=embedding_columns(dim))
        block.insert(0, "phonemeLabel", labels)
        block.insert(0, "recordingId", rid)
        block["recordedDate"] = "2026-01-01"
        if plant_contrast is not None:
            is_pos = block["phonemeLabel"].isin(plant_contrast.positive_phones).to_numpy()
            block.loc[is_pos, "hubert_0"] = block.loc[is_pos, "hubert_0"] + plant_magnitude
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def test_compute_dprime_table_one_row_per_recording_with_token_counts() -> None:
    rng = np.random.default_rng(2)
    df = _synthetic_embeddings(rng, n_recordings=8)
    table = compute_dprime_table(df)
    assert len(table) == 8
    assert set(table["recordingId"]) == {f"rec_{r:02d}" for r in range(8)}
    nasality = contrast_by_key("nasality")
    assert (table[nasality.n_positive_field] == 6).all()
    assert (table[nasality.n_negative_field] == 6).all()
    assert nasality.dprime_field in table.columns


def test_planted_contrast_yields_high_dprime() -> None:
    rng = np.random.default_rng(3)
    nasality = contrast_by_key("nasality")
    df = _synthetic_embeddings(rng, plant_contrast=nasality, plant_magnitude=10.0)
    table = compute_dprime_table(df, direction_estimation=LEAVE_ONE_RECORDING_OUT)
    assert table[nasality.dprime_field].median() > 4.0


def test_leave_one_recording_out_reduces_in_sample_inflation() -> None:
    rng = np.random.default_rng(4)
    df = _synthetic_embeddings(rng, dim=64, n_recordings=10)  # pure noise
    nasality = contrast_by_key("nasality")
    grand = compute_dprime_table(df, direction_estimation=GRAND_MEAN)
    loro = compute_dprime_table(df, direction_estimation=LEAVE_ONE_RECORDING_OUT)
    assert loro[nasality.dprime_field].median() < grand[nasality.dprime_field].median()


def test_contrast_below_min_tokens_is_missing() -> None:
    rng = np.random.default_rng(5)
    df = _synthetic_embeddings(rng, n_recordings=6)
    # stridency phones are absent from the synthetic labels -> 0 tokens -> NaN.
    stridency = contrast_by_key("stridency")
    table = compute_dprime_table(df)
    assert table[stridency.dprime_field].isna().all()
    assert (table[stridency.n_positive_field] == 0).all()


def test_min_tokens_constant_is_five() -> None:
    assert MIN_TOKENS_PER_CLASS == 5


def test_detect_embedding_columns_orders_numerically() -> None:
    df = pd.DataFrame(
        {"hubert_0": [1.0], "hubert_10": [1.0], "hubert_2": [1.0], "recordingId": ["r"]}
    )
    assert detect_embedding_columns(df) == ["hubert_0", "hubert_2", "hubert_10"]


# --- frame timing + pooling ---------------------------------------------------

def test_frame_center_times_are_evenly_spaced_within_clip() -> None:
    centers = frame_center_times(num_frames=5, audio_duration_sec=0.1)
    np.testing.assert_allclose(centers, np.array([0.01, 0.03, 0.05, 0.07, 0.09]))


def test_frame_center_times_empty_for_no_frames() -> None:
    assert frame_center_times(0, 1.0).size == 0


def test_pool_phone_embedding_means_frames_in_window() -> None:
    emb = np.array([[0.0], [2.0], [4.0], [6.0]])
    centers = np.array([0.01, 0.03, 0.05, 0.07])
    vector, n = pool_phone_embedding(emb, centers, start_sec=0.02, end_sec=0.06)
    assert n == 2
    np.testing.assert_allclose(vector, np.array([3.0]))  # mean of frames at 0.03, 0.05


def test_pool_phone_embedding_falls_back_to_nearest_frame() -> None:
    emb = np.array([[10.0], [20.0], [30.0]])
    centers = np.array([0.01, 0.03, 0.05])
    # Window shorter than the frame stride with no centre inside.
    vector, n = pool_phone_embedding(emb, centers, start_sec=0.0315, end_sec=0.0325)
    assert n == 1
    np.testing.assert_allclose(vector, np.array([20.0]))


def test_pool_phone_embedding_none_when_no_frames() -> None:
    vector, n = pool_phone_embedding(np.empty((0, 1)), np.empty(0), 0.0, 1.0)
    assert vector is None
    assert n == 0

"""Orchestration for the HuBERT phonological-subspace experiment.

Single responsibility: drive the two data-prep layers end to end and persist
them, reusing the existing MFA phone boundaries from the eGeMAPS phoneme parquet.

Layer 1  hubert_phone_embeddings.parquet      one mean-pooled 768-d vector / phone
Layer 2  hubert_dprime_by_recording.parquet   one d-prime / contrast / recording

This stays strictly data-prep: no cycle-phase join, statistics, or plotting
(those live in the Analysis project, mirroring how the eGeMAPS phoneme parquet
is joined to the cycle calendar there). Embeddings are written after each
recording so an interrupted run resumes from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.hubert_dprime import (
    compute_dprime_table,
    embedding_columns,
)
from speech_feature_extraction.phoneme_prosody_experiment.hubert_extract import (
    HubertFeatureExtractor,
)
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    HUBERT_DPRIME_FILENAME,
    HUBERT_PHONE_EMBEDDINGS_FILENAME,
    PHONEME_PROSODY_FEATURES_FILENAME,
)

LOGGER = logging.getLogger(__name__)

HUBERT_EXTRACTOR_VERSION = "hubert_phone_v1"

_PHONE_META_COLUMNS = (
    "recordingId",
    "userId",
    "recordedDate",
    "phonemeIndex",
    "phonemeLabel",
    "startSec",
    "endSec",
)


def load_aligned_phonemes(phoneme_parquet: Path) -> pd.DataFrame:
    """Load qc-passing phone rows (boundaries + labels) from the eGeMAPS parquet."""
    df = pd.read_parquet(phoneme_parquet)
    df = df[df["qc_recording_ok"] == True]  # noqa: E712
    keep = [c for c in _PHONE_META_COLUMNS if c in df.columns]
    return df[keep].sort_values(["recordingId", "phonemeIndex"]).reset_index(drop=True)


def _find_wav(audio_dir: Path, recording_id: str) -> Path | None:
    matches = list(audio_dir.rglob(f"{recording_id}.wav"))
    return matches[0] if matches else None


def _embedding_rows_for_recording(
    recording_id: str,
    phones: pd.DataFrame,
    wav_path: Path,
    extractor: HubertFeatureExtractor,
) -> pd.DataFrame | None:
    """Extract one mean-pooled embedding per phone for a single recording."""
    waveform = extractor.load_waveform(wav_path)
    frame_embeddings, frame_centers = extractor.extract_frame_embeddings(waveform)
    if frame_embeddings.shape[0] == 0:
        LOGGER.warning("No HuBERT frames for %s; skipping", recording_id)
        return None

    emb_cols = embedding_columns(extractor.embedding_dim)
    vectors: list[np.ndarray] = []
    pooled_counts: list[int] = []
    for phone in phones.itertuples(index=False):
        pooled = extractor.pool_phone(
            frame_embeddings, frame_centers, float(phone.startSec), float(phone.endSec)
        )
        if pooled is None:
            return None
        vectors.append(pooled.vector)
        pooled_counts.append(pooled.n_frames_pooled)

    emb_frame = pd.DataFrame(np.vstack(vectors).astype(np.float32), columns=emb_cols)
    meta_frame = phones.reset_index(drop=True).copy()
    meta_frame["modelName"] = extractor.model_name
    meta_frame["hubertLayer"] = extractor.layer_name
    meta_frame["embeddingDim"] = extractor.embedding_dim
    meta_frame["extractorVersion"] = HUBERT_EXTRACTOR_VERSION
    meta_frame["nFramesPooled"] = pooled_counts
    return pd.concat([meta_frame, emb_frame], axis=1)


def build_phone_embeddings(
    phoneme_df: pd.DataFrame,
    audio_dir: Path,
    output_path: Path,
    extractor: HubertFeatureExtractor,
    force: bool = False,
) -> Path:
    """Build hubert_phone_embeddings.parquet (resumable, atomic per recording)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    base_frames: list[pd.DataFrame] = []
    if output_path.exists() and not force:
        existing = pd.read_parquet(output_path)
        if not existing.empty:
            done_ids = set(existing["recordingId"].unique())
            base_frames = [existing]
            LOGGER.info("Resuming: %d recordings already embedded", len(done_ids))

    recording_ids = [rid for rid in phoneme_df["recordingId"].unique() if rid not in done_ids]
    LOGGER.info("Embedding %d recordings", len(recording_ids))

    processed: list[pd.DataFrame] = []
    for index, recording_id in enumerate(recording_ids, start=1):
        wav_path = _find_wav(audio_dir, recording_id)
        if wav_path is None:
            LOGGER.warning("No WAV for %s under %s; skipping", recording_id, audio_dir)
            continue
        phones = phoneme_df[phoneme_df["recordingId"] == recording_id]
        LOGGER.info("[%d/%d] %s (%d phones)", index, len(recording_ids), recording_id, len(phones))
        try:
            rec_frame = _embedding_rows_for_recording(recording_id, phones, wav_path, extractor)
        except Exception:  # noqa: BLE001 - one bad recording must not abort the run
            LOGGER.exception("Failed to embed %s", recording_id)
            continue
        if rec_frame is None:
            continue
        processed.append(rec_frame)
        _atomic_write_parquet(pd.concat(base_frames + processed, ignore_index=True), output_path)

    if not (base_frames or processed):
        LOGGER.warning("No embeddings produced; writing empty contract file")
        if not output_path.exists():
            pd.DataFrame(columns=list(_PHONE_META_COLUMNS)).to_parquet(output_path, index=False)
    LOGGER.info("Embeddings written: %s", output_path)
    return output_path


def build_dprime_table(embeddings_path: Path, output_path: Path) -> Path:
    """Build hubert_dprime_by_recording.parquet from the embeddings parquet."""
    embeddings_df = pd.read_parquet(embeddings_path)
    if embeddings_df.empty:
        raise ValueError(f"Embeddings parquet is empty: {embeddings_path}")
    table = compute_dprime_table(embeddings_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(table, output_path)
    LOGGER.info("d-prime table written: %s (%d recordings)", output_path, len(table))
    _log_dprime_coverage(table)
    return output_path


def _log_dprime_coverage(table: pd.DataFrame) -> None:
    """Log per-contrast coverage and median d-prime as a sanity summary."""
    for col in (c for c in table.columns if c.startswith("dprime_")):
        coverage = float(table[col].notna().mean()) if len(table) else 0.0
        median = float(table[col].median(skipna=True)) if len(table) else float("nan")
        LOGGER.info("  %-22s coverage=%4.0f%% median_dprime=%.3f", col, coverage * 100, median)


def run_hubert_pipeline(
    output_dir: Path,
    audio_dir: Path,
    phoneme_parquet: Path | None = None,
    model_name: str = "facebook/hubert-base-ls960",
    layer: int | None = None,
    device: str | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Run both layers; returns (embeddings_path, dprime_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    phoneme_parquet = phoneme_parquet or output_dir / PHONEME_PROSODY_FEATURES_FILENAME
    phoneme_df = load_aligned_phonemes(phoneme_parquet)
    LOGGER.info(
        "Loaded %d qc-passing phones across %d recordings",
        len(phoneme_df),
        phoneme_df["recordingId"].nunique(),
    )

    extractor = HubertFeatureExtractor(model_name=model_name, layer=layer, device=device)
    embeddings_path = build_phone_embeddings(
        phoneme_df=phoneme_df,
        audio_dir=audio_dir,
        output_path=output_dir / HUBERT_PHONE_EMBEDDINGS_FILENAME,
        extractor=extractor,
        force=force,
    )
    dprime_path = build_dprime_table(
        embeddings_path=embeddings_path,
        output_path=output_dir / HUBERT_DPRIME_FILENAME,
    )
    return embeddings_path, dprime_path


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write to a temp file then rename so an interrupt cannot corrupt output."""
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)

"""Frozen HuBERT phone-level embedding extraction.

Single responsibility: turn one WAV plus its MFA phone boundaries into one
mean-pooled embedding vector per phone, following stage 2 of the
phonological-subspace method (Muller et al. 2026): "frame-level embeddings are
extracted from frozen HuBERT-base and averaged over each phone interval to
produce phone-level mean embeddings."

The model is run once per recording (HuBERT needs surrounding context, exactly
like the openSMILE full-recording extraction in ``segment_features.py``); each
phone then claims the frames whose centre falls in its interval. The timing and
pooling helpers are pure NumPy so they can be unit-tested without torch.

torch / torchaudio / transformers are imported lazily so the rest of the
experiment package imports without the optional ``hubert`` extra installed.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL_NAME = "facebook/hubert-base-ls960"
TARGET_SAMPLE_RATE_HZ = 16000
# HuBERT-base downsamples 16 kHz audio by a factor of 320 (conv stride product),
# i.e. one frame per 20 ms (~50 Hz). Frame centres are derived from the actual
# returned frame count and audio duration so they stay calibrated for any clip.
HUBERT_FRAME_STRIDE_SEC = 0.020


def frame_center_times(num_frames: int, audio_duration_sec: float) -> np.ndarray:
    """Centre time (seconds) of each HuBERT frame.

    Frame centres are spread evenly across the clip from the actual frame count,
    which self-calibrates to HuBERT's ~20 ms stride without drifting over long
    recordings.
    """
    if num_frames <= 0 or audio_duration_sec <= 0:
        return np.empty(0, dtype=float)
    step = audio_duration_sec / num_frames
    return (np.arange(num_frames, dtype=float) + 0.5) * step


def pool_phone_embedding(
    frame_embeddings: np.ndarray,
    frame_centers: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray | None, int]:
    """Mean-pool the frames whose centre falls in ``[start_sec, end_sec)``.

    If no frame centre lands inside a (very short) interval, the single frame
    whose centre is nearest the interval midpoint is used so every aligned phone
    still yields an embedding. Returns ``(vector, n_frames_pooled)``; the vector
    is ``None`` only when there are no frames at all.
    """
    if frame_embeddings.shape[0] == 0:
        return None, 0

    in_window = (frame_centers >= start_sec) & (frame_centers < end_sec)
    n_pooled = int(np.sum(in_window))
    if n_pooled > 0:
        return frame_embeddings[in_window].mean(axis=0), n_pooled

    midpoint = (start_sec + end_sec) / 2.0
    nearest = int(np.argmin(np.abs(frame_centers - midpoint)))
    return frame_embeddings[nearest], 1


@dataclass(frozen=True)
class PhoneEmbedding:
    """One mean-pooled phone embedding with its pooling QC count."""

    vector: np.ndarray
    n_frames_pooled: int


class HubertFeatureExtractor:
    """Wrap a frozen HuBERT model for phone-level embedding extraction.

    Args:
        model_name: HuggingFace checkpoint (default ``facebook/hubert-base-ls960``).
        layer: hidden layer to read. ``None`` (default) uses the final hidden
            layer (``last_hidden_state``), consistent with the paper; an int
            selects ``hidden_states[layer]`` for layer-sensitivity checks.
        device: torch device string. Defaults to ``cuda`` if available else cpu.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        layer: int | None = None,
        device: str | None = None,
    ) -> None:
        try:
            self._torch = importlib.import_module("torch")
            self._torchaudio = importlib.import_module("torchaudio")
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "HuBERT extraction needs the optional 'hubert' extra. Install with: "
                "pip install -e '.[hubert]'  (torch, torchaudio, transformers)"
            ) from error

        self._model_name = model_name
        self._layer = layer
        self._device = device or ("cuda" if self._torch.cuda.is_available() else "cpu")

        self._feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_name)
        model = transformers.AutoModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model.to(self._device)
        self._embedding_dim = int(model.config.hidden_size)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def layer_name(self) -> str:
        """Human-readable layer identifier recorded in output lineage."""
        return "final" if self._layer is None else f"hidden_{self._layer}"

    @property
    def sampling_rate(self) -> int:
        return TARGET_SAMPLE_RATE_HZ

    def load_waveform(self, audio_path: Path) -> np.ndarray:
        """Load a WAV as mono 16 kHz float32 samples.

        Audio I/O uses soundfile (native WAV, no codec backend) and resampling
        uses torchaudio's codec-free functional resampler, so the heavy
        ``torchaudio.load`` / TorchCodec path is avoided.
        """
        import soundfile

        waveform, sample_rate = soundfile.read(str(audio_path), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != TARGET_SAMPLE_RATE_HZ:
            tensor = self._torch.from_numpy(np.ascontiguousarray(waveform))
            tensor = self._torchaudio.functional.resample(
                tensor, sample_rate, TARGET_SAMPLE_RATE_HZ
            )
            waveform = tensor.cpu().numpy()
        return np.ascontiguousarray(waveform, dtype=np.float32)

    def extract_frame_embeddings(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run frozen HuBERT once over a waveform.

        Returns ``(frame_embeddings, frame_centers)`` where ``frame_embeddings``
        is ``(num_frames, embedding_dim)`` float32 and ``frame_centers`` is the
        centre time (seconds) of each frame.
        """
        inputs = self._feature_extractor(
            waveform,
            sampling_rate=TARGET_SAMPLE_RATE_HZ,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self._device)
        with self._torch.no_grad():
            outputs = self._model(
                input_values,
                output_hidden_states=self._layer is not None,
            )
        hidden = (
            outputs.last_hidden_state
            if self._layer is None
            else outputs.hidden_states[self._layer]
        )
        frame_embeddings = hidden.squeeze(0).to(self._torch.float32).cpu().numpy()
        duration_sec = waveform.shape[0] / TARGET_SAMPLE_RATE_HZ
        centers = frame_center_times(frame_embeddings.shape[0], duration_sec)
        return frame_embeddings, centers

    def pool_phone(
        self,
        frame_embeddings: np.ndarray,
        frame_centers: np.ndarray,
        start_sec: float,
        end_sec: float,
    ) -> PhoneEmbedding | None:
        """Mean-pool one phone interval; ``None`` if there are no frames."""
        vector, n_pooled = pool_phone_embedding(
            frame_embeddings, frame_centers, start_sec, end_sec
        )
        if vector is None:
            return None
        return PhoneEmbedding(vector=vector, n_frames_pooled=n_pooled)

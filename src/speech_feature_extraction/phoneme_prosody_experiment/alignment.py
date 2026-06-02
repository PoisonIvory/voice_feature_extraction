"""Forced alignment integration for prosody phoneme extraction.

This module wraps MFA (Montreal Forced Aligner) to produce phone-level
alignments from prosody WAV recordings. Alignments are output as TextGrid
files and parsed into our segment data structures.

MFA installation (conda recommended):
    conda install -c conda-forge montreal-forced-aligner
    mfa model download acoustic english_mfa
    mfa model download dictionary english_us_mfa
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from speech_feature_extraction.phoneme_prosody_experiment.rainbow_profile import (
    AlignedPhonemeSegment,
)

LOGGER = logging.getLogger(__name__)

RAINBOW_PASSAGE_SENTENCE_ONE = (
    "When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow."
)
RAINBOW_PASSAGE_SENTENCE_TWO = (
    "The rainbow is a division of white light into many beautiful colors."
)
RAINBOW_PASSAGE_TEXT = (
    f"{RAINBOW_PASSAGE_SENTENCE_ONE} {RAINBOW_PASSAGE_SENTENCE_TWO} "
    "These take the shape of a long round arch, with its path high above, and its two ends "
    "apparently beyond the horizon. There is, according to legend, a boiling pot of gold at one end. "
    "People look, but no one ever finds it. When a man looks for something beyond his reach, his "
    "friends say he is looking for the pot of gold at the end of the rainbow."
)
RAINBOW_PASSAGE_MEDIUM_TEXT = f"{RAINBOW_PASSAGE_SENTENCE_ONE} {RAINBOW_PASSAGE_SENTENCE_TWO}"
RAINBOW_PASSAGE_SHORT_TEXT = RAINBOW_PASSAGE_SENTENCE_ONE

MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_DICTIONARY = "english_us_mfa"

ALIGNMENT_ENGINE_MFA = "mfa"
PHONE_TIER_NAME = "phones"
WORD_TIER_NAME = "words"


@dataclass(frozen=True)
class AlignmentResult:
    """Result of forced alignment for one recording."""

    recording_id: str
    audio_path: Path
    textgrid_path: Path | None
    segments: tuple[AlignedPhonemeSegment, ...]
    word_segments: tuple[WordSegment, ...]
    alignment_engine: str
    alignment_version: str
    success: bool
    error_message: str | None


@dataclass(frozen=True)
class WordSegment:
    """One word interval from alignment."""

    word: str
    start_sec: float
    end_sec: float


def check_mfa_available() -> tuple[bool, str]:
    """Check if MFA is installed and models are available."""
    try:
        command = [*_resolve_mfa_base_command(), "version"]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            error_text = (result.stderr or result.stdout or "MFA command failed").strip()
            return False, error_text
        version = result.stdout.strip()
        return True, version
    except FileNotFoundError:
        return False, "MFA not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "MFA version check timed out"


def check_mfa_models_available() -> tuple[bool, str]:
    """Check if required MFA models are downloaded."""
    try:
        base_command = _resolve_mfa_base_command()
        result = subprocess.run(
            [*base_command, "model", "list", "acoustic"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "Failed listing acoustic models").strip()
        if MFA_ACOUSTIC_MODEL not in result.stdout:
            return False, f"Acoustic model '{MFA_ACOUSTIC_MODEL}' not found. Run: mfa model download acoustic {MFA_ACOUSTIC_MODEL}"

        result = subprocess.run(
            [*base_command, "model", "list", "dictionary"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return False, (result.stderr or result.stdout or "Failed listing dictionary models").strip()
        if MFA_DICTIONARY not in result.stdout:
            return False, f"Dictionary '{MFA_DICTIONARY}' not found. Run: mfa model download dictionary {MFA_DICTIONARY}"

        return True, "Models available"
    except Exception as e:
        return False, f"Error checking models: {e}"


def align_recording(
    audio_path: Path,
    recording_id: str,
    output_dir: Path,
    transcription: str | None = None,
) -> AlignmentResult:
    """Run MFA alignment on a single recording.

    Args:
        audio_path: Path to the WAV file.
        recording_id: Unique identifier for this recording.
        output_dir: Directory to write TextGrid output.
        transcription: Text transcription. Defaults to Rainbow Passage.

    Returns:
        AlignmentResult with parsed phone segments.
    """
    if transcription is None:
        transcriptions = _candidate_transcriptions_for_duration(_read_wav_duration_sec(audio_path))
    else:
        transcriptions = (transcription,)

    available, version_or_error = check_mfa_available()
    if not available:
        return AlignmentResult(
            recording_id=recording_id,
            audio_path=audio_path,
            textgrid_path=None,
            segments=(),
            word_segments=(),
            alignment_engine=ALIGNMENT_ENGINE_MFA,
            alignment_version="unknown",
            success=False,
            error_message=version_or_error,
        )

    models_ok, models_msg = check_mfa_models_available()
    if not models_ok:
        return AlignmentResult(
            recording_id=recording_id,
            audio_path=audio_path,
            textgrid_path=None,
            segments=(),
            word_segments=(),
            alignment_engine=ALIGNMENT_ENGINE_MFA,
            alignment_version=version_or_error,
            success=False,
            error_message=models_msg,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mfa_corpus_") as corpus_dir:
        corpus_path = Path(corpus_dir)
        wav_dest = corpus_path / f"{recording_id}.wav"
        lab_dest = corpus_path / f"{recording_id}.lab"

        shutil.copy2(audio_path, wav_dest)

        with tempfile.TemporaryDirectory(prefix="mfa_output_") as mfa_output_dir:
            mfa_output_path = Path(mfa_output_dir)
            attempt_errors: list[str] = []

            for attempt_idx, transcript_candidate in enumerate(transcriptions, start=1):
                lab_dest.write_text(transcript_candidate.strip(), encoding="utf-8")

                try:
                    result = subprocess.run(
                        [
                            *_resolve_mfa_base_command(),
                            "align",
                            "--clean",
                            "--single_speaker",
                            str(corpus_path),
                            MFA_DICTIONARY,
                            MFA_ACOUSTIC_MODEL,
                            str(mfa_output_path),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode != 0:
                        attempt_errors.append(result.stderr[:500] if result.stderr else "Unknown MFA error")
                        LOGGER.warning(
                            "MFA alignment attempt %d failed for %s",
                            attempt_idx,
                            recording_id,
                        )
                        continue

                    textgrid_src = mfa_output_path / f"{recording_id}.TextGrid"
                    if not textgrid_src.exists():
                        possible_files = list(mfa_output_path.rglob("*.TextGrid"))
                        if possible_files:
                            textgrid_src = possible_files[0]
                        else:
                            attempt_errors.append("No TextGrid output found")
                            continue

                    textgrid_dest = output_dir / f"{recording_id}.TextGrid"
                    shutil.copy2(textgrid_src, textgrid_dest)

                    segments, word_segments = parse_textgrid(textgrid_dest)

                    return AlignmentResult(
                        recording_id=recording_id,
                        audio_path=audio_path,
                        textgrid_path=textgrid_dest,
                        segments=segments,
                        word_segments=word_segments,
                        alignment_engine=ALIGNMENT_ENGINE_MFA,
                        alignment_version=version_or_error,
                        success=True,
                        error_message=None,
                    )

                except subprocess.TimeoutExpired:
                    attempt_errors.append("MFA alignment timed out (>5 minutes)")
                except Exception as e:
                    LOGGER.exception("Unexpected error during MFA alignment for %s", recording_id)
                    attempt_errors.append(str(e)[:500])

            return AlignmentResult(
                recording_id=recording_id,
                audio_path=audio_path,
                textgrid_path=None,
                segments=(),
                word_segments=(),
                alignment_engine=ALIGNMENT_ENGINE_MFA,
                alignment_version=version_or_error,
                success=False,
                error_message=" | ".join(attempt_errors)[:500] if attempt_errors else "Unknown MFA error",
            )


def parse_textgrid(textgrid_path: Path) -> tuple[tuple[AlignedPhonemeSegment, ...], tuple[WordSegment, ...]]:
    """Parse a TextGrid file into phone and word segments.

    Args:
        textgrid_path: Path to the TextGrid file.

    Returns:
        Tuple of (phone_segments, word_segments).
    """
    try:
        from praatio import textgrid
    except ImportError as e:
        raise ImportError(
            "praatio is required for TextGrid parsing. Install with: pip install praatio"
        ) from e

    tg = textgrid.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

    phone_segments: list[AlignedPhonemeSegment] = []
    word_segments: list[WordSegment] = []

    phone_tier = _find_tier(tg, PHONE_TIER_NAME, ["phones", "phone", "Phone", "Phones"])
    if phone_tier is not None:
        for interval in phone_tier.entries:
            label = interval.label.strip()
            if not label or label in ("", "sil", "sp", "spn", "<eps>"):
                continue
            phone_segments.append(
                AlignedPhonemeSegment(
                    phoneme_label=label,
                    start_sec=float(interval.start),
                    end_sec=float(interval.end),
                )
            )

    word_tier = _find_tier(tg, WORD_TIER_NAME, ["words", "word", "Word", "Words"])
    if word_tier is not None:
        for interval in word_tier.entries:
            label = interval.label.strip()
            if not label or label in ("", "sil", "sp", "spn", "<eps>"):
                continue
            word_segments.append(
                WordSegment(
                    word=label,
                    start_sec=float(interval.start),
                    end_sec=float(interval.end),
                )
            )

    return tuple(phone_segments), tuple(word_segments)


def _find_tier(tg, primary_name: str, fallback_names: list[str]):
    """Find a tier by name with fallbacks."""
    if primary_name in tg.tierNames:
        return tg.getTier(primary_name)
    for name in fallback_names:
        if name in tg.tierNames:
            return tg.getTier(name)
    return None


def align_batch(
    recordings: Iterable[tuple[Path, str]],
    output_dir: Path,
    transcription: str | None = None,
) -> list[AlignmentResult]:
    """Align multiple recordings.

    Args:
        recordings: Iterable of (audio_path, recording_id) tuples.
        output_dir: Directory to write TextGrid outputs.
        transcription: Shared transcription for all recordings.

    Returns:
        List of AlignmentResult objects.
    """
    results = []
    for audio_path, recording_id in recordings:
        result = align_recording(
            audio_path=audio_path,
            recording_id=recording_id,
            output_dir=output_dir,
            transcription=transcription,
        )
        results.append(result)
        if result.success:
            LOGGER.info("Aligned %s: %d phones", recording_id, len(result.segments))
        else:
            LOGGER.warning("Failed to align %s: %s", recording_id, result.error_message)
    return results


def _resolve_mfa_base_command() -> list[str]:
    """Resolve MFA invocation command from environment or installed tooling."""
    explicit_command = os.getenv("SPEECH_MFA_COMMAND")
    if explicit_command:
        return shlex.split(explicit_command)

    direct_mfa = shutil.which("mfa")
    if direct_mfa:
        return [direct_mfa]

    micromamba = shutil.which("micromamba") or "/opt/homebrew/bin/micromamba"
    if Path(micromamba).exists():
        return [micromamba, "run", "-n", os.getenv("SPEECH_MFA_ENV", "aligner"), "mfa"]

    return ["mfa"]


def _read_wav_duration_sec(audio_path: Path) -> float | None:
    """Read WAV duration in seconds for transcript candidate selection."""
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            if frame_rate <= 0:
                return None
            return wav_file.getnframes() / frame_rate
    except Exception:
        return None


def _candidate_transcriptions_for_duration(duration_sec: float | None) -> tuple[str, ...]:
    """Pick likely transcript variants based on recording duration."""
    if duration_sec is None:
        return (RAINBOW_PASSAGE_TEXT, RAINBOW_PASSAGE_MEDIUM_TEXT, RAINBOW_PASSAGE_SHORT_TEXT)

    # Empirical fit for current prosody_reading clips (~3-13 seconds).
    if duration_sec <= 8.0:
        return (RAINBOW_PASSAGE_SHORT_TEXT, RAINBOW_PASSAGE_MEDIUM_TEXT, RAINBOW_PASSAGE_TEXT)
    if duration_sec <= 16.0:
        return (RAINBOW_PASSAGE_MEDIUM_TEXT, RAINBOW_PASSAGE_SHORT_TEXT, RAINBOW_PASSAGE_TEXT)
    return (RAINBOW_PASSAGE_TEXT, RAINBOW_PASSAGE_MEDIUM_TEXT, RAINBOW_PASSAGE_SHORT_TEXT)

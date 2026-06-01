import wave
from pathlib import Path

from speech_feature_extraction.audio_qc import inspect_wav, sha256_file


def test_sha256_file_hashes_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"abc")

    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_inspect_wav_reads_basic_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    qc = inspect_wav(path)

    assert qc["qc_audio_readable"] is True
    assert qc["qc_sample_rate_hz"] == 16000
    assert qc["qc_channel_count"] == 1
    assert qc["qc_duration_sec"] == 1


def test_inspect_wav_rejects_low_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "sample_8k.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00\x00" * 8000)

    qc = inspect_wav(path)

    assert qc["qc_audio_readable"] is False
    assert qc["qc_sample_rate_hz"] == 8000
    assert qc["qc_failure_reason"] == "sample_rate_too_low:8000<16000"
    assert "sample_rate_too_low" in qc["qc_warning_codes"]

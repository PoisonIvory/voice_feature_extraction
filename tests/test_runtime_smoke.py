import importlib.util
import wave
from pathlib import Path

from speech_feature_extraction.opensmile_egemaps import OpenSmileEgemapsExtractor


def test_runtime_dependencies_importable() -> None:
    missing = [
        dependency
        for dependency in ("opensmile", "appwrite")
        if importlib.util.find_spec(dependency) is None
    ]
    assert not missing, f"Missing runtime dependencies: {', '.join(missing)}"


def test_opensmile_runtime_smoke_extracts_features(tmp_path: Path) -> None:
    path = tmp_path / "smoke.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    extractor = OpenSmileEgemapsExtractor()
    features, lld_qc = extractor.extract_file_with_qc(path)

    assert features["qc_opensmile_egemaps_success"] is True
    assert features["qc_feature_count_egemaps"] == 88
    assert features["qc_feature_count_egemaps_expected"] == 88
    assert lld_qc.total_frames > 0

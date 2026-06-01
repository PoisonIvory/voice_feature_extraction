from pathlib import Path

from speech_feature_extraction.opensmile_egemaps import OpenSmileEgemapsExtractor


class _FakeMask:
    def __init__(self, values: list[bool]) -> None:
        self._values = values

    def any(self) -> bool:
        return any(self._values)


class _FakeRow:
    def __init__(self, data: dict[str, float]) -> None:
        self._data = data
        self.index = list(data.keys())

    def isna(self) -> _FakeMask:
        return _FakeMask([value != value for value in self._data.values()])

    def to_dict(self) -> dict[str, float]:
        return dict(self._data)


class _FakeILoc:
    def __init__(self, row: _FakeRow) -> None:
        self._row = row

    def __getitem__(self, _: int) -> _FakeRow:
        return self._row


class _FakeFrame:
    def __init__(self, data: dict[str, float]) -> None:
        self.empty = False
        self.iloc = _FakeILoc(_FakeRow(data))
        self._size = 1

    def __len__(self) -> int:
        return self._size


class _FakeSmile:
    def __init__(self, frame: _FakeFrame, feature_names: list[str]) -> None:
        self._frame = frame
        self.feature_names = feature_names
        self.config_name = "eGeMAPSv02"
        self.config_path = "/tmp/eGeMAPSv02.conf"

    def process_file(self, _: str) -> _FakeFrame:
        return self._frame


def _build_extractor(frame: _FakeFrame, feature_names: list[str]) -> OpenSmileEgemapsExtractor:
    extractor = OpenSmileEgemapsExtractor.__new__(OpenSmileEgemapsExtractor)
    extractor._smile = _FakeSmile(frame, feature_names)
    extractor._feature_set_name = "opensmile.FeatureSet.eGeMAPSv02"
    extractor._feature_level_name = "opensmile.FeatureLevel.Functionals"
    extractor._sampling_rate_hz = 16000
    extractor._resample = True
    extractor._channels = 0
    extractor._mixdown = False
    return extractor


def _assert_raises_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError as error:
        assert message in str(error)
        return
    raise AssertionError("Expected ValueError was not raised")


def test_extract_file_rejects_nan_features() -> None:
    frame = _FakeFrame({"f1": 1.0, "f2": float("nan")})
    extractor = _build_extractor(frame=frame, feature_names=["f1", "f2"])

    _assert_raises_value_error(
        lambda: extractor.extract_file(Path("/tmp/sample.wav")),
        message="NaN features",
    )


def test_extract_file_rejects_schema_mismatch() -> None:
    frame = _FakeFrame({"f1": 1.0, "f2": 2.0})
    extractor = _build_extractor(frame=frame, feature_names=["f1", "f3"])

    _assert_raises_value_error(
        lambda: extractor.extract_file(Path("/tmp/sample.wav")),
        message="schema mismatch",
    )


def test_extract_file_rejects_feature_count_mismatch() -> None:
    feature_names = [f"f{i}" for i in range(87)]
    frame = _FakeFrame({name: float(i) for i, name in enumerate(feature_names)})
    extractor = _build_extractor(frame=frame, feature_names=feature_names)

    _assert_raises_value_error(
        lambda: extractor.extract_file(Path("/tmp/sample.wav")),
        message="runtime feature count",
    )

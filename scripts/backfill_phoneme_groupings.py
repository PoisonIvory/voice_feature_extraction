"""Append articulatory grouping columns to an existing phoneme-features parquet.

The grouping columns (manner/place/voicing/height/broad_class) are a
deterministic function of the normalized phoneme label, which is already stored
as ``phonemeLabel``. Recordings extracted before grouping was wired into the
pipeline can therefore be enriched in place without re-running forced alignment
and feature extraction. New runs already emit these columns natively.

The output column order is reconciled to PhonemeRowData so a backfilled parquet
is schema-identical to a freshly extracted one. The write is atomic (temp file
then rename) so an interruption cannot corrupt the dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.pipeline import PhonemeRowData
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_EXPERIMENT_DATA_ROOT,
    PHONEME_PROSODY_FEATURES_FILENAME,
)
from speech_feature_extraction.phoneme_prosody_experiment.taxonomy import group_phoneme

DEFAULT_PARQUET = Path(PHONEME_PROSODY_EXPERIMENT_DATA_ROOT) / PHONEME_PROSODY_FEATURES_FILENAME
GROUPING_COLUMNS = ("phonemeManner", "phonemePlace", "phonemeVoicing", "phonemeHeight", "phonemeBroadClass")


def backfill(df: pd.DataFrame) -> pd.DataFrame:
    """Add grouping columns derived from phonemeLabel, in dataclass order."""
    groupings = df["phonemeLabel"].map(group_phoneme)
    df = df.copy()
    df["phonemeManner"] = groupings.map(lambda g: g.manner)
    df["phonemePlace"] = groupings.map(lambda g: g.place)
    df["phonemeVoicing"] = groupings.map(lambda g: g.voicing)
    df["phonemeHeight"] = groupings.map(lambda g: g.height)
    df["phonemeBroadClass"] = groupings.map(lambda g: g.broad_class)

    ordered = [c for c in PhonemeRowData.__dataclass_fields__ if c in df.columns]
    return df[ordered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    missing = [c for c in GROUPING_COLUMNS if c not in df.columns]
    if not missing:
        print(f"All grouping columns already present in {args.parquet}; nothing to do.")
        return

    enriched = backfill(df)
    tmp_path = args.parquet.with_name(args.parquet.name + ".tmp")
    enriched.to_parquet(tmp_path, index=False)
    tmp_path.replace(args.parquet)

    print(f"Backfilled {len(missing)} grouping column(s) into {args.parquet}")
    print(f"Rows: {len(enriched)} | columns: {len(enriched.columns)}")


if __name__ == "__main__":
    main()

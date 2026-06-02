"""Phoneme-order consistency check for the prosody phoneme-features parquet.

Verifies that forced alignment produces consistent phoneme ordering across
recordings (the same Rainbow Passage sentences 2-3), independent of timing.
Uses longest-common-subsequence (LCS) ratio so insertions/deletions do not
positionally penalise an otherwise-correct order.

Run after each extraction chunk to confirm consistency holds as recordings
accumulate, before any downstream pattern analysis.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from speech_feature_extraction.phoneme_prosody_experiment.rainbow_inventory import (
    PROSODY_CANONICAL_ARPABET_SEQUENCE,
)
from speech_feature_extraction.phoneme_prosody_experiment.schema import (
    PHONEME_PROSODY_EXPERIMENT_DATA_ROOT,
    PHONEME_PROSODY_FEATURES_FILENAME,
)

DEFAULT_PARQUET = Path(PHONEME_PROSODY_EXPERIMENT_DATA_ROOT) / PHONEME_PROSODY_FEATURES_FILENAME
# An aligned read of the wrong content keeps order but drops LCS well below this.
LCS_OUTLIER_THRESHOLD = 0.90


def lcs_ratio(a: list[str], b: list[str]) -> float:
    """Longest-common-subsequence length divided by the longer sequence."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] else max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m] / max(n, m)


def sequences_by_recording(df: pd.DataFrame) -> dict[str, list[str]]:
    """Ordered phoneme-label sequence per recording, by phonemeIndex."""
    sequences: dict[str, list[str]] = {}
    for recording_id, group in df.sort_values("phonemeIndex").groupby("recordingId"):
        sequences[str(recording_id)] = group["phonemeLabel"].astype(str).tolist()
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    args = parser.parse_args()

    if not args.parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {args.parquet}")

    df = pd.read_parquet(args.parquet)
    if df.empty:
        print("Parquet is empty; nothing to check yet.")
        return

    canonical = list(PROSODY_CANONICAL_ARPABET_SEQUENCE)
    sequences = sequences_by_recording(df)
    n = len(sequences)

    print(f"Parquet: {args.parquet}")
    print(f"Recordings: {n} | canonical length: {len(canonical)}")
    print()

    counts = [len(s) for s in sequences.values()]
    print(
        f"Phone counts: min={min(counts)} median={int(statistics.median(counts))} "
        f"max={max(counts)} | exactly canonical ({len(canonical)}): "
        f"{sum(c == len(canonical) for c in counts)}/{n}"
    )

    vs_canon = {rid: lcs_ratio(seq, canonical) for rid, seq in sequences.items()}
    vals = sorted(vs_canon.values())
    print(
        f"LCS vs canonical: min={vals[0]:.3f} median={statistics.median(vals):.3f} "
        f"max={vals[-1]:.3f}"
    )

    outliers = {rid: r for rid, r in vs_canon.items() if r < LCS_OUTLIER_THRESHOLD}
    print(f"Order outliers (LCS < {LCS_OUTLIER_THRESHOLD:.2f}): {len(outliers)}")
    for rid, r in sorted(outliers.items(), key=lambda kv: kv[1]):
        print(f"  {rid}: {r:.3f}")

    if "qc_recording_ok" in df.columns:
        per_rec_ok = df.groupby("recordingId")["qc_recording_ok"].first()
        print()
        print(f"Recording alignment QC: ok={int(per_rec_ok.sum())}/{len(per_rec_ok)}")
        failed = per_rec_ok[~per_rec_ok].index.tolist()
        if failed:
            print(f"  flagged: {failed}")


if __name__ == "__main__":
    main()

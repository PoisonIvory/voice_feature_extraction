# voice_feature_extraction

Standalone Python pipeline for reproducibly pulling saved Decibelle Appwrite WAV recordings and extracting standardized openSMILE eGeMAPSv02 features.

The first canonical pass intentionally keeps the feature surface small:

- Appwrite Storage `audio` is the source of truth for saved WAV files.
- Appwrite `voice_recordings` is left-joined as preferred metadata.
- Only `vowel` and `prosody` recordings are in scope.
- The canonical extractor is `opensmile.FeatureSet.eGeMAPSv02` at `opensmile.FeatureLevel.Functionals`.
- Extraction uses a standardized openSMILE runtime policy (`sampling_rate=16000`, `resample=True`).
- `ComParE_2016` is deferred to a later optional discovery artifact.
- Parquet files are the local source of truth; CSV/XLSX exports can be generated later from Parquet.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Fill in `APPWRITE_API_KEY` in `.env`. The key needs read access to the `period_tracker_db` database and `audio` storage bucket.

## Commands

Write a storage-first audit without extracting features:

```sh
extract-speech-features audit
```

Download in-scope WAVs and extract openSMILE eGeMAPSv02 features:

```sh
extract-speech-features extract
```

Publish an immutable snapshot bundle from existing processed parquet outputs:

```sh
extract-speech-features publish-snapshot --snapshot-id 2026-06-01 --update-latest
```

Process a small smoke-test batch:

```sh
extract-speech-features extract --limit 2
```

`--limit` now runs a partial extraction without deleting prior completed recordings in the canonical parquet.

Generated artifacts are written under `data/processed/` and local WAV cache files under `data/raw_audio/`. These paths are ignored by git.

## Outputs

- `data/processed/voice_features_v3_recordings.parquet`: one row per completed recording with metadata, lineage, QC, and eGeMAPSv02 features prefixed with `egemaps_`.
- `data/processed/voice_features_v3_audit.parquet`: skipped files, failures, metadata warnings, and extraction status.

Each completed extraction row records lineage fields including:
- extractor version and extraction timestamp
- openSMILE package version
- feature set and level
- openSMILE config file identity
- sample-rate and resampling settings
- SHA256 of the processed audio file

## Snapshot Publishing Contract

Snapshot publishing is a separate workflow from extraction and writes immutable bundles under:

- `exports/snapshots/speech-features/v3/<snapshot>/voice_features_v3_recordings.parquet`
- `exports/snapshots/speech-features/v3/<snapshot>/voice_features_v3_audit.parquet`
- `exports/snapshots/speech-features/v3/<snapshot>/manifest.json`

`manifest.json` includes:

- `manifest_version` (`1.0`)
- provenance (`source_commit`, `pipeline_version`, `opensmile_version`, `python_version`)
- per-file integrity (`content_sha256`, `file_size_bytes`) and schema hashes
- required consumer contract (`required_core_columns`, `required_feature_prefixes`, `required_feature_count_by_prefix`)

Environment variables:

- `SPEECH_SNAPSHOT_ROOT` (default `exports/snapshots`)
- `SPEECH_SNAPSHOT_ID` (optional explicit snapshot ID)
- `SPEECH_UPDATE_LATEST` (`true`/`false`, default `false`)
- `SPEECH_SNAPSHOT_PIPELINE_VERSION` (optional provenance override)
- `SPEECH_SNAPSHOT_OPENSMILE_VERSION` (optional provenance override)
- `SPEECH_SNAPSHOT_SOURCE_COMMIT` (optional provenance override)

## Reproducibility Notes

- The project pins `opensmile` to `<3` and `>=2.5` to avoid accidental cross-major upgrades.
- If the openSMILE runtime schema changes, extraction now fails instead of silently writing mismatched feature columns.

## Development

```sh
pytest
```
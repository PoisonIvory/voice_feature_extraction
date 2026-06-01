# voice_feature_extraction

Standalone Python pipeline for reproducibly pulling saved Decibelle Appwrite WAV recordings and extracting standardized openSMILE eGeMAPSv02 features.

The first canonical pass intentionally keeps the feature surface small:

- Appwrite Storage `audio` is the source of truth for saved WAV files.
- Appwrite `voice_recordings` is left-joined as preferred metadata.
- Only `vowel` and `prosody` recordings are in scope.
- The canonical extractor is `opensmile.FeatureSet.eGeMAPSv02` at `opensmile.FeatureLevel.Functionals`.
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

Process a small smoke-test batch:

```sh
extract-speech-features extract --limit 2
```

Generated artifacts are written under `data/processed/` and local WAV cache files under `data/raw_audio/`. These paths are ignored by git.

## Outputs

- `data/processed/voice_features_v3_recordings.parquet`: one row per completed recording with metadata, lineage, QC, and eGeMAPSv02 features prefixed with `egemaps_`.
- `data/processed/voice_features_v3_audit.parquet`: skipped files, failures, metadata warnings, and extraction status.

## Development

```sh
pytest
```
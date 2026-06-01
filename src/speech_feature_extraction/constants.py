"""Shared pipeline constants."""

EXTRACTOR_VERSION = "v3.1-opensmile-egemaps"

APPWRITE_DATABASE_ID = "period_tracker_db"
APPWRITE_VOICE_RECORDINGS_COLLECTION_ID = "voice_recordings"
APPWRITE_AUDIO_BUCKET_ID = "audio"

IN_SCOPE_TASK_TYPES = {"vowel", "prosody"}
OPENSMILE_EGEMAPS_PREFIX = "egemaps_"
OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT = 88

RECORDINGS_PARQUET = "voice_features_v3_recordings.parquet"
AUDIT_PARQUET = "voice_features_v3_audit.parquet"

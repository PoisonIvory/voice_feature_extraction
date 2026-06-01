"""Shared pipeline constants."""

EXTRACTOR_VERSION = "v4.0-daily-task-separated-median"

APPWRITE_DATABASE_ID = "period_tracker_db"
APPWRITE_VOICE_RECORDINGS_COLLECTION_ID = "voice_recordings"
APPWRITE_AUDIO_BUCKET_ID = "audio"

IN_SCOPE_TASK_TYPES = {"vowel", "prosody"}
OPENSMILE_EGEMAPS_PREFIX = "egemaps_"
OPENSMILE_EGEMAPS_EXPECTED_FEATURE_COUNT = 88

DAILY_FEATURES_PARQUET = "voice_features_v4_daily.parquet"
RECORDINGS_STAGING_PARQUET = "voice_features_v4_recordings_staging.parquet"
AUDIT_PARQUET = "voice_features_v4_audit.parquet"

# Task-specific quality control thresholds
# Based on ASHA protocols, MDVP reference values, and eGeMAPS documentation.
# NOTE: MDVP perturbation cutoffs are used as screening guidance only; they are
# not directly transferable across all software implementations.
# References:
#   - Eyben et al. 2015, "The Geneva Minimalistic Acoustic Parameter Set"
#   - ASHA Expert Panel 2018, "Recommended Protocols for Instrumental Assessment of Voice"
#   - Praat MDVP thresholds for jitter/shimmer pathology detection

VOWEL_QC_THRESHOLDS = {
    # Duration requirements (ASHA recommends 3-5s sustained vowel, analyze middle 2-3s)
    "min_duration_sec": 2.0,
    "max_duration_sec": 15.0,
    # Voiced ratio: sustained vowel should be almost entirely voiced (>90%)
    "min_voiced_ratio": 0.90,
    # F0 stability: coefficient of variation should be low for sustained phonation
    "max_f0_coefficient_of_variation": 0.20,
    # Jitter screening threshold: MDVP reference is 1.04%, relaxed for QC flagging
    "max_jitter_percent": 1.5,
    # Shimmer screening threshold: MDVP reference is 3.81%, relaxed for QC flagging
    "max_shimmer_percent": 4.5,
    # Shimmer dB screening threshold: MDVP reference is 0.350 dB
    "max_shimmer_db": 0.50,
}

PROSODY_QC_THRESHOLDS = {
    # Duration: connected speech needs longer samples for temporal patterns
    "min_duration_sec": 2.5,
    "max_duration_sec": 30.0,
    # Voiced ratio: prosody should have both voiced and unvoiced (pauses)
    "min_voiced_ratio": 0.30,
    "max_voiced_ratio": 0.95,
    # F0 variation is expected in prosody - no max CoV constraint
    # Jitter/shimmer less meaningful for connected speech - not enforced
}

# Shared audio quality thresholds (apply to all tasks)
AUDIO_QC_THRESHOLDS = {
    "min_sample_rate_hz": 16000,
    "max_clipping_ratio": 0.001,  # Block if >0.1% samples clip
}

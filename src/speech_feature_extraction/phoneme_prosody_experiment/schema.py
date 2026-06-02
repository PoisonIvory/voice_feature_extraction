"""Schema primitives for the isolated prosody phoneme experiment dataset."""

PHONEME_PROSODY_EXPERIMENT_DATA_ROOT = "data/experimental/phoneme_prosody"
PHONEME_PROSODY_FEATURES_FILENAME = "prosody_phoneme_features.parquet"

PHONEME_PROSODY_LINEAGE_FIELDS = (
    "recordingId",
    "userId",
    "recordedDate",
    "taskType",
    "audioHash",
    "extractorVersion",
    "alignmentEngine",
    "alignmentVersion",
)

PHONEME_PROSODY_ALIGNMENT_FIELDS = (
    "phonemeIndex",
    "phonemeLabel",
    "wordLabel",
    "startSec",
    "endSec",
    "durationSec",
    "alignmentScoreRaw",
    "alignmentQuality",
)

PHONEME_PROSODY_CONTEXT_FIELDS = (
    "prevPhonemeLabel",
    "nextPhonemeLabel",
    "coarticulationContext",
    "isAdjacentToNasal",
)

PHONEME_PROSODY_BOUNDARY_FIELDS = (
    "trimPolicyMs",
    "analysisStartSec",
    "analysisEndSec",
    "analysisDurationSec",
)

PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS = (
    "rainbowOccurrenceIndex",
    "rainbowExpectedPositionRatio",
    "rainbowObservedPositionRatio",
    "rainbowPositionDeltaRatio",
    "rainbowTimingConsistent",
)

PHONEME_PROSODY_GROUPING_FIELDS = (
    "phonemeClassPrimary",
    "phonemeClassTags",
)

PHONEME_PROSODY_FEATURE_VALUE_FIELDS = (
    "segment_mfcc2_mean",
    "segment_h1h2_mean",
    "segment_f1_bandwidth_mean",
)

PHONEME_PROSODY_FEATURE_QC_FIELDS = (
    "qc_segment_ok",
    "qc_segment_reason",
    "qc_numFrames",
    "qc_minFramesRequired",
)

PHONEME_PROSODY_REQUIRED_FIELDS = (
    *PHONEME_PROSODY_LINEAGE_FIELDS,
    *PHONEME_PROSODY_ALIGNMENT_FIELDS,
    *PHONEME_PROSODY_CONTEXT_FIELDS,
    *PHONEME_PROSODY_BOUNDARY_FIELDS,
    *PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS,
    *PHONEME_PROSODY_GROUPING_FIELDS,
    *PHONEME_PROSODY_FEATURE_VALUE_FIELDS,
    *PHONEME_PROSODY_FEATURE_QC_FIELDS,
)

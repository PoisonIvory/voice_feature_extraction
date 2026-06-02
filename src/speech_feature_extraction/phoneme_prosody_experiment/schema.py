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

# Deferred for the MVP: process_batch never builds a rainbow template, so
# these fields are always None. They are retained in the contract so the
# parquet shape is stable when template matching is enabled later.
PHONEME_PROSODY_RAINBOW_PROFILE_FIELDS = (
    "rainbowOccurrenceIndex",
    "rainbowExpectedPositionRatio",
    "rainbowObservedPositionRatio",
    "rainbowPositionDeltaRatio",
    "rainbowTimingConsistent",
)

# Middle grouping between recording and phoneme unit. phonemeClassPrimary /
# phonemeClassTags carry the granular phone and coarticulation overlap tags;
# the articulatory dimensions below let downstream analysis roll phonemes up by
# whichever grouping is most informative.
PHONEME_PROSODY_GROUPING_FIELDS = (
    "phonemeClassPrimary",
    "phonemeClassTags",
    "phonemeManner",
    "phonemePlace",
    "phonemeVoicing",
    "phonemeHeight",
    "phonemeBroadClass",
)

PHONEME_PROSODY_FEATURE_VALUE_FIELDS = (
    "segment_mfcc2_mean",
    "segment_h1h2_mean",
    "segment_f1_bandwidth_mean",
    "segment_f0_mean",
)

PHONEME_PROSODY_FEATURE_QC_FIELDS = (
    "qc_segment_ok",
    "qc_segment_reason",
    "qc_numFrames",
    "qc_minFramesRequired",
    "qc_label_canonical",
)

# Recording-level alignment QC, stamped identically on every row of a
# recording. Lets downstream consumers filter mis-aligned recordings (where the
# transcript fallback force-aligned the wrong content) instead of having them
# silently pollute phoneme-level patterns.
PHONEME_PROSODY_RECORDING_QC_FIELDS = (
    "qc_recording_coverage_ratio",
    "qc_recording_unexpected_phones",
    "qc_recording_ok",
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
    *PHONEME_PROSODY_RECORDING_QC_FIELDS,
)

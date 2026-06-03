"""Schema primitives for the isolated prosody phoneme experiment dataset."""

PHONEME_PROSODY_EXPERIMENT_DATA_ROOT = "data/experimental/phoneme_prosody"
PHONEME_PROSODY_FEATURES_FILENAME = "prosody_phoneme_features_v3.parquet"

# HuBERT phonological-subspace experiment outputs (parallel to the eGeMAPS
# phoneme features, sharing the same MFA boundaries). The unsuffixed names are
# the canonical HuBERT-base outputs; additional frozen backbones are written to
# per-model suffixed names so they can share the same MFA boundaries without
# clobbering each other (see the *_for_model helpers below).
HUBERT_PHONE_EMBEDDINGS_FILENAME = "hubert_phone_embeddings.parquet"
HUBERT_DPRIME_FILENAME = "hubert_dprime_by_recording.parquet"


def hubert_model_slug(model_name: str) -> str:
    """Filesystem-safe slug for an SSL checkpoint id (its last path component).

    ``microsoft/wavlm-base`` -> ``wavlm-base``. Lets the multi-backbone
    robustness check name its outputs per model.
    """
    return model_name.rsplit("/", 1)[-1]


def hubert_embeddings_filename(model_name: str) -> str:
    """Per-backbone phone-embeddings parquet filename for ``model_name``."""
    return f"hubert_phone_embeddings__{hubert_model_slug(model_name)}.parquet"


def hubert_dprime_filename(model_name: str) -> str:
    """Per-backbone d-prime-by-recording parquet filename for ``model_name``."""
    return f"hubert_dprime__{hubert_model_slug(model_name)}.parquet"

# Canonical eGeMAPSv02 Low-Level Descriptor names (openSMILE order).
EGEMAPS_LLD_NAMES = (
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
)

AGGREGATE_STATS = ("mean", "median", "std")


def lld_value_field(lld_name: str, stat: str) -> str:
    """Map an openSMILE LLD column name to a parquet feature field."""
    stem = lld_name.removesuffix("_sma3nz").removesuffix("_sma3")
    stem = stem.replace(".", "_").replace("-", "_")
    return f"segment_{stem}_{stat}"

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

PHONEME_PROSODY_FEATURE_VALUE_FIELDS = tuple(
    lld_value_field(name, stat)
    for name in EGEMAPS_LLD_NAMES
    for stat in AGGREGATE_STATS
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

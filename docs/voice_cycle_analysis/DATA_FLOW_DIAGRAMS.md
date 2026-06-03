# Data Flow Diagrams

Use the first diagram for the main presentation. The appendix diagrams expose
more implementation detail without making the main slide carry the full method.

## Main Slide: End-to-End Data Flow

```mermaid
flowchart LR
    APP["Mobile app<br/>daily voice tasks"] --> AW["Appwrite<br/>Storage + voice metadata"]
    AW --> WAV["SpeechFeatureExtraction<br/>downloads WAVs here"]

    WAV --> CANON["Whole-WAV acoustics<br/>openSMILE eGeMAPSv02<br/>daily vowel + prosody features"]
    WAV --> GROUPS["Phoneme grouping layer<br/>MFA boundaries + articulatory taxonomy"]
    WAV --> PHONES["Granular phoneme layer<br/>one acoustic row per aligned phoneme"]

    CANON --> ANALYSIS["Cycle analysis<br/>join voice + hormones + wearable context"]
    GROUPS --> ANALYSIS
    PHONES --> ANALYSIS

    CONTEXT["Physiology + context<br/>Inito hormones, Oura,<br/>cycle calendar"] --> ANALYSIS

    ANALYSIS --> OUT["Presentation outputs<br/>localization, controls,<br/>figures, mechanistic model"]
```

**Talk track:** the app captures repeated voice samples; Appwrite is the cloud
source of truth; this repository pulls the WAVs into a reproducible local
pipeline; three analysis streams ask increasingly specific questions about where
cycle-linked voice variation lives.

## Appendix A: Appwrite to Canonical Daily Features

```mermaid
flowchart TD
    A["Appwrite Storage<br/>audio bucket: saved WAV files"] --> C["Build manifest"]
    B["Appwrite Database<br/>voice_recordings metadata"] --> C

    C --> D{"In scope?<br/>target user + vowel/prosody"}
    D -- "no" --> AUDIT["Audit parquet<br/>skip reason + metadata warnings"]
    D -- "yes" --> E["Download or reuse cached WAV<br/>data/raw_audio"]

    E --> F["WAV QC<br/>readable audio, duration,<br/>sample properties, clipping"]
    F --> G["SHA256 + lineage<br/>reproducibility fields"]
    G --> H["openSMILE eGeMAPSv02<br/>Functionals, 16 kHz resampling"]
    H --> I["Task-specific QC<br/>vowel/prosody thresholds"]
    I --> J["Recording staging parquet<br/>one row per recording"]
    J --> K["Daily aggregation<br/>median by user + UTC day"]
    K --> L["voice_features_v4_daily.parquet<br/>vowel_egemaps_* + prosody_egemaps_*"]
    I --> AUDIT
```

**What this diagram answers:** how a saved mobile recording becomes the
recording-level audit and the daily eGeMAPS feature table used for the broad
cycle scan.

## Appendix B: Mechanistic Analysis Streams

```mermaid
flowchart LR
    VoiceSource["Cached Appwrite WAVs + metadata<br/>vowel and connected-speech tasks"]
    PhaseContext["Shared phase context<br/>cycle calendar + Inito hormones + Oura"]

    subgraph WholeRecording["STREAM 1: WHOLE-WAV / DAILY ACOUSTICS"]
        direction TB
        WholeExtract["openSMILE eGeMAPSv02 Functionals<br/>full vowel + prosody WAVs"]
        WholeDaily["Daily feature table<br/>per-day median vowel_egemaps_*<br/>and prosody_egemaps_*"]
        WholeAnalysis["Broad cycle scan<br/>follicular vs luteal effects,<br/>PdG coupling, Oura cross-check"]
        WholeFinding["Finding<br/>coherent cycle signal;<br/>cover/timbre channels move most"]
        WholeExtract --> WholeDaily --> WholeAnalysis --> WholeFinding
    end

    subgraph PhonemeGroups["STREAM 2: PHONEME GROUPINGS / TAXONOMY"]
        direction TB
        GroupSubset["Clean connected-speech subset<br/>fixed Rainbow passage"]
        GroupAlign["MFA forced alignment<br/>phone boundaries in TextGrid"]
        GroupTaxonomy["Taxonomy labels<br/>manner, place, voicing,<br/>vowel class, nasal, diphthong"]
        GroupAnalysis["Group-level localization<br/>compare interpretable phoneme classes<br/>across cycle phase"]
        GroupFinding["Finding<br/>signal is broadly shared across classes;<br/>nasal + diphthong groups stand out"]
        GroupSubset --> GroupAlign --> GroupTaxonomy --> GroupAnalysis --> GroupFinding
    end

    subgraph GranularPhones["STREAM 3: GRANULAR PHONEME ROWS"]
        direction TB
        PhoneSubset["Same aligned connected-speech subset<br/>fixed passage controls wording"]
        PhoneFeatures["Per-phoneme acoustic table<br/>one row per aligned phoneme<br/>with QC + timing + features"]
        PhoneControls["Granular controls<br/>per-recording de-meaning,<br/>F0 residualization, within-recording contrasts"]
        PhoneAnalysis["Residual checks<br/>which individual phoneme effects remain<br/>after recording-level offsets are removed?"]
        PhoneFinding["Finding<br/>dominant effect is global;<br/>bounded residual structure remains"]
        PhoneSubset --> PhoneFeatures --> PhoneControls --> PhoneAnalysis --> PhoneFinding
    end

    IntegratedModel["Integrated model<br/>progesterone-linked acoustic-surface shift<br/>visible at whole-WAV scale,<br/>organized by phoneme classes,<br/>and stress-tested at phoneme-row granularity"]

    VoiceSource --> WholeExtract
    VoiceSource --> GroupSubset
    VoiceSource --> PhoneSubset

    PhaseContext -.-> WholeAnalysis
    PhaseContext -.-> GroupAnalysis
    PhaseContext -.-> PhoneAnalysis

    WholeFinding --> IntegratedModel
    GroupFinding --> IntegratedModel
    PhoneFinding --> IntegratedModel
```

**What this diagram answers:** how the project triangulates the same longitudinal
voice source from three interpretable angles: whole-WAV acoustics, phoneme
grouping/taxonomy, and granular phoneme-row controls.

# Data Flow Diagrams

Use the first diagram for the main presentation. The appendix diagrams expose
more implementation detail without making the main slide carry the full method.

## Main Slide: End-to-End Data Flow

```mermaid
flowchart LR
    APP["Mobile app<br/>daily voice tasks"] --> AW["Appwrite<br/>Storage + voice metadata"]
    AW --> WAV["SpeechFeatureExtraction<br/>downloads WAVs here"]

    WAV --> CANON["Canonical acoustics<br/>openSMILE eGeMAPSv02<br/>daily vowel + prosody features"]
    WAV --> PHON["Phoneme acoustics<br/>MFA boundaries + openSMILE frames<br/>phoneme-level feature table"]
    WAV --> SSL["SSL representation check<br/>HuBERT phone embeddings<br/>phonological d-prime"]

    CANON --> ANALYSIS["Cycle analysis<br/>join voice + hormones + wearable context"]
    PHON --> ANALYSIS
    SSL --> ANALYSIS

    CONTEXT["Physiology + context<br/>Inito hormones, Oura,<br/>cycle calendar"] --> ANALYSIS

    ANALYSIS --> OUT["Presentation outputs<br/>localization, controls,<br/>figures, mechanistic model"]

    classDef source fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e;
    classDef extraction fill:#fef9c3,stroke:#ca8a04,color:#713f12;
    classDef analysis fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef output fill:#f3e8ff,stroke:#7e22ce,color:#581c87;

    class APP,AW,CONTEXT source;
    class WAV,CANON,PHON,SSL extraction;
    class ANALYSIS analysis;
    class OUT output;
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

    classDef cloud fill:#e0f2fe,stroke:#0369a1,color:#0c4a6e;
    classDef qc fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    classDef data fill:#dcfce7,stroke:#16a34a,color:#14532d;
    classDef process fill:#fef9c3,stroke:#ca8a04,color:#713f12;

    class A,B cloud;
    class F,I,AUDIT qc;
    class J,K,L data;
    class C,D,E,G,H process;
```

**What this diagram answers:** how a saved mobile recording becomes the
recording-level audit and the daily eGeMAPS feature table used for the broad
cycle scan.

## Appendix B: Mechanistic Analysis Streams

```mermaid
flowchart LR
    VoiceSource["Cached Appwrite WAVs + metadata<br/>vowel and connected-speech tasks"]
    PhaseContext["Shared phase context<br/>cycle calendar + Inito hormones + Oura"]

    subgraph WholeRecording["STREAM 1: WHOLE-RECORDING / DAILY ACOUSTICS"]
        direction TB
        WholeExtract["openSMILE eGeMAPSv02 Functionals<br/>vowel + prosody recordings"]
        WholeDaily["Daily feature table<br/>per-day median vowel_egemaps_*<br/>and prosody_egemaps_*"]
        WholeAnalysis["Broad cycle scan<br/>follicular vs luteal effects,<br/>PdG coupling, Oura cross-check"]
        WholeFinding["Finding<br/>coherent cycle signal;<br/>cover/timbre channels move most"]
        WholeExtract --> WholeDaily --> WholeAnalysis --> WholeFinding
    end

    subgraph PhonemeGrain["STREAM 2: PHONEME-GRAIN ACOUSTICS"]
        direction TB
        PhonemeSubset["Clean connected-speech subset<br/>fixed Rainbow passage"]
        PhonemeAlign["MFA forced alignment<br/>phone boundaries in TextGrid"]
        PhonemeFeatures["openSMILE LLD frames<br/>assigned to phoneme windows"]
        PhonemeAnalysis["Phoneme analyses<br/>global shifts, de-meaned residuals,<br/>within-recording contrasts"]
        PhonemeFinding["Finding<br/>mostly global acoustic setting;<br/>diphthong + nasal residuals"]
        PhonemeSubset --> PhonemeAlign --> PhonemeFeatures --> PhonemeAnalysis --> PhonemeFinding
    end

    subgraph SSLSubspace["STREAM 3: SSL PHONOLOGICAL SUBSPACE"]
        direction TB
        SSLSubset["Same aligned connected-speech subset<br/>reuse MFA phone boundaries"]
        SSLEmbeddings["Frozen speech embeddings<br/>HuBERT phone vectors<br/>with backbone robustness checks"]
        SSLDPrime["D-prime by recording<br/>phonological contrast separability"]
        SSLAnalysis["Phase + hormone checks<br/>contrast effects and profile stability"]
        SSLFinding["Finding<br/>no broad phonological-subspace collapse"]
        SSLSubset --> SSLEmbeddings --> SSLDPrime --> SSLAnalysis --> SSLFinding
    end

    IntegratedModel["Integrated model<br/>progesterone-linked acoustic-surface shift<br/>mostly global across phoneme inventory<br/>with bounded residual structure"]

    VoiceSource --> WholeExtract
    VoiceSource --> PhonemeSubset
    VoiceSource --> SSLSubset

    PhaseContext -.-> WholeAnalysis
    PhaseContext -.-> PhonemeAnalysis
    PhaseContext -.-> SSLAnalysis

    WholeFinding --> IntegratedModel
    PhonemeFinding --> IntegratedModel
    SSLFinding --> IntegratedModel
```

**What this diagram answers:** how the project triangulates the same longitudinal
voice source from three angles: broad interpretable acoustics, phoneme-level
localization, and a self-supervised representation negative control.

# Phoneme Prosody Experiment: A Complete Guide

This document walks you through the phoneme prosody experiment pipeline - a system designed to extract acoustic features from individual phonemes (speech sounds) in recordings. The goal is to track subtle changes in speech patterns over time.

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [The Big Picture](#the-big-picture)
3. [Core Concepts](#core-concepts)
4. [File-by-File Breakdown](#file-by-file-breakdown)
5. [The Pipeline Flow](#the-pipeline-flow)
6. [Key Data Structures](#key-data-structures)
7. [Longitudinal Analysis](#longitudinal-analysis)
8. [Putting It All Together](#putting-it-all-together)

---

## What This System Does

Imagine you record someone reading a standardized passage (the "Rainbow Passage"). This system:

1. **Aligns** the audio to text, identifying exactly when each sound (phoneme) occurs
2. **Extracts** acoustic features from each tiny phoneme segment
3. **Assesses** the quality of each measurement
4. **Aggregates** data over time to track longitudinal changes

---

## The Big Picture

```mermaid
flowchart TB
    subgraph Input
        A[Audio Recording<br/>WAV file] 
        B[Transcription<br/>Rainbow Passage text]
    end
    
    subgraph "Forced Alignment"
        C[Montreal Forced Aligner<br/>MFA]
        D[TextGrid Output<br/>Phone boundaries]
    end
    
    subgraph "Phoneme Processing"
        E[Segment Feature Extraction<br/>openSMILE LLDs]
        F[Quality Assessment<br/>Good/Marginal/Poor]
    end
    
    subgraph "Rainbow Profile"
        G[Template Comparison<br/>Expected vs Observed timing]
    end
    
    subgraph Output
        H[Parquet Dataset<br/>Per-phoneme features]
        I[Phoneme Aggregates<br/>Daily summaries]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    E --> F
    F --> H
    D --> G
    G --> H
    H --> I
```

---

## Core Concepts

### What is a Phoneme?

A **phoneme** is the smallest unit of sound in speech. For example:
- The word "cat" has 3 phonemes: /K/ /AE/ /T/
- The word "rainbow" has 6 phonemes: /R/ /EY/ /N/ /B/ /OW/

This system uses **ARPAbet notation**, a standard way to represent American English phonemes:

| ARPAbet | Sound | Example |
|---------|-------|---------|
| AA | "ah" | f**a**ther |
| AE | "a" | c**a**t |
| M | "m" | **m**other |
| N | "n" | **n**ose |
| NG | "ng" | si**ng** |
| EY | "ay" | s**ay** |
| IY | "ee" | s**ee** |

### What is Forced Alignment?

**Forced alignment** takes an audio recording and its transcription, then figures out exactly when each word and phoneme occurs. Think of it like automatic subtitles, but at the phoneme level.

```mermaid
sequenceDiagram
    participant Audio as Audio Recording
    participant Text as "When the sunlight..."
    participant MFA as MFA Aligner
    participant TG as TextGrid
    
    Audio->>MFA: Audio signal
    Text->>MFA: Transcription
    MFA->>MFA: Acoustic model matching
    MFA->>TG: Time-aligned segments
    Note over TG: W: 0.10-0.18s<br/>EH: 0.18-0.25s<br/>N: 0.25-0.32s
```

### Why the Rainbow Passage?

The **Rainbow Passage** is a standardized text used in speech research. It contains all the phonemes of American English in natural contexts. Because everyone reads the same text, we can:

1. Compare the same phonemes across different speakers
2. Track the same phonemes for one speaker over time
3. Build "templates" of expected phoneme timing

---

## File-by-File Breakdown

### 1. `schema.py` - The Data Dictionary

This file defines what columns appear in the output dataset. Think of it as the "contract" for what data the pipeline produces.

```mermaid
classDiagram
    class OutputSchema {
        <<Phoneme Row>>
        +LINEAGE_FIELDS
        +ALIGNMENT_FIELDS
        +CONTEXT_FIELDS
        +BOUNDARY_FIELDS
        +RAINBOW_PROFILE_FIELDS
        +FEATURE_VALUE_FIELDS
        +FEATURE_QC_FIELDS
    }
    
    class LineageFields {
        recordingId
        userId
        recordedDate
        taskType
        audioHash
        extractorVersion
    }
    
    class AlignmentFields {
        phonemeIndex
        phonemeLabel
        wordLabel
        startSec
        endSec
        durationSec
        alignmentQuality
    }
    
    class FeatureFields {
        segment_mfcc2_mean
        segment_h1h2_mean
        segment_f1_bandwidth_mean
    }
    
    OutputSchema --> LineageFields
    OutputSchema --> AlignmentFields
    OutputSchema --> FeatureFields
```

**Key insight**: The schema is organized into logical groups:
- **Lineage**: Where did this data come from?
- **Alignment**: When does this phoneme occur?
- **Context**: What's before/after this phoneme?
- **Features**: What are the acoustic measurements?
- **QC**: Can we trust this measurement?

---

### 2. `taxonomy.py` - Phoneme Normalization and Classification

This file standardizes phone labels and assigns phoneme classes/tags.

Alignment uses the ARPAbet-native `english_us_arpa` models, so MFA emits
ARPAbet phones directly with optional stress markers (e.g., "AH0", "AH1").
`normalize_phoneme_label()`:
- Strips stress suffixes and uppercases (e.g., "AH0" -> "AH")
- Falls back to an IPA-to-ARPAbet map only as a defensive measure if a
  non-ARPAbet symbol ever appears

```python
normalize_phoneme_label("ah1")  # Returns "AH"
normalize_phoneme_label("M")    # Returns "M"
```

Canonical-label guard: `is_canonical_phoneme()` reports whether a normalized
label is in the ARPAbet inventory. The pipeline records this per row as
`qc_label_canonical`, so any unmapped phone is surfaced in QC instead of
silently leaking into `phonemeLabel`.

`classify_phoneme()` also assigns a granular `phonemeClassPrimary`, overlap
`phonemeClassTags` (nasal-coupled, pharyngeal-engaged, oral-anterior,
voiceless-frication), and a nasal `coarticulationContext` from the
neighboring phones.

---

### 3. `alignment.py` - MFA Integration

This file wraps the Montreal Forced Aligner (MFA), an external tool that does the heavy lifting of phoneme alignment. It uses the ARPAbet-native models so phones come out as ARPAbet directly:

```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

**Recorded transcription scope**: the prosody task records only sentences 2-3 of the Rainbow Passage (`PROSODY_CANONICAL_TRANSCRIPTION`): "The rainbow is a division of white light into many beautiful colors. These take the shape of a long round arch, with its path high above, and its two ends apparently beyond the horizon." Alignment also tries duration-based candidates if the canonical text does not align.

```mermaid
flowchart LR
    subgraph "align_recording()"
        A[Input WAV + ID] --> B{MFA Available?}
        B -->|No| C[Return Error]
        B -->|Yes| D{Models Downloaded?}
        D -->|No| E[Return Error]
        D -->|Yes| F[Create Temp Corpus]
        F --> G[Run MFA align]
        G --> H{Success?}
        H -->|No| I[Return Error]
        H -->|Yes| J[Parse TextGrid]
        J --> K[Return AlignmentResult]
    end
```

**What happens under the hood**:

1. Checks if MFA is installed and models are available
2. Creates a temporary "corpus" folder with the audio and transcription
3. Runs MFA's alignment command
4. Parses the resulting TextGrid file
5. Returns structured segment data

**The TextGrid format**: MFA outputs a Praat TextGrid file with tiers for words and phones. Each tier contains intervals with start time, end time, and label.

---

### 4. `alignment_quality.py` - Quality Assessment

Not all alignments are created equal. Short segments or unusual timing might indicate alignment errors. This module assesses quality.

```mermaid
flowchart TD
    subgraph "Quality Scoring"
        A[Segment] --> B[Duration Score]
        A --> C[Voiced Ratio Score]
        A --> D[Position Delta Score]
        
        B --> E[Combined Score]
        C --> E
        D --> E
        
        E --> F{Score >= 0.75}
        F -->|Yes| G[GOOD]
        F -->|No| H{Score >= 0.4}
        H -->|Yes| I[MARGINAL]
        H -->|No| J[POOR]
    end
    
    subgraph "Thresholds"
        K[min_duration_good: 40ms]
        L[min_duration_marginal: 25ms]
        M[max_position_delta_good: 5%]
    end
```

**Quality factors**:

1. **Duration**: Phonemes under 25ms are suspicious - they might be alignment artifacts
2. **Voiced Ratio**: For voiced sounds, we expect some voicing - low ratios suggest problems
3. **Position Delta**: If a phoneme occurs far from where we expect it in the passage, something might be wrong

---

### 5. `segment_features.py` - Acoustic Feature Extraction

This is where the actual acoustic analysis happens. openSMILE LLDs are
extracted **once over the whole recording** (`extract_recording_frames`), and
each phoneme then claims the frames whose center falls inside its trimmed
window (`aggregate_window`). Running openSMILE on per-phoneme audio slices is
avoided on purpose: its analysis windows need surrounding context, so a tiny
clip returns a single all-NaN "Segment too short" placeholder frame instead of
real features.

```mermaid
flowchart TB
    subgraph "Once per recording"
        R[Full recording WAV] --> S[openSMILE eGeMAPSv02<br/>LLD extraction]
        S --> T[Frame table<br/>one row per 10ms + center time]
    end

    subgraph "Per phoneme"
        A[Phoneme Boundaries<br/>0.100s - 0.180s] --> B[Apply Trim Policy<br/>default 0ms = whole phoneme]
        B --> D[Analysis Window]
        T --> F[Select frames with center in window]
        D --> F
        F --> I[Compute Aggregates<br/>Mean, Median]
    end

    subgraph "Output Features"
        I --> J[MFCC2: Spectral shape]
        I --> K[H1-H2: Voice quality]
        I --> L[F1 Bandwidth: Formant precision]
        I --> M[F0: Pitch]
    end
```

**Why the trim policy defaults to 0 ms?**

At phoneme transitions the acoustic signal is "blended" between sounds, so an optional trim can isolate the steady-state portion. However, the vowel-formant literature finds that averaging across the whole interval (the "Full" method) is the most reliable, and a fixed 20 ms-per-side trim would discard most of a typical 70 ms phoneme. So the default trim is 0 ms (whole-phoneme averaging), which also makes the 4-frame (40 ms) QC threshold act as a clean ">=40 ms" inclusion criterion. Coarticulation is captured separately via `coarticulationContext` rather than by trimming. A positive `trim_policy_ms` re-enables steady-state trimming; because frames come from the full-file extraction, trimming only narrows which frames a phoneme keeps, never the audio handed to openSMILE.

**Why a 4-frame minimum?** At a 10 ms hop, 4 frames = 40 ms, the minimum phoneme duration for reliable formant/spectral measurement in the phonetics literature (30 ms is only the forced-alignment floor). Shorter phones still get features computed but are flagged `qc_segment_ok = False`.

**Key features extracted**:

| Feature | What It Measures | Clinical Relevance |
|---------|------------------|-------------------|
| MFCC2 | Spectral envelope shape | Changes with resonance patterns |
| H1-H2 | Breathiness/pressed voice | Voice quality changes |
| F1 Bandwidth | First formant precision | Articulatory precision |
| F0 | Fundamental frequency | Pitch control |

---

### 6. `rainbow_profile.py` - Template Matching

The Rainbow Passage is standardized, so we know approximately where each phoneme should occur. This module compares observed timing against expected timing.

> Deferred for the MVP: `process_batch` does not build or pass a template, so the `rainbow*` columns are always `None` in the current output. The fields stay in the schema so the parquet shape is stable when template matching is enabled later. Enabling it first requires reconciling the occurrence-key ordering between `_occurrence_key` (raw segment order) and `summarize_alignment_against_template` (normalized, filtered, time-sorted).

```mermaid
flowchart LR
    subgraph "Template Building"
        A[Reference Recording] --> B[Align Phonemes]
        B --> C[Compute Position Ratios]
        C --> D[RainbowPassageTemplate]
    end
    
    subgraph "Template Comparison"
        E[New Recording] --> F[Align Phonemes]
        F --> G[Compute Observed Ratios]
        G --> H[Compare to Template]
        H --> I[Position Delta per Phoneme]
    end
    
    subgraph "Metrics"
        I --> J[Coverage Ratio<br/>% of expected phones found]
        I --> K[Sequence Match Ratio<br/>% in correct order]
        I --> L[Timing Consistency<br/>Within 10% tolerance?]
    end
```

**Position ratio**: Instead of absolute times, we use relative positions. The 50th phoneme at 10s in a 100s recording has position ratio 0.10. This accounts for different speaking rates.

---

### 7. `rainbow_inventory.py` - Expected Phoneme Sequence

This file contains the "ground truth" ARPAbet phoneme sequences. Because the prosody task records only sentences 2-3, coverage validation uses the canonical subset, while the full passage is kept for reference only.

```python
# Recorded subset (sentences 2-3) -> used by validate_phone_coverage().
PROSODY_CANONICAL_ARPABET_SEQUENCE = (
    # "The rainbow is a division of white light..."
    "DH", "AH", "R", "EY", "N", "B", "OW",
    # ... continues through "...beyond the horizon."
)

# Full passage -> reference only.
RAINBOW_PASSAGE_ARPABET_SEQUENCE = (...)
```

**Why this matters**:
- We can detect missing phonemes (maybe the speaker skipped a word)
- We can detect extra phonemes (maybe alignment hallucinated sounds)
- We can count expected occurrences for statistical power calculations

`validate_phone_coverage()` and `get_expected_phone_count()` operate on the recorded sentences-2-3 subset so coverage checks match what was actually spoken.

---

### 8. `biomarkers.py` - Longitudinal Analysis

Once we have per-phoneme features, we aggregate them to track changes over time.

```mermaid
flowchart TB
    subgraph "Daily Aggregation"
        A[Per-Phoneme Features] --> B[Group by User + Date + Phoneme]
        B --> C[DailyPhonemeAggregate]
    end
    
    subgraph "Trajectory Building"
        C --> D[Group by User + Phoneme]
        D --> E[Sort by Date]
        E --> F[PhonemeTrajectory<br/>Values over time]
    end
    
    subgraph "Reliability Checks"
        C --> G[Count observations]
        G --> H{>= 5 observations?}
        H -->|Yes| I[qc_reliable = True]
        H -->|No| J[qc_reliable = False]
    end
```

---

### 9. `pipeline.py` - The Orchestrator

This ties everything together into a single processing flow.

```mermaid
flowchart TB
    subgraph "process_recording()"
        A[RecordingMetadata] --> B[Compute Audio Hash]
        B --> C[align_recording]
        C --> D{Success?}
        D -->|No| E[Return Empty]
        D -->|Yes| F[For Each Segment]
        
        F --> G[Normalize Phoneme Label]
        F --> H[aggregate_window]
        F --> I[assess_segment_quality]
        F --> J[Match Rainbow Template]
        
        G --> K[Build PhonemeRowData]
        H --> K
        I --> K
        J --> K
        
        K --> L[Collect All Rows]
    end
    
    subgraph "process_batch()"
        M[List of Recordings] --> N[For Each Recording]
        N --> O[process_recording]
        O --> P[Accumulate Rows]
        P --> Q[Convert to DataFrame]
        Q --> R[Write Parquet]
    end
```

---

## The Pipeline Flow

Here's the complete data flow from audio to aggregates:

```mermaid
flowchart TB
    subgraph "Stage 1: Input"
        A[WAV Audio File]
        B[Recording Metadata<br/>user_id, date, etc.]
    end
    
    subgraph "Stage 2: Alignment"
        C[MFA Forced Alignment]
        D[TextGrid with Phone Boundaries]
    end
    
    subgraph "Stage 3: Per-Phoneme Processing"
        E[Phoneme Label Normalization]
        F[Boundary Trimming]
        G[openSMILE Feature Extraction]
        H[Quality Assessment]
        I[Rainbow Template Matching]
    end
    
    subgraph "Stage 4: Output"
        J[PhonemeRowData Objects]
        K[prosody_phoneme_features.parquet]
    end
    
    subgraph "Stage 5: Aggregates"
        L[Daily Phoneme Aggregates]
        M[Phoneme Trajectories]
    end
    
    A --> C
    B --> C
    C --> D
    D --> E
    D --> F
    F --> G
    G --> H
    D --> I
    E --> J
    G --> J
    H --> J
    I --> J
    J --> K
    K --> L
    L --> M
```

---

## Key Data Structures

### PhonemeRowData

One row per aligned phoneme in the output dataset:

```mermaid
classDiagram
    class PhonemeRowData {
        <<dataclass>>
        %% Lineage
        +str recordingId
        +str userId
        +str recordedDate
        +str taskType
        +str audioHash
        +str extractorVersion
        +str alignmentEngine
        +str alignmentVersion
        
        %% Alignment
        +int phonemeIndex
        +str phonemeLabel
        +str wordLabel
        +float startSec
        +float endSec
        +float durationSec
        +float alignmentScoreRaw
        +str alignmentQuality
        
        %% Context
        +str prevPhonemeLabel
        +str nextPhonemeLabel
        
        %% Features
        +float segment_mfcc2_mean
        +float segment_h1h2_mean
        +float segment_f1_bandwidth_mean
        +float segment_f0_mean
        
        %% QC
        +bool qc_segment_ok
        +str qc_segment_reason
        +int qc_numFrames
    }
```

### AlignmentResult

Returned from the MFA alignment step:

```mermaid
classDiagram
    class AlignmentResult {
        <<dataclass>>
        +str recording_id
        +Path audio_path
        +Path textgrid_path
        +tuple~AlignedPhonemeSegment~ segments
        +tuple~WordSegment~ word_segments
        +str alignment_engine
        +str alignment_version
        +bool success
        +str error_message
    }
    
    class AlignedPhonemeSegment {
        <<dataclass>>
        +str phoneme_label
        +float start_sec
        +float end_sec
    }
    
    class WordSegment {
        <<dataclass>>
        +str word
        +float start_sec
        +float end_sec
    }
    
    AlignmentResult "1" --> "*" AlignedPhonemeSegment
    AlignmentResult "1" --> "*" WordSegment
```

---

## Longitudinal Analysis

### Per-Phoneme Tracking

By collecting data over days/weeks/months, we can detect trends for individual phonemes:

```mermaid
gantt
    title Phoneme /M/ MFCC2 Trajectory Example
    dateFormat  YYYY-MM-DD
    section MFCC2 Values
    Day 1 (12.3)    :done, 2024-01-01, 1d
    Day 2 (12.1)    :done, 2024-01-02, 1d
    Day 3 (11.8)    :done, 2024-01-03, 1d
    Day 4 (11.5)    :active, 2024-01-04, 1d
    Day 5 (11.2)    :crit, 2024-01-05, 1d
```

### Reliability Flags

Not all aggregates are equally reliable:

| Observations | Reliability | Reason |
|--------------|-------------|--------|
| >= 5 | Reliable | Sufficient data for stable mean |
| < 5 | Unreliable | High variance, interpret with caution |

---

## Putting It All Together

### Example: Processing One Recording

```python
from speech_feature_extraction.phoneme_prosody_experiment import (
    RecordingMetadata,
    process_recording,
    SegmentFeatureExtractor,
)
from pathlib import Path

# 1. Define the recording
metadata = RecordingMetadata(
    recording_id="rec_001",
    user_id="user_123",
    recorded_date="2024-01-15",
    task_type="rainbow_passage",
    audio_path=Path("/path/to/recording.wav"),
)

# 2. Process it
rows = process_recording(
    metadata=metadata,
    alignments_dir=Path("/output/alignments"),
)

# 3. Each row is one phoneme with full features
for row in rows[:5]:
    print(f"{row.phonemeLabel}: {row.startSec:.3f}s - {row.endSec:.3f}s")
    print(f"  MFCC2: {row.segment_mfcc2_mean}")
    print(f"  Quality: {row.alignmentQuality}")
```

### Example: Computing Trajectories

```python
import pandas as pd
from speech_feature_extraction.phoneme_prosody_experiment import (
    compute_daily_phoneme_aggregates,
    compute_phoneme_trajectories,
)

# Load the extracted features
df = pd.read_parquet("prosody_phoneme_features.parquet")

# Daily aggregates per phoneme
daily_phonemes = compute_daily_phoneme_aggregates(df)

# Longitudinal trajectories
trajectories = compute_phoneme_trajectories(daily_phonemes)

for traj in trajectories:
    if traj.phoneme_label == "M":
        print(f"User {traj.user_id} - /M/ over {traj.total_days} days")
        print(f"  MFCC2 trend: {traj.mfcc2_values}")
        print(f"  Reliable days: {traj.reliable_days}/{traj.total_days}")
```

---

## Summary

The phoneme prosody experiment pipeline extracts acoustic features from speech at the phoneme level. It combines:

1. **Forced alignment** (MFA) to locate phoneme boundaries
2. **Phoneme normalization** for consistent ARPAbet labels
3. **Acoustic feature extraction** (openSMILE) for measurements
4. **Quality assessment** to flag unreliable data
5. **Template matching** for standardized passage comparison
6. **Phoneme aggregation** for longitudinal tracking

The modular design allows each component to be tested and improved independently while maintaining a clean data contract through the schema definitions.

### Future Extensions

The current implementation already includes per-phoneme features, phoneme
groupings (`phonemeClassPrimary`/`phonemeClassTags`), and nasal
`coarticulationContext`. Future versions could add:

- **Rainbow template matching** wired into `process_batch` (currently deferred; `rainbow*` fields are `None`)
- **Clinical biomarkers** derived from phoneme class contrasts
- **ComParE_2016** or Praat/Parselmouth as additional feature extractors

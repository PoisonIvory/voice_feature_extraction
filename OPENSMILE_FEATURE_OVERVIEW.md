# openSMILE Feature Overview (Current Pipeline)

This project currently extracts **openSMILE eGeMAPSv02 Functionals** at recording level:

- Feature set: `opensmile.FeatureSet.eGeMAPSv02`
- Feature level: `opensmile.FeatureLevel.Functionals`
- Expected feature count: `88`
- Sampling rate: `16000` Hz (`resample=True`)
- Prefix in output parquet: `egemaps_`

This aligns with the canonical first-pass scope in `USER_STORIES.md` (eGeMAPSv02 as primary interpretable set, ComParE as deferred optional).

## Feature Domains and Counts

- **Pitch (F0) statistics**: 10
- **Loudness statistics**: 10
- **General spectral + MFCC (voiced+unvoiced frames)**: 10
- **Voice quality / perturbation (jitter, shimmer, HNR, harmonics)**: 10
- **Formant-related (F1-F3 frequency/bandwidth/amplitude)**: 18
- **Voiced-only spectral/MFCC (`*V*`)**: 18
- **Unvoiced-only spectral (`*UV*`)**: 5
- **Temporal/segment cadence features**: 6
- **Equivalent sound level**: 1

Total: **88**

## Full Feature List (eGeMAPSv02 Functionals)

### 1) Pitch (F0)

- `F0semitoneFrom27.5Hz_sma3nz_amean`
- `F0semitoneFrom27.5Hz_sma3nz_stddevNorm`
- `F0semitoneFrom27.5Hz_sma3nz_percentile20.0`
- `F0semitoneFrom27.5Hz_sma3nz_percentile50.0`
- `F0semitoneFrom27.5Hz_sma3nz_percentile80.0`
- `F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2`
- `F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope`
- `F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope`
- `F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope`
- `F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope`

### 2) Loudness

- `loudness_sma3_amean`
- `loudness_sma3_stddevNorm`
- `loudness_sma3_percentile20.0`
- `loudness_sma3_percentile50.0`
- `loudness_sma3_percentile80.0`
- `loudness_sma3_pctlrange0-2`
- `loudness_sma3_meanRisingSlope`
- `loudness_sma3_stddevRisingSlope`
- `loudness_sma3_meanFallingSlope`
- `loudness_sma3_stddevFallingSlope`

### 3) General Spectral and MFCC

- `spectralFlux_sma3_amean`
- `spectralFlux_sma3_stddevNorm`
- `mfcc1_sma3_amean`
- `mfcc1_sma3_stddevNorm`
- `mfcc2_sma3_amean`
- `mfcc2_sma3_stddevNorm`
- `mfcc3_sma3_amean`
- `mfcc3_sma3_stddevNorm`
- `mfcc4_sma3_amean`
- `mfcc4_sma3_stddevNorm`

### 4) Voice Quality and Perturbation

- `jitterLocal_sma3nz_amean`
- `jitterLocal_sma3nz_stddevNorm`
- `shimmerLocaldB_sma3nz_amean`
- `shimmerLocaldB_sma3nz_stddevNorm`
- `HNRdBACF_sma3nz_amean`
- `HNRdBACF_sma3nz_stddevNorm`
- `logRelF0-H1-H2_sma3nz_amean`
- `logRelF0-H1-H2_sma3nz_stddevNorm`
- `logRelF0-H1-A3_sma3nz_amean`
- `logRelF0-H1-A3_sma3nz_stddevNorm`

### 5) Formant-Related (Vocal Tract Resonance Proxies)

- `F1frequency_sma3nz_amean`
- `F1frequency_sma3nz_stddevNorm`
- `F1bandwidth_sma3nz_amean`
- `F1bandwidth_sma3nz_stddevNorm`
- `F1amplitudeLogRelF0_sma3nz_amean`
- `F1amplitudeLogRelF0_sma3nz_stddevNorm`
- `F2frequency_sma3nz_amean`
- `F2frequency_sma3nz_stddevNorm`
- `F2bandwidth_sma3nz_amean`
- `F2bandwidth_sma3nz_stddevNorm`
- `F2amplitudeLogRelF0_sma3nz_amean`
- `F2amplitudeLogRelF0_sma3nz_stddevNorm`
- `F3frequency_sma3nz_amean`
- `F3frequency_sma3nz_stddevNorm`
- `F3bandwidth_sma3nz_amean`
- `F3bandwidth_sma3nz_stddevNorm`
- `F3amplitudeLogRelF0_sma3nz_amean`
- `F3amplitudeLogRelF0_sma3nz_stddevNorm`

### 6) Voiced-Only Spectral/MFCC

- `alphaRatioV_sma3nz_amean`
- `alphaRatioV_sma3nz_stddevNorm`
- `hammarbergIndexV_sma3nz_amean`
- `hammarbergIndexV_sma3nz_stddevNorm`
- `slopeV0-500_sma3nz_amean`
- `slopeV0-500_sma3nz_stddevNorm`
- `slopeV500-1500_sma3nz_amean`
- `slopeV500-1500_sma3nz_stddevNorm`
- `spectralFluxV_sma3nz_amean`
- `spectralFluxV_sma3nz_stddevNorm`
- `mfcc1V_sma3nz_amean`
- `mfcc1V_sma3nz_stddevNorm`
- `mfcc2V_sma3nz_amean`
- `mfcc2V_sma3nz_stddevNorm`
- `mfcc3V_sma3nz_amean`
- `mfcc3V_sma3nz_stddevNorm`
- `mfcc4V_sma3nz_amean`
- `mfcc4V_sma3nz_stddevNorm`

### 7) Unvoiced-Only Spectral

- `alphaRatioUV_sma3nz_amean`
- `hammarbergIndexUV_sma3nz_amean`
- `slopeUV0-500_sma3nz_amean`
- `slopeUV500-1500_sma3nz_amean`
- `spectralFluxUV_sma3nz_amean`

### 8) Temporal/Segment Cadence

- `loudnessPeaksPerSec`
- `VoicedSegmentsPerSec`
- `MeanVoicedSegmentLengthSec`
- `StddevVoicedSegmentLengthSec`
- `MeanUnvoicedSegmentLength`
- `StddevUnvoicedSegmentLength`

### 9) Global Level

- `equivalentSoundLevel_dBp`

## Additional QC Metrics Computed in This Repo (Not in the 88)

The pipeline also computes LLD-derived QC metrics:

- `voiced_ratio`
- `f0_cov`
- `jitter_mean`
- `shimmer_db_mean`
- `total_frames`
- `voiced_frames`

These are for task quality control and are not part of the canonical 88 eGeMAPSv02 functionals.

## Optional Geometry-Derived Feature Block

The extractor now supports an optional second feature block built from the existing
eGeMAPS formant means (`F1/2/3frequency_sma3nz_amean`).

- Enable in CLI with `extract --include-geometry-derived`
- Output prefix: `egemaps_geom_`
- Added fields:
  - `egemaps_geom_f1_f2_delta_hz_amean`
  - `egemaps_geom_f2_f3_delta_hz_amean`
  - `egemaps_geom_f1_f3_delta_hz_amean`
  - `egemaps_geom_f2_f1_ratio_amean`
  - `egemaps_geom_f3_f2_ratio_amean`
  - `egemaps_geom_f3_f1_ratio_amean`
  - `egemaps_geom_formant_spacing_hz_amean`
  - `egemaps_geom_apparent_vtl_cm_amean`

The VTL value is an acoustic proxy from formant spacing (`VTL ~= c / (2*dF)`), not an anatomical measurement.

## Literature-Grounded Assessment

## Is 88 "too light"?

For a reproducible, interpretable baseline, **88 eGeMAPS functionals is a strong base**, not underpowered by default:

- eGeMAPS is explicitly designed as a compact expert-curated clinical/paralinguistic feature set.
- Reviews of neurological and psychiatric voice biomarkers frequently treat GeMAPS/eGeMAPS as standard baseline sets.
- openSMILE documentation positions ComParE (`6373`) as broad/high-dimensional and eGeMAPS (`88`) as compact/interpretable.

## Geometry / vocal-tract changes: what you already capture

You already include indirect geometric proxies:

- Formant frequencies: `F1frequency`, `F2frequency`, `F3frequency`
- Formant bandwidths: `F1/2/3bandwidth`
- Formant amplitudes relative to source: `F1/2/3amplitudeLogRelF0`

These are relevant because vocal-tract resonances (formants) shift with vocal-tract shape/length changes.

## Likely missing if geometry is a priority

If your specific research question is geometric/articulatory change tracking, gaps are mainly in **derived metrics**, not necessarily a missing base extractor:

- Explicit formant ratios/intervals (e.g., `F2/F1`, `F3/F2`)
- Formant dispersion measures (`F2-F1`, `F3-F2`, average spacing)
- VTL-oriented estimates from multi-formant spacing (apparent vocal tract length proxies)
- Task-conditioned vowel-space metrics (if multiple controlled vowels are available)
- Dynamic formant trajectory features beyond summary functionals

## Recommendation for this repository

Given current scope and `USER_STORIES.md`, keep the current canonical dataset on eGeMAPSv02 and consider:

1. **Keep eGeMAPSv02 as the default production baseline** (current choice is methodologically solid and reproducible).
2. **Add a second, optional "geometry-derived" feature block** computed from existing formant outputs (simple and interpretable; no explosion to 6k+ dimensions).
3. **Add optional ComParE_2016 experiments only after baseline stability** (already aligned with your methodology story).

This keeps the main pipeline simple and auditable while adding articulatory sensitivity where your research question needs it.

## Sources Reviewed

- openSMILE Python README (feature-set sizes and supported levels): [audeering/opensmile-python README](https://raw.githubusercontent.com/audeering/opensmile-python/main/README.rst)
- Neurological disorder voice review (GeMAPS/ComParE commonly used): [Frontiers 2022 systematic review](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.842301/full)
- Depression voice biomarker systematic review (prosodic/spectral/perturbation markers): [Journal of Voice 2025 review PDF](https://orbi.umons.ac.be/bitstream/20.500.12907/53444/1/1-s2.0-S0892199725001870-main.pdf)
- Formant/VTL methodology and cautions for geometric inference: [Behavior Research Methods practical guide](https://pmc.ncbi.nlm.nih.gov/articles/PMC11525281/)
- Example benchmarking eGeMAPS vs larger sets in dementia pipeline: [Automatic Detection of Alzheimer’s Disease Using Spontaneous Speech](https://pmc.ncbi.nlm.nih.gov/articles/PMC9056005/)

# Voice-Cycle Data Story Brief

## Core Message

Menstrual-cycle voice change is measurable, biologically coherent, and localized: in this dataset, the dominant signal behaves like a progesterone-linked **phonatory/filter setting** (vocal-fold cover plus timbre channel), not a reorganization of vocal-tract geometry or phonological category structure.

## Talk Goal (10-12 minutes)

Present one defensible arc:

1. **Wide detection:** a real cycle-linked voice signal exists.
2. **Narrow localization:** the signal is strongest in source/cover and timbre channels, with tract geometry mostly spared.
3. **Mechanistic stress tests:** phoneme-grain and SSL-representation analyses show where the signal is and where it is not.
4. **Translational implication:** voice can be framed as a non-invasive, ecologically valid biomarker candidate for hormone-sensitive states.

## Storyline Structure (Big Picture -> Narrow)

### Act 1: Why this question matters

- Voice is a multisystem output shaped by respiratory, phonatory, articulatory, and neural control.
- Hormonal transitions are known to affect tissue, autonomic state, and CNS modulation, but cycle-linked voice evidence is still sparse and often calendar-only.
- This project asks: **where in speech does cycle-linked variation actually live?**

### Act 2: Dataset and design strengths

- Broad longitudinal layer: `59` voice days across `9` cycles, with `29` voice+hormone overlap days and wearable context.
- Mechanistic layer: `71` clean connected-speech recordings over `53` days, with `2` phase-balanced cycles (`2026-01`, `2026-02`) used as the strongest internal replication lens.
- Multi-source triangulation: voice + measured hormones (E3G/PdG) + Oura + cycle calendar.
- Multi-grain triangulation: whole-recording eGeMAPS -> phoneme-level acoustics -> SSL phonological d-prime.

### Act 3: Wide scan found a coherent signal

From `REPORT.md` (independent broad analysis):

- Largest follicular-vs-luteal effects concentrated in peripheral/mucosal features (for example shimmer variability `g=0.97`, H1-A3 `g=0.73`), while mean F0 behaved comparatively stable.
- Composite instability index increased from menstrual/early-follicular to late-luteal (means: menstrual `-0.37`, early-follicular `-0.09`, peri-ovulatory `0.21`, early-luteal `0.16`, late-luteal `0.30` z).
- PdG tracked central/peripheral voice metrics (for example pitch range rho `-0.52`, partial `-0.37` after temperature control).
- Oura independent cross-check matched expected cycle physiology (temperature deviation `g=1.14`; HRV `g=-0.46` in luteal).

Interpretation: there is a biologically plausible signal worth mechanistic narrowing, not isolated feature noise.

### Act 4: Localization analysis isolated where the signal lives

From `LOCALIZATION_FINDINGS.md` + tables:

- **Reed vs tube split:** focused dissociation test significant in both tasks:
  - Prosody: `p=0.0078`
  - Vowel: `p=0.0488`
- **Sensitivity-floor stress test:** measurement system clearly detects large formant motion unrelated to cycle, yet cycle-shift in F1/F2 is tiny:
  - F1 within-recording variability / cycle shift ratio: `~105x`
  - F2 within-recording variability / cycle shift ratio: `~36x`
- **Hormone attribution:** average absolute partial coupling stronger for peripheral vs central under PdG:
  - Peripheral PdG: `0.237`
  - Central PdG: `0.172`
- **Premenstrual concentration (directional):**
  - MFCC2 premenstrual minus follicular: `+1.32 z`
  - Pitch variability premenstrual minus follicular: `+0.83 z`

Interpretation: strongest evidence supports a cover/phonatory mechanism with central-control modulation layered on top.

### Act 5: Phoneme-grain analysis separated global setting from segment-specific residuals

From `PHONEME_PROSODY_FINDINGS.md` + tables:

- **Global setting result:** luteal shifts were broad and highly reproducible across balanced cycles:
  - MFCC2 (voiced): `+1.42` / `+1.34 SD` (Jan/Feb)
  - H1-H2 (voiced): `+0.77` / `+0.71 SD`
- **After per-recording de-meaning (key confound control):**
  - Most class effects collapse toward zero.
  - Two residuals survive:
    - Diphthong H1-H2 residual: Cliff's delta `0.312`, `q=0.023`
    - Nasal MFCC2 residual: Cliff's delta `0.361`, `q=0.049`
- **Within-recording contrasts mostly phase-flat**, supporting "global setting, not broad category reorganization."
- **Held-out cycle phase decode:** balanced accuracy `0.877` with full phoneme profile vs `0.846` with global means only -> profile adds a small increment beyond global signal.

Interpretation: cycle effect is mostly recording-global with limited, mechanistically plausible phoneme-specific residuals.

### Act 6: SSL phonological-subspace analysis served as a negative control

From `HUBERT_SUBSPACE_FINDINGS.md` + tables:

- HuBERT-base composite consonant d-prime effect: Cliff's delta `-0.121` (negligible), `q=0.836`.
- `0/8` contrasts survive BH-FDR in any of 3 backbones.
- Cross-backbone profile cosine remained high (`min=0.959`), so the null pattern is architecture-stable.
- Directional (non-significant) privileged trends still align with mechanism priors:
  - Nasality vs PdG partial rho `-0.337`
  - Voicing vs PdG partial rho `+0.242`

Interpretation: cycle-linked change appears stronger in acoustic surface channels than in phonological separability structure, which is exactly what a specificity control should show.

### Act 7: Integrated interpretation

The convergent model:

- **Primary signal:** progesterone-linked shift in phonatory/cover plus timbre channel.
- **Spatial pattern:** mostly global across phoneme inventory.
- **Residual structure:** diphthong open-quotient and nasal timbre enrichments.
- **What is spared:** tract-geometry changes are small relative to measurement sensitivity; phonological separability remains largely stable.
- **Clinical-translation hypothesis:** voice may index hormone-sensitive states with a combination of global and targeted features.

## Strongest Findings vs Directional Findings

### High-confidence (best for main claims)

- Dissociation evidence for moved vs spared feature sets (prosody and vowel).
- Sensitivity-floor demonstration that "geometry flatness" is not an instrument artifact.
- Reproducible Jan/Feb within-cycle shifts for global MFCC2/H1-H2 channels.
- Multi-backbone null for phonological-subspace collapse.

### Directional / hypothesis-generating (present with caveats)

- Premenstrual amplification in central-control-linked measures.
- PdG rate-of-change coupling to cover variability.
- Privileged d-prime hormone trends (nasality down, voicing up with PdG).

## Novel Contributions to Emphasize

1. **Three-layer triangulation** across interpretable acoustics, phoneme-grain decomposition, and frozen-SSL representational metrics.
2. **Confound-aware localization protocol** (within-cycle normalization + per-recording de-meaning + within-recording contrasts).
3. **Negative-control framing**: showing where expected effects do *not* appear strengthens causal localization.
4. **Measured-hormone integration** (not calendar-only), directly addressing a known gap in related cycle-voice work.

## How to Position Relative to Closely Related Work

Use the Kervin et al. 2025 thread as context:

- Agreement: both datasets support cycle-linked glottal/cover-linked movement.
- Advancement here: measured hormones + multi-grain analytic decomposition.
- Refinement: this project supports a "rate/sensitivity-aware" interpretation but keeps it directional when sample is limited.

Use this sentence:

> "This study extends prior daily voice-cycle work by adding measured hormones and by explicitly separating global phonatory shifts from phoneme-specific and representational effects."

## How to Frame for Professor Patel / VOxx Priorities

Anchor to the VOxx language:

- Voice as a **dynamic, non-invasive biomarker**.
- **Everyday/ecological recording** context.
- Integration of **acoustics + AI + physiological context data**.
- Focus on **subtle biological transitions** and early signal detection.

Anchor to Patel's prosody/acoustics lineage:

- Explicitly call out prosody-sensitive features (pitch range, spectral tilt, timing proxies).
- Emphasize interpretable source-filter grounding and clear controls for over-claiming.
- Highlight translational path: biomarker discovery -> longitudinal monitoring -> personalized intervention hypotheses.

## Likely Expert Questions and Tight Answers

### "How do you know this is not recording-condition drift?"

- Between-cycle drift handled with within-cycle normalization and date-partial analyses.
- Per-recording de-meaning and within-recording contrasts explicitly cancel recording-level offsets.
- Multi-method convergence (whole-file, phoneme, and SSL negative control) argues against a single nuisance explanation.

### "Why trust any null claims with this sample?"

- Null claims are not from single p-values; they are supported by sensitivity-floor checks and architecture-stable patterns.
- The analysis distinguishes "no detectable effect in this channel" from "no biological effect anywhere."

### "What is clinically meaningful today?"

- Not a diagnostic claim yet.
- The immediate result is a mechanistic map plus testable endpoints for prospective replication.

## Figure Sequence for Presentation

Recommended visual order (all source tables are present in `Analysis/outputs/.../tables`):

1. Coverage and overlap (from `localization/tables/summary.json` and phase-balance fields).
2. Localization map (from `localization/tables/feature_effects.csv`).
3. Sensitivity floor (from `localization/tables/sensitivity_floor.csv`).
4. Hormone pathways (from `localization/tables/hormone_coupling.csv` and `peripheral_vs_central.csv`).
5. Rate variability (from `localization/tables/variability_by_pdg_rate.csv`).
6. Premenstrual window (from `localization/tables/premenstrual_window.csv`).
7. Phoneme localization forest (from `phoneme/tables/localization.csv`).
8. Diphthong residual check (from `phoneme/tables/diphthong_f0_residual.csv`).
9. Within-recording contrasts (from `phoneme/tables/within_recording_contrasts.csv`).
10. HuBERT phase forest (from `hubert/tables/phase_contrasts.csv`).
11. Cross-backbone robustness (from `hubert/tables/inter_backbone_rho.csv` and `profile_cosine.csv`).

If you want pre-rendered images before presenting, run:

`python -m src.analysis.localization.figures`

`python -m src.analysis.phoneme.figures`

`python -m src.analysis.hubert.figures`

## One-Slide Claim Ladder (for discussion slide)

- **Claim 1 (strong):** Cycle-linked voice variation is present and coherent in this participant.
- **Claim 2 (strong):** Variation localizes primarily to cover/timbre channels, with geometry largely spared.
- **Claim 3 (moderate):** Signal is predominantly global across phoneme inventory with two localized residual enrichments.
- **Claim 4 (moderate):** Representational separability remains mostly stable, supporting acoustic-surface specificity.
- **Claim 5 (directional):** Late-luteal and rate-of-change effects suggest sensitivity dynamics worth prospective testing.

## Immediate Next Studies (if asked)

1. Replicate with additional participants and per-cycle phase balance as the first design constraint.
2. Pre-register diphthong H1-H2 and nasal MFCC2 residual endpoints.
3. Add direct nasal patency/nasalance measures to test the nasal residual mechanism.
4. Add structured symptom scales to directly link central-channel changes to symptom windows.
5. Keep fixed-passage token control for SSL analyses to preserve d-prime interpretability.

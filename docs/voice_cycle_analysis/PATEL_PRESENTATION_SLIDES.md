# Voice-Cycle Presentation Deck (Markdown Script)

Use this as a slide-build blueprint. Keep one primary visual per slide, one headline claim, and 2-3 supporting bullets max.

If you want pre-rendered figures before slide assembly, run in `Analysis`:

`python -m src.analysis.localization.figures`

`python -m src.analysis.phoneme.figures`

`python -m src.analysis.hubert.figures`

## Slide 1 — Title and Thesis

**Title:** Where does menstrual-cycle variation live in voice?

**Headline claim:** The dominant signal behaves like a progesterone-linked phonatory/filter setting, not a broad reorganization of tract geometry or phonological separability.

**Visual:** No plot (clean title slide).

**Talk track (20-30 sec):**
- This talk moves from broad detection to mechanistic localization.
- The key result is not just that voice changes, but where the change appears and where it does not.

## Slide 2 — Why this question is timely

**Headline claim:** Voice is a multisystem health signal, and hormone-linked transitions are still under-characterized.

**Visual:** Short concept diagram (source-filter + central control), or text-only with three pathways.

**Bullets:**
- Voice reflects respiratory, phonatory, articulatory, and neural systems.
- Hormonal transitions likely affect both tissue state and control dynamics.
- Goal: locate cycle-linked effects with interpretable, confound-aware analysis.

## Slide 3 — Data and design

**Headline claim:** The project combines dense longitudinal sampling with measured hormones and wearable context.

**Visual:** Use the main-slide diagram in `DATA_FLOW_DIAGRAMS.md`, optionally followed by the coverage timeline from broad analysis plus a small stats panel.

**Numbers to show:**
- `59` voice days across `9` cycles (broad layer).
- `71` clean recordings, `53` days, `2` phase-balanced cycles (mechanistic layer).
- `29` voice+hormone overlap days.

**Talk track (30 sec):**
- Emphasize repeated within-person design and phase-balanced-cycle logic.

## Slide 4 — Wide scan: signal is real and structured

**Headline claim:** The earliest full-feature sweep identified coherent cycle-linked movement, concentrated away from mean pitch architecture.

**Visual source table:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/feature_effects.csv` (rank by absolute effect size and color by axis).

**Numbers to annotate:**
- Example large effects: shimmer variability `g=0.97`, H1-A3 `g=0.73`.
- Instability index mean rises to late-luteal peak (`0.30 z`).

## Slide 5 — Localization: reed vs tube

**Headline claim:** Dissociation testing supports stronger movement in cover/timbre-linked channels than in geometry-linked channels.

**Visual source table:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/feature_effects.csv` (forest plot grouped by family).

**Numbers to annotate:**
- Focused dissociation p-values: prosody `0.0078`, vowel `0.0488`.

**Talk track (35 sec):**
- Explain moved vs spared set logic.
- State this as strongest mechanistic evidence.

## Slide 6 — Sensitivity-floor check ("dog that did not bark")

**Headline claim:** Geometry flatness is not a measurement artifact.

**Visual source table:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/sensitivity_floor.csv`.

**Numbers to annotate:**
- F1 detectable movement ratio: `~105x` larger within-recording than cycle shift.
- F2 detectable movement ratio: `~36x`.

**Talk track (30 sec):**
- The system clearly detects large formant changes in other contexts, yet cycle-related F1/F2 shifts stay minimal.

## Slide 7 — Hormone attribution and pathway split

**Headline claim:** Progesterone-linked coupling is stronger than estrogen for key peripheral channels.

**Visual source tables:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/hormone_coupling.csv` and `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/peripheral_vs_central.csv`.

**Numbers to annotate:**
- Mean abs partial rho (PdG): peripheral `0.237` vs central `0.172`.
- Example strong PdG couplings: MFCC2 partial rho `0.408`, pitch range partial rho `-0.494`.

## Slide 8 — Dynamics: rate and premenstrual concentration

**Headline claim:** Directional evidence suggests a sensitivity dynamic, with stronger expression near the late-luteal window.

**Visual option A source:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/variability_by_pdg_rate.csv`

**Visual option B source:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/localization/tables/premenstrual_window.csv`

**Numbers to annotate:**
- Premenstrual minus follicular: MFCC2 `+1.32 z`; pitch variability `+0.83 z`.

**Talk track (25 sec):**
- Present as directional/hypothesis-generating, not definitive.

## Slide 9 — Phoneme-grain decomposition

**Headline claim:** Most of the cycle signal is global across phoneme inventory; two residual enrichments survive strict controls.

**Visual source table:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/phoneme/tables/localization.csv`.

**Numbers to annotate:**
- Global within-cycle shifts:
  - MFCC2 voiced: `+1.42` / `+1.34 SD`
  - H1-H2 voiced: `+0.77` / `+0.71 SD`
- Residuals after de-meaning:
  - Diphthong H1-H2: delta `0.312`, `q=0.023`
  - Nasal MFCC2: delta `0.361`, `q=0.049`

## Slide 10 — SSL representational negative control

**Headline claim:** Cycle effects do not present as broad phonological-subspace collapse.

**Visual source tables:** `/Users/ivyhamilton/Decibelle/Analysis/outputs/hubert/tables/phase_contrasts.csv` and `/Users/ivyhamilton/Decibelle/Analysis/outputs/hubert/tables/profile_cosine.csv`.

**Numbers to annotate:**
- Consonant composite: delta `-0.121`, `q=0.836`.
- `0/8` contrasts BH-significant across all backbones.
- Profile cosine minimum `0.959` across backbones.

**Talk track (30 sec):**
- This is a specificity result: acoustic surface shifts do not imply representational collapse.

## Slide 11 — Integrated model and confidence ladder

**Headline claim:** Convergent evidence supports a localized, mostly global acoustic-surface mechanism with bounded residual structure.

**Visual:** Simple model diagram with 3 tiers:
- Global setting (strong)
- Phoneme residuals (moderate)
- Representational separability (mostly null)

**On-slide confidence ladder:**
- Strong: localization + sensitivity-floor.
- Strong: global phoneme-level replication across balanced cycles.
- Moderate: diphthong/nasal residuals.
- Directional: rate and late-luteal sensitivity effects.

## Slide 12 — What this enables next

**Headline claim:** The result is a mechanistic map and a replication-ready protocol.

**Visual:** Roadmap slide (3-5 bullets).

**Bullets:**
- Pre-register diphthong H1-H2 and nasal MFCC2 residual endpoints.
- Add direct nasal patency/nasalance and symptom-state capture.
- Replicate with phase-balanced cycles as a hard inclusion criterion.
- Scale to cohort with each participant as her own control.

**Close sentence:**
- "The contribution is not only a signal; it is a way to localize biological voice change while constraining over-interpretation."

## Slide Style Notes (readability-first)

- One claim per slide.
- One main visual per slide; avoid figure mosaics.
- Put exact numbers in callout labels on the plot, not in dense bullet text.
- Keep legend language consistent: follicular vs luteal, global vs residual, strong vs directional.
- Use confidence tags directly on slide titles where needed: `[Strong]`, `[Moderate]`, `[Directional]`.

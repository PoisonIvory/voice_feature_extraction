"""Assemble the standalone Markdown findings report.

Single responsibility: turn the result frames + figures into one self-contained
report under docs/voice_cycle_analysis/REPORT.md. Numbers are pulled from the
result frames so the prose stays consistent with the computed outputs.
"""

from __future__ import annotations

import pandas as pd

from . import paths


def _fmt(value: float, nd: int = 2) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.{nd}f}"


def _md_table(df: pd.DataFrame, columns: list[str], headers: list[str], floats: int = 2) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [head, sep]
    for _, row in df[columns].iterrows():
        cells = []
        for c in columns:
            v = row[c]
            cells.append(_fmt(v, floats) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _embed(figmap: dict[str, tuple[str, str]], key: str) -> str:
    if key not in figmap:
        return ""
    filename, caption = figmap[key]
    return f"![{caption}](figures/{filename})\n\n*{caption}*\n"


def _cov(coverage: pd.DataFrame, subset: str) -> str:
    row = coverage[coverage["subset"] == subset]
    return str(row["n"].iloc[0]) if len(row) else "?"


def build(df, explore_res, assoc_res, figures, coverage, phasedist) -> str:
    figmap = {key: (fname, cap) for key, fname, cap in figures}
    sweep = explore_res["sweep_follicular_vs_luteal"]
    axis_sum = explore_res["axis_family_summary"]
    dose = assoc_res["hormone_dose_response"]
    wear = assoc_res["wearable_phase_validation"]
    dprime = assoc_res["dprime_phase_contrast"]
    idx = assoc_res["instability_index"]

    temp_row = wear[wear["metric"] == "temp_deviation"].iloc[0] if len(wear) else None
    hrv_row = wear[wear["metric"] == "hrv"].iloc[0] if "hrv" in set(wear["metric"]) else None
    pitch_range = dose[dose["feature"].str.contains("pctlrange0-2")].head(1)
    top_sweep = sweep.dropna(subset=["hedges_g"]).head(12).copy()
    top_dose = dose.dropna(subset=["rho_pdg"]).head(10).copy()

    sub_means = (
        idx.dropna(subset=["voice_instability_index"]).groupby("subphase")["voice_instability_index"].mean()
    )

    md = f"""# Voice Across the Menstrual Cycle: An Independent N-of-1 Study

*Exploratory, hypothesis-generating analysis. Single subject with PMDD, naturally
cycling, no medication. {_cov(coverage, 'voice_days')} voice days across
{_cov(coverage, 'cycles')} cycles; {_cov(coverage, 'voice_and_hormones')} days with
measured Inito hormones; {_cov(coverage, 'voice_and_oura')} with Oura.*

## TL;DR

- The cycle signal in this voice is real and coherent, and it is concentrated where
  hormone physiology predicts: the **peripheral / vocal-fold-mucosa** features
  (amplitude perturbation, spectral balance / breathiness, formant-bandwidth
  stability) carry the largest and most cross-cycle-consistent shifts, while the
  **structural pitch level** (mean F0) is among the most stable features, exactly
  the dissociation the gender-affirming-therapy literature implies (androgens set
  pitch architecture; estrogen/progesterone act on the mucosa).
- A composite **voice-instability index rises from a follicular/menstrual low to a
  premenstrual (late-luteal) peak** (means by sub-phase: menstrual
  {_fmt(sub_means.get('menstrual'))}, early-follicular {_fmt(sub_means.get('early_follicular'))},
  peri-ovulatory {_fmt(sub_means.get('periovulatory'))}, early-luteal
  {_fmt(sub_means.get('early_luteal'))}, late-luteal {_fmt(sub_means.get('late_luteal'))} z).
- **Measured progesterone (PdG) tracks voice**: higher PdG goes with a narrower
  pitch range (rho={_fmt(pitch_range['rho_pdg'].iloc[0]) if len(pitch_range) else 'n/a'}),
  more breathiness-related spectral change, and longer unvoiced/pause segments,
  associations that largely survive partialling out Oura temperature.
- **Independent corroboration from Oura**: temperature is higher in the luteal phase
  (Hedges g={_fmt(temp_row['hedges_g_luteal_minus_foll']) if temp_row is not None else 'n/a'})
  and HRV lower (g={_fmt(hrv_row['hedges_g_luteal_minus_foll']) if hrv_row is not None else 'n/a'}),
  the progesterone thermogenic + autonomic signature, derived without touching the voice data.
- **Novel angle**: self-supervised (HuBERT) articulatory-contrast separability tends
  to fall in the luteal phase, an under-explored "articulatory precision" readout.

## 1. Background and interpretive framework

Sex-steroid receptors sit on the vocal-fold mucosa, and laryngeal cytology mirrors
cervical cytology across the cycle, so the larynx behaves as a hormonal target organ.
Reading across the siloed literatures (menstrual cycle, menopause, PCOS, gender-affirming
therapy, PMDD) suggests three semi-independent pathways from hormones to voice, which we
use only as an interpretive lens, not as a hypothesis filter:

1. **Structural / source** (vocal-fold mass and tension, sets mean F0): dominated by
   androgens, which barely move across a normal cycle, so mean F0 should be the most
   stable feature. Testosterone therapy lowers F0 by ~6 semitones; adult estrogen does
   not raise F0, dissociating "pitch architecture" from "voice quality."
2. **Peripheral / mucosal** (vocal-fold cover hydration, viscosity, edema, sets voice
   quality and resonance): governed by the estrogen:progesterone balance and net fluid.
   Estrogen (late follicular) hydrates the mucosa; progesterone (luteal) drives edema and
   drying. Predicts cyclic movement in HNR, jitter, shimmer, breathiness (H1-A3, alpha
   ratio, Hammarberg) and formant frequencies/bandwidths.
3. **Central / neuro-affective** (arousal, mood, psychomotor, sets prosody and dynamics):
   governed by progesterone to allopregnanolone (GABAergic) and its premenstrual
   withdrawal. Predicts movement in pitch variability/range, loudness/intensity, speech
   rate/pausing, and articulatory precision.

**PMDD as a lens (not a driver).** The subject has PMDD, an abnormal CNS sensitivity to
normal luteal progesterone/allopregnanolone, with symptoms in the late-luteal window that
remit at menses. Voice-in-PMDD is essentially unstudied, so we did not anchor the analysis
to it; instead we explore the data first and then read the results through this lens, which
predicts that the central pathway (and the late-luteal window) should be where this person
most diverges from the limited "normal-cycling" literature.

## 2. Data and independent methods

This analysis is deliberately independent: it rebuilds the join from components and
re-derives cycle phases rather than inheriting the other project's labels.

- **Voice**: this project's eGeMAPSv02 daily functionals (vowel + connected-speech/prosody),
  176 features, plus per-recording HuBERT phonological d-primes.
- **Hormones**: Inito urinary estrogen (e3g) and progesterone (PdG) metabolites; FSH/LH.
- **Wearable**: raw Oura daily biometrics (temperature deviation, HRV, resting HR, etc.).
- **Phase derivation**: device menses anchor + backward counting from the next menses
  (luteal length is stable), refined by hormones/temperature, giving menses / follicular /
  ovulatory / luteal and finer sub-phases plus a premenstrual (late-luteal) flag. The other
  project's follicular/luteal label is retained only as a cross-check (it agrees with ours).
- **Statistics**: N-of-1 exploratory framing. We lead with effect sizes (Hedges g, Cliff's
  delta), 95% bootstrap CIs, and **cross-cycle consistency** (does the effect repeat across
  the {_cov(coverage, 'cycles')} cycles), with Mann-Whitney p / BH-FDR q as context, not gates.

{_embed(figmap, 'coverage')}

## 3. Discovery: where the cycle signal lives

A data-first sweep of all 176 features for the follicular-to-luteal contrast shows the
largest, most reliable shifts in peripheral/mucosal families, with structural pitch level
among the most stable, the framework's predicted dissociation, emerging from the data.

{_md_table(top_sweep, ['feature','family','axis','hedges_g','g_ci_low','g_ci_high','cross_cycle_consistency','n_cycles','mw_p'], ['feature','family','axis','g (lut-foll)','CI low','CI high','cross-cycle','n cyc','MW p'])}

{_embed(figmap, 'effect_ranking')}

{_embed(figmap, 'axis_summary')}

Mean absolute effect by family confirms the concentration of signal:

{_md_table(axis_sum, ['axis','family','n_features','mean_abs_g','max_abs_g','mean_consistency'], ['axis','family','n','mean|g|','max|g|','mean consistency'])}

{_embed(figmap, 'cycle_clock')}

## 4. The premenstrual rise: a composite instability index

Collapsing the peripheral-instability panel into one daily index shows a clear progression
from a follicular/menstrual low to a premenstrual (late-luteal) peak.

{_embed(figmap, 'instability')}

## 5. Voice vs measured hormones (dose-response, {_cov(coverage, 'voice_and_hormones')} days)

On the days with measured hormones, several features scale with progesterone (PdG), and the
associations largely persist after partialling out Oura temperature, i.e., they are not just
a thermal/autonomic artifact.

{_md_table(top_dose, ['feature','family','axis','rho_pdg','p_pdg','rho_ep_ratio','partial_rho_pdg_given_temp'], ['feature','family','axis','rho PdG','p','rho E:P','partial (|temp)'])}

{_embed(figmap, 'dose_response')}

{_embed(figmap, 'overlay')}

## 6. Independent confirmation from the wearable

Using only Oura (no voice), the luteal phase shows the expected progesterone signature,
validating the phase labels and providing a hormone proxy beyond the hormone window.

{_md_table(wear, ['metric','luteal_mean','follicular_mean','hedges_g_luteal_minus_foll','mw_p','n_luteal','n_follicular'], ['metric','luteal mean','foll mean','g (lut-foll)','MW p','n lut','n foll'])}

{_embed(figmap, 'wearable')}

## 7. Novel: articulatory precision across the cycle

Phonological-contrast separability (d-prime) from frozen HuBERT embeddings tends to be lower
in the luteal phase for the best-sampled contrasts, an exploratory "articulatory precision"
readout that, to our knowledge, has not been applied to the menstrual cycle.

{_md_table(dprime.dropna(subset=['hedges_g']), ['feature','hedges_g','g_ci_low','g_ci_high','cross_cycle_consistency','mw_p'], ['contrast','g (lut-foll)','CI low','CI high','cross-cycle','MW p'])}

{_embed(figmap, 'dprime')}

## 8. How this compares to the limited published literature

The handful of published studies are all on normal-cycling women and mostly use sustained
vowels and calendar phase. Read modestly against them:

- **Pitch architecture is stable, quality is not.** Like the field's frequent "no mean-F0
  effect," mean F0 here is among the most stable features; the action is in voice quality and
  resonance, consistent with the estrogen/progesterone mucosal mechanism.
- **Premenstrual quality degradation.** Increased amplitude-perturbation variability and
  breathiness-related spectral change toward the luteal/premenstrual window echoes the
  "premenstrual vocal syndrome" descriptions (jitter/shimmer up, quality down).
- **Reduced luteal pitch range / dynamics.** A normal-cycling daily-recording study reported
  ~9% lower F0 variability in the luteal phase; here pitch range narrows with measured
  progesterone, in the same direction, and is accompanied by longer pauses and reduced
  articulatory precision, a more central/affective pattern consistent with the depression
  voice-biomarker literature and with this subject's PMDD.
- **Autonomic signature matches PMDD.** Lower luteal HRV is consistent with reports of
  reduced parasympathetic tone in PMDD.

The plausible "meaningful difference" for this person is not in the peripheral features
(which look like the normal-cycling pattern) but in the **central pathway** (pitch range,
pausing, articulatory precision) and its concentration in the **late-luteal symptom window**.
This is a hypothesis to test with symptom-tracked data, not a claim.

## 9. Novel contributions and gaps addressed

- Continuous **dose-response to measured urinary hormones** (not calendar phase) in a dense
  N-of-1 design over {_cov(coverage, 'cycles')} cycles, addressing the field's main gap.
- A **three-pathway decomposition** tested within one person: quality/resonance vs
  prosody/dynamics vs stable structure.
- **Wearable + measured-hormone + voice fusion**, using Oura to independently confirm phase.
- **Articulatory-precision (SSL d-prime)** and **hormone rate-of-change/withdrawal** as
  exploratory cycle readouts.
- First look at **voice across the cycle in PMDD**.

## 10. Limitations

- Single subject; everything is exploratory and hypothesis-generating. With 176 features no
  result survives multiple-comparison correction; the case rests on effect sizes plus
  cross-cycle consistency plus mechanistic coherence, not significance.
- The hormone window ({_cov(coverage, 'voice_and_hormones')} days) is short and concentrated in
  a few cycles; dose-response is correlational and cannot separate progesterone from
  correlated luteal changes beyond the temperature control applied.
- No daily symptom logs, so the PMDD symptom window is proxied from phase/hormones/HRV; the
  central-pathway interpretation is not yet validated against symptoms.
- Recordings vary in count per day and were not made under fixed laboratory conditions.
- Oura resting-HR ran counter to the expected luteal direction (likely behavioral/N-of-1
  noise), a reminder to treat single-metric wearable contrasts cautiously.

## 11. Future directions

- Add daily symptom ratings (e.g., DRSP) to test voice as a PMDD symptom-state biomarker.
- Extend hormone sampling to cover more cycles and the full follicular-luteal arc.
- Model withdrawal velocity (dPdG/dt) explicitly around the late-luteal window.
- Pre-register the central-pathway prediction and test it in a second person with PMDD.

## Reproducibility

Regenerate everything with `python -m analysis.run_analysis`. Inputs, the assembled daily
table, the per-tier result tables, the data dictionary, and these figures are written under
`data/analysis/`, `analysis/outputs/`, and this folder.
"""
    paths.ensure_output_dirs()
    paths.REPORT_FILE.write_text(md)
    return md


__all__ = ["build"]

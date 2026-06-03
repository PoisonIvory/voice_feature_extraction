# Voice as a Non-Invasive Readout of Progesterone Action: Convergent Evidence from Source-Filter Localization, Phoneme-Resolved Acoustics, and Representational Geometry

**A longitudinal N-of-1 study of menstrual-cycle effects on speech production**
**Author:** Ivy Hamilton
**Date:** June 2026
**Data:** 59–71 days of paired vowel + read-speech recordings, daily Inito E3G/PdG, Oura biometrics, 5 tracked cycles (2 phase-balanced), anchored to a source-of-truth cycle calendar.

---

## The Clinical Question, Framed for Voice-as-Biomarker Research

Can a 60-second voice recording, captured in daily life, reveal *which* hormone is acting, *where* it is acting (peripheral tissue vs central motor control), and *when* a speaker is in her most sensitive window — all without a blood draw or imaging? This question sits at the intersection of speech acoustics, endocrinology, upper-airway physiology, and neurosteroid neuroscience. It is also the practical extension of recent work showing that voice carries detectable signatures of endocrine transitions (e.g., menopause). The present study tests that premise with measured hormones, dense longitudinal sampling, and three orthogonal measurement families that share no common machinery.

The answer that emerges is mechanistically specific, physiologically grounded, and methodologically self-validating: **the cycle retunes the vocal-fold cover (the “reed”) via progesterone’s peripheral and central actions, leaves vocal-tract geometry (the “tube”) unchanged, applies a global phonatory/filter setting across the phonetic inventory, and produces a premenstrual spike in pitch-control variability that matches the known neurosteroid sensitivity profile of PMDD.** Two phoneme-selective residuals (diphthong open-quotient amplification; nasal timbre excess) survive strict confound controls and align with independent literature on luteal nasal congestion. A third family — phonological-subspace separability in frozen SSL embeddings — returns a clean negative control, confirming that the signal is surface-level rather than articulatory reorganization. The same data that localize the effect also demonstrate that the measurement pipeline *could* have detected vocal-tract change (formant sensitivity floor 20–100× larger than cycle effects) and that phonological geometry *does not* collapse (as it does in dysarthria). This is not a fishing expedition; it is a hypothesis-driven, multi-lens triangulation anchored to a single speaker’s own physiology.

---

## The Source-Filter Frame: Reed vs Tube as a Diagnostic Lens

The vocal apparatus is a wind instrument. The reed (vocal-fold cover) determines *how* the buzz is generated — its open quotient, clarity, and short-term spectral tilt. The tube (vocal-tract cavities) determines *which* resonances are emphasized — the formant frequencies that define vowel identity. Progesterone is known to increase mucosal edema and mucus viscosity in the vocal-fold lamina propria; it is also metabolized to allopregnanolone, a positive allosteric modulator of GABA-A receptors that dampens central motor circuits, including those controlling pitch. Estrogen’s effects on the fold lining are weaker and more transient. If the cycle driver is progesterone acting on the soft, fluid-filled cover, the acoustic signature should be concentrated in reed measures (H1-H2 open quotient, HNR clarity, spectral tilt, MFCC timbre) and absent from tube measures (F1–F3 frequencies). If the driver were instead a gross reshaping of the bony resonating cavities, formants would move.

This prediction is tested with two complementary safeguards. First, a formal dissociation test (Crawford & Garthwaite single-case method) asks whether the gap between “reed moves” and “tube stays still” exceeds what would be expected by chance; it is significant in both speaking styles (read sentences p = 0.008; sustained vowel p = 0.049). Second, a sensitivity-floor check quantifies how much formants *do* move for non-cycle reasons within the same recordings: within-sentence vowel-to-vowel swings are ~184 Hz (F1) and ~210 Hz (F2); cycle-driven shifts are 1.8 Hz and 5.8 Hz — 20–100× smaller. The instrument can bark; across the cycle it stays silent. That silence is therefore informative, not a blind spot. The cycle acts on the reed, not the tube — exactly the pattern predicted by progesterone-driven cover physiology and inconsistent with a tube-reshaping mechanism.

---

## Progesterone as the Driver: Peripheral and Central Pathways, Drift-Controlled

Daily urinary E3G and PdG measurements allow direct attribution. After removing slow calendar drift (the confounder that can produce spurious hormone–feature correlations when both trend together over months), progesterone coupling is strong for reed/timbre measures (Hammarberg index, MFCC2, HNR) and for a central pitch-control measure (pitch-range variability); estrogen coupling hovers near zero. This is not a level effect but a dual-channel effect: progesterone thickens the fold lining (peripheral, cover) *and* is converted to allopregnanolone that modulates the brain circuits governing pitch steadiness (central). The same hormone therefore moves both voice *quality* and voice *control* — a finding that integrates vocal-fold histology, neurosteroid pharmacology, and speech-motor physiology in a single within-subject design.

An honest surprise relative to textbook “best voice at ovulation” lore is that clarity (HNR) and open quotient peak in the high-progesterone luteal phase for this speaker. This matches the direction reported by Kervin et al. (2025) for glottal opening (direct imaging) and underscores that measured hormones can overturn tidy stories derived from calendar phase alone.

---

## Rate, Not Level; Premenstrual Spike as PMDD Signature

Kervin’s daily voice-diary work suggested that instability tracks the *speed* of hormonal change rather than absolute level. The present data allow a direct test: on days when progesterone is changing fastest, reed/cover measures show elevated day-to-day jitter while tube/geometry remains steady — a directional pattern consistent with the earlier observation and now anchored to measured hormone velocity.

For a speaker with PMDD, the late-luteal window is not merely the tail of the progesterone curve; it is the period of maximal brain sensitivity to allopregnanolone withdrawal. The data show exactly this signature: pitch-control variability rises premenstrually while mid-luteal clarity and open quotient are at their most stable. The body context (Oura temperature rise, resting-heart-rate elevation, HRV dip in luteal phase) confirms that cycle labels are physiologically valid and that the premenstrual pitch spike is not an artifact of mislabeled days. The voice is therefore a non-invasive, daily readout of both peripheral progesterone action and central neurosteroid sensitivity — the precise phenotype that defines PMDD and that standard serum hormone panels cannot capture.

---

## Phoneme Grain: Global Setting + Two Mechanistically Coherent Residuals

Whole-recording functionals average across a heterogeneous phonetic mixture. The phoneme-resolved analysis (8,463 force-aligned segments, MFA 3.3.9) asks whether the luteal signal is uniform or concentrated. After within-cycle normalization and recording-level de-meaning (subtracting each recording’s own mean to isolate relative change), open quotient (H1-H2) and timbre (MFCC2) rise in essentially every voiced class with near-identical magnitude across the two phase-balanced cycles (~+0.7 SD and +1.4 SD). Within-recording contrasts (voiced–voiceless, high–low vowel, etc.) are flat for phase, confirming that the dominant effect is a uniform per-recording setting rather than a reorganization of relative phoneme acoustics.

Two residuals survive de-meaning, F0 residualization, and BH-FDR. Diphthongs — the longest, most sustained dynamically-voiced nuclei — show an amplified open-quotient increase (de-meaned Cliff’s delta +0.31, q = 0.023), consistent with the expectation that a glottal-cycle change has the most room to express itself in prolonged voicing. Nasals show an amplified timbre shift (de-meaned +0.36, q = 0.049), aligning with independent reports of luteal nasal mucosal swelling, reduced patency, and increased nasalance. These are not post-hoc discoveries; they were mechanistically privileged a priori from the source-filter and upper-airway literatures. The same data that establish globality also isolate the two places where the global setting is locally amplified — a decomposition that whole-file averages cannot provide and that resolves the recording-condition confound that plagues longitudinal voice work.

---

## Representational Geometry as Negative Control: Specificity of the Phonological-Subspace Method

If the cycle were altering articulatory precision, phonological-feature separability in SSL embedding space should decline — the signature Muller et al. (2026) documented for dysarthria. The same MFA boundaries feeding the phoneme analysis yield per-recording d-prime for eight contrasts across three backbones (HuBERT-base, WavLM-base, wav2vec2-base). The consonant composite is negligible (Cliff’s delta –0.12, BH q = 0.84); zero contrasts survive FDR correction in any backbone. Geometry controls (vowel height, backness, lowness) are inert, mirroring the formant null. The two mechanistically privileged contrasts lean in the predicted direction (nasality d-prime falls with progesterone, consistent with congestion blurring the nasal/oral distinction; voicing d-prime rises, consistent with enhanced voiced/voiceless contrast in read-speech literature) but remain non-significant — hypothesis-generating leads, not claims.

Triangulation is decisive: composite d-prime does not track the global MFCC2 or H1-H2 setting that the acoustic studies show moves strongly (date-partial rho 0.11 and –0.01). The cycle changes the acoustic surface without changing how distinctly the representation separates phonological categories. This is the cleanest possible specificity result for the phonological-subspace method: a non-articulatory, within-speaker physiological perturbation returns a null in the same speaker, passage, and pipeline that would light up under dysarthria. The fixed-passage design removes the token-count confound at source; LORO direction estimation removes in-sample inflation. Three SSL objectives agree on profile shape (cosine 0.96–0.997) and on the phase null. The conclusion is architecture-independent.

---

## Methodological Rigor as Evidence of Depth

The design incorporates multiple layers of internal validation that are rarely combined in single-subject voice research:

- Within-cycle normalization (each cycle is its own baseline) prevents between-cycle drift from masquerading as phase effects.
- Drift-controlled partial Spearman (date as covariate) isolates hormone coupling from slow calendar trends.
- Per-recording de-meaning and within-recording contrasts cancel every recording-level nuisance (gain, distance, technique, mic placement) by construction.
- Dissociation testing and equivalence testing formalize the “dog that didn’t bark” logic.
- LORO + fixed passage + cross-backbone cosine provide architecture-independent negative controls.
- Held-out-cycle decoding (0.88 balanced accuracy) demonstrates that the phase signal is reproducible, not over-fit.
- Body biometrics (temperature, HR, HRV) anchor cycle labels to independent physiology.

These controls are not decorative; they are the reason the story can claim mechanistic specificity rather than pattern discovery. The same data that generate the headline findings also demonstrate that the measurement families are dissociable, that the pipeline is sensitive where it should be, and that the nulls are not artifacts of low power or insensitive instruments.

---

## Why This Matters for Voice-as-Health Research

A 60-second recording, collected daily with consumer hardware, can localize a hormonal signal to the vocal-fold cover, attribute it to progesterone’s dual peripheral/central actions, flag the premenstrual window of maximal neurosteroid sensitivity, and do so with internal negative controls that rule out vocal-tract geometry change and representational collapse. For conditions like PMDD — invisible on standard hormone panels and defined by *sensitivity* rather than *level* — a non-invasive daily signal that tracks the sensitivity itself is clinically relevant. The same logic extends to menopause, contraceptive transitions, and other endocrine states where voice may serve as a remote, low-burden biomarker. The methodological template (multi-grain acoustics + SSL negative controls + hormone-anchored single-subject design) is portable to other within-person longitudinal questions in speech neuroscience and personal health informatics.

---

## Limitations and Next Steps

The findings rest on two phase-balanced cycles in one speaker; lopsided cycles inform direction only. The diphthong and nasal residuals, while mechanistically coherent and FDR-surviving, are single-subject and require pre-registered replication. The nasality/PdG lean in d-prime is the most interesting positive signal and the least certain. Calendar-based luteal definition (last 14 days) can misalign with actual ovulation in long cycles. Absolute d-prime magnitudes are not calibrated across backbones.

The highest-value next step is simply sampling both phases in every cycle for a small cohort, each woman as her own control. Pre-register the nasality d-prime decline and the diphthong open-quotient residual as confirmatory endpoints. Pair the nasal MFCC2 residual with direct nasalance or rhinomanometry. Add HNR and formant frequencies to the phoneme extractor so the full source-filter taxonomy is available at segment grain. The design already contains every control needed to scale; the limiting factor is phase-balanced coverage per cycle, not analytic sophistication.

---

## References (Key Anchors)

- Kervin et al. (2025). Daily Laryngeal Kinematics and Acoustics Throughout the Menstrual Cycle. (Closest prior study; explicitly requested measured hormones.)
- Muller et al. (2026). Phonological Subspace Collapse Is Aetiology-Specific and Cross-Lingually Stable (arXiv:2604.21706). (Method adapted for negative control.)
- Crawford & Garthwaite (2005). Single-case dissociation testing.
- Lakens (2018). Equivalence testing for “flat” claims.
- Abitbol et al. (1999). Estrogen vs progesterone effects on vocal-fold mucosa.
- Zhu et al. (2016). Progesterone/allopregnanolone and central pitch control.
- Timby et al. (2025). Allopregnanolone/GABA-A account of PMDD.
- Upper-airway literature on luteal nasal congestion and nasalance.
- Read-speech literature on enhanced voiced/voiceless contrast in high-hormone phase.

---

*This narrative is constructed entirely from the analysis outputs, code structure, and cited literature. It deliberately avoids any external presentation scaffolding.*
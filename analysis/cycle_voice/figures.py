"""Generate the data-story figures for the report.

Single responsibility: turn the assembled table and result frames into PNG
figures saved under the report's figures directory. Returns (key, filename,
caption) tuples so the report can embed them in order.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import paths

SUBPHASE_ORDER = [
    "menstrual",
    "early_follicular",
    "late_follicular",
    "periovulatory",
    "early_luteal",
    "late_luteal",
]
AXIS_COLORS = {
    "structural_source": "#4C72B0",
    "peripheral_mucosal": "#C44E52",
    "central_neuroaffective": "#55A868",
    "unassigned": "#999999",
}
plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight", "font.size": 10})


def _save(fig, name: str) -> str:
    path = paths.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return name


def _short(feature: str) -> str:
    return feature.replace("_egemaps_", " ").replace("_sma3nz", "").replace("_sma3", "")


def fig_coverage_timeline(df: pd.DataFrame) -> tuple[str, str, str]:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    lanes = [("Oura", "has_oura", "#55A868"), ("Hormones (e3g/PdG)", "has_hormones", "#C44E52"), ("Voice", "has_voice", "#4C72B0")]
    for y, (label, col, color) in enumerate(lanes):
        days = df.loc[df[col], "date"]
        ax.scatter(days, [y] * len(days), s=12, color=color, marker="s")
    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels([l[0] for l in lanes])
    ax.set_title("Data coverage across the study window (N-of-1)")
    ax.grid(axis="x", alpha=0.3)
    return ("coverage", _save(fig, "fig01_coverage_timeline.png"),
            "Daily availability of voice, measured hormone, and Oura data. The hormone-quantified window (29 voice+hormone days) is the densest overlap.")


def fig_effect_ranking(sweep: pd.DataFrame, top: int = 20) -> tuple[str, str, str]:
    d = sweep.dropna(subset=["hedges_g"]).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1))
    colors = [AXIS_COLORS.get(a, "#999") for a in d["axis"]]
    err = np.vstack([d["hedges_g"] - d["g_ci_low"], d["g_ci_high"] - d["hedges_g"]])
    ax.barh(range(len(d)), d["hedges_g"], color=colors, alpha=0.85)
    ax.errorbar(d["hedges_g"], range(len(d)), xerr=err, fmt="none", ecolor="#333", elinewidth=0.8, capsize=2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([_short(f) for f in d["feature"]], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Hedges g  (luteal - follicular)")
    ax.set_title("Largest voice-feature shifts, luteal vs follicular (95% bootstrap CI)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in AXIS_COLORS.values()]
    ax.legend(handles, [k.replace("_", " ") for k in AXIS_COLORS], fontsize=7, loc="lower right")
    return ("effect_ranking", _save(fig, "fig02_effect_ranking.png"),
            "Top features by absolute effect size for the follicular-to-luteal contrast, colored by interpretive axis. Peripheral/mucosal features dominate the largest, most reliable shifts.")


def fig_axis_summary(summary: pd.DataFrame) -> tuple[str, str, str]:
    d = summary.sort_values("mean_abs_g")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1))
    colors = [AXIS_COLORS.get(a, "#999") for a in d["axis"]]
    ax.barh(range(len(d)), d["mean_abs_g"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{f} ({a.split('_')[0]})" for f, a in zip(d["family"], d["axis"])], fontsize=8)
    ax.set_xlabel("Mean |Hedges g| across features in family")
    ax.set_title("Where the cycle signal concentrates (by feature family / axis)")
    return ("axis_summary", _save(fig, "fig03_axis_family_summary.png"),
            "Average absolute luteal-vs-follicular effect per feature family. Perturbation and spectral-balance (peripheral/mucosal) families carry the strongest mean signal; structural pitch level is among the most stable.")


def _ordered_subphase(s: pd.Series) -> pd.Categorical:
    return pd.Categorical(s, SUBPHASE_ORDER, ordered=True)


def fig_cycle_clock(df: pd.DataFrame, features: list[str]) -> tuple[str, str, str]:
    voice = df[df["has_voice"]].copy()
    voice["subphase"] = _ordered_subphase(voice["subphase"])
    feats = features[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, feat in zip(axes.ravel(), feats):
        grp = voice.groupby("subphase", observed=True)[feat]
        m = grp.mean()
        sem = grp.std() / np.sqrt(grp.count())
        x = range(len(m))
        ax.errorbar(x, m.to_numpy(), yerr=sem.to_numpy(), marker="o", color="#C44E52", capsize=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels([s.replace("_", "\n") for s in m.index], fontsize=7)
        ax.set_title(_short(feat), fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Cycle clock: feature means by sub-phase (mean +/- SEM)", y=1.0)
    fig.tight_layout()
    return ("cycle_clock", _save(fig, "fig04_cycle_clock.png"),
            "Top features traced across the cycle sub-phases (menstrual through late-luteal). The follicular-to-luteal drift and premenstrual extremes are visible feature by feature.")


def fig_instability_index(idx: pd.DataFrame) -> tuple[str, str, str]:
    d = idx.dropna(subset=["voice_instability_index"]).copy()
    d["subphase"] = _ordered_subphase(d["subphase"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    grp = d.groupby("subphase", observed=True)["voice_instability_index"]
    m, sem = grp.mean(), grp.std() / np.sqrt(grp.count())
    axes[0].bar(range(len(m)), m.to_numpy(), yerr=sem.to_numpy(), color="#C44E52", alpha=0.85, capsize=3)
    axes[0].set_xticks(range(len(m)))
    axes[0].set_xticklabels([s.replace("_", "\n") for s in m.index], fontsize=8)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("Voice instability index (z)")
    axes[0].set_title("Composite voice instability by sub-phase")
    sc = axes[1].scatter(d["cycle_day_from_ovulation"], d["voice_instability_index"],
                         c=d["has_hormones"].map({True: "#C44E52", False: "#999999"}))
    axes[1].axvline(0, color="#4C72B0", ls="--", lw=1, label="ovulation (approx)")
    axes[1].set_xlabel("Cycle day from ovulation (negative=follicular)")
    axes[1].set_ylabel("Voice instability index (z)")
    axes[1].set_title("Instability vs cycle position")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return ("instability", _save(fig, "fig05_instability_index.png"),
            "A composite of peripheral-instability features rises from a follicular/menstrual low to a premenstrual (late-luteal) peak. Red points are days with measured hormones.")


def fig_hormone_dose_response(df: pd.DataFrame, dose: pd.DataFrame) -> tuple[str, str, str]:
    sub = df[df["has_voice"] & df["has_hormones"]].copy()
    feats = dose.dropna(subset=["rho_pdg"]).head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), feats):
        row = dose[dose["feature"] == feat].iloc[0]
        ax.scatter(sub["pdg"], sub[feat], color="#C44E52")
        ax.set_xlabel("PdG (progesterone metabolite)")
        ax.set_ylabel(_short(feat), fontsize=8)
        ax.set_title(f"rho={row['rho_pdg']:.2f}, p={row['p_pdg']:.3f}, partial|temp={row['partial_rho_pdg_given_temp']:.2f}", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Voice features vs measured progesterone (29 voice+hormone days)", y=1.0)
    fig.tight_layout()
    return ("dose_response", _save(fig, "fig06_hormone_dose_response.png"),
            "The voice features most strongly associated with measured progesterone (PdG). Associations largely persist after partialling out Oura temperature.")


def fig_wearable_validation(df: pd.DataFrame) -> tuple[str, str, str]:
    sub = df[df["has_oura"]].copy()
    sub = sub[sub["phase"].isin(["follicular", "luteal"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (metric, title) in zip(axes, [("temp_deviation", "Oura temperature deviation"), ("hrv", "Oura HRV")]):
        data = [sub.loc[sub["phase"] == ph, metric].dropna() for ph in ("follicular", "luteal")]
        ax.boxplot(data, labels=["follicular", "luteal"], showmeans=True)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Independent phase confirmation from Oura (progesterone signature)", y=1.0)
    fig.tight_layout()
    return ("wearable", _save(fig, "fig07_wearable_validation.png"),
            "Oura temperature is elevated and HRV reduced in the luteal phase, independently corroborating the hormone-anchored phase labels without using voice.")


def fig_dprime(dp_daily: pd.DataFrame, contrast: pd.DataFrame) -> tuple[str, str, str]:
    feats = contrast.dropna(subset=["hedges_g"]).head(4)["feature"].tolist()
    dd = dp_daily.copy()
    dd["subphase"] = _ordered_subphase(dd["subphase"])
    fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 4))
    axes = np.atleast_1d(axes)
    for ax, feat in zip(axes, feats):
        grp = dd.groupby("subphase", observed=True)[feat]
        m, sem = grp.mean(), grp.std() / np.sqrt(grp.count())
        ax.errorbar(range(len(m)), m.to_numpy(), yerr=sem.to_numpy(), marker="o", color="#55A868", capsize=3)
        ax.set_xticks(range(len(m)))
        ax.set_xticklabels([s.replace("_", "\n") for s in m.index], fontsize=6, rotation=0)
        ax.set_title(feat.replace("dprime_", "d' "), fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Articulatory precision (HuBERT d-prime) across the cycle", y=1.02)
    fig.tight_layout()
    return ("dprime", _save(fig, "fig08_dprime_articulatory.png"),
            "Phonological-contrast separability from self-supervised embeddings tends to fall in the luteal phase, suggesting reduced articulatory precision; exploratory and novel for cycle research.")


def fig_hormone_overlay(idx: pd.DataFrame, df: pd.DataFrame) -> tuple[str, str, str]:
    win = idx[idx["has_hormones"]].dropna(subset=["voice_instability_index"]).sort_values("date")
    if win.empty:
        return ("overlay", "", "")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(win["date"], win["voice_instability_index"], "-o", color="#C44E52", label="voice instability index")
    ax.set_ylabel("Voice instability index (z)", color="#C44E52")
    ax.axhline(0, color="#C44E52", lw=0.6, alpha=0.5)
    ax2 = ax.twinx()
    ax2.plot(win["date"], win["pdg"], "-s", color="#4C72B0", alpha=0.7, label="PdG")
    ax2.plot(win["date"], win["e3g"] / 10.0, "-^", color="#55A868", alpha=0.6, label="e3g/10")
    ax2.set_ylabel("PdG  /  e3g (scaled)", color="#4C72B0")
    ax.set_title("Voice instability tracks measured progesterone within the hormone window")
    lines = [l for l in (ax.get_lines() + ax2.get_lines()) if not l.get_label().startswith("_")]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    return ("overlay", _save(fig, "fig09_hormone_overlay.png"),
            "Within the measured-hormone window, the daily voice-instability index rises and falls with progesterone (PdG), the integrative N-of-1 view.")


def generate_all(df: pd.DataFrame, explore_res: dict, assoc_res: dict) -> list[tuple[str, str, str]]:
    paths.ensure_output_dirs()
    sweep = explore_res["sweep_follicular_vs_luteal"]
    top_feats = sweep.dropna(subset=["hedges_g"]).head(6)["feature"].tolist()
    figs = [
        fig_coverage_timeline(df),
        fig_effect_ranking(sweep),
        fig_axis_summary(explore_res["axis_family_summary"]),
        fig_cycle_clock(df, top_feats),
        fig_instability_index(assoc_res["instability_index"]),
        fig_hormone_dose_response(df, assoc_res["hormone_dose_response"]),
        fig_wearable_validation(df),
        fig_dprime(assoc_res["dprime_daily"], assoc_res["dprime_phase_contrast"]),
        fig_hormone_overlay(assoc_res["instability_index"], df),
    ]
    return [f for f in figs if f[1]]

"""Tier 0 discovery + Tier A phase contrasts.

Data-first: sweep every voice feature for phase structure, rank by effect size
and cross-cycle consistency, and summarize correlation structure with PCA. The
three-axis framework is attached only for grouping/interpretation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import stats
from .features import classify_feature


def contrast_sweep(
    df: pd.DataFrame,
    features: list[str],
    pos_mask: pd.Series,
    neg_mask: pd.Series,
    pos_label: str,
    neg_label: str,
) -> pd.DataFrame:
    """Effect-size sweep of pos vs neg over all features (one row per feature)."""
    tmp = df.copy()
    tmp["grp"] = np.where(pos_mask, "pos", np.where(neg_mask, "neg", None))
    rows = []
    for feat in features:
        a = tmp.loc[tmp["grp"] == "pos", feat].to_numpy()
        b = tmp.loc[tmp["grp"] == "neg", feat].to_numpy()
        na = int(np.isfinite(a).sum())
        nb = int(np.isfinite(b).sum())
        meta = classify_feature(feat)
        g = stats.hedges_g(a, b)
        ci_lo, ci_hi = stats.bootstrap_ci_g(a, b)
        consistency, n_cycles = stats.cross_cycle_consistency(tmp, feat, "grp", "pos", "neg")
        rows.append(
            {
                "feature": feat,
                "task": meta.task if meta else "",
                "family": meta.family if meta else "",
                "axis": meta.axis if meta else "",
                f"n_{pos_label}": na,
                f"n_{neg_label}": nb,
                "hedges_g": g,
                "g_ci_low": ci_lo,
                "g_ci_high": ci_hi,
                "cliffs_delta": stats.cliffs_delta(a, b),
                "mw_p": stats.mann_whitney_p(a, b),
                "cross_cycle_consistency": consistency,
                "n_cycles": n_cycles,
            }
        )
    out = pd.DataFrame(rows)
    out["q_value"] = stats.benjamini_hochberg(out["mw_p"].to_numpy())
    out["abs_g"] = out["hedges_g"].abs()
    out["ci_excludes_zero"] = (out["g_ci_low"] > 0) | (out["g_ci_high"] < 0)
    return out.sort_values("abs_g", ascending=False).reset_index(drop=True)


def axis_family_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Mean |effect| and consistency per axis and family, to see where signal concentrates."""
    grp = sweep.groupby(["axis", "family"]).agg(
        n_features=("feature", "size"),
        mean_abs_g=("abs_g", "mean"),
        max_abs_g=("abs_g", "max"),
        mean_consistency=("cross_cycle_consistency", "mean"),
        n_ci_excludes_zero=("ci_excludes_zero", "sum"),
    )
    return grp.sort_values("mean_abs_g", ascending=False).reset_index()


def _matrix(df: pd.DataFrame, features: list[str], min_coverage: float = 0.8):
    voice = df[df["has_voice"]].copy()
    keep = [f for f in features if voice[f].notna().mean() >= min_coverage]
    mat = voice[keep]
    mat = mat.fillna(mat.mean())
    return voice, keep, mat


def run_pca(df: pd.DataFrame, features: list[str], n_components: int = 5):
    """PCA over voice-day feature matrix; returns (scores, loadings, variance)."""
    voice, keep, mat = _matrix(df, features)
    if mat.shape[0] < 5 or mat.shape[1] < 3:
        empty = pd.DataFrame()
        return empty, empty, empty
    x = StandardScaler().fit_transform(mat.to_numpy())
    k = min(n_components, x.shape[1], x.shape[0])
    pca = PCA(n_components=k, random_state=7)
    scores = pca.fit_transform(x)
    score_cols = [f"PC{i+1}" for i in range(k)]
    scores_df = voice[["date", "phase", "subphase", "cycle_day_from_ovulation", "has_hormones"]].copy()
    for i, c in enumerate(score_cols):
        scores_df[c] = scores[:, i]
    loadings = pd.DataFrame(pca.components_.T, index=keep, columns=score_cols)
    loadings = loadings.reset_index().rename(columns={"index": "feature"})
    variance = pd.DataFrame(
        {"component": score_cols, "explained_variance_ratio": pca.explained_variance_ratio_}
    )
    return scores_df, loadings, variance


def compute(df: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    voice = df["has_voice"]
    foll = df["phase"].eq("follicular")
    lut = df["phase"].eq("luteal")
    premenstrual = df["subphase"].eq("late_luteal")
    ovul = df["phase"].eq("ovulatory")

    results: dict[str, pd.DataFrame] = {}
    results["sweep_follicular_vs_luteal"] = contrast_sweep(df, features, lut & voice, foll & voice, "luteal", "follicular")
    results["sweep_premenstrual_vs_follicular"] = contrast_sweep(df, features, premenstrual & voice, foll & voice, "premenstrual", "follicular")
    results["sweep_ovulatory_vs_luteal"] = contrast_sweep(df, features, ovul & voice, lut & voice, "ovulatory", "luteal")
    results["axis_family_summary"] = axis_family_summary(results["sweep_follicular_vs_luteal"])

    scores, loadings, variance = run_pca(df, features)
    results["pca_scores"] = scores
    results["pca_loadings"] = loadings
    results["pca_variance"] = variance
    return results

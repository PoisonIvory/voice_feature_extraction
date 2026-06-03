"""Small-sample statistics helpers for an N-of-1 exploratory design.

Single responsibility: effect sizes, nonparametric tests, bootstrap CIs,
Benjamini-Hochberg FDR, partial correlation, and cross-cycle consistency. We
lead with effect sizes and CIs; p/q values are reported as context, not gates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference (a - b) with small-sample correction."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if pooled_var <= 0:
        return float("nan")
    d = (a.mean() - b.mean()) / np.sqrt(pooled_var)
    correction = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    return float(d * correction)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Nonparametric effect size in [-1, 1]; robust to non-normal small samples."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = sum((a[:, None] > b[None, :]).sum(axis=1))
    less = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((greater - less) / (len(a) * len(b)))


def mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    try:
        return float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def bootstrap_ci_g(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    gs = np.empty(n_boot)
    for i in range(n_boot):
        gs[i] = hedges_g(rng.choice(a, len(a), replace=True), rng.choice(b, len(b), replace=True))
    gs = gs[np.isfinite(gs)]
    if gs.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5)))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH-FDR adjusted q-values; NaNs preserved."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return out
    pv = p[mask]
    order = np.argsort(pv)
    n = len(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q_full = np.empty(n)
    q_full[order] = np.clip(q, 0, 1)
    out[mask] = q_full
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return (float("nan"), float("nan"), int(m.sum()))
    rho, p = stats.spearmanr(x[m], y[m])
    return (float(rho), float(p), int(m.sum()))


def partial_spearman(x: np.ndarray, y: np.ndarray, covar: np.ndarray) -> tuple[float, float, int]:
    """Spearman partial correlation of x, y controlling for covar (rank residuals)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    covar = np.asarray(covar, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(covar)
    if m.sum() < 5:
        return (float("nan"), float("nan"), int(m.sum()))
    rx, ry, rc = (stats.rankdata(v[m]) for v in (x, y, covar))
    design = np.column_stack([np.ones(m.sum()), rc])
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    rho, p = stats.spearmanr(res_x, res_y)
    return (float(rho), float(p), int(m.sum()))


def cross_cycle_consistency(
    df: pd.DataFrame, feature: str, group_col: str, pos: str, neg: str
) -> tuple[float, int]:
    """Fraction of cycles whose (pos - neg) median difference matches the overall sign."""
    overall = df.loc[df[group_col] == pos, feature].median() - df.loc[df[group_col] == neg, feature].median()
    if not np.isfinite(overall) or overall == 0:
        return (float("nan"), 0)
    agree = 0
    total = 0
    for _, g in df.groupby("cycle_start_date"):
        a = g.loc[g[group_col] == pos, feature]
        b = g.loc[g[group_col] == neg, feature]
        if a.notna().sum() >= 1 and b.notna().sum() >= 1:
            diff = a.median() - b.median()
            if np.isfinite(diff) and diff != 0:
                total += 1
                if np.sign(diff) == np.sign(overall):
                    agree += 1
    if total == 0:
        return (float("nan"), 0)
    return (agree / total, total)

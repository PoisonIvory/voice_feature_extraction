"""Tier B (hormone dose-response), Tier C (wearable), Tier D (articulatory/novel).

Tier B correlates each voice feature with measured hormones (e3g, PdG, E:P,
PdG rate-of-change), and controls for Oura temperature via partial correlation.
Tier C uses Oura temp/HRV to independently confirm phase and act as a hormone
proxy beyond the hormone window. Tier D brings in the HuBERT articulatory
d-primes and a composite voice-instability index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import paths, stats
from .explore import contrast_sweep
from .features import DPRIME_COLUMNS, classify_feature

# Compact peripheral-instability panel for the composite index (z-scored mean).
_INSTABILITY_FEATURES = [
    "prosody_egemaps_shimmerLocaldB_sma3nz_stddevNorm",
    "prosody_egemaps_jitterLocal_sma3nz_stddevNorm",
    "prosody_egemaps_logRelF0-H1-A3_sma3nz_amean",
    "prosody_egemaps_F2bandwidth_sma3nz_stddevNorm",
    "prosody_egemaps_F3bandwidth_sma3nz_stddevNorm",
]


def hormone_dose_response(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    sub = df[df["has_voice"] & df["has_hormones"]].copy()
    rows = []
    for feat in features:
        meta = classify_feature(feat)
        rho_e, p_e, n_e = stats.spearman(sub[feat], sub["e3g"])
        rho_p, p_p, _ = stats.spearman(sub[feat], sub["pdg"])
        rho_r, p_r, _ = stats.spearman(sub[feat], sub["ep_ratio"])
        rho_roc, p_roc, _ = stats.spearman(sub[feat], sub["pdg_roc"])
        pr_p, pr_pp, n_pr = stats.partial_spearman(sub[feat], sub["pdg"], sub["temp_deviation"])
        rows.append(
            {
                "feature": feat,
                "family": meta.family if meta else "",
                "axis": meta.axis if meta else "",
                "n": n_e,
                "rho_e3g": rho_e, "p_e3g": p_e,
                "rho_pdg": rho_p, "p_pdg": p_p,
                "rho_ep_ratio": rho_r, "p_ep_ratio": p_r,
                "rho_pdg_roc": rho_roc, "p_pdg_roc": p_roc,
                "partial_rho_pdg_given_temp": pr_p, "p_partial": pr_pp, "n_partial": n_pr,
            }
        )
    out = pd.DataFrame(rows)
    out["q_pdg"] = stats.benjamini_hochberg(out["p_pdg"].to_numpy())
    out["abs_rho_pdg"] = out["rho_pdg"].abs()
    return out.sort_values("abs_rho_pdg", ascending=False).reset_index(drop=True)


def wearable_phase_validation(df: pd.DataFrame) -> pd.DataFrame:
    """Independently confirm the luteal hormonal signature from Oura."""
    sub = df[df["has_oura"]]
    foll = sub["phase"].eq("follicular")
    lut = sub["phase"].eq("luteal")
    rows = []
    for metric, expected in [("temp_deviation", "higher in luteal"), ("hrv", "lower in luteal"),
                              ("resting_hr", "higher in luteal"), ("breath_rate", "higher in luteal")]:
        if metric not in sub.columns:
            continue
        a = sub.loc[lut, metric].to_numpy()
        b = sub.loc[foll, metric].to_numpy()
        rows.append(
            {
                "metric": metric,
                "expected": expected,
                "luteal_mean": float(np.nanmean(a)) if np.isfinite(a).any() else float("nan"),
                "follicular_mean": float(np.nanmean(b)) if np.isfinite(b).any() else float("nan"),
                "hedges_g_luteal_minus_foll": stats.hedges_g(a, b),
                "mw_p": stats.mann_whitney_p(a, b),
                "n_luteal": int(np.isfinite(a).sum()),
                "n_follicular": int(np.isfinite(b).sum()),
            }
        )
    return pd.DataFrame(rows)


def voice_vs_wearable(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Oura temp/HRV as a hormone proxy: correlate voice features on voice+Oura days."""
    sub = df[df["has_voice"] & df["has_oura"]].copy()
    rows = []
    for feat in features:
        meta = classify_feature(feat)
        rho_t, p_t, n_t = stats.spearman(sub[feat], sub["temp_deviation"])
        rho_h, p_h, _ = stats.spearman(sub[feat], sub["hrv"])
        rows.append(
            {
                "feature": feat, "family": meta.family if meta else "", "axis": meta.axis if meta else "",
                "n": n_t, "rho_temp": rho_t, "p_temp": p_t, "rho_hrv": rho_h, "p_hrv": p_h,
            }
        )
    out = pd.DataFrame(rows)
    out["abs_rho_temp"] = out["rho_temp"].abs()
    return out.sort_values("abs_rho_temp", ascending=False).reset_index(drop=True)


def dprime_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean articulatory d-prime joined to phase/hormone context."""
    dp = pd.read_parquet(paths.HUBERT_DPRIME)
    dp["date"] = pd.to_datetime(dp["recordedDate"])
    cols = [c for c in DPRIME_COLUMNS if c in dp.columns]
    daily = dp.groupby("date")[cols].mean().reset_index()
    keep = ["date", "cycle_start_date", "phase", "subphase", "cycle_day_from_ovulation",
            "has_hormones", "pdg", "e3g", "ep_ratio"]
    return daily.merge(df[keep], on="date", how="left")


def dprime_phase_contrast(dp_daily: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DPRIME_COLUMNS if c in dp_daily.columns]
    foll = dp_daily["phase"].eq("follicular")
    lut = dp_daily["phase"].eq("luteal")
    return contrast_sweep(dp_daily, cols, lut, foll, "luteal", "follicular")


def composite_instability_index(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day z-scored mean of the peripheral-instability panel (voice days)."""
    feats = [f for f in _INSTABILITY_FEATURES if f in df.columns]
    voice = df[df["has_voice"]].copy()
    z = pd.DataFrame(index=voice.index)
    for f in feats:
        s = voice[f]
        if s.notna().sum() >= 3 and s.std(ddof=0) > 0:
            z[f] = (s - s.mean()) / s.std(ddof=0)
    voice["voice_instability_index"] = z.mean(axis=1)
    return voice[["date", "phase", "subphase", "cycle_day_from_ovulation", "has_hormones",
                  "pdg", "e3g", "ep_ratio", "voice_instability_index"]]


def compute(df: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    results["hormone_dose_response"] = hormone_dose_response(df, features)
    results["wearable_phase_validation"] = wearable_phase_validation(df)
    results["voice_vs_wearable"] = voice_vs_wearable(df, features)
    dp_daily = dprime_daily(df)
    results["dprime_daily"] = dp_daily
    results["dprime_phase_contrast"] = dprime_phase_contrast(dp_daily)
    results["instability_index"] = composite_instability_index(df)
    return results

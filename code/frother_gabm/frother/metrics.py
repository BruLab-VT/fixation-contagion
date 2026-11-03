"""Statistics (bootstrap CIs, Hedges g), diversity, and SIS estimation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_mean(x, n_boot=3000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    if x.size == 1:
        return (float(x[0]),) * 3
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def contrast(a, b, n_boot=3000, seed=0) -> dict:
    """mean(a) - mean(b) with percentile-bootstrap CI and Hedges' g."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    if a.size < 2 or b.size < 2:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "g": np.nan,
                "n_a": int(a.size), "n_b": int(b.size), "excludes_zero": False}
    rng = np.random.default_rng(seed)
    d = (rng.choice(a, (n_boot, a.size)).mean(1) - rng.choice(b, (n_boot, b.size)).mean(1))
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                 / (a.size + b.size - 2))
    g = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    J = 1 - 3 / (4 * (a.size + b.size) - 9)
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    return {"diff": float(a.mean() - b.mean()), "lo": lo, "hi": hi,
            "g": float(g * J) if np.isfinite(g) else np.nan,
            "n_a": int(a.size), "n_b": int(b.size),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def cosine_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    return v @ v.T


def dpp_diversity(v, floor=1e-7) -> float:
    """Mean log-eigenvalue of the shifted cosine Gram matrix; higher = more diverse."""
    v = np.asarray(v, float)
    if v.ndim != 2 or len(v) < 2:
        return np.nan
    s = (cosine_matrix(v) + 1.0) / 2.0
    eig = np.linalg.eigvalsh(s)
    return float(np.mean(np.log(np.maximum(eig, floor))))


def feature_entropy(feature_lists: list[set]) -> float:
    """Shannon entropy (bits) of the feature-frequency distribution over a pool."""
    counts = {}
    for fs in feature_lists:
        for f in fs:
            counts[f] = counts.get(f, 0) + 1
    if not counts:
        return np.nan
    p = np.array(list(counts.values()), float); p /= p.sum()
    return float(-(p * np.log2(p)).sum())

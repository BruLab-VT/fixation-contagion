"""Re-grade every stored concept with the CURRENT extractor (LLM flag + text
verification) and rebuild the per-unit tables from the concepts.

No generation is repeated: concept texts are read from concepts.csv, LLM
extraction calls are served from the cache when the prompt is unchanged, and
verification is pure text. Embedding-based columns (dpp, similarity) are kept
from the existing units.csv because they do not depend on features.

A copy of the pre-rescore files is kept as *.before_rescore.csv the first time.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd

from .extract import verify
from .metrics import feature_entropy
from .schema import S1_BUNDLE, S1_FLAW
from .store import concept_text


def _backup(path):
    b = path.replace(".csv", ".before_rescore.csv")
    if os.path.exists(path) and not os.path.exists(b):
        shutil.copy(path, b)


VERIFY_ONLY = False     # set by rescore_all: apply text verification to the stored flags, no LLM


def _regrade(sess, df: pd.DataFrame, label: str) -> pd.DataFrame:
    texts = [concept_text(str(t), str(d)) for t, d in zip(df["title"].fillna(""), df["description"].fillna(""))]
    print(f"[rescore] {label}: re-grading {len(texts)} concepts" + (" (verification only)" if VERIFY_ONLY else ""))
    if VERIFY_ONLY:
        feats = [verify(set(json.loads(f)) if isinstance(f, str) else set(), t)
                 for f, t in zip(df["features"], texts)]
    else:
        feats = sess.extractor.extract_many(texts, seed=7)
    df = df.copy()
    df["features_before"] = df["features"]
    df["features"] = [json.dumps(sorted(f)) for f in feats]
    return df


def _sets(col):
    return col.apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())


def _keep_cols(old: pd.DataFrame, new: pd.DataFrame, key_cols, keep):
    """Carry embedding-based columns from the old units table by unit key."""
    if old is None:
        return new
    o = old.set_index(key_cols)
    n = new.set_index(key_cols)
    for c in keep:
        if c in o.columns:
            n[c] = o[c].reindex(n.index)
    return n.reset_index()


# --------------------------------------------------------------------------
def rescore_study1(sess, d):
    p = os.path.join(d, "concepts.csv")
    if not os.path.exists(p):
        return
    c = _regrade(sess, pd.read_csv(p), "study1"); fs = _sets(c["features"])
    c["marker_hits"] = fs.apply(lambda s: len(s & set(S1_BUNDLE)))
    c["flaw"] = fs.apply(lambda s: int(S1_FLAW in s))
    for m in S1_BUNDLE:
        c[f"has_{m}"] = fs.apply(lambda s: int(m in s))
    _backup(p); c.to_csv(p, index=False)
    old = pd.read_csv(os.path.join(d, "units.csv")); _backup(os.path.join(d, "units.csv"))
    rows = []
    for (brief, ex, rep), g in c.groupby(["brief", "exposure", "rep"]):
        gf = _sets(g["features"])
        rows.append({"job": "study1", "brief": brief, "exposure": ex, "rep": rep,
                     "marker_hits": g["marker_hits"].mean(), "full_bundle": (g["marker_hits"] == 3).mean(),
                     "flaw": g["flaw"].mean(),
                     **{f"adopt_{m}": g[f"has_{m}"].mean() for m in S1_BUNDLE},
                     "coverage": (gf.apply(len) > 0).mean(), "mean_features": gf.apply(len).mean()})
    new = _keep_cols(old, pd.DataFrame(rows), ["brief", "exposure", "rep"], ["self_first_sim", "dpp"])
    new.to_csv(os.path.join(d, "units.csv"), index=False)


def rescore_D2(sess, d):
    p = os.path.join(d, "concepts.csv")
    if not os.path.exists(p):
        return
    c = _regrade(sess, pd.read_csv(p), "D2"); fs = _sets(c["features"])
    _backup(p); c.to_csv(p, index=False)
    old = pd.read_csv(os.path.join(d, "units.csv")); _backup(os.path.join(d, "units.csv"))
    rows = []
    T = int(c["round"].max())
    for (mode, rep), g in c.groupby(["mode", "rep"]):
        f1 = list(_sets(g[g["round"] == 1]["features"])); fl = list(_sets(g[g["round"] == T]["features"]))
        rows.append({"job": "D2", "instruction": "revise", "mode": mode, "dose": int(g["dose"].iloc[0]), "seeded": 0, "rep": rep,
                     "entropy_round1": feature_entropy(f1), "entropy_last": feature_entropy(fl),
                     "unique_round1": len(set().union(*f1)), "unique_last": len(set().union(*fl)),
                     "mean_features_last": float(np.mean([len(x) for x in fl]))})
    new = _keep_cols(old, pd.DataFrame(rows), ["mode", "rep"], ["dpp_round1", "dpp_last"])
    new.to_csv(os.path.join(d, "units.csv"), index=False)


def rescore_profile(sess, d):
    p = os.path.join(d, "concepts.csv")
    if not os.path.exists(p):
        return
    from .extract import fixture_check, wild_precision_check
    gates = {}
    if not VERIFY_ONLY:
        gates["fixture"] = fixture_check(sess.extractor)
    c = _regrade(sess, pd.read_csv(p), "profile")
    _backup(p); c.to_csv(p, index=False)
    if not VERIFY_ONLY:
        pr = c[c["component"] == "prior"].head(60)
        texts = [concept_text(str(t), str(dd)) for t, dd in zip(pr["title"].fillna(""), pr["description"].fillna(""))]
        wp = wild_precision_check(sess.extractor, texts, os.path.join(d, "wild_precision.csv"))
        gates["wild_suspect"] = wp[wp.suspect]["feature"].tolist()
    with open(os.path.join(d, "grader_gates.json"), "w") as f:
        json.dump(gates, f, indent=2, default=float)


def rescore_all(sess, cfg, jobs=None, verify_only=False):
    global VERIFY_ONLY
    VERIFY_ONLY = verify_only
    for name, fn in (("study1", rescore_study1), ("D2", rescore_D2), ("profile", rescore_profile)):
        if jobs and name not in jobs:
            continue
        d = os.path.join(cfg.results_dir, name)
        if os.path.isdir(d):
            try:
                fn(sess, d)
            except Exception as e:      # noqa: BLE001
                print(f"[rescore] {name} failed: {e!r}")
    print("[rescore] done")

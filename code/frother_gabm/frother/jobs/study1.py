"""study1 — individual fixation, replicated on the new model.

brief {verbatim, neutral} x exposure {none, single, triple, memory}
x reps x iterations. Exemplars carry the three-feature bundle plus the flaw
(app_connected). Example shown at EVERY iteration, placed last. Unit = one run.

H1a exposure raises excess bundle adoption; H1a' dose; H1b brief moderates;
H1c flaw is NOT copied; H1d memory produces self-fixation in feature space.
"""
from __future__ import annotations

import json

import numpy as np

from ..metrics import dpp_diversity, cosine_matrix
from ..schema import EXAMPLE_HEADERS, S1_BUNDLE, S1_EXAMPLES, S1_FLAW, S1_FLAW_SENTENCE
from ..session import seed_of
from ..store import JobStore

JOB = "study1"
EXPOSURES = ("none", "single", "triple", "memory")


def _unit(sess, store, key, brief, exposure, rep):
    cfg = sess.cfg
    n_ex = {"none": 0, "single": 1, "triple": 3, "memory": 0}[exposure]
    ex_block = ""
    if n_ex:
        exs = [e + S1_FLAW_SENTENCE for e in S1_EXAMPLES[:n_ex]]
        ex_block = EXAMPLE_HEADERS["solo"][n_ex] + "\n" + "\n\n".join(exs)
    history, crows = [], []
    for it in range(cfg.study1_iterations):
        mem = ""
        if exposure == "memory" and history:
            mem = "Your previous concepts in this session:\n" + "\n\n".join(
                f"{i+1}. {h['text']}" for i, h in enumerate(history))
        c = sess.generate(brief=brief, blocks=[mem, ex_block],
                          seed=seed_of(cfg.master_seed, JOB, brief, exposure, rep, it),
                          ask="Propose your concept now." if not history else "Propose your next concept now.")
        history.append(c)
        crows.append({"job": JOB, "unit": key, "brief": brief, "exposure": exposure,
                      "rep": rep, "iteration": it + 1, "title": c["title"],
                      "description": c["description"], "rationale": c["rationale"],
                      "features": c["features"],
                      "marker_hits": len(set(c["features"]) & set(S1_BUNDLE)),
                      "flaw": int(S1_FLAW in c["features"]),
                      **{f"has_{m}": int(m in c["features"]) for m in S1_BUNDLE}})
    vecs = sess.embed([h["text"] for h in history])
    sim = cosine_matrix(vecs)
    self_first = float(sim[0, 1:].mean()) if len(history) > 1 else np.nan
    row = {"job": JOB, "brief": brief, "exposure": exposure, "rep": rep,
           "marker_hits": float(np.mean([r["marker_hits"] for r in crows])),
           "full_bundle": float(np.mean([r["marker_hits"] == 3 for r in crows])),
           "flaw": float(np.mean([r["flaw"] for r in crows])),
           **{f"adopt_{m}": float(np.mean([r[f"has_{m}"] for r in crows])) for m in S1_BUNDLE},
           "self_first_sim": self_first,
           "dpp": dpp_diversity(vecs),
           "coverage": float(np.mean([len(r["features"]) > 0 for r in crows])),
           "mean_features": float(np.mean([len(r["features"]) for r in crows]))}
    store.commit(key, row, crows)


def run(sess, cfg):
    store = JobStore(cfg.results_dir, JOB)
    keys, spec = [], {}
    for brief in ("neutral", "verbatim"):
        for ex in EXPOSURES:
            for rep in range(cfg.study1_reps):
                k = f"{brief}|{ex}|{rep}"; keys.append(k); spec[k] = (brief, ex, rep)
    store.run_units(keys, lambda k: _unit(sess, store, k, *spec[k]), cfg.workers)

"""D2 — convergence vs contagion (the missing control).

Instruction fixed to `revise`, NO seed. mode x reps:
    nominal : agents revise alone (no teammate concepts shown)
    int3    : each agent sees 3 random teammates' latest concepts
    int7    : each agent sees every teammate's latest concept
Fully connected team of team_size, d2_rounds rounds, own history capped at d_history.
All features are tracked; exposures are recorded so the endogenous-contagion
analysis can estimate R0 per feature in every cell.

H6a interacting teams converge on more features than nominal teams (rise of
    the top endogenous features higher; pool entropy lower at the last round);
H6b R0 for the top endogenous feature is >= 1 under interaction and < 1 nominal;
H6c visibility is a dose: int7 >= int3 >= nominal on convergence.
"""
from __future__ import annotations

import numpy as np

from ..metrics import dpp_diversity, feature_entropy
from ..schema import EXAMPLE_HEADERS, PERSONAS
from ..session import seed_of
from ..store import JobStore
from ..team_core import ASK, MEM_HEAD

JOB = "D2"
MODES = {"nominal": 0, "int3": 3, "int7": 7}


def _unit(sess, store, key, mode, rep):
    cfg = sess.cfg
    n, T, dose = cfg.team_size, cfg.d2_rounds, MODES[mode]
    rng = np.random.default_rng(seed_of(cfg.master_seed, JOB, mode, rep))
    personas = list(PERSONAS[:n]) + [f"{PERSONAS[i % len(PERSONAS)]} (2)" for i in range(len(PERSONAS), n)]
    rng.shuffle(personas)
    hist = {a: [] for a in range(n)}
    crows, exposures = [], []
    for t in range(T):
        latest = {a: hist[a][-1] for a in range(n) if hist[a]}
        for a in range(n):
            others = [b for b in latest if b != a]
            rng.shuffle(others)
            vis = others[:dose]
            own = hist[a][-cfg.d_history:] if cfg.d_history > 0 else hist[a]
            mem = ""
            if own:
                mem = MEM_HEAD["revise"] + "\n" + "\n\n".join(f"{i+1}. {h['text']}" for i, h in enumerate(own))
            obs = ""
            if vis:
                obs = EXAMPLE_HEADERS["teammates"] + "\n" + "\n\n".join(latest[b]["text"] for b in vis)
            ask = "Propose your concept now." if not hist[a] else ASK["revise"]
            c = sess.generate(brief="neutral", blocks=[mem, obs], persona=personas[a],
                              setting=("participating in a team design session" if dose else
                                       "working alone on a design session"),
                              seed=seed_of(cfg.master_seed, JOB, mode, rep, t, a), ask=ask)
            hist[a].append(c)
            for b in vis:
                exposures.append({"round": t + 1, "viewer": a, "source": b})
            crows.append({"job": JOB, "unit": key, "instruction": "revise", "mode": mode, "dose": dose,
                          "seeded": 0, "rep": rep, "round": t + 1, "agent": a,
                          "visible": sorted(vis), "features": c["features"], "title": c["title"],
                          "description": c["description"], "rationale": c["rationale"]})
    last = [hist[a][-1] for a in range(n)]
    first = [hist[a][0] for a in range(n)]
    v_last, v_first = sess.embed([h["text"] for h in last]), sess.embed([h["text"] for h in first])
    row = {"job": JOB, "instruction": "revise", "mode": mode, "dose": dose, "seeded": 0, "rep": rep,
           "entropy_round1": feature_entropy([set(h["features"]) for h in first]),
           "entropy_last": feature_entropy([set(h["features"]) for h in last]),
           "dpp_round1": dpp_diversity(v_first), "dpp_last": dpp_diversity(v_last),
           "unique_round1": len(set().union(*[set(h["features"]) for h in first])),
           "unique_last": len(set().union(*[set(h["features"]) for h in last])),
           "mean_features_last": float(np.mean([len(h["features"]) for h in last]))}
    store.commit(key, row, crows, extra={"exposures.csv": [
        {"unit": key, "instruction": "revise", "mode": mode, "dose": dose, "seeded": 0, "rep": rep, **e} for e in exposures]})


def run(sess, cfg):
    store = JobStore(cfg.results_dir, JOB)
    keys, spec = [], {}
    for mode in MODES:
        for rep in range(cfg.d2_reps):
            k = f"{mode}|{rep}"; keys.append(k); spec[k] = (mode, rep)
    store.run_units(keys, lambda k: _unit(sess, store, k, *spec[k]), cfg.workers)

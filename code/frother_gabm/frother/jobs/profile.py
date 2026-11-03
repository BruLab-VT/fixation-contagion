"""profile — the attractive-feature protocol for one model.

Four elicitations, each run in `prof_halves` independent replicates (halves)
for split-half reliability. All concepts are stored UNGRADED (features empty)
when --no-extract is set; grading is batched afterwards with `rescore` using a
fixed grader model, so every subject model is measured with the same instrument.

  prior   solo, unexposed, neutral brief                 -> default set (base rates)
  drift   solo, own-history memory, 8 concepts           -> drift attractors (rise 1 -> 8)
  social  8-agent teams, revise, see-all, 8 rounds       -> social attractors (R0 per feature)
          + revise-alone control teams                    (elaboration drift)
  scan    one exemplar per feature: 4 fixed shared      -> exposure response (copy / avoid)
          + this model's 10 rarest hardware features        under the neutral instruction
          + this model's most common mechanism (contrast on its own default)
          The 4 fixed + own default are ALSO run under the directive instruction
          ("build directly on the example") -> copyability ceiling; the gap between
          directive and neutral is the model's disposition to differentiate.

Candidate selection for the scan uses vocabulary-pattern detection on the
prior concepts (no grader needed inside the job) and is written to
scan_candidates.json for the record.
"""
from __future__ import annotations

import json
from collections import Counter

from ..extract import pattern_features
from ..schema import EXAMPLE_HEADERS, SCHEMA, stimulus
from ..session import seed_of
from ..store import JobStore
from ..team_core import run_team
from .permission import ASKS

JOB = "profile"
FIXED_SCAN = ("integrated_heating", "magnetic_drive", "ultrasonic", "double_wall_insulation")
NOT_SCANNABLE = {"one_touch_auto", "app_connected", "self_cleaning", "dishwasher_safe",
                 "auto_stop_sensing", "froth_texture_setting", "cold_froth_mode", "temperature_control"}


def _crow(base, c, **kw):
    return {**base, **kw, "title": c["title"], "description": c["description"], "rationale": c["rationale"],
            "features": c["features"], "features_pattern": c["features_pattern"]}


def _prior(sess, store, key, half, i):
    c = sess.generate(brief="neutral", blocks=[], seed=seed_of(sess.cfg.master_seed, JOB, "prior", half, i))
    store.commit(key, None, [_crow({"job": JOB, "component": "prior", "half": half, "mode": "solo", "rep": i,
                                    "round": 1, "agent": 0, "exemplar": ""}, c)])


def _drift(sess, store, key, half, r):
    cfg = sess.cfg
    hist, crows = [], []
    for it in range(cfg.prof_drift_iters):
        mem = ""
        if hist:
            mem = "Your previous concepts in this session:\n" + "\n\n".join(f"{i+1}. {h['text']}" for i, h in enumerate(hist))
        c = sess.generate(brief="neutral", blocks=[mem], seed=seed_of(cfg.master_seed, JOB, "drift", half, r, it),
                          ask="Propose your concept now." if not hist else "Propose your next concept now.")
        hist.append(c)
        crows.append(_crow({"job": JOB, "component": "drift", "half": half, "mode": "memory", "rep": r,
                            "round": it + 1, "agent": 0, "exemplar": ""}, c))
    store.commit(key, None, crows)


def _social(sess, store, key, half, mode, r):
    cfg = sess.cfg
    dose = 0 if mode == "nominal" else cfg.team_size - 1
    tag = {"job": JOB, "component": "social", "half": half, "mode": mode, "rep": r, "exemplar": ""}
    crows, ex = run_team(sess, n=cfg.team_size, rounds=cfg.prof_rounds, dose=dose,
                         seed_parts=(JOB, "social", half, mode, r), tag=tag)
    store.commit(key, None, crows, extra={"exposures.csv": ex} if ex else None)


def _scan(sess, store, key, half, feature, i, instruction="neutral"):
    block = EXAMPLE_HEADERS["prior"] + "\n" + stimulus(0, feature)
    c = sess.generate(brief="neutral", blocks=[block], ask=ASKS[instruction],
                      seed=seed_of(sess.cfg.master_seed, JOB, "scan", instruction, half, feature, i))
    store.commit(key, None, [_crow({"job": JOB, "component": "scan", "half": half, "mode": instruction, "rep": i,
                                    "round": 1, "agent": 0, "exemplar": feature}, c)])


def select_scan_candidates(store, cfg) -> dict:
    import pandas as pd
    c = pd.read_csv(store.path("concepts.csv"))
    c = c[c["component"] == "prior"]
    fp = c["features_pattern"].apply(lambda s: set(json.loads(s)) if isinstance(s, str) else set())
    n = max(1, len(fp))
    rate = {f.fid: sum(f.fid in s for s in fp) / n for f in SCHEMA.features}
    hardware = [f.fid for f in SCHEMA.features if f.hardware and f.fid not in FIXED_SCAN and f.fid not in NOT_SCANNABLE]
    rarest = sorted(hardware, key=lambda f: (rate[f], f))[:cfg.prof_scan_rarest]
    mechs = [f.fid for f in SCHEMA.features if f.facet == "mechanism"]
    top_default = max(mechs, key=lambda f: rate[f])
    return {"fixed": list(FIXED_SCAN), "rarest": rarest, "top_default": top_default,
            "pattern_rates": rate, "n_prior_concepts": int(n)}


def run(sess, cfg):
    store = JobStore(cfg.results_dir, JOB)
    H = cfg.prof_halves
    # phase 1: prior + drift + social (independent of candidates)
    keys, spec = [], {}
    for h in range(H):
        for i in range(cfg.prof_prior_n):
            k = f"prior|{h}|{i}"; keys.append(k); spec[k] = (_prior, (h, i))
        for r in range(cfg.prof_drift_runs):
            k = f"drift|{h}|{r}"; keys.append(k); spec[k] = (_drift, (h, r))
        for r in range(cfg.prof_social_teams):
            k = f"social|{h}|int|{r}"; keys.append(k); spec[k] = (_social, (h, "interacting", r))
        for r in range(cfg.prof_nominal_teams):
            k = f"social|{h}|nom|{r}"; keys.append(k); spec[k] = (_social, (h, "nominal", r))
    store.run_units(keys, lambda k: spec[k][0](sess, store, k, *spec[k][1]), cfg.workers)
    # phase 2: choose candidates from prior concepts, then scan
    cand = select_scan_candidates(store, cfg)
    with open(store.path("scan_candidates.json"), "w") as f:
        json.dump(cand, f, indent=2)
    exemplars = cand["fixed"] + cand["rarest"] + [cand["top_default"]]
    store.log(f"scan exemplars ({len(exemplars)}): fixed={cand['fixed']} rarest={cand['rarest']} default={cand['top_default']}")
    keys, spec = [], {}
    for h in range(H):
        for feat in exemplars:
            for i in range(cfg.prof_scan_reps):
                k = f"scan|{h}|{feat}|{i}"; keys.append(k); spec[k] = (_scan, (h, feat, i))
        for feat in cand["fixed"] + [cand["top_default"]]:          # copyability ceiling
            for i in range(cfg.prof_scan_reps):
                k = f"scand|{h}|{feat}|{i}"; keys.append(k); spec[k] = (_scan, (h, feat, i, "directive"))
    store.run_units(keys, lambda k: spec[k][0](sess, store, k, *spec[k][1]), cfg.workers)
    with open(store.path("model.json"), "w") as f:
        json.dump({"model": cfg.model, "chat_kwargs": cfg.chat_kwargs, "temperature": cfg.temperature,
                   "extracted_in_job": cfg.extract}, f, indent=2)
    store.log("profile complete" + ("" if cfg.extract else " (ungraded; run rescore with the grader model)"))

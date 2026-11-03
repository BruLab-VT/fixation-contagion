#!/usr/bin/env python3
"""Run the experiment jobs.

    python run_all.py all                 # study1, P and D2 in order, then analysis
    python run_all.py P D2                # only these jobs (then analysis)
    python run_all.py analyze             # figures/tables/report from existing CSVs
    python run_all.py all --quick         # tiny smoke-size run of everything
    python run_all.py check               # is the LLM endpoint reachable?
    python run_all.py rescore             # re-grade all stored concepts with the current
                                          # extractor + verification, rebuild units.csv, analyze
    python run_all.py profile --no-extract        # attractive-feature protocol for one model (ungraded)
    python run_all.py profile-analyze              # per-model profile matrix from graded concepts
    python run_all.py cross --profile-root results_profiles   # cross-model comparison

Jobs: study1, P, D2, profile. Every job resumes: rerunning skips finished units.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from frother.config import add_config_args, config_from_args
from frother.session import Session
from frother.analyze import analyze_all
from frother.jobs import study1, teams_dynamics, profile, permission
from frother.rescore import rescore_all

JOBS = {"study1": study1.run, "P": permission.run, "D2": teams_dynamics.run, "profile": profile.run}
ORDER = ["study1", "P", "D2"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", nargs="+", help="all | check | analyze | any of " + " ".join(ORDER))
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--profile-root", default="results_profiles", help="cross: directory holding one results dir per model")
    p.add_argument("--models-tsv", default="slurm/models.tsv", help="cross: model metadata (family, size, reasoning)")
    p.add_argument("--verify-only", action="store_true",
                   help="rescore: apply text verification to stored flags without any LLM calls")
    add_config_args(p)
    a = p.parse_args()
    cfg = config_from_args(a)
    sess = Session(cfg)

    want = a.jobs
    if "check" in want:
        ok = sess.check()
        sys.exit(0 if ok else 1)
    if want == ["analyze"]:
        analyze_all(cfg)
        return
    if want == ["rescore"]:
        rescore_all(sess, cfg, verify_only=a.verify_only)
        if os.path.isdir(os.path.join(cfg.results_dir, "profile")):
            from frother.profile_analysis import profile_model
            try:
                profile_model(cfg.results_dir, cfg.figures_dir)
            except Exception as e:      # noqa: BLE001
                print("[profile-analyze]", e)
        analyze_all(cfg)
        return
    if want == ["profile-analyze"]:
        from frother.profile_analysis import profile_model
        s_ = profile_model(cfg.results_dir, cfg.figures_dir)
        print(json.dumps({k: v for k, v in s_.items() if k != "scan"}, indent=2, default=float))
        return
    if want == ["cross"]:
        from frother.profile_analysis import cross_model
        cross_model(a.profile_root, cfg.figures_dir, a.models_tsv)
        return
    if want == ["profile"]:
        jobs = ["profile"]
    else:
        jobs = ORDER if "all" in want else [j for j in ORDER if j in want]
    unknown = [j for j in want if j not in JOBS and j not in ("all", "analyze", "rescore", "profile-analyze", "cross")]
    if unknown:
        sys.exit(f"unknown job(s): {unknown}. Choose from {ORDER}")
    if not sess.check():
        sys.exit("Endpoint check failed. Is Ollama running? (open the Ollama app or run `ollama serve`)")

    t0 = time.time()
    for j in jobs:
        print(f"\n===== JOB {j} ({time.strftime('%H:%M:%S')}) =====")
        JOBS[j](sess, cfg)
        print(f"[{j}] LLM calls made: {sess.client.n_calls}, served from cache: {sess.client.n_cached}")
    print(f"\nall jobs finished in {(time.time() - t0) / 60:.1f} min")
    if not a.no_analyze and jobs != ["profile"]:
        analyze_all(cfg)


if __name__ == "__main__":
    main()

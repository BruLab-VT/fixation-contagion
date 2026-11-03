#!/usr/bin/env python3
"""End-to-end check with a fake LLM (no Ollama needed). ~1 minute.
Runs study1, P and D2 at tiny size into results_smoke/, then the analysis,
the re-grader and the analysis again."""
import os
import shutil
import sys

from frother.config import Config
from frother.session import Session
from frother.analyze import analyze_all
from frother.jobs import study1, permission, teams_dynamics
from frother.rescore import rescore_all

cfg = Config(backend="mock", results_dir="results_smoke", figures_dir="figures_smoke",
             cache_dir="cache_smoke").quick().resolve()
for d in (cfg.results_dir, cfg.figures_dir, cfg.cache_dir):
    shutil.rmtree(d, ignore_errors=True)
cfg.resolve()
sess = Session(cfg)
for name, job in (("study1", study1), ("P", permission), ("D2", teams_dynamics)):
    print(f"--- {name}")
    job.run(sess, cfg)
analyze_all(cfg)
print("--- rescore")
rescore_all(sess, cfg)
analyze_all(cfg)
need = ["results_smoke/study1/units.csv", "results_smoke/study1/concepts.csv",
        "results_smoke/study1/units.before_rescore.csv", "figures_smoke/study1_fixation.png",
        "results_smoke/P/units.csv", "figures_smoke/P_permission.png",
        "results_smoke/D2/units.csv", "results_smoke/D2/exposures.csv", "figures_smoke/D2_convergence.png",
        "results_smoke/D2/table_endogenous_D2.csv", "figures_smoke/D2_endogenous_trajectories.png",
        "results_smoke/REPORT.md"]
missing = [p for p in need if not os.path.exists(p)]
print("\nSMOKE TEST:", "PASS" if not missing else f"FAIL, missing {missing}")
sys.exit(1 if missing else 0)

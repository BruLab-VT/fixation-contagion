#!/usr/bin/env python3
"""End-to-end check of the profile protocol with the fake LLM: three fake
'models' run ungraded, are batch-graded, profiled, and compared."""
import os, shutil, sys, json
from frother.config import Config
from frother.session import Session
from frother.jobs import profile
from frother.rescore import rescore_all
from frother.profile_analysis import profile_model, cross_model

root = "results_profiles_smoke"; figs = "figures_profiles_smoke"
shutil.rmtree(root, ignore_errors=True); shutil.rmtree(figs, ignore_errors=True); shutil.rmtree("cache_psmoke", ignore_errors=True)
for name in ("fake-a", "fake-b", "fake-c"):
    cfg = Config(backend="mock", model=name, results_dir=f"{root}/{name}", figures_dir=figs,
                 cache_dir="cache_psmoke", extract=False, embed=False, master_seed=hash(name) % 1000).quick().resolve()
    sess = Session(cfg)
    profile.run(sess, cfg)                                   # ungraded
    cfg.extract = True; cfg.extract_model = "fake-grader"
    rescore_all(Session(cfg), cfg, jobs=["profile"])          # batched grading
    profile_model(cfg.results_dir, figs)
os.makedirs("arc", exist_ok=True)
cross_model(root, figs, "arc/models.tsv" if os.path.exists("arc/models.tsv") else None)
need = [f"{root}/fake-a/profile/profile_matrix.csv", f"{root}/fake-a/profile/scan_candidates.json",
        f"{root}/CROSS_MODEL_REPORT.md", f"{figs}/cross_default.png", f"{figs}/profile_fake-a.png"]
missing = [p for p in need if not os.path.exists(p)]
print("PROFILE SMOKE:", "PASS" if not missing else f"FAIL {missing}")
sys.exit(1 if missing else 0)

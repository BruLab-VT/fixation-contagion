# frother_gabm_paper (v0.7.0)

The code behind *Fixation Contagion*


## Files

```
run_all.py                 command line: run jobs, analyze, rescore, profile-analyze, cross
smoke_test.py              mock-LLM self-test of study1, P, D2 (~1 min)      -> SMOKE TEST: PASS
smoke_profile.py           mock-LLM self-test of profile + cross-model report -> PROFILE SMOKE: PASS
frother/config.py          run sizes, temperature 0.7, 400-token cap, paths, --quick sizes
frother/schema.py          the 36 features in 5 facets, word lists, design brief, base example, personas
frother/llm.py             Ollama / vLLM / mock clients, sqlite request cache, embeddings
frother/extract.py         two-stage feature coder (JSON call + word-list verification) and grader gates
frother/session.py         builds one prompt, calls the model, parses and codes the concept
frother/store.py           units.csv / concepts.csv writers; done.txt makes every job resumable
frother/team_core.py       one team session (used by D2 logic and profile); the revise prompt text
frother/metrics.py         bootstrap CI, Hedges' g contrast, cosine matrix, DPP diversity, entropy
frother/jobs/study1.py     job study1
frother/jobs/permission.py job P (holds the three instructions and the exemplar builder)
frother/jobs/teams_dynamics.py  job D2
frother/jobs/profile.py    job profile
frother/analyze.py         REPORT.md, tables and figures for study1, P and D2
frother/contagion.py       adoption / abandonment per round, p, c, gamma, R0 and dose-response for D2
frother/profile_analysis.py per-model profile and cross-model report for Study 3
frother/rescore.py         re-grade stored concepts with a fixed grader model (study1, D2, profile)
slurm/                     ARC wrappers for job profile: models.tsv, submit_profiles.sh, run_profile.sh,
                           run_grade.sh, preflight.sh, watch.sh, README_ARC.md, account.conf (template)
```

## Run

```
pip3 install -r requirements.txt
python3 smoke_test.py && python3 smoke_profile.py     # no model needed
python3 run_all.py check                              # Ollama reachable?
python3 run_all.py study1 P D2                        # the laptop studies, then analysis
python3 run_all.py analyze                            # re-draw from existing CSVs
bash slurm/submit_profiles.sh                         # Study 3 on ARC (edit slurm/account.conf first)
python3 run_all.py cross --profile-root results_profiles
```

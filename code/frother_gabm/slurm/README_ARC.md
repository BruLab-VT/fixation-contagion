# Attractive-feature profiles on ARC (Falcon)

One job per model serves it with vLLM and runs the protocol ungraded; one
grading job afterwards grades every model with the same grader and builds the
per-model and cross-model results. Everything resumes.

## One-time setup

```
scp ~/Downloads/frother_gabm.zip <pid>@falcon1.arc.vt.edu:~/      # from your Mac
ssh <pid>@falcon1.arc.vt.edu
unzip -o -q frother_gabm.zip && cd frother_gabm
module reset; module load Miniforge3
python3 -m pip install --user -r requirements.txt
nano slurm/account.conf            # ONE line: ACCOUNT=your_allocation  (see: quota)
bash slurm/preflight.sh            # must end with ALL CHECKS PASSED
```

## Step 1 — shake-out (tiny jobs, 30 min each)

```
QUICK=1 bash slurm/submit_profiles.sh
bash slurm/watch.sh -f             # Ctrl-C to stop watching
```

A model that loads prints `[vllm] ready` and `profile complete` in
`logs/profile-<model>-<job>.log`; one that does not prints the last 40 lines of
its vLLM log with the reason. Fix or drop that row in `slurm/models.tsv`.
Then clear the tiny results (they must not be resumed by the real run):

```
scancel -u $USER
rm -rf results_profiles cache_profiles figures_profiles logs/model_env_*
```

## Step 2 — the real runs, with grading queued to follow

```
GRADE_AFTER=1 bash slurm/submit_profiles.sh
```

Submits all twelve model jobs (each with its own GPUs/memory/wall-time) and a
grading job that starts after they have all finished, succeeded or not. Or by
tier / by name: `bash slurm/submit_profiles.sh core`, `... extension`,
`... gpt-oss-20b phi-4`, then `sbatch --account=<acct> slurm/run_grade.sh` when
everything is in.

Watch: `bash slurm/watch.sh -f`. Resubmit anything that hit its time limit
with the same command (finished units are skipped), optionally
`TIME_OVERRIDE=3:00:00`.

## Step 3 — results

```
results_profiles/<model>/profile/profile_matrix.csv, profile_summary.json,
                                 profile_reliability.csv, grader_gates.json
results_profiles/CROSS_MODEL_REPORT.md, cross_<property>.csv, *_spearman.csv
figures_profiles/profile_<model>.png, cross_<property>.png, cross_dendrogram.png
```

Copy home: `scp -r <pid>@falcon1.arc.vt.edu:~/frother_gabm/results_profiles ~/Desktop/`

## Locally generated models

A model profiled on the Mac (Ollama, `--no-extract`) can be dropped into
`results_profiles/<name>/` on ARC before the grading job; use the `name` from
`slurm/models.tsv` so the cross-model analysis finds its metadata.

## Wall-times (about 2x the expected run)

| models | GPUs | expected | requested |
|---|---|---|---|
| llama-3.1-8b, ministral-8b | 1 | ~30 min | 1:00 |
| gemma-3-12b, qwen2.5-14b, phi-4, gpt-oss-20b | 1 | ~40 min | 1:30 |
| mistral-small-24b, gemma-3-27b, qwen3-32b-nothink, gpt-oss-120b | 2 | ~1 h | 2:00 |
| llama-3.3-70b | 4 | ~1.5 h | 4:00 |
| qwen3-32b-think | 2 | ~3 h | 6:00 |
| grading | 1 | ~1.5 h | 3:00 |

`QUICK=1` jobs are submitted at 0:30. `PARTITION_OVERRIDE=l40s_preemptable_q`
in `account.conf` usually shortens waits; a preempted job is just a resubmit.

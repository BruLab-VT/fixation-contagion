#!/bin/bash
# ============================================================================
# Submit the attractive-feature profile jobs, one per model, each with ITS OWN
# GPUs, memory and wall-time from slurm/models.tsv.
#
#     bash slurm/submit_profiles.sh                    # every model
#     bash slurm/submit_profiles.sh core               # one tier (core|extension)
#     bash slurm/submit_profiles.sh gpt-oss-20b phi-4  # named models
#     QUICK=1 bash slurm/submit_profiles.sh            # tiny shake-out of every model (30 min jobs)
#     GRADE_AFTER=1 bash slurm/submit_profiles.sh      # also queue the grading job to run after all of them
#     DRY_RUN=1 bash slurm/submit_profiles.sh          # print the sbatch lines only
#     TIME_OVERRIDE=3:00:00 bash slurm/submit_profiles.sh core   # longer wall-time (resubmits resume)
#
# Each model's hardware and flags are written to logs/model_env_<name>.sh and
# sourced by the job, rather than passed through `sbatch --export` (which
# splits on commas and would break JSON chat kwargs).
# ============================================================================

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

if [[ -f slurm/account.conf ]]; then
  _ENV_ACCOUNT="${ACCOUNT:-}"
  # shellcheck disable=SC1091
  source slurm/account.conf
  [[ -n "$_ENV_ACCOUNT" ]] && ACCOUNT="$_ENV_ACCOUNT"
fi
PROJECT_DIR="${PROJECT_DIR:-$PWD}"
ACCOUNT="${ACCOUNT:-}"
PARTITION="${PARTITION_OVERRIDE:-l40s_normal_q}"
RESULTS_ROOT="${RESULTS_ROOT:-results_profiles}"
QUICK="${QUICK:-0}"; DRY_RUN="${DRY_RUN:-0}"; GRADE_AFTER="${GRADE_AFTER:-0}"
PROFILE_ARGS="${PROFILE_ARGS:-}"
[[ "$QUICK" == "1" ]] && PROFILE_ARGS="$PROFILE_ARGS --quick"
mkdir -p logs "$RESULTS_ROOT"

if [[ "$ACCOUNT" == "<account_name>" || -z "$ACCOUNT" ]]; then
  echo "FATAL: no Slurm account. Edit ONE line: slurm/account.conf -> ACCOUNT=your_alloc"
  echo "  your allocations:"; quota 2>/dev/null | head -8
  exit 1
fi

# which rows?
SELECT=("$@")
want() {  # name tier -> 0 if selected
  [[ ${#SELECT[@]} -eq 0 ]] && return 0
  for s in "${SELECT[@]}"; do [[ "$s" == "$1" || "$s" == "$2" || "$s" == "all" ]] && return 0; done
  return 1
}

echo "=================================================================="
echo "SUBMITTING  account=$ACCOUNT  partition=$PARTITION  results=$RESULTS_ROOT  quick=$QUICK"
echo "=================================================================="
rm -f logs/.submitted_jobids
tail -n +2 slurm/models.tsv | while IFS= read -r row; do
  name=$(echo "$row" | cut -f1);  path=$(echo "$row" | cut -f2);  served=$(echo "$row" | cut -f3)
  gpus=$(echo "$row" | cut -f4);  cpus=$(echo "$row" | cut -f5);  mem=$(echo "$row" | cut -f6)
  wt=$(echo "$row" | cut -f7);    tier=$(echo "$row" | cut -f11)
  ck=$(echo "$row" | cut -f12);   sflags=$(echo "$row" | cut -f13); rargs=$(echo "$row" | cut -f14)
  [[ -z "$name" ]] && continue
  want "$name" "$tier" || continue
  [[ "$QUICK" == "1" ]] && wt="0:30:00"
  [[ -n "${TIME_OVERRIDE:-}" ]] && wt="$TIME_OVERRIDE"
  if [[ ! -d "$path" ]]; then echo "SKIP   $name  (weights not found: $path)"; continue; fi

  ENVFILE="logs/model_env_${name}.sh"
  {
    echo "MODEL_NAME=$name"
    echo "MODEL_PATH=$path"
    echo "SERVED_NAME=$served"
    echo "TENSOR_PARALLEL=$gpus"
    printf "SERVE_FLAGS=%q\n" "$sflags"
    printf "CHAT_KWARGS=%q\n" "$ck"
    printf "RUN_ARGS=%q\n" "$rargs"
    printf "PROFILE_ARGS=%q\n" "$PROFILE_ARGS"
    echo "RESULTS_ROOT=$RESULTS_ROOT"
  } > "$ENVFILE"

  echo "SUBMIT $name  [$PARTITION gpu:l40s:$gpus cpus=$cpus mem=$mem time=$wt]"
  CMD=(sbatch --account="$ACCOUNT" --job-name="$name" --partition="$PARTITION"
       --gres="gpu:l40s:$gpus" --cpus-per-task="$cpus" --mem="$mem" --time="$wt"
       --export="ALL,PROJECT_DIR=$PROJECT_DIR,MODEL_ENV_FILE=$PROJECT_DIR/$ENVFILE"
       slurm/run_profile.sh)
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  DRY: %q ' "${CMD[@]}"; echo
  else
    OUT=$("${CMD[@]}") && { echo "  -> $OUT"; echo "${OUT##* }" >> logs/.submitted_jobids; } || echo "  -> SUBMIT FAILED: $OUT"
  fi
done

if [[ "$GRADE_AFTER" == "1" && "$DRY_RUN" != "1" && -f logs/.submitted_jobids ]]; then
  DEP=$(paste -sd: logs/.submitted_jobids)
  echo "SUBMIT grading job  [after: $DEP]"
  OUT=$(sbatch --account="$ACCOUNT" --dependency="afterany:${DEP}" \
        --export="ALL,PROJECT_DIR=$PROJECT_DIR,RESULTS_ROOT=$RESULTS_ROOT" slurm/run_grade.sh) \
    && echo "  -> $OUT" || echo "  -> SUBMIT FAILED: $OUT"
fi
rm -f logs/.submitted_jobids

echo "=================================================================="
echo "watch with:  bash slurm/watch.sh        (or -f to refresh every 30s)"
echo "resume    :  re-run this same command; finished units are skipped."
echo "=================================================================="

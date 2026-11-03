#!/bin/bash
#SBATCH --job-name=grade
#SBATCH --account=<account_name>
#SBATCH --partition=l40s_normal_q
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/grade-%j.log
#SBATCH --error=logs/grade-%j.log

# ============================================================================
# GRADE EVERYTHING with one fixed grader (qwen2.5-14b), then build the
# per-model profiles and the cross-model comparison.
#
#     sbatch --account=<acct> slurm/run_grade.sh
# or queue it automatically after the model jobs:
#     GRADE_AFTER=1 bash slurm/submit_profiles.sh
#
# Re-runnable: grading calls are cached, so a second pass is cheap.
# ============================================================================

set -uo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/frother_gabm}"
RESULTS_ROOT="${RESULTS_ROOT:-results_profiles}"
GRADER_PATH="${GRADER_PATH:-/common/data/models/qwen--Qwen2.5-14B-Instruct}"
GRADER_NAME="${GRADER_NAME:-qwen2.5-14b-grader}"
export PYTHONUNBUFFERED=1
PORT="${PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 1000) ))}"
API_KEY="grader-$(hostname)-${SLURM_JOB_ID:-local}"

cd "$PROJECT_DIR" || { echo "FATAL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs figures_profiles
echo "=================================================================="
echo "GRADING  job ${SLURM_JOB_ID:-local} on $(hostname)  started $(date -Is)"
echo "grader   : $GRADER_PATH  ->  $RESULTS_ROOT/*/profile/concepts.csv"
echo "=================================================================="

module reset 2>/dev/null
module load Miniforge3 2>/dev/null || module load Anaconda3 2>/dev/null
PY="$(command -v python3)"
[[ -z "$PY" ]] && { echo "FATAL: no python3 after Miniforge3"; exit 1; }
"$PY" -c "import requests, pandas, numpy, matplotlib, scipy" 2>/dev/null || { echo "FATAL: stack incomplete in $PY -- $PY -m pip install --user -r requirements.txt"; exit 1; }
module load vLLM 2>/dev/null || { echo "FATAL: module load vLLM failed"; exit 1; }

VLLM_PID=""
cleanup() { local rc=$?; if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID"; sleep 5; kill -9 "$VLLM_PID" 2>/dev/null; fi; echo "finished : $(date -Is) exit=$rc"; exit $rc; }
trap cleanup EXIT INT TERM

VLLM_LOG="logs/vllm-grader-${SLURM_JOB_ID:-local}.log"
vllm serve "$GRADER_PATH" --served-model-name "$GRADER_NAME" --api-key "$API_KEY" --host 127.0.0.1 --port "$PORT" \
  --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.90 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
READY=0
for i in $(seq 1 90); do
  kill -0 "$VLLM_PID" 2>/dev/null || { echo "FATAL: grader vLLM died:"; tail -40 "$VLLM_LOG"; exit 1; }
  curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { READY=1; echo "[vllm] grader ready after $((i*10))s"; break; }
  sleep 10
done
[[ $READY -eq 1 ]] || { echo "FATAL: grader not ready"; tail -40 "$VLLM_LOG"; exit 1; }

export LLM_API_KEY="$API_KEY"
for d in "$RESULTS_ROOT"/*/; do
  m=$(basename "$d")
  [[ -f "$d/profile/concepts.csv" ]] || { echo "-- $m: no concepts.csv, skipped"; continue; }
  echo "== grading $m  ($(date +%H:%M:%S))"
  "$PY" run_all.py rescore --backend openai --base-url "http://127.0.0.1:${PORT}" \
    --model "$GRADER_NAME" --extract-model "$GRADER_NAME" --no-embed --workers 32 \
    --results-dir "$d" --figures-dir figures_profiles --cache-dir cache_profiles/grader \
    || echo "!! grading failed for $m"
done
echo "== cross-model analysis"
"$PY" run_all.py cross --profile-root "$RESULTS_ROOT" --figures-dir figures_profiles --models-tsv slurm/models.tsv
echo "grading and cross-model analysis finished at $(date -Is)"

#!/bin/bash
#SBATCH --job-name=prof
#SBATCH --account=<account_name>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/profile-%x-%j.log
#SBATCH --error=logs/profile-%x-%j.log

# ============================================================================
# ONE MODEL of the attractive-feature protocol: serve it with vLLM on this
# node's GPUs, run the protocol against 127.0.0.1 (ungraded), stop the server.
#
# Do not sbatch this directly -- partition, GPUs, memory and wall-time differ
# per model and are injected by the submitter:
#
#     bash slurm/submit_profiles.sh               # every row of models.tsv
#     bash slurm/submit_profiles.sh gpt-oss-20b   # one model
#     bash slurm/submit_profiles.sh core          # one tier
#
# EVERYTHING RESUMES. If the job hits its wall-clock limit, resubmit: finished
# units are read from results_profiles/<model>/profile/done.txt and only the
# remainder is generated.
# ============================================================================

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/frother_gabm}"

# The submitter wrote this model's hardware and flags to a file that we
# source, rather than passing them through `sbatch --export` (a comma-separated
# list that would split JSON such as {"a":1,"b":2}). The file is also an
# exact record of what this job was told to serve.
if [[ -n "${MODEL_ENV_FILE:-}" && -f "$MODEL_ENV_FILE" ]]; then
  echo "[env] model settings from $MODEL_ENV_FILE"
  # shellcheck disable=SC1090
  source "$MODEL_ENV_FILE"
fi

MODEL_NAME="${MODEL_NAME:?MODEL_NAME not set -- submit via slurm/submit_profiles.sh}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH not set -- submit via slurm/submit_profiles.sh}"
SERVED_NAME="${SERVED_NAME:-$MODEL_NAME}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
SERVE_FLAGS="${SERVE_FLAGS:-}"
CHAT_KWARGS="${CHAT_KWARGS:-}"
RUN_ARGS="${RUN_ARGS:-}"
PROFILE_ARGS="${PROFILE_ARGS:-}"
RESULTS_ROOT="${RESULTS_ROOT:-results_profiles}"
WORKERS="${WORKERS:-32}"

# Slurm captures stdout through a pipe and Python line-buffers only to a tty;
# without this the log stays empty for minutes and a healthy job looks hung.
export PYTHONUNBUFFERED=1

PORT="${PORT:-$(( 8000 + (${SLURM_JOB_ID:-0} % 1000) ))}"
API_KEY="profile-$(hostname)-${SLURM_JOB_ID:-local}"

cd "$PROJECT_DIR" || { echo "FATAL: PROJECT_DIR not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs "$RESULTS_ROOT"

echo "=================================================================="
echo "model    : $MODEL_NAME  (served as $SERVED_NAME)"
echo "job      : ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "started  : $(date -Is)"
echo "weights  : $MODEL_PATH"
echo "serving  : tp=$TENSOR_PARALLEL  ctx=$MAX_MODEL_LEN  seqs=$MAX_NUM_SEQS  port=$PORT"
echo "flags    : [$SERVE_FLAGS]  chat_kwargs [$CHAT_KWARGS]  run_args [$RUN_ARGS $PROFILE_ARGS]"
echo "results  : $RESULTS_ROOT/$MODEL_NAME"
echo "=================================================================="

# ---------------------------------------------------------------- env -----
# TWO SEPARATE ENVIRONMENTS. vLLM is a MODULE whose Python lacks pandas etc.;
# the base Miniforge environment has those but not vLLM. Record the experiment
# interpreter by ABSOLUTE PATH first, because `module load vLLM` prepends its
# own Python to PATH. `module reset` (not purge) keeps the site defaults that
# make the vLLM module visible.
module reset 2>/dev/null
module load Miniforge3 2>/dev/null || module load Anaconda3 2>/dev/null
PY="$(command -v python3)"
if [[ -z "$PY" ]]; then echo "FATAL: no python3 after Miniforge3"; exit 1; fi
echo "[env] experiment python : $PY"
if ! "$PY" -c "import requests, pandas, numpy, matplotlib" 2>/dev/null; then
  echo "FATAL: the experiment stack is incomplete in $PY"
  echo "  install with: $PY -m pip install --user -r requirements.txt"
  exit 1
fi
echo "[env] experiment stack  : OK"

module load "${VLLM_MODULE:-vLLM}" 2>/dev/null || { echo "FATAL: 'module load vLLM' failed. Check: module spider vllm"; exit 1; }
command -v vllm >/dev/null || { echo "FATAL: vllm not on PATH after loading the module"; exit 1; }
echo "[env] vLLM module       : loaded ($(command -v vllm))"

if [[ ! -d "$MODEL_PATH" ]]; then echo "FATAL: weights not found at $MODEL_PATH"; exit 1; fi

# ------------------------------------------------------------ cleanup -----
VLLM_PID=""; WATCH_PID=""
cleanup() {
  local rc=$?
  if [[ -n "$WATCH_PID" ]] && kill -0 "$WATCH_PID" 2>/dev/null; then sleep 2; kill "$WATCH_PID" 2>/dev/null; fi
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[cleanup] stopping vLLM (pid $VLLM_PID)"
    kill "$VLLM_PID" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$VLLM_PID" 2>/dev/null
  fi
  echo "finished : $(date -Is)  exit=$rc"
  exit $rc
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------- vLLM -----
VLLM_LOG="logs/vllm-${MODEL_NAME}-${SLURM_JOB_ID:-local}.log"
echo "[vllm] starting (log: $VLLM_LOG)"
# shellcheck disable=SC2086
vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --api-key "$API_KEY" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  $SERVE_FLAGS \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# Poll until the server answers a REAL authenticated request.
echo "[vllm] waiting for readiness (pid $VLLM_PID)"
READY=0
for i in $(seq 1 180); do            # up to 30 minutes for the big models
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "FATAL: vLLM died during startup. Last 40 lines of $VLLM_LOG:"; tail -40 "$VLLM_LOG"; exit 1
  fi
  if curl -sf -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
    READY=1; echo "[vllm] ready after $((i * 10))s"; break
  fi
  sleep 10
done
if [[ "$READY" -ne 1 ]]; then
  echo "FATAL: vLLM not ready within 30 minutes. Last 40 lines of $VLLM_LOG:"; tail -40 "$VLLM_LOG"; exit 1
fi
SERVED=$(curl -s -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:${PORT}/v1/models" | tr ',' '\n' | grep -o '"id":"[^"]*"' | head -1)
echo "[vllm] serving $SERVED (expected \"id\":\"$SERVED_NAME\")"

# A second, independent sign of life from OUTSIDE the experiment process.
(
  while true; do
    sleep "${SERVER_SAMPLE_EVERY:-300}"
    util=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | paste -sd' | ' -)
    echo "[$(date +%H:%M:%S)] SERVER  gpu[${util:-n/a}]  vllm-log $(wc -l < "$VLLM_LOG" 2>/dev/null || echo 0) lines"
  done
) &
WATCH_PID=$!

# ------------------------------------------------------------ execute -----
# DATA ONLY: concepts are stored ungraded (--no-extract). Grading happens once,
# with one fixed grader, in slurm/run_grade.sh after every model has landed.
export LLM_API_KEY="$API_KEY"
CK=(); [[ -n "$CHAT_KWARGS" ]] && CK=(--chat-kwargs "$CHAT_KWARGS")
echo "[run] starting profile for $MODEL_NAME (progress also in $RESULTS_ROOT/$MODEL_NAME/profile/progress.log)"
# shellcheck disable=SC2086
"$PY" run_all.py profile \
  --backend openai --base-url "http://127.0.0.1:${PORT}" --model "$SERVED_NAME" \
  --no-extract --no-embed --workers "$WORKERS" --temperature 0.7 \
  --results-dir "$RESULTS_ROOT/$MODEL_NAME" --figures-dir figures_profiles \
  --cache-dir "cache_profiles/$MODEL_NAME" \
  "${CK[@]}" $RUN_ARGS $PROFILE_ARGS
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "=================================================================="
  echo "MODEL $MODEL_NAME EXITED $RC -- partial results are on disk and the run RESUMES on resubmit."
  echo "=================================================================="
fi
exit $RC

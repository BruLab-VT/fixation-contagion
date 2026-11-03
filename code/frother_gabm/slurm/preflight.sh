#!/bin/bash
# Run on a LOGIN NODE before submitting. Every check fails in seconds here;
# the same failure inside a queued job costs a queue cycle.
#     bash slurm/preflight.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
FAIL=0; ok(){ echo "  [ OK ] $1"; }; bad(){ echo "  [FAIL] $1"; FAIL=1; }; warn(){ echo "  [warn] $1"; }
echo "=================================================================="; echo "frother_gabm pre-flight  ·  $(date -Is)"; echo "=================================================================="
echo; echo "1. Files"
for f in frother/__init__.py run_all.py smoke_profile.py slurm/models.tsv slurm/account.conf slurm/run_profile.sh slurm/run_grade.sh slurm/submit_profiles.sh; do
  [[ -f "$f" ]] && ok "$f" || bad "$f missing"; done
echo; echo "2. Experiment stack (base Miniforge)"
module reset 2>/dev/null; module load Miniforge3 2>/dev/null || module load Anaconda3 2>/dev/null
PY="$(command -v python3)"
if [[ -z "$PY" ]]; then bad "no python3 after Miniforge3"; else
  ok "interpreter: $PY"
  for m in requests pandas numpy matplotlib scipy; do "$PY" -c "import $m" 2>/dev/null && ok "import $m" || bad "import $m  (fix: $PY -m pip install --user -r requirements.txt)"; done
  "$PY" -c "import tabulate" 2>/dev/null && ok "import tabulate" || warn "import tabulate -- reports fall back to plain tables"
fi
echo; echo "3. Package"
"$PY" -c "import sys; sys.path.insert(0,'.'); import frother, frother.jobs.profile, frother.profile_analysis; print('  [ OK ] frother imports (v'+frother.__version__+')')" 2>/dev/null || bad "frother does not import"
echo; echo "4. Offline smoke test (no GPU)"
if "$PY" smoke_profile.py > /tmp/frother_smoke.$$ 2>&1; then ok "smoke_profile: PASS"; rm -rf results_profiles_smoke figures_profiles_smoke cache_psmoke; else bad "smoke_profile failed -- see /tmp/frother_smoke.$$"; tail -10 /tmp/frother_smoke.$$; fi
echo; echo "5. Model weights"
tail -n +2 slurm/models.tsv | while IFS= read -r row; do n=$(echo "$row"|cut -f1); p=$(echo "$row"|cut -f2); [[ -d "$p" ]] && echo "  [ OK ] $n" || echo "  [FAIL] $n  -> $p not found"; done
echo; echo "6. vLLM module"
if module load vLLM 2>/dev/null; then ok "module load vLLM"; command -v vllm >/dev/null && ok "vllm on PATH: $(command -v vllm)" || bad "vllm not on PATH"; else bad "module load vLLM failed -- try: module spider vllm"; fi
echo; echo "7. Slurm account"
ACCT="${ACCOUNT:-}"; [[ -z "$ACCT" && -f slurm/account.conf ]] && ACCT=$(grep -m1 '^ACCOUNT=' slurm/account.conf | cut -d= -f2)
if [[ "$ACCT" == "<account_name>" || -z "$ACCT" ]]; then bad "still the placeholder -- edit ONE line in slurm/account.conf"; quota 2>/dev/null | head -8; else ok "account: $ACCT"; fi
echo; echo "=================================================================="
[[ $FAIL -eq 0 ]] && echo "ALL CHECKS PASSED -- submit with:  QUICK=1 bash slurm/submit_profiles.sh   (then the real run)" || echo "SOME CHECKS FAILED -- fix the [FAIL] items above."
echo "=================================================================="
exit $FAIL

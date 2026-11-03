#!/bin/bash
# Live status of every profile job from a LOGIN NODE.
#     bash slurm/watch.sh        # one snapshot
#     bash slurm/watch.sh -f     # refresh every 30s until Ctrl-C
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
RESULTS_ROOT="${RESULTS_ROOT:-results_profiles}"
FOLLOW=0; [[ "${1:-}" == "-f" ]] && FOLLOW=1
snapshot() {
  echo "=================================================================="
  echo "frother profiles  ·  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=================================================================="
  echo; echo "QUEUE"
  squeue -u "$USER" -o "  %.10i %.20j %.14P %.9T %.10M %.10L %R" 2>/dev/null | head -20
  echo; echo "PROGRESS ($RESULTS_ROOT/<model>/profile/progress.log, last line each)"
  for d in "$RESULTS_ROOT"/*/; do
    [[ -d "$d" ]] || continue
    m=$(basename "$d"); p="$d/profile/progress.log"
    if [[ -f "$p" ]]; then
      done_n=$(wc -l < "$d/profile/done.txt" 2>/dev/null || echo 0)
      printf "  %-20s units done %5s   %s\n" "$m" "$done_n" "$(tail -1 "$p" | cut -c1-90)"
    else
      printf "  %-20s (not started)\n" "$m"
    fi
  done
  echo; echo "GRADED PROFILES"
  ls "$RESULTS_ROOT"/*/profile/profile_matrix.csv 2>/dev/null | sed 's|/profile/profile_matrix.csv||; s|.*/|  |' || echo "  (none yet)"
  [[ -f "$RESULTS_ROOT/CROSS_MODEL_REPORT.md" ]] && echo "  cross-model report: $RESULTS_ROOT/CROSS_MODEL_REPORT.md"
}
if [[ $FOLLOW -eq 1 ]]; then while true; do clear; snapshot; sleep 30; done; else snapshot; fi

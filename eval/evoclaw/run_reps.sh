#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/run_reps.sh

Launch repeat-N independent trials for one compression arm.

Required environment: all run_chain.sh requirements.
Optional:
  ARM=pipeline REPS=3 MILESTONES=6 TIMEOUT=10800
  TRIAL_PREFIX=repeat LOG_DIR=<path> LAUNCH_DELAY=1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
export REPO
: "${EVOCLAW:?Set EVOCLAW to the EvoClaw checkout root}"
: "${EVOCLAW_REPO_NAME:?Set EVOCLAW_REPO_NAME}"
: "${EVOCLAW_IMAGE:?Set EVOCLAW_IMAGE}"
: "${EVOCLAW_DATA_ROOT:?Set EVOCLAW_DATA_ROOT}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
bash "$HERE/deploy.sh" "$EVOCLAW"
ARM="${ARM:-pipeline}"
REPS="${REPS:-3}"
MILESTONES="${MILESTONES:-6}"
TIMEOUT="${TIMEOUT:-10800}"
TRIAL_PREFIX="${TRIAL_PREFIX:-repeat}"
LOG_DIR="${LOG_DIR:-$REPO/eval/evoclaw/results/$TRIAL_PREFIX/logs}"
LAUNCH_DELAY="${LAUNCH_DELAY:-1}"
[[ "$REPS" =~ ^[1-9][0-9]*$ ]] || { echo "REPS must be positive" >&2; exit 2; }
mkdir -p "$LOG_DIR"

PIDS=()
for (( index=1; index<=REPS; index++ )); do
  TRIAL="${TRIAL_PREFIX}_rep${index}"
  nohup env \
    COMPACT_STRATEGY="$ARM" \
    MILESTONES="$MILESTONES" \
    TIMEOUT="$TIMEOUT" \
    TRIAL_NAME="$TRIAL" \
    bash "$HERE/run_chain.sh" > "$LOG_DIR/$TRIAL.log" 2>&1 &
  PIDS+=("$!")
  echo "[run_reps] launched trial=$TRIAL pid=$!"
  sleep "$LAUNCH_DELAY"
done
printf '%s\n' "${PIDS[@]}" > "$LOG_DIR/pids.txt"
echo "[run_reps] launched ${#PIDS[@]} trial(s); logs=$LOG_DIR"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/run_fork_seed.sh

Launch a detached seed run that stops at an explicitly chosen context size.
The resulting trial can be cloned with prepare_fork_at_compact.py and resumed
with run_fork_resume.sh.

Required environment:
  MYAGENT_STOP_AT_CONTEXT   positive context-token threshold
  all run_chain.sh requirements

Optional: TAG, MILESTONES, TIMEOUT, EVOCLAW_CONFIG, EVOCLAW_RESULTS_ROOT.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
[[ "${MYAGENT_STOP_AT_CONTEXT:-}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MYAGENT_STOP_AT_CONTEXT must be an explicit positive integer" >&2
  exit 2
}

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
TAG="${TAG:-fork_seed_$(date -u +%Y%m%dT%H%M%SZ)}"
RESULTS_ROOT="${EVOCLAW_RESULTS_ROOT:-$REPO/eval/evoclaw/results}"
OUTPUT="$RESULTS_ROOT/$TAG"
CONFIG="${EVOCLAW_CONFIG:-$HERE/e2e_config_snapshot_cut.yaml}"
mkdir -p "$OUTPUT"

nohup env \
  EVOCLAW_CONFIG="$CONFIG" \
  COMPACT_STRATEGY=none \
  MYAGENT_ARM_LABEL=fork_seed \
  MYAGENT_STOP_AT_CONTEXT="$MYAGENT_STOP_AT_CONTEXT" \
  MILESTONES="${MILESTONES:-6}" \
  TIMEOUT="${TIMEOUT:-10800}" \
  TRIAL_NAME="$TAG" \
  bash "$HERE/run_chain.sh" > "$OUTPUT/seed.log" 2>&1 &

PID=$!
printf '%s\n' "$PID" > "$OUTPUT/seed.pid"
echo "[fork-seed] started tag=$TAG pid=$PID log=$OUTPUT/seed.log"

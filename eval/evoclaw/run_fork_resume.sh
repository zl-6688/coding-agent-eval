#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/run_fork_resume.sh <fork_full|fork_sm> <trial-root>

Resume a prepared fork-at-compact EvoClaw trial. Provider credentials, endpoint,
and model must already be exported; this script never loads credential files.

Required environment: EVOCLAW plus the variables documented by env.sh.
Optional: REPO, EVOCLAW_PYTHON, MYAGENT_INSTANCE_ID.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

ARM="${1:-}"
TRIAL_ROOT="${2:-}"
[[ -n "$ARM" && -n "$TRIAL_ROOT" ]] || { usage >&2; exit 2; }
case "$ARM" in
  fork_full) SESSION_MEMORY=0 ;;
  fork_sm) SESSION_MEMORY=1 ;;
  *) echo "Unknown arm: $ARM" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
: "${EVOCLAW:?Set EVOCLAW to the EvoClaw checkout root}"
# shellcheck disable=SC1091
source "$HERE/env.sh"
PYTHON_BIN="${EVOCLAW_PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }

export MYAGENT_REPO="$REPO"
export MYAGENT_INSTANCE_ID="${MYAGENT_INSTANCE_ID:-fork-eval}"
export COMPACT_STRATEGY=pipeline
export MYAGENT_ARM_LABEL="$ARM"
export MYAGENT_SESSION_MEMORY="$SESSION_MEMORY"
export MYAGENT_STOP_AT_CONTEXT=""

echo "[run_fork_resume] arm=$ARM session_memory=$SESSION_MEMORY trial=$TRIAL_ROOT"
cd "$EVOCLAW"
exec "$PYTHON_BIN" -m harness.e2e.run_e2e --resume-trial "$TRIAL_ROOT"

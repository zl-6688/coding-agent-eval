#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/run_chain.sh [-- extra EvoClaw arguments]

Run one EvoClaw chain with the myagent adapter.

Required exported variables:
  EVOCLAW                 EvoClaw checkout root
  EVOCLAW_REPO_NAME       dataset/repository identifier
  EVOCLAW_IMAGE           benchmark container image
  EVOCLAW_DATA_ROOT       root containing srs/ and e2e_trial/
  ANTHROPIC_API_KEY or UNIFIED_API_KEY
  ANTHROPIC_BASE_URL
  MODEL_ID

Optional variables:
  REPO                    coding-agent-eval root (derived from this script)
  EVOCLAW_PYTHON          Python executable (default: python3)
  EVOCLAW_SRS_ROOT        SRS directory (default: EVOCLAW_DATA_ROOT/srs)
  COMPACT_STRATEGY        none, pipeline, or truncate (default: none)
  MILESTONES              milestone count (default: 1)
  PROMPT_VERSION          EvoClaw prompt version (default: v1)
  TIMEOUT                 seconds (default: 1800)
  TRIAL_NAME              unique trial name (derived from strategy)
  EVOCLAW_CONFIG          optional harness config path
  EVOCLAW_FORCE           1 to pass --force (default: 0)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
elif [[ $# -gt 0 ]]; then
  echo "Unexpected arguments. Put EvoClaw arguments after --." >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
: "${EVOCLAW:?Set EVOCLAW to the EvoClaw checkout root}"
: "${EVOCLAW_REPO_NAME:?Set EVOCLAW_REPO_NAME}"
: "${EVOCLAW_IMAGE:?Set EVOCLAW_IMAGE}"
: "${EVOCLAW_DATA_ROOT:?Set EVOCLAW_DATA_ROOT}"

# shellcheck disable=SC1091
source "$HERE/env.sh"

PYTHON_BIN="${EVOCLAW_PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
}
[[ -d "$EVOCLAW/harness/e2e" ]] || { echo "Invalid EVOCLAW checkout: $EVOCLAW" >&2; exit 2; }

export MYAGENT_REPO="$REPO"
export MYAGENT_INSTANCE_ID="${MYAGENT_INSTANCE_ID:-$EVOCLAW_REPO_NAME}"
export COMPACT_STRATEGY="${COMPACT_STRATEGY:-none}"
MILESTONES="${MILESTONES:-1}"
PROMPT_VERSION="${PROMPT_VERSION:-v1}"
TIMEOUT="${TIMEOUT:-1800}"
TRIAL_NAME="${TRIAL_NAME:-trial_${COMPACT_STRATEGY}}"
SRS_ROOT="${EVOCLAW_SRS_ROOT:-$EVOCLAW_DATA_ROOT/srs}"

ARGS=(
  --repo-name "$EVOCLAW_REPO_NAME"
  --image "$EVOCLAW_IMAGE"
  --srs-root "$SRS_ROOT"
  --workspace-root "$EVOCLAW_DATA_ROOT"
  --agent myagent
  --model "$MODEL_ID"
  --prompt-version "$PROMPT_VERSION"
  --milestones "$MILESTONES"
  --timeout "$TIMEOUT"
  --trial-name "$TRIAL_NAME"
)
if [[ -n "${EVOCLAW_CONFIG:-}" ]]; then
  ARGS+=(--config "$EVOCLAW_CONFIG")
fi
if [[ "${EVOCLAW_FORCE:-0}" == "1" ]]; then
  ARGS+=(--force)
fi
ARGS+=("${EXTRA_ARGS[@]}")

echo "[run_chain] strategy=$COMPACT_STRATEGY trial=$TRIAL_NAME milestones=$MILESTONES"
cd "$EVOCLAW"
exec "$PYTHON_BIN" -m harness.e2e.run_e2e "${ARGS[@]}"

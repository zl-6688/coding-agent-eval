#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash eval/evoclaw/run_three_arms.sh run <tag> [milestones] [max-turns]
  bash eval/evoclaw/run_three_arms.sh collect <tag>

Run or collect the none/pipeline/truncate comparison. `run` launches three
provider-backed EvoClaw trials, waits for them, then collects traces, verdicts,
curves, and a cost summary. `collect` performs only the deterministic collection.

Required environment:
  EVOCLAW, EVOCLAW_REPO_NAME, EVOCLAW_DATA_ROOT, EVOCLAW_PYTHON
  plus run_chain.sh provider/image variables when using `run`

Optional: REPO, TIMEOUT, EVOCLAW_RESULTS_ROOT, MYAGENT_TRACES.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

COMMAND="${1:-}"
TAG="${2:-}"
[[ "$COMMAND" == "run" || "$COMMAND" == "collect" ]] || { usage >&2; exit 2; }
[[ -n "$TAG" ]] || { usage >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
export REPO
: "${EVOCLAW_REPO_NAME:?Set EVOCLAW_REPO_NAME}"
: "${EVOCLAW_DATA_ROOT:?Set EVOCLAW_DATA_ROOT}"
PYTHON_BIN="${EVOCLAW_PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }

RESULTS_ROOT="${EVOCLAW_RESULTS_ROOT:-$REPO/eval/evoclaw/results}"
OUTPUT="$RESULTS_ROOT/$TAG"
TRIALS_ROOT="$EVOCLAW_DATA_ROOT/e2e_trial"
TRACE_SOURCE="${MYAGENT_TRACES:-/opt/myagent/.traces}"
ARMS=(none pipeline truncate)

find_trial() {
  find "$TRIALS_ROOT" -maxdepth 1 -type d -name "${TAG}_$1*" -print -quit 2>/dev/null
}

find_container() {
  local arm="$1" name
  while IFS= read -r name; do
    case "$name" in
      "$EVOCLAW_REPO_NAME-${TAG}_${arm}"*) printf '%s\n' "$name"; return 0 ;;
    esac
  done < <(docker ps -a --format '{{.Names}}')
  return 1
}

if [[ "$COMMAND" == "run" ]]; then
  : "${EVOCLAW:?Set EVOCLAW to the EvoClaw checkout root}"
  : "${EVOCLAW_IMAGE:?Set EVOCLAW_IMAGE}"
  # Fail before launching any paid work if provider configuration is incomplete.
  # shellcheck disable=SC1091
  source "$HERE/env.sh"
  bash "$HERE/deploy.sh" "$EVOCLAW"
  MILESTONES="${3:-6}"
  MAX_TURNS="${4:-400}"
  mkdir -p "$OUTPUT/logs"
  PIDS=()
  for arm in "${ARMS[@]}"; do
    if [[ -n "$(find_trial "$arm")" ]]; then
      echo "Refusing to overwrite an existing trial for tag=$TAG arm=$arm" >&2
      exit 2
    fi
    trial="${TAG}_${arm}"
    nohup env \
      MYAGENT_MAX_TURNS="$MAX_TURNS" \
      COMPACT_STRATEGY="$arm" \
      MYAGENT_ARM_LABEL="$arm" \
      MILESTONES="$MILESTONES" \
      TIMEOUT="${TIMEOUT:-10800}" \
      TRIAL_NAME="$trial" \
      bash "$HERE/run_chain.sh" > "$OUTPUT/logs/$trial.log" 2>&1 &
    PIDS+=("$!")
    echo "[three-arms] launched arm=$arm trial=$trial pid=$!"
  done
  printf '%s\n' "${PIDS[@]}" > "$OUTPUT/pids.txt"
  RUN_FAILURE=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || RUN_FAILURE=1
  done
  if [[ "$RUN_FAILURE" == "1" ]]; then
    echo "[three-arms] at least one trial exited non-zero; collecting available evidence" >&2
  fi
fi

mkdir -p "$OUTPUT/traces"
: > "$OUTPUT/verdicts.jsonl"
FOUND=0
for arm in "${ARMS[@]}"; do
  container="$(find_container "$arm" || true)"
  trial_root="$(find_trial "$arm")"
  if [[ -n "$container" ]]; then
    docker cp "$container:$TRACE_SOURCE/." "$OUTPUT/traces/" >/dev/null
    FOUND=1
  fi
  if [[ -n "$trial_root" ]]; then
    "$PYTHON_BIN" "$HERE/verdicts_bridge.py" --trial "$trial_root" >> "$OUTPUT/verdicts.jsonl"
    FOUND=1
  fi
done
[[ "$FOUND" == "1" ]] || { echo "No trials or containers found for tag=$TAG" >&2; exit 1; }

"$PYTHON_BIN" "$REPO/eval/compression_eval/extract_curves.py" \
  --traces "$OUTPUT/traces" \
  --verdicts "$OUTPUT/verdicts.jsonl" \
  --out "$OUTPUT/curves.png" > "$OUTPUT/curve.txt"
"$PYTHON_BIN" "$REPO/eval/compression_eval/peak_cost.py" "$OUTPUT/traces" \
  > "$OUTPUT/summary.txt"
echo "[three-arms] collected evidence in $OUTPUT"

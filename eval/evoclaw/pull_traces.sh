#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/pull_traces.sh <container> [destination]

Copy agent JSONL traces out of an EvoClaw container. The destination defaults
to eval/evoclaw/results/manual-traces under this repository.

Environment:
  MYAGENT_TRACES   container trace directory (default: /opt/myagent/.traces)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

CONTAINER="${1:-}"
[[ -n "$CONTAINER" ]] || { usage >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
DESTINATION="${2:-$REPO/eval/evoclaw/results/manual-traces}"
SOURCE_DIR="${MYAGENT_TRACES:-/opt/myagent/.traces}"

command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
docker exec "$CONTAINER" test -d "$SOURCE_DIR" || {
  echo "Trace directory not found in $CONTAINER: $SOURCE_DIR" >&2
  exit 1
}
mkdir -p "$DESTINATION"
docker cp "$CONTAINER:$SOURCE_DIR/." "$DESTINATION/"
COUNT="$(find "$DESTINATION" -maxdepth 1 -type f -name 'run_*.jsonl' | wc -l | tr -d ' ')"
echo "[pull_traces] copied $COUNT trace file(s) to $DESTINATION"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/smoke_container.sh

Run a live, minimal container smoke for adapter installation, agent execution,
session persistence, and resume continuity. This command makes provider calls.

Required exported variables:
  EVOCLAW, SMOKE_IMAGE
  ANTHROPIC_API_KEY or UNIFIED_API_KEY
  ANTHROPIC_BASE_URL, MODEL_ID

Optional:
  REPO (derived), EVOCLAW_PYTHON=python3, SMOKE_CONTAINER=ace-agent-smoke
  MYAGENT_MAX_TURNS=12, COMPACT_STRATEGY=none

The command never loads a credential file or prints credential metadata.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
: "${EVOCLAW:?Set EVOCLAW to the EvoClaw checkout root}"
: "${SMOKE_IMAGE:?Set SMOKE_IMAGE to an EvoClaw-compatible base image}"
# shellcheck disable=SC1091
source "$HERE/env.sh"

PYTHON_BIN="${EVOCLAW_PYTHON:-python3}"
CONTAINER="${SMOKE_CONTAINER:-ace-agent-smoke}"
SESSION_ID="smoke-$(date -u +%s)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
command -v docker >/dev/null 2>&1 || { echo "docker is required" >&2; exit 2; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }

bash "$HERE/deploy.sh" "$EVOCLAW"
if docker ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER"; then
  echo "Container already exists: $CONTAINER (choose SMOKE_CONTAINER or remove it explicitly)" >&2
  exit 2
fi

docker run -d --name "$CONTAINER" -v "$REPO:/opt/myagent-src:ro" \
  "$SMOKE_IMAGE" tail -f /dev/null >/dev/null
docker exec "$CONTAINER" python3 --version >/dev/null
docker exec "$CONTAINER" sh -c \
  'id fakeroot >/dev/null 2>&1 || useradd -m -u 1001 fakeroot 2>/dev/null || adduser -D -u 1001 fakeroot 2>/dev/null || true'
docker exec "$CONTAINER" sh -c 'chown -R fakeroot /testbed 2>/dev/null || true'
docker exec --user fakeroot -e HOME=/home/fakeroot "$CONTAINER" sh -c \
  'git config --global user.email "agent-smoke@example.invalid" && git config --global user.name "agent-smoke" && git config --global --add safe.directory /testbed'

(
  cd "$EVOCLAW"
  MYAGENT_REPO="$REPO" "$PYTHON_BIN" -c \
    "from harness.e2e.agents.myagent import MyAgentFramework; print(MyAgentFramework().get_container_init_script('myagent'))"
) > "$TEMP_DIR/init.py"
docker cp "$TEMP_DIR/init.py" "$CONTAINER:/tmp/init.py" >/dev/null
docker exec "$CONTAINER" python3 /tmp/init.py

ENV_ARGS=(
  -e HOME=/home/fakeroot
  -e ANTHROPIC_BASE_URL
  -e MODEL_ID
  -e COMPACT_STRATEGY="${COMPACT_STRATEGY:-none}"
  -e AGENT_WORKDIR=/testbed
  -e MYAGENT_MAX_TURNS="${MYAGENT_MAX_TURNS:-12}"
  -e MYAGENT_INSTANCE_ID=container-smoke
)
[[ -z "${ANTHROPIC_API_KEY:-}" ]] || ENV_ARGS+=(-e ANTHROPIC_API_KEY)
[[ -z "${UNIFIED_API_KEY:-}" ]] || ENV_ARGS+=(-e UNIFIED_API_KEY)

run_agent() {
  local command="$1"
  docker exec -i --user fakeroot "${ENV_ARGS[@]}" -w /testbed "$CONTAINER" \
    /bin/sh -c "/usr/local/bin/myagent $command --session-id $SESSION_ID"
}

printf '%s' 'Create SMOKE.txt containing the line "milestone done". Commit it and create git tag agent-impl-smoke-1. Remember the marker ZEBRA-42.' \
  | run_agent run
docker exec --user fakeroot -e HOME=/home/fakeroot "$CONTAINER" test -s \
  "/home/fakeroot/.myagent/$SESSION_ID.json"
docker exec --user fakeroot -e HOME=/home/fakeroot "$CONTAINER" sh -c \
  'cd /testbed && test -f SMOKE.txt && git tag | grep -Fx agent-impl-smoke-1'

RESUME_OUTPUT="$(
  printf '%s' 'Return only the marker I asked you to remember in the previous turn.' \
    | run_agent resume
)"
printf '%s\n' "$RESUME_OUTPUT"
grep -Fq 'ZEBRA-42' <<< "$RESUME_OUTPUT" || {
  echo "[smoke] resume response did not preserve the marker" >&2
  exit 1
}

echo "[smoke] run and resume completed; inspect the response above for ZEBRA-42."
echo "[smoke] cleanup is explicit: docker rm -f $CONTAINER"

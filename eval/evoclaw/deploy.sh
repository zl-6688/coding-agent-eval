#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash eval/evoclaw/deploy.sh <evoclaw-clone> [options]

Deploy the versioned myagent adapter into an EvoClaw checkout and register it.

Options:
  --allow-domain DOMAIN  add an explicit model endpoint host to the container whitelist
  --cap-eval-cpus        cap evaluator CPU requests to the host CPU count
  -h, --help             show this help without inspecting the checkout

Environment:
  EVOCLAW_PYTHON         Python used to apply text patches (default: python3)
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

EVOCLAW="${1:-${EVOCLAW:-}}"
[[ $# -eq 0 ]] || shift
ALLOW_DOMAIN="${EVOCLAW_ALLOW_DOMAIN:-}"
CAP_EVAL_CPUS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-domain)
      [[ $# -ge 2 ]] || { echo "--allow-domain requires a value" >&2; exit 2; }
      ALLOW_DOMAIN="$2"
      shift 2
      ;;
    --cap-eval-cpus)
      CAP_EVAL_CPUS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EVOCLAW" || ! -d "$EVOCLAW/harness/e2e/agents" ]]; then
  echo "Expected an EvoClaw checkout containing harness/e2e/agents: $EVOCLAW" >&2
  exit 2
fi
if [[ -n "$ALLOW_DOMAIN" && ! "$ALLOW_DOMAIN" =~ ^[A-Za-z0-9.-]+$ ]]; then
  echo "Invalid --allow-domain value: $ALLOW_DOMAIN" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${EVOCLAW_PYTHON:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
}

sed 's/\r$//' "$HERE/myagent.py" > "$EVOCLAW/harness/e2e/agents/myagent.py"

"$PYTHON_BIN" - "$EVOCLAW" "$ALLOW_DOMAIN" "$CAP_EVAL_CPUS" <<'PY'
from __future__ import annotations

import io
import re
import sys
from pathlib import Path


root = Path(sys.argv[1])
allow_domain = sys.argv[2]
cap_eval_cpus = sys.argv[3] == "1"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


init_path = root / "harness/e2e/agents/__init__.py"
if init_path.exists():
    source = read(init_path)
    line = "from harness.e2e.agents.myagent import MyAgentFramework  # noqa: F401"
    if line not in source:
        source = source.rstrip() + "\n" + line + "\n"
        write(init_path, source)

base_path = root / "harness/e2e/agents/base.py"
if base_path.exists():
    source = read(base_path)
    line = "    from harness.e2e.agents import myagent  # noqa: F401"
    if line not in source:
        anchor = "    from harness.e2e.agents import openhands  # noqa: F401"
        if anchor not in source:
            raise SystemExit("Could not locate framework registration anchor in agents/base.py")
        source = source.replace(anchor, anchor + "\n" + line, 1)
        write(base_path, source)

for relative in ("harness/e2e/run_e2e.py", "harness/e2e/run_milestone.py"):
    path = root / relative
    if not path.exists():
        continue
    source = read(path)
    pattern = re.compile(r"choices=\[(?P<body>[^\]]*\"openhands\"[^\]]*)\]")

    def add_choice(match: re.Match[str]) -> str:
        body = match.group("body").rstrip()
        if '"myagent"' in body:
            return match.group(0)
        separator = " " if body.endswith(",") else ", "
        return f'choices=[{body}{separator}"myagent"]'

    updated, count = pattern.subn(add_choice, source, count=1)
    if count == 0 and '"myagent"' not in source:
        raise SystemExit(f"Could not locate --agent choices in {relative}")
    if updated != source:
        write(path, updated)

if allow_domain:
    setup_path = root / "harness/e2e/container_setup.py"
    source = read(setup_path)
    literal = repr(allow_domain)
    if literal not in source:
        anchor = "WHITELISTED_DOMAINS = ["
        if anchor not in source:
            raise SystemExit("Could not locate WHITELISTED_DOMAINS")
        source = source.replace(anchor, f"{anchor}\n    {literal},", 1)
        write(setup_path, source)

if cap_eval_cpus:
    evaluator = root / "harness/e2e/evaluator.py"
    source = read(evaluator)
    source = source.replace(
        'self.docker_cpus = metadata.get("docker_cpus", 16)',
        'self.docker_cpus = min(metadata.get("docker_cpus", 16), __import__("os").cpu_count() or 16)',
    )
    source = source.replace(
        "            self.docker_cpus = 16\n",
        '            self.docker_cpus = min(16, __import__("os").cpu_count() or 16)\n',
    )
    write(evaluator, source)
PY

echo "[deploy] myagent adapter registered in $EVOCLAW"
if [[ -n "$ALLOW_DOMAIN" ]]; then
  echo "[deploy] container whitelist includes $ALLOW_DOMAIN"
fi

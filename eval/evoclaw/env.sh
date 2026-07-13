#!/usr/bin/env bash
# Validate an explicitly exported provider environment. This file never reads
# credential files and never prints credential contents or shape information.

_ace_evoclaw_env_usage() {
  cat <<'EOF'
Usage: source eval/evoclaw/env.sh

Required exported variables:
  ANTHROPIC_API_KEY or UNIFIED_API_KEY
  ANTHROPIC_BASE_URL
  MODEL_ID

The helper only validates existing shell variables. It does not load a file.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  _ace_evoclaw_env_usage
  return 0 2>/dev/null || exit 0
fi

_ace_evoclaw_missing=()
if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${UNIFIED_API_KEY:-}" ]]; then
  _ace_evoclaw_missing+=("ANTHROPIC_API_KEY or UNIFIED_API_KEY")
fi
[[ -n "${ANTHROPIC_BASE_URL:-}" ]] || _ace_evoclaw_missing+=("ANTHROPIC_BASE_URL")
[[ -n "${MODEL_ID:-}" ]] || _ace_evoclaw_missing+=("MODEL_ID")

if (( ${#_ace_evoclaw_missing[@]} > 0 )); then
  printf '[env] missing required exported variable: %s\n' "${_ace_evoclaw_missing[@]}" >&2
  unset _ace_evoclaw_missing
  return 2 2>/dev/null || exit 2
fi

export ANTHROPIC_BASE_URL MODEL_ID
[[ -z "${ANTHROPIC_API_KEY:-}" ]] || export ANTHROPIC_API_KEY
[[ -z "${UNIFIED_API_KEY:-}" ]] || export UNIFIED_API_KEY
echo "[env] provider endpoint, model, and credential are configured."
unset _ace_evoclaw_missing

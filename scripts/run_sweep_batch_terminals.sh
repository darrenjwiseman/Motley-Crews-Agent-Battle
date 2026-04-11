#!/usr/bin/env bash
# Batch sweep with per-shard logs and optional macOS Terminal tail windows.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
PY=python3
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PY="$REPO_ROOT/.venv/bin/python"
fi
CFG=${1:-}
if [[ -z "${CFG}" ]]; then
  if [[ -f config/sweep.toml ]]; then
    CFG=config/sweep.toml
  else
    CFG=config/sweep.example.toml
  fi
else
  shift
fi
exec "$PY" -m motley_crews_play.eval_sweep --batch --batch-terminals --config "$CFG" "$@"

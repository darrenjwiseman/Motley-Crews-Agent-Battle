#!/usr/bin/env bash
# Run weight sweep from repo root (prefers .venv if present).
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
fi
exec "$PY" -m motley_crews_play.eval_sweep --config "$CFG"

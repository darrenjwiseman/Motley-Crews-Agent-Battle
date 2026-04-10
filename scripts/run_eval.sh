#!/usr/bin/env bash
# One-click / one-command batch evaluation from repo root.
# Double-click: on macOS, use run_eval.command instead (opens Terminal).
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
  if [[ -f config/eval.toml ]]; then
    CFG=config/eval.toml
  else
    CFG=config/eval.example.toml
  fi
fi
exec "$PY" -m motley_crews_play.eval_cli --config "$CFG"

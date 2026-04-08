#!/usr/bin/env bash
# Create/update .venv, install play dependencies, launch the Pygame UI.
# Usage: ./scripts/launch_game.sh [--seed N] [other motley_crews_play args]
# Env: SKIP_PIP_INSTALL=1 skips pip (faster if you know deps are current).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [[ ! -x "$PY" ]]; then
  echo "Creating virtual environment in .venv ..."
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found. Install Python 3.11+ from python.org or Homebrew." >&2
    exit 1
  fi
  python3 -m venv "$VENV"
fi

if [[ "${SKIP_PIP_INSTALL:-}" != "1" ]]; then
  echo "Checking dependencies (requirements-play.txt) ..."
  "$PIP" install --disable-pip-version-check -r "$ROOT/requirements-play.txt"
fi

exec "$PY" -m motley_crews_play --ui "$@"

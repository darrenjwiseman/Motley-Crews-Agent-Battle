#!/usr/bin/env bash
# macOS: double-click to run weight sweep from scripts/.
cd "$(dirname "$0")" || exit 1
exec bash ./run_sweep.sh "$@"

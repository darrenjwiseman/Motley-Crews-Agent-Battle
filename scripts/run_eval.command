#!/usr/bin/env bash
# macOS: double-click in Finder to run evaluation (uses Terminal).
cd "$(dirname "$0")" || exit 1
exec bash ./run_eval.sh "$@"

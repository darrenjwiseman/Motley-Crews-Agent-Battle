#!/usr/bin/env bash
# macOS: double-click to run batched weight sweep from scripts/.
cd "$(dirname "$0")" || exit 1
exec bash ./run_sweep_batch.sh "$@"

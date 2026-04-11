#!/usr/bin/env bash
# macOS: batch sweep with Terminal tail windows for each shard.
cd "$(dirname "$0")" || exit 1
exec bash ./run_sweep_batch_terminals.sh "$@"

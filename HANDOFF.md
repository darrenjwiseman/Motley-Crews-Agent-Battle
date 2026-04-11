# Session handoff

Update this file when you stop mid-task so the next session (human or agent) can continue without re-scanning the repo.

## Current goal

**Batch weight sweep** (`run_sweep_batch` in `eval_sweep.py`) is implemented: splits seeds across parallel subprocesses, per-shard logs, optional macOS Terminal `tail -f` windows (`--batch-terminals`), **sequential per-shard calibration** with scaled ETAs and **aggregated wall time** (`max` over shards), **`Proceed? [Y/n]`** after ETA (`--batch-yes` / `--yes` to skip), **work-weighted aggregate progress bar** on the supervisor during full shard runs. **`config/sweep.example.wide_separation.toml`** adds ~50% stronger weight separation vs `sweep.example.toml` for experiments. Core policies / Pygame unchanged.

## Next steps

1. Optional: tune sweep `seed_count` / variants when results cluster ~50% vs stock; use wide-separation example or generate variant grids externally.
2. Optional: tune phased preview timing (`cpu_delay_ms` / phase fractions in `pygame_app.py`) if CPU/human previews feel slow.
3. Optional: if combined move+action previews should show LOS from the post-move position, compute paths using `preview_after_move` when `move.actor_slot == action.actor_slot`.

## Files in progress

| File | Status |
|------|--------|
| — | Nothing blocked |

## Commands verified this session

```text
.venv/bin/python -m pytest tests/test_eval_sweep.py -q
```

## Notes / blockers

- **`motley_crews_play/eval_sweep.py`**: non-batch `run_from_toml` unchanged in spirit (calibration/ETA). **`run_sweep_batch`** merges shard CSVs into `*_master` outputs; **`BatchSweepAborted`** when user declines at prompt (CLI exits 0). **`tomli-w`** in `requirements.txt` for shard TOML writes.
- **Batch CLI**: `--batch`, `--batch-shards`, `--batch-terminals`, `--batch-log-dir`, `--batch-yes` / `--yes`.
- **Scripts**: `scripts/run_sweep_batch.sh`, `run_sweep_batch.command`, `run_sweep_batch_terminals.sh`, `run_sweep_batch_terminals.command` (repo root `scripts/`).
- **Configs**: `config/sweep.example.toml` (12 presets), `config/sweep.example.wide_separation.toml` (same labels, wider numeric separation); outputs default to distinct paths in the wide example (`results_wide.csv`).
- **`ParameterizedHeuristicPolicy`** / **`HeuristicWeights`**: see prior notes in git history and `policies.py`; sweep rows still parsed via **`heuristic_weights_from_spec`**.
- **`run_match`**: policy-driven setup unless `setup_random=True`.
- Generated sweep outputs under `config/sweep_out/` remain gitignored.

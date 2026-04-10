# Session handoff

Update this file when you stop mid-task so the next session (human or agent) can continue without re-scanning the repo.

## Current goal

Headless **evaluation** and **weight sweep** tooling is in place (`motley_crews_play/evaluation.py`, `eval_cli.py`, `eval_sweep.py`, configs under `config/`). Pygame/rules status unchanged from prior handoff.

## Next steps

1. Optional: tune phased preview timing (`cpu_delay_ms` / phase fractions in `pygame_app.py`) if CPU/human previews feel slow.
2. Optional: if combined move+action previews should show LOS from the post-move position, compute paths using `preview_after_move` for the acting piece when `move.actor_slot == action.actor_slot`.
3. Optional: run larger sweeps locally (`config/sweep.toml`); adjust `[parallel].game_progress_interval` if logs are too chatty or too quiet.

## Files in progress

| File | Status |
|------|--------|
| — | Nothing blocked |

## Commands verified this session

```text
.venv/bin/python -m pytest tests/ -q
```

## Notes / blockers

- **Evaluation / sweep (recent work)**  
  - **`evaluation.py`**: paired seeds, Wilson intervals, round-robin, Elo, behavior stats from logs.  
  - **`eval_cli.py`**: TOML config (`config/eval.example.toml`), `python -m motley_crews_play.eval_cli --config …`.  
  - **`eval_sweep.py`**: `[[variants]]` weight sweeps; **process pool** (`[parallel].workers`); **calibration** + ETA (`[run].calibration_seed_count`, `[output].time_estimate`); **progress**: sequential = live `\r` bar + tallies; parallel = variant bar/tally on completion **plus** per-game lines every `[parallel].game_progress_interval` games (default 50) from workers on stdout so long runs are not silent.  
  - Scripts: `scripts/run_eval.sh`, `run_eval.command`, `run_sweep.sh`, `run_sweep.command`.  
  - Generated sweep outputs are gitignored under `config/sweep_out/` (see `.gitignore`).

- **VP / draw / FX / path highlights** (older): `motley_crews_env` syncs score from dead units; simultaneous 4–4 VP is a draw; `motley_crews_play` has dual-layer highlights, phased CPU/human previews (`highlight_geometry.py`), and no recurring **move** range/path in preview phases when a turn includes both move and action (action-only range/path for those phases).

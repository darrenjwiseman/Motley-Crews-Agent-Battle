# Session handoff

Update this file when you stop mid-task so the next session (human or agent) can continue without re-scanning the repo.

## Current goal

Headless **evaluation** and **weight sweep** tooling is in place. **`ParameterizedHeuristicPolicy`** now supports **per-class** (`w_class` / `w_knight`…), **group** (`group_melee`, `group_mage`, `group_arbalist`), and **deployment** (`deploy_forward`, `deploy_center`) weights; **`run_match`** uses **`choose_setup`** during setup unless `setup_random=True`. Pygame/rules core unchanged.

## Next steps

1. Optional: tune phased preview timing (`cpu_delay_ms` / phase fractions in `pygame_app.py`) if CPU/human previews feel slow.
2. Optional: if combined move+action previews should show LOS from the post-move position, compute paths using `preview_after_move` for the acting piece when `move.actor_slot == action.actor_slot`.
3. Run sweeps with trimmed `[[variants]]` in `config/sweep.toml` (see `config/sweep.example.toml` for full preset list); adjust `[parallel].game_progress_interval` if logs are too chatty or too quiet.

## Files in progress

| File | Status |
|------|--------|
| — | Nothing blocked |

## Commands verified this session

```text
.venv/bin/python -m pytest tests/ -q
```

## Notes / blockers

- **Parameterized heuristic weights** (`motley_crews_play/policies.py`): `HeuristicWeights` adds class/group/deploy fields; play scoring multiplies the primary score by a **geometric mean** of `effective_actor_weight` over move/action actors; **`heuristic_weights_from_spec`** parses TOML rows (including `w_class` arrays). **`choose_setup`** on `ParameterizedHeuristicPolicy` / `RandomPolicy`; **`score_setup_placement`** for deployment heuristic.
- **`run_match`** (`match.py`): policy-driven alternating placement by default; **`setup_random=True`** restores legacy `complete_setup_random` (tests / RNG parity).
- **Eval / sweep**: `eval_cli.py` passes full policy tables through `heuristic_weights_from_spec`; `eval_sweep.py` stores each variant’s **full spec** in CSV; sweep **report** includes a **“How to interpret this report”** section (paired sides, ~50% vs stock, Wilson `unclear`, identical rows).
- **`config/sweep.example.toml`**: documents optional weight keys and lists **12** example `[[variants]]` presets (copy to `sweep.toml`, delete unused blocks).
- **`evaluation.py`**: paired seeds, Wilson intervals, round-robin, Elo, behavior stats from logs (unchanged).
- **`eval_sweep.py`**: process pool, calibration/ETA, parallel progress lines (unchanged behavior aside from variant CSV columns).
- Scripts: `scripts/run_eval.sh`, `run_eval.command`, `run_sweep.sh`, `run_sweep.command`. Generated sweep outputs under `config/sweep_out/` are gitignored.
- **VP / draw / FX / path highlights** (older): `motley_crews_env` syncs score from dead units; simultaneous 4–4 VP is a draw; `motley_crews_play` has dual-layer highlights, phased CPU/human previews (`highlight_geometry.py`), and no recurring **move** range/path in preview phases when a turn includes both move and action (action-only range/path for those phases).

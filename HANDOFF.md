# Session handoff

Update this file when you stop mid-task so the next session (human or agent) can continue without re-scanning the repo.

## Current goal

**Pygame + highlights: converted units (`actor_team`).** Human UI now keys selection, menus, move/attack/special picking, idle long-press / drag-to-move, and board highlights on **roster `(team, slot)`** plus `actor_team` on legal actions (engine already emits these). Converted figures (e.g. White Mage **Convert** on an opponent roster slot) are clickable and fully playable. **`highlight_geometry.path_cells_for_turn`** uses `move.actor_team` / `action.actor_team` so preview paths match the correct figure. Brief manual tests OK.

## Next steps

1. Optional: if combined move+action previews should show LOS from the **post-move** position only when the same figure moves and acts, align path preview with `preview_after_move` when move+action share the same actor identity.
2. Optional: tune phased preview timing (`cpu_delay_ms` / phase fractions in `pygame_app.py`) if CPU/human previews feel slow.
3. Optional (sweep): tune `seed_count` / weight variants when eval results cluster; see `config/sweep.example.wide_separation.toml` and `eval_sweep.py` batch mode (`--batch`, shard logs, etc.).

## Files in progress

| File | Status |
|------|--------|
| — | Nothing blocked |

## Commands verified this session

```text
.venv/bin/python -m py_compile motley_crews_play/pygame_app.py motley_crews_play/highlight_geometry.py
.venv/bin/python -m pytest tests/test_motley_crews_env.py tests/test_engine_movement_attack.py tests/test_engine_edge_cases.py tests/test_engine_specials.py tests/test_engine_setup.py -q
```

## Notes / blockers

- **Convert UI**: `MotleyCrewsUI` tracks `_play_actor_team` with `_play_slot`; `_actor_team_slot_at_cell` finds figures by `controller == current_player` on either roster. Helpers `_move_matches_actor` / `_action_matches_actor` mirror engine `actor_team` semantics (`None` ⇒ current player’s roster).
- **Batch sweep** (`run_sweep_batch` in `motley_crews_play/eval_sweep.py`): parallel shards, per-shard logs, optional macOS Terminal tails, calibration ETAs, `Proceed? [Y/n]`, work-weighted progress bar — see prior git history and `config/sweep.example*.toml`. Outputs under `config/sweep_out/` gitignored.

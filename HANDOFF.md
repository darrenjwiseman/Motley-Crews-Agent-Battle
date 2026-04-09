# Session handoff

Update this file when you stop mid-task so the next session (human or agent) can continue without re-scanning the repo.

## Current goal

Pygame UX polish and rules-engine behavior are in good shape; next work is whatever the roadmap/README prioritizes (e.g. further UI, agents, or spec edge cases).

## Next steps

1. Optional: tune phased preview timing (`cpu_delay_ms` / phase fractions in `pygame_app.py`) if CPU/human previews feel slow.
2. Optional: if combined move+action previews should show LOS from the post-move position, compute paths using `preview_after_move` for the acting piece when `move.actor_slot == action.actor_slot`.

## Files in progress

| File | Status |
|------|--------|
| — | Nothing blocked |

## Commands verified this session

```text
.venv/bin/python -m pytest tests/ -q
```

## Notes / blockers

- **VP / draw / FX / path highlights** (recent work): `motley_crews_env` syncs score from dead units; simultaneous 4–4 VP is a draw; `motley_crews_play` has dual-layer highlights, phased CPU/human previews (`highlight_geometry.py`), and no recurring **move** range/path in preview phases when a turn includes both move and action (action-only range/path for those phases).

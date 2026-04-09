#!/usr/bin/env python3
"""Play a full match with two random legal-action policies (smoke test)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow running without installing the package
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from motley_crews_env.engine import begin_setup, complete_setup_random, initial_state, legal_actions, step
from motley_crews_env.types import MatchPhase


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--max-turns", type=int, default=5000, help="Safety cap on half-turns")
    args = p.parse_args()
    rng = random.Random(args.seed)
    state = initial_state()
    if state.match_phase == int(MatchPhase.PENDING_SETUP):
        state = begin_setup(
            state,
            coin_flip_winner=rng.randint(0, 1),
            winner_chooses_first_setup=rng.choice([True, False]),
        )
    state = complete_setup_random(state, rng)
    n = 0
    while not state.done and n < args.max_turns:
        acts = legal_actions(state)
        if not acts:
            print("Stalemate: no legal actions", file=sys.stderr)
            break
        state = step(state, rng.choice(acts)).state
        n += 1
    if state.done:
        print(f"Finished in {n} plies. Winner: player {state.winner}  scores {state.score}")
    else:
        print(f"Stopped after {n} plies (cap). Scores {state.score}")


if __name__ == "__main__":
    main()

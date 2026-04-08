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

from motley_crews_env.engine import initial_state, legal_actions, step


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    p.add_argument("--max-turns", type=int, default=5000, help="Safety cap on half-turns")
    args = p.parse_args()
    rng = random.Random(args.seed)
    state = initial_state()
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

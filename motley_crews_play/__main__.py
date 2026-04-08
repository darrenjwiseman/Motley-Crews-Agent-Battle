"""CLI: headless CPU vs CPU / scripted matches (no Pygame)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from motley_crews_play.match import MatchLogger, run_match
from motley_crews_play.policies import RandomPolicy, ScriptedHeuristicPolicy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ui",
        action="store_true",
        help="Open the Pygame board (install pygame; see requirements-play.txt)",
    )
    p.add_argument(
        "--mode",
        choices=("cpu-cpu", "random-random"),
        default="cpu-cpu",
        help="cpu-cpu: heuristic vs heuristic; random-random: random vs random",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=5000, dest="max_plies")
    p.add_argument("--verbose", action="store_true", help="print each ply log entry")
    args = p.parse_args()

    if args.ui:
        try:
            from motley_crews_play.pygame_app import run as run_ui
        except ImportError as e:
            print(
                "Pygame is required for --ui. Install: pip install -r requirements-play.txt",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        run_ui(seed=args.seed)
        return

    if args.mode == "cpu-cpu":
        pa = ScriptedHeuristicPolicy()
        pb = ScriptedHeuristicPolicy()
    else:
        pa = RandomPolicy()
        pb = RandomPolicy()

    log: MatchLogger | None = MatchLogger() if args.verbose else None
    result = run_match(pa, pb, seed=args.seed, max_plies=args.max_plies, log=log)
    s = result.final_state
    if args.verbose and log:
        for e in log.entries:
            print(e)
    if s.done:
        print(f"Finished in {result.plies} plies. Winner: player {s.winner}  scores {s.score}")
    else:
        reason = "max_plies" if result.stopped_early else "no legal actions / stalemate"
        print(f"Stopped ({reason}) after {result.plies} plies. Scores {s.score}")


if __name__ == "__main__":
    main()

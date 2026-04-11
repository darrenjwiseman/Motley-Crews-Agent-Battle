"""
Load a TOML config and run batch evaluation (pairwise or round-robin).

Usage::

    python -m motley_crews_play.eval_cli --config config/eval.toml
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from motley_crews_play.evaluation import (
    EloTracker,
    MatchBehaviorStats,
    PairwiseRecord,
    evaluate_pair_swapped,
    evaluate_pair_with_logs,
    round_robin,
    win_rate_with_wilson,
)
from motley_crews_play.policies import (
    ParameterizedHeuristicPolicy,
    Policy,
    RandomPolicy,
    ScriptedHeuristicPolicy,
    heuristic_weights_from_spec,
)


class EvalConfigError(ValueError):
    pass


def _req(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise EvalConfigError(f"Missing [{section}] key {key!r}")
    return d[key]


def _policy_from_table(row: dict[str, Any]) -> Policy:
    name = _req(row, "name", "policies")
    ptype = _req(row, "type", f"policies[{name}]")
    if ptype == "random":
        return RandomPolicy()
    if ptype == "scripted_heuristic":
        return ScriptedHeuristicPolicy()
    if ptype == "parameterized_heuristic":
        w = heuristic_weights_from_spec(row)
        return ParameterizedHeuristicPolicy(w)
    raise EvalConfigError(f"Unknown policy type {ptype!r} for {name!r}")


def load_policies(toml: dict[str, Any]) -> dict[str, Policy]:
    tables = toml.get("policies")
    if not isinstance(tables, list) or not tables:
        raise EvalConfigError("Need at least one [[policies]] table")
    out: dict[str, Policy] = {}
    for row in tables:
        if not isinstance(row, dict):
            raise EvalConfigError("Each [[policies]] entry must be a table")
        name = str(row["name"])
        if name in out:
            raise EvalConfigError(f"Duplicate policy name {name!r}")
        out[name] = _policy_from_table(row)
    return out


@dataclass(frozen=True, slots=True)
class RunParams:
    seeds: Sequence[int]
    max_plies: int


def load_run_section(toml: dict[str, Any]) -> RunParams:
    run = toml.get("run")
    if not isinstance(run, dict):
        raise EvalConfigError("Missing [run] table")
    seed_start = int(_req(run, "seed_start", "run"))
    seed_count = int(_req(run, "seed_count", "run"))
    if seed_count < 1:
        raise EvalConfigError("[run] seed_count must be >= 1")
    max_plies = int(_req(run, "max_plies", "run"))
    if max_plies < 1:
        raise EvalConfigError("[run] max_plies must be >= 1")
    seeds = list(range(seed_start, seed_start + seed_count))
    return RunParams(seeds=seeds, max_plies=max_plies)


def _print_pairwise(name: str, rec: PairwiseRecord, wilson: bool) -> None:
    p_hat, lo, hi = win_rate_with_wilson(rec)
    print(f"\n=== {name} (from focus perspective) ===")
    print(f"  wins={rec.wins} losses={rec.losses} draws={rec.draws} timeouts={rec.timeouts}")
    print(f"  games={rec.games}  decided(w+l)={rec.wins + rec.losses}")
    if wilson and rec.wins + rec.losses > 0:
        print(f"  win rate (decided): {p_hat:.4f}  Wilson 95% [{lo:.4f}, {hi:.4f}]")


def _behavior_summary(stats: list[MatchBehaviorStats]) -> None:
    if not stats:
        return
    n = len(stats)
    avg_plies = sum(s.plies for s in stats) / n
    avg_sp = sum(s.special_rate for s in stats) / n
    avg_mv = sum(s.move_rate for s in stats) / n
    print("  behavior (mean over games): plies/game={:.1f}  special_rate={:.3f}  move_rate={:.3f}".format(
        avg_plies, avg_sp, avg_mv
    ))


def run_from_toml(toml: dict[str, Any], *, out_stream: Any = None) -> None:
    out = out_stream if out_stream is not None else sys.stdout
    policies = load_policies(toml)
    runp = load_run_section(toml)
    mode = toml.get("mode") or {}
    if not isinstance(mode, dict):
        raise EvalConfigError("Missing [mode] table")
    kind = str(mode.get("kind", "pairwise")).lower()
    output = toml.get("output") or {}
    if not isinstance(output, dict):
        output = {}
    print_wilson = bool(output.get("print_wilson", True))
    want_elo = bool(output.get("elo", False))
    elo_k = float(output.get("elo_k", 32.0))
    elo_base = float(output.get("elo_base", 1500.0))
    want_behavior = bool(output.get("behavior", False))

    if kind == "pairwise":
        pw = toml.get("pairwise") or {}
        if not isinstance(pw, dict):
            raise EvalConfigError("Missing [pairwise] table")
        focus_n = str(_req(pw, "focus", "pairwise"))
        opp_n = str(_req(pw, "opponent", "pairwise"))
        if focus_n not in policies or opp_n not in policies:
            raise EvalConfigError("pairwise focus/opponent must name defined policies")
        focus_p = policies[focus_n]
        opp_p = policies[opp_n]
        if want_behavior:
            rec, behaviors = evaluate_pair_with_logs(
                focus_p, opp_p, runp.seeds, max_plies=runp.max_plies
            )
            print(f"Pairwise: {focus_n} vs {opp_n}  seeds={list(runp.seeds)[0]}..{list(runp.seeds)[-1]}", file=out)
            _print_pairwise(f"{focus_n} vs {opp_n}", rec, print_wilson)
            _behavior_summary(behaviors)
        else:
            rec = evaluate_pair_swapped(focus_p, opp_p, runp.seeds, max_plies=runp.max_plies)
            print(
                f"Pairwise: {focus_n} vs {opp_n}  seeds={list(runp.seeds)[0]}..{list(runp.seeds)[-1]}",
                file=out,
            )
            _print_pairwise(f"{focus_n} vs {opp_n}", rec, print_wilson)
        if want_elo:
            print(
                "\n  (Elo is intended for round_robin; pairwise uses win rate above.)",
                file=out,
            )
        return

    if kind == "round_robin":
        rr_sec = toml.get("round_robin") or {}
        if not isinstance(rr_sec, dict):
            raise EvalConfigError("Missing [round_robin] table")
        names_raw = _req(rr_sec, "policy_names", "round_robin")
        if not isinstance(names_raw, list) or len(names_raw) < 2:
            raise EvalConfigError("[round_robin].policy_names must list at least two names")
        names = [str(x) for x in names_raw]
        for n in names:
            if n not in policies:
                raise EvalConfigError(f"round_robin policy {n!r} not defined in [[policies]]")
        entries = [(n, policies[n]) for n in names]
        print(
            f"Round-robin: {', '.join(names)}  seeds={list(runp.seeds)[0]}..{list(runp.seeds)[-1]}",
            file=out,
        )
        result = round_robin(entries, runp.seeds, max_plies=runp.max_plies)
        for i, name_i in enumerate(names):
            print(f"\n--- {name_i} ---", file=out)
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                rec = result.rows[i].vs[name_j]
                _print_pairwise(f"{name_i} vs {name_j}", rec, print_wilson)
        if want_elo:
            elo = EloTracker(k=elo_k, base=elo_base).ratings_from_round_robin(result)
            print("\n=== Elo (from round-robin game outcomes) ===", file=out)
            for n in sorted(elo.ratings, key=lambda k: -elo.ratings[k]):
                print(f"  {n}: {elo.ratings[n]:.1f}", file=out)
        if want_behavior:
            print("\n(behavior=true not yet aggregated for full round_robin; use pairwise.)", file=out)
        return

    raise EvalConfigError(f"Unknown mode.kind {kind!r}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to TOML config (default: config/eval.toml or config/eval.example.toml under cwd)",
    )
    args = p.parse_args(argv)
    cwd = Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        for candidate in (cwd / "config" / "eval.toml", cwd / "config" / "eval.example.toml"):
            if candidate.is_file():
                cfg_path = candidate
                break
        if cfg_path is None:
            print(
                "No config found. Copy config/eval.example.toml to config/eval.toml "
                "or pass --config PATH",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        raise SystemExit(2)
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        run_from_toml(data)
    except EvalConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

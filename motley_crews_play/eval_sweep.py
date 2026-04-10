"""
Run many heuristic weight variants on the same seeds; write CSV + Markdown report.

Usage::

    python -m motley_crews_play.eval_sweep --config config/sweep.toml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from motley_crews_play.evaluation import PairwiseRecord, perspective_outcome, win_rate_with_wilson
from motley_crews_play.match import run_match
from motley_crews_play.policies import HeuristicWeights, ParameterizedHeuristicPolicy, RandomPolicy, ScriptedHeuristicPolicy


class SweepConfigError(ValueError):
    pass


def _req(d: dict[str, Any], key: str, section: str) -> Any:
    if key not in d:
        raise SweepConfigError(f"Missing [{section}] key {key!r}")
    return d[key]


@dataclass(frozen=True, slots=True)
class RunParams:
    seeds: Sequence[int]
    max_plies: int


def load_run_section(toml: dict[str, Any]) -> RunParams:
    run = toml.get("run")
    if not isinstance(run, dict):
        raise SweepConfigError("Missing [run] table")
    seed_start = int(_req(run, "seed_start", "run"))
    seed_count = int(_req(run, "seed_count", "run"))
    if seed_count < 1:
        raise SweepConfigError("[run] seed_count must be >= 1")
    max_plies = int(_req(run, "max_plies", "run"))
    if max_plies < 1:
        raise SweepConfigError("[run] max_plies must be >= 1")
    seeds = list(range(seed_start, seed_start + seed_count))
    return RunParams(seeds=seeds, max_plies=max_plies)


def load_variants(toml: dict[str, Any]) -> list[dict[str, Any]]:
    v = toml.get("variants")
    if not isinstance(v, list) or not v:
        raise SweepConfigError("Need at least one [[variants]] table")
    out: list[dict[str, Any]] = []
    for row in v:
        if not isinstance(row, dict):
            raise SweepConfigError("Each [[variants]] entry must be a table")
        label = str(_req(row, "label", "variants"))
        vp_scale = float(_req(row, "vp_scale", label))
        damage_scale = float(_req(row, "damage_scale", label))
        win_bonus = float(_req(row, "win_bonus", label))
        out.append(
            {
                "label": label,
                "vp_scale": vp_scale,
                "damage_scale": damage_scale,
                "win_bonus": win_bonus,
            }
        )
    return out


def load_output_section(toml: dict[str, Any]) -> tuple[Path, Path, int, bool]:
    out = toml.get("output") or {}
    if not isinstance(out, dict):
        out = {}
    csv_path = Path(str(out.get("csv_path", "config/sweep_out/results.csv")))
    report_path = Path(str(out.get("report_path", "config/sweep_out/report.md")))
    top_n = int(out.get("summary_top_n", 5))
    time_estimate = bool(out.get("time_estimate", True))
    return csv_path, report_path, top_n, time_estimate


def load_parallel_workers(toml: dict[str, Any]) -> int:
    sec = toml.get("parallel") or {}
    if not isinstance(sec, dict):
        return 1
    w = int(sec.get("workers", 1))
    return w


def load_parallel_game_progress_interval(toml: dict[str, Any]) -> int:
    """
    When using a process pool, each worker runs ``4 * seed_count`` games before returning.
    Emit a status line to stderr every N completed games (0 = off). Ignored for sequential runs.
    """
    sec = toml.get("parallel") or {}
    if not isinstance(sec, dict):
        return 50
    v = sec.get("game_progress_interval")
    if v is None:
        return 50
    return max(0, int(v))


def resolve_worker_count(requested: int, n_variants: int) -> int:
    """``requested`` 0 => min(cpu_count, variants); else clamp to [1, variants]."""
    if n_variants < 1:
        return 1
    if requested <= 0:
        requested = os.cpu_count() or 1
    return max(1, min(requested, n_variants))


def load_calibration_seed_count(toml: dict[str, Any]) -> int:
    run = toml.get("run")
    if not isinstance(run, dict):
        return 0
    v = run.get("calibration_seed_count")
    if v is None:
        return 0
    return max(0, int(v))


def classify_vs_anchor(wilson_lo: float, wilson_hi: float) -> str:
    """vs stock heuristic: strong / weak / unclear."""
    if wilson_lo > 0.5:
        return "strong"
    if wilson_hi < 0.5:
        return "weak"
    return "unclear"


@dataclass
class VariantResult:
    label: str
    vp_scale: float
    damage_scale: float
    win_bonus: float
    vs_heuristic: PairwiseRecord
    vs_random: PairwiseRecord

    def row_dict(self) -> dict[str, Any]:
        ph, loh, hih = win_rate_with_wilson(self.vs_heuristic)
        pr, lor, hir = win_rate_with_wilson(self.vs_random)
        cls = classify_vs_anchor(loh, hih)
        return {
            "label": self.label,
            "vp_scale": self.vp_scale,
            "damage_scale": self.damage_scale,
            "win_bonus": self.win_bonus,
            "vs_heuristic_wins": self.vs_heuristic.wins,
            "vs_heuristic_losses": self.vs_heuristic.losses,
            "vs_heuristic_draws": self.vs_heuristic.draws,
            "vs_heuristic_timeouts": self.vs_heuristic.timeouts,
            "vs_heuristic_win_rate": f"{ph:.6f}",
            "vs_heuristic_wilson_lo": f"{loh:.6f}",
            "vs_heuristic_wilson_hi": f"{hih:.6f}",
            "vs_heuristic_class": cls,
            "vs_random_wins": self.vs_random.wins,
            "vs_random_losses": self.vs_random.losses,
            "vs_random_draws": self.vs_random.draws,
            "vs_random_timeouts": self.vs_random.timeouts,
            "vs_random_win_rate": f"{pr:.6f}",
            "vs_random_wilson_lo": f"{lor:.6f}",
            "vs_random_wilson_hi": f"{hir:.6f}",
        }


def _progress_bar(done: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "[?]"
    filled = int(round(width * done / total))
    filled = min(max(filled, 0), width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def _is_tty(stream: Any) -> bool:
    return bool(getattr(stream, "isatty", lambda: False)())


def _emit_sequential_progress(
    file: TextIO,
    *,
    global_done: int,
    global_total: int,
    variant_label: str,
    rec_h: PairwiseRecord,
    rec_r: PairwiseRecord,
    tty: bool,
) -> None:
    bar = _progress_bar(global_done, global_total)
    he = f"{rec_h.wins}W/{rec_h.losses}L"
    if rec_h.draws or rec_h.timeouts:
        he += f"/{rec_h.draws}D/{rec_h.timeouts}T"
    rr = f"{rec_r.wins}W/{rec_r.losses}L"
    if rec_r.draws or rec_r.timeouts:
        rr += f"/{rec_r.draws}D/{rec_r.timeouts}T"
    text = f"{bar}  {variant_label}  vs_heur {he}  vs_rand {rr}"
    if tty:
        pad = max(0, 120 - len(text))
        file.write("\r" + text + " " * pad)
        file.flush()
    else:
        file.write(text + "\n")
        file.flush()


def _maybe_parallel_stderr_progress(
    label: str,
    gi: int,
    g_in_v: int,
    rec_h: PairwiseRecord,
    rec_r: PairwiseRecord,
    every: int,
) -> None:
    if every <= 0 or g_in_v <= 0:
        return
    if gi != 1 and gi != g_in_v and gi % every != 0:
        return
    he = f"{rec_h.wins}W/{rec_h.losses}L"
    rr = f"{rec_r.wins}W/{rec_r.losses}L"
    # Child process: use stdout so progress appears in the same stream as "Full evaluation:"
    # (stderr is easy to miss when copying logs).
    print(
        f"  [sweep:{label}] games {gi}/{g_in_v}  vs_heur {he}  vs_rand {rr}",
        file=sys.stdout,
        flush=True,
    )


def _evaluate_variant_streaming(
    spec: dict[str, Any],
    seeds: list[int],
    max_plies: int,
    *,
    on_game: Callable[[int, int, str, PairwiseRecord, PairwiseRecord], None] | None,
    parallel_progress_every: int = 0,
) -> VariantResult:
    """Same pairing protocol as :func:`_evaluate_variant_packed`, with optional per-game callback.

    When ``on_game`` is None and ``parallel_progress_every`` > 0 (worker processes), print
    occasional lines to stdout so long runs are not silent until a variant finishes.
    """
    label = spec["label"]
    w = HeuristicWeights(
        vp_scale=spec["vp_scale"],
        damage_scale=spec["damage_scale"],
        win_bonus=spec["win_bonus"],
    )
    pol = ParameterizedHeuristicPolicy(w)
    anchor_h = ScriptedHeuristicPolicy()
    anchor_r = RandomPolicy()
    rec_h = PairwiseRecord()
    rec_r = PairwiseRecord()
    g_in_v = 4 * len(seeds)
    gi = 0
    for seed in seeds:
        r0 = run_match(pol, anchor_h, seed=seed, max_plies=max_plies)
        rec_h.add(perspective_outcome(r0, 0))
        gi += 1
        if on_game:
            on_game(gi, g_in_v, label, rec_h, rec_r)
        else:
            _maybe_parallel_stderr_progress(
                label, gi, g_in_v, rec_h, rec_r, parallel_progress_every
            )
        r1 = run_match(anchor_h, pol, seed=seed, max_plies=max_plies)
        rec_h.add(perspective_outcome(r1, 1))
        gi += 1
        if on_game:
            on_game(gi, g_in_v, label, rec_h, rec_r)
        else:
            _maybe_parallel_stderr_progress(
                label, gi, g_in_v, rec_h, rec_r, parallel_progress_every
            )
    for seed in seeds:
        r0 = run_match(pol, anchor_r, seed=seed, max_plies=max_plies)
        rec_r.add(perspective_outcome(r0, 0))
        gi += 1
        if on_game:
            on_game(gi, g_in_v, label, rec_h, rec_r)
        else:
            _maybe_parallel_stderr_progress(
                label, gi, g_in_v, rec_h, rec_r, parallel_progress_every
            )
        r1 = run_match(anchor_r, pol, seed=seed, max_plies=max_plies)
        rec_r.add(perspective_outcome(r1, 1))
        gi += 1
        if on_game:
            on_game(gi, g_in_v, label, rec_h, rec_r)
        else:
            _maybe_parallel_stderr_progress(
                label, gi, g_in_v, rec_h, rec_r, parallel_progress_every
            )
    return VariantResult(
        label=label,
        vp_scale=spec["vp_scale"],
        damage_scale=spec["damage_scale"],
        win_bonus=spec["win_bonus"],
        vs_heuristic=rec_h,
        vs_random=rec_r,
    )


def _evaluate_variant_packed(
    packed: tuple[Any, ...],
) -> VariantResult:
    """
    Top-level entry point for :class:`ProcessPoolExecutor` (must be picklable).

    ``packed`` is ``(spec, seeds, max_plies)`` or ``(spec, seeds, max_plies, game_progress_interval)``.
    """
    if len(packed) == 3:
        spec, seeds_t, max_plies = packed
        interval = 0
    else:
        spec, seeds_t, max_plies, interval = packed
    return _evaluate_variant_streaming(
        spec,
        list(seeds_t),
        max_plies,
        on_game=None,
        parallel_progress_every=interval,
    )


def run_sweep(
    toml: dict[str, Any],
    *,
    seeds: Sequence[int] | None = None,
    workers: int | None = None,
    progress: Any = None,
) -> list[VariantResult]:
    runp = load_run_section(toml)
    seed_list = list(seeds) if seeds is not None else list(runp.seeds)
    variants = load_variants(toml)
    n = len(variants)
    prog = progress if progress is not None else sys.stdout
    parallel_req = load_parallel_workers(toml)
    w_req = parallel_req if workers is None else workers
    wcount = resolve_worker_count(w_req, n)

    if n == 0:
        return []

    tty = _is_tty(prog)
    games_per_variant = 4 * len(seed_list)
    total_games = n * games_per_variant

    gp_interval = load_parallel_game_progress_interval(toml)

    if wcount == 1 or n == 1:
        results: list[VariantResult] = []
        for i, spec in enumerate(variants):
            off = i * games_per_variant

            def make_on(off_: int) -> Callable[[int, int, str, PairwiseRecord, PairwiseRecord], None]:
                def _on_game(
                    gi: int,
                    giv: int,
                    lbl: str,
                    rh: PairwiseRecord,
                    rr: PairwiseRecord,
                ) -> None:
                    _emit_sequential_progress(
                        prog,
                        global_done=off_ + gi,
                        global_total=total_games,
                        variant_label=lbl,
                        rec_h=rh,
                        rec_r=rr,
                        tty=tty,
                    )

                return _on_game

            vr = _evaluate_variant_streaming(
                spec, seed_list, runp.max_plies, on_game=make_on(off)
            )
            results.append(vr)
        if tty:
            prog.write("\n")
            prog.flush()
        return results

    packed = [(spec, tuple(seed_list), runp.max_plies, gp_interval) for spec in variants]
    ordered: list[VariantResult | None] = [None] * n
    completed: list[VariantResult] = []
    try:
        with ProcessPoolExecutor(max_workers=wcount) as ex:
            futs = {ex.submit(_evaluate_variant_packed, p): i for i, p in enumerate(packed)}
            done = 0
            for fut in as_completed(futs):
                i = futs[fut]
                vr = fut.result()
                ordered[i] = vr
                done += 1
                completed.append(vr)
                print(_progress_bar(done, n), file=prog, flush=True)
                print("  --- wins tally (sorted by label) ---", file=prog, flush=True)
                for row in sorted(completed, key=lambda x: x.label):
                    h2 = f"{row.vs_heuristic.wins}W/{row.vs_heuristic.losses}L"
                    r2 = f"{row.vs_random.wins}W/{row.vs_random.losses}L"
                    print(f"  {row.label}  vs_heur {h2}  vs_rand {r2}", file=prog, flush=True)
    except (PermissionError, NotImplementedError, OSError) as e:
        print(
            f"Process pool unavailable ({type(e).__name__}: {e}); falling back to sequential.",
            file=prog,
            flush=True,
        )
        return run_sweep(
            toml,
            seeds=seed_list,
            workers=1,
            progress=prog,
        )
    return [ordered[i] for i in range(n)]


def _sort_key(vr: VariantResult) -> tuple[float, float]:
    _, lo, _ = win_rate_with_wilson(vr.vs_heuristic)
    ph, _, _ = win_rate_with_wilson(vr.vs_heuristic)
    return (-lo, -ph)


def write_csv(path: Path, rows: list[VariantResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dicts = [r.row_dict() for r in rows]
    if not dicts:
        return
    fieldnames = list(dicts[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(dicts)


def write_report(
    path: Path,
    rows: list[VariantResult],
    *,
    seed_lo: int,
    seed_hi: int,
    top_n: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=_sort_key)
    strong = []
    weak = []
    unclear = []
    for r in rows:
        _, lo, hi = win_rate_with_wilson(r.vs_heuristic)
        c = classify_vs_anchor(lo, hi)
        if c == "strong":
            strong.append(r)
        elif c == "weak":
            weak.append(r)
        else:
            unclear.append(r)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Heuristic weight sweep report",
        "",
        f"- Generated: {ts}",
        f"- Seeds: `{seed_lo}` .. `{seed_hi}` (same for every variant; paired sides per seed)",
        f"- Variants: {len(rows)}",
        "",
        "## Summary (vs stock `ScriptedHeuristicPolicy`)",
        "",
        f"- **Strong** (Wilson 95% lower bound > 0.5): **{len(strong)}**",
        f"- **Weak** (Wilson 95% upper bound < 0.5): **{len(weak)}**",
        f"- **Unclear** (interval crosses 0.5): **{len(unclear)}**",
        "",
        "Strong variants (best Wilson lower bound first):",
        "",
    ]
    strong_sorted = sorted(strong, key=_sort_key)
    for r in strong_sorted[:top_n]:
        _, lo, hi = win_rate_with_wilson(r.vs_heuristic)
        ph, _, _ = win_rate_with_wilson(r.vs_heuristic)
        lines.append(
            f"- `{r.label}` — win rate {ph:.4f}, Wilson [{lo:.4f}, {hi:.4f}]  "
            f"(vp_scale={r.vp_scale}, damage_scale={r.damage_scale}, win_bonus={r.win_bonus})"
        )
    if not strong:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "Weakest variants (lowest Wilson lower bound among those measured):",
            "",
        ]
    )
    weak_sorted = sorted(rows, key=lambda x: win_rate_with_wilson(x.vs_heuristic)[1])  # by lo ascending
    for r in weak_sorted[:top_n]:
        _, lo, hi = win_rate_with_wilson(r.vs_heuristic)
        ph, _, _ = win_rate_with_wilson(r.vs_heuristic)
        lines.append(
            f"- `{r.label}` — win rate {ph:.4f}, Wilson [{lo:.4f}, {hi:.4f}]  "
            f"(vp_scale={r.vp_scale}, damage_scale={r.damage_scale}, win_bonus={r.win_bonus})"
        )

    lines.extend(["", "## Full table (sorted by Wilson lower bound vs heuristic, descending)", ""])
    lines.append(
        "| label | vs heur WR | Wilson lo | Wilson hi | class | vs random WR | "
        "vp_scale | damage_scale | win_bonus |"
    )
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for r in sorted_rows:
        ph, loh, hih = win_rate_with_wilson(r.vs_heuristic)
        pr, _, _ = win_rate_with_wilson(r.vs_random)
        cls = classify_vs_anchor(loh, hih)
        lines.append(
            f"| {r.label} | {ph:.4f} | {loh:.4f} | {hih:.4f} | {cls} | {pr:.4f} | "
            f"{r.vp_scale} | {r.damage_scale} | {r.win_bonus} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "**Interpretation:** `vs random` should stay near 1.0; if it drops, treat the run as suspect. "
            "Increase `seed_count` if many variants are **unclear**.",
            "",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_from_toml(
    toml: dict[str, Any],
    *,
    progress: Any = None,
    workers_override: int | None = None,
) -> tuple[Path, Path]:
    runp = load_run_section(toml)
    csv_path, report_path, top_n, time_estimate = load_output_section(toml)
    variants = load_variants(toml)
    prog = progress if progress is not None else sys.stdout
    workers_eff = workers_override if workers_override is not None else load_parallel_workers(toml)

    seed_count = len(runp.seeds)
    cal_n = load_calibration_seed_count(toml)
    if (
        time_estimate
        and cal_n > 0
        and cal_n < seed_count
        and len(variants) > 0
    ):
        cal_seeds = list(runp.seeds[:cal_n])
        print(
            f"Calibration: all {len(variants)} variants × {cal_n} seeds "
            f"(timing only; full run uses {seed_count} seeds).",
            file=prog,
            flush=True,
        )
        t0 = time.perf_counter()
        run_sweep(toml, seeds=cal_seeds, workers=workers_eff, progress=prog)
        t_cal = time.perf_counter() - t0
        scale = float(seed_count) / float(cal_n)
        est = t_cal * scale
        print(
            f"Calibration wall time: {t_cal:.1f}s  →  estimated full run: {est:.1f}s "
            f"({est / 60.0:.1f} min), assuming time scales ~linearly with seed_count.",
            file=prog,
            flush=True,
        )

    print("Full evaluation:", file=prog, flush=True)
    wc = resolve_worker_count(workers_eff, len(variants))
    gp = load_parallel_game_progress_interval(toml)
    if wc > 1 and len(variants) > 1:
        if gp > 0:
            print(
                f"  Per-variant games: {4 * seed_count} (parallel: per-game lines every "
                f"{gp} games below; variant bar + tally when each variant finishes).",
                file=prog,
                flush=True,
            )
        else:
            print(
                "  Parallel mode: stdout may stay quiet until a variant finishes; "
                "set [parallel].game_progress_interval = 50 (or use workers=1 for a live bar).",
                file=prog,
                flush=True,
            )
    rows = run_sweep(toml, workers=workers_eff, progress=prog)
    write_csv(csv_path, rows)
    write_report(
        report_path,
        rows,
        seed_lo=int(runp.seeds[0]),
        seed_hi=int(runp.seeds[-1]),
        top_n=top_n,
    )
    return csv_path, report_path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to TOML (default: config/sweep.toml or config/sweep.example.toml under cwd)",
    )
    p.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        metavar="N",
        help="Process pool size (overrides [parallel].workers). 0 = min(cpu_count, variants). Default: from config.",
    )
    args = p.parse_args(argv)
    cwd = Path.cwd()
    cfg_path = args.config
    if cfg_path is None:
        for candidate in (cwd / "config" / "sweep.toml", cwd / "config" / "sweep.example.toml"):
            if candidate.is_file():
                cfg_path = candidate
                break
        if cfg_path is None:
            print(
                "No sweep config found. Copy config/sweep.example.toml to config/sweep.toml "
                "or pass --config PATH",
                file=sys.stderr,
            )
            raise SystemExit(2)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        raise SystemExit(2)
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    try:
        csv_p, report_p = run_from_toml(data, workers_override=args.workers)
    except SweepConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"Wrote {csv_p}", flush=True)
    print(f"Wrote {report_p}", flush=True)


if __name__ == "__main__":
    main()

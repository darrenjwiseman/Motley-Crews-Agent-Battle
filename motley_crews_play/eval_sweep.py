"""
Run many heuristic weight variants on the same seeds; write CSV + Markdown report.

Usage::

    python -m motley_crews_play.eval_sweep --config config/sweep.toml
    python -m motley_crews_play.eval_sweep --config config/sweep.toml --batch
    python -m motley_crews_play.eval_sweep --config config/sweep.toml --batch --batch-terminals
    python -m motley_crews_play.eval_sweep --config config/sweep.toml --batch --batch-yes
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import tomli_w
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from motley_crews_play.evaluation import PairwiseRecord, perspective_outcome, win_rate_with_wilson
from motley_crews_play.match import run_match
from motley_crews_play.policies import (
    ParameterizedHeuristicPolicy,
    RandomPolicy,
    ScriptedHeuristicPolicy,
    heuristic_weights_from_spec,
)


class SweepConfigError(ValueError):
    pass


class BatchSweepAborted(Exception):
    """User chose not to proceed after batch calibration (supervisor prompt)."""


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
        float(_req(row, "vp_scale", label))
        float(_req(row, "damage_scale", label))
        float(_req(row, "win_bonus", label))
        out.append(dict(row))
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


_CSV_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "vs_heuristic_wins",
        "vs_heuristic_losses",
        "vs_heuristic_draws",
        "vs_heuristic_timeouts",
        "vs_heuristic_win_rate",
        "vs_heuristic_wilson_lo",
        "vs_heuristic_wilson_hi",
        "vs_heuristic_class",
        "vs_random_wins",
        "vs_random_losses",
        "vs_random_draws",
        "vs_random_timeouts",
        "vs_random_win_rate",
        "vs_random_wilson_lo",
        "vs_random_wilson_hi",
    }
)


def _split_seeds(seeds: list[int], n_shards: int) -> list[list[int]]:
    """Split ``seeds`` into ``n_shards`` contiguous index ranges; no empty shards when len(seeds) >= n_shards."""
    n = len(seeds)
    if n_shards < 1:
        raise SweepConfigError("n_shards must be >= 1")
    if n == 0:
        return []
    n_shards_eff = min(n_shards, n)
    out: list[list[int]] = []
    for i in range(n_shards_eff):
        lo = i * n // n_shards_eff
        hi = (i + 1) * n // n_shards_eff
        out.append(seeds[lo:hi])
    return out


def _merge_pairwise_records(*records: PairwiseRecord) -> PairwiseRecord:
    out = PairwiseRecord()
    for r in records:
        out.wins += r.wins
        out.losses += r.losses
        out.draws += r.draws
        out.timeouts += r.timeouts
    return out


def _numbered_path(path: Path, index1: int) -> Path:
    """``results.csv`` → ``results_1.csv`` (1-based index)."""
    return path.parent / f"{path.stem}_{index1}{path.suffix}"


def _master_path(path: Path) -> Path:
    """``results.csv`` → ``results_master.csv``."""
    return path.parent / f"{path.stem}_master{path.suffix}"


def _parse_csv_spec_value(raw: str) -> Any:
    raw = raw.strip()
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        try:
            return [float(p) for p in parts]
        except ValueError:
            return parts
    rl = raw.lower()
    if rl in ("true", "false"):
        return rl == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


def variant_results_from_csv(path: Path) -> list[VariantResult]:
    """Rebuild :class:`VariantResult` rows from a sweep CSV (for batch aggregation)."""
    with path.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    out: list[VariantResult] = []
    for row in rows:
        label = row.get("label", "")
        spec: dict[str, Any] = {}
        for k, v in row.items():
            if k is None or k == "label" or k in _CSV_METRIC_KEYS:
                continue
            if v is None or v == "":
                continue
            spec[k] = _parse_csv_spec_value(v)
        vh = PairwiseRecord(
            wins=int(row["vs_heuristic_wins"]),
            losses=int(row["vs_heuristic_losses"]),
            draws=int(row["vs_heuristic_draws"]),
            timeouts=int(row["vs_heuristic_timeouts"]),
        )
        vrnd = PairwiseRecord(
            wins=int(row["vs_random_wins"]),
            losses=int(row["vs_random_losses"]),
            draws=int(row["vs_random_draws"]),
            timeouts=int(row["vs_random_timeouts"]),
        )
        out.append(VariantResult(label=label, spec=spec, vs_heuristic=vh, vs_random=vrnd))
    return out


def classify_vs_anchor(wilson_lo: float, wilson_hi: float) -> str:
    """vs stock heuristic: strong / weak / unclear."""
    if wilson_lo > 0.5:
        return "strong"
    if wilson_hi < 0.5:
        return "weak"
    return "unclear"


def _variant_spec_csv_value(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return v


@dataclass
class VariantResult:
    label: str
    spec: dict[str, Any]
    vs_heuristic: PairwiseRecord
    vs_random: PairwiseRecord

    def row_dict(self) -> dict[str, Any]:
        ph, loh, hih = win_rate_with_wilson(self.vs_heuristic)
        pr, lor, hir = win_rate_with_wilson(self.vs_random)
        cls = classify_vs_anchor(loh, hih)
        out: dict[str, Any] = {"label": self.label}
        for k in sorted(self.spec.keys()):
            out[k] = _variant_spec_csv_value(self.spec[k])
        out.update(
            {
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
        )
        return out


def merge_variant_results(rows_by_shard: list[list[VariantResult]]) -> list[VariantResult]:
    """Merge variants by ``label``; ``spec`` must match across shards (first wins)."""
    by_label: dict[str, tuple[dict[str, Any], list[VariantResult]]] = {}
    for shard_rows in rows_by_shard:
        for vr in shard_rows:
            if vr.label not in by_label:
                by_label[vr.label] = (vr.spec, [])
            spec0, group = by_label[vr.label]
            if vr.spec != spec0:
                raise SweepConfigError(
                    f"Variant {vr.label!r}: spec mismatch across batch shards (merge aborted)"
                )
            group.append(vr)
    merged: list[VariantResult] = []
    for label, (spec0, group) in by_label.items():
        hs = [x.vs_heuristic for x in group]
        rs = [x.vs_random for x in group]
        merged.append(
            VariantResult(
                label=label,
                spec=dict(spec0),
                vs_heuristic=_merge_pairwise_records(*hs),
                vs_random=_merge_pairwise_records(*rs),
            )
        )
    if not rows_by_shard:
        return []
    order = [r.label for r in rows_by_shard[0]]
    rank = {lb: i for i, lb in enumerate(order)}
    merged.sort(key=lambda x: rank.get(x.label, 10**9))
    return merged


def _progress_bar(done: int, total: int, width: int = 36) -> str:
    if total <= 0:
        return "[?]"
    filled = int(round(width * done / total))
    filled = min(max(filled, 0), width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


# Batch supervisor: poll shard logs at most this often (within 30–60s UX window).
BATCH_SUPERVISOR_POLL_INTERVAL_S = 45.0

_RE_SWEEP_PARALLEL = re.compile(r"\[sweep:([^\]]+)\] games (\d+)/(\d+)")
_RE_SWEEP_SEQUENTIAL = re.compile(r"^\s*\[[#\-]+\]\s+(\d+)/(\d+)\s+\S+\s+vs_heur")


def _games_done_from_shard_log(
    text: str,
    *,
    n_variants: int,
    games_per_variant: int,
) -> int:
    """
    Estimate completed games in one shard subprocess from its log (stdout).

    Parallel pool lines: ``[sweep:label] games gi/g_in_v``. Sequential lines:
    ``[###] global/total  label  vs_heur ...`` (non-tty / log file mode).
    """
    games_per_shard = n_variants * games_per_variant
    if games_per_shard <= 0:
        return 0
    if "[sweep:" in text:
        last_by_label: dict[str, int] = {}
        for raw in text.splitlines():
            line = raw.replace("\r", "")
            m = _RE_SWEEP_PARALLEL.search(line)
            if m:
                last_by_label[m.group(1)] = int(m.group(2))
        out = sum(last_by_label.values())
    else:
        out = 0
        for raw in text.splitlines():
            line = raw.replace("\r", "")
            m = _RE_SWEEP_SEQUENTIAL.match(line)
            if m:
                out = max(out, int(m.group(1)))
    return min(max(out, 0), games_per_shard)


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
    w = heuristic_weights_from_spec(spec)
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
    spec_out = {k: v for k, v in spec.items() if k != "label"}
    return VariantResult(
        label=label,
        spec=spec_out,
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
    fieldnames: list[str] = []
    seen: set[str] = set()
    for d in dicts:
        for k in d:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
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
    batch_note: str | None = None,
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
    ]
    if batch_note:
        lines.extend([f"- {batch_note}", ""])
    lines.extend(
        [
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
    )
    strong_sorted = sorted(strong, key=_sort_key)
    for r in strong_sorted[:top_n]:
        _, lo, hi = win_rate_with_wilson(r.vs_heuristic)
        ph, _, _ = win_rate_with_wilson(r.vs_heuristic)
        lines.append(
            f"- `{r.label}` — win rate {ph:.4f}, Wilson [{lo:.4f}, {hi:.4f}]  "
            f"(vp_scale={r.spec['vp_scale']}, damage_scale={r.spec['damage_scale']}, "
            f"win_bonus={r.spec['win_bonus']})"
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
            f"(vp_scale={r.spec['vp_scale']}, damage_scale={r.spec['damage_scale']}, "
            f"win_bonus={r.spec['win_bonus']})"
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
            f"{r.spec['vp_scale']} | {r.spec['damage_scale']} | {r.spec['win_bonus']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## How to interpret this report",
            "",
            "- **Paired sides:** For each seed, the variant plays `ScriptedHeuristicPolicy` as player 0, "
            "then as player 1 (same seed). `vs heur WR` is wins / (wins + losses); draws are excluded "
            "from that denominator.",
            "",
            "- **~0.5 vs stock is not a coin flip:** It means the variant wins as often as it loses under "
            "this seating. When a variant matches stock behavior, expect **about 50%**—that is **even "
            "with the anchor**, not RNG from the runner.",
            "",
            "- **Wilson interval and `unclear`:** Strong/weak use the 95% Wilson interval vs 0.5. If the "
            "point estimate is exactly 0.5, the interval **crosses** 0.5, so the class is **unclear** "
            "even with many seeds.",
            "",
            "- **`vs random`:** Sanity check that the heuristic beats random; it is **not** used for "
            "strong/weak vs stock.",
            "",
            "- **Identical rows across variants:** Same numbers mean **identical win/loss outcomes** on "
            "this run—not statistically independent rows. Often, nearby weights did not change chosen "
            "moves; a wider sweep or other metrics can expose differences.",
            "",
            "- **More seeds:** Larger `seed_count` helps when there is a **real** edge to measure; it "
            "will not separate variants that play identically.",
            "",
            "**Quick checks:** `vs random` should stay near 1.0; if it drops, treat the run as suspect. "
            "If many variants stay **unclear** because effects are small, try more seeds or stronger "
            "weight separation.",
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


def _batch_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    root = str(_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = root
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _default_batch_log_dir(csv_path: Path) -> Path:
    """Directory for per-shard logs, alongside configured CSV (e.g. ``results_batch_logs``)."""
    return csv_path.parent / f"{csv_path.stem}_batch_logs"


def _escape_applescript_string_literal(s: str) -> str:
    """Escape for use inside AppleScript ``"..."`` string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _open_macos_tail_windows(log_paths: list[Path], *, progress: Any) -> None:
    """Open Terminal.app windows running ``tail -f`` on each log path."""
    for i, lp in enumerate(log_paths):
        posix = _escape_applescript_string_literal(str(lp.resolve()))
        script = (
            'tell application "Terminal"\n'
            f'    do script "tail -f " & quoted form of "{posix}"\n'
            "end tell\n"
        )
        try:
            subprocess.run(
                ["osascript", "-"],
                input=script.encode("utf-8"),
                cwd=str(_ROOT),
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as e:
            print(
                f"Could not open Terminal for {lp.name} ({type(e).__name__}: {e}). "
                "Run: tail -f <path> manually.",
                file=progress,
                flush=True,
            )
        if i + 1 < len(log_paths):
            time.sleep(0.25)


def _build_shard_eval_toml(
    base_toml: dict[str, Any],
    *,
    chunk: list[int],
    csv_path: Path,
    report_path: Path,
    shard_index: int,
    per_shard: int,
) -> dict[str, Any]:
    """TOML for one batch shard (full chunk range; optional ``run_sweep(..., seeds=...)`` for calibration)."""
    shard_toml = copy.deepcopy(base_toml)
    run_sec = shard_toml.setdefault("run", {})
    if not isinstance(run_sec, dict):
        raise SweepConfigError("[run] must be a table")
    run_sec["seed_start"] = int(chunk[0])
    run_sec["seed_count"] = len(chunk)
    run_sec["calibration_seed_count"] = 0
    out_sec = shard_toml.setdefault("output", {})
    if not isinstance(out_sec, dict):
        out_sec = {}
        shard_toml["output"] = out_sec
    out_sec["csv_path"] = str(_numbered_path(csv_path, shard_index + 1).resolve())
    out_sec["report_path"] = str(_numbered_path(report_path, shard_index + 1).resolve())
    out_sec["time_estimate"] = False
    par_sec = shard_toml.setdefault("parallel", {})
    if not isinstance(par_sec, dict):
        par_sec = {}
        shard_toml["parallel"] = par_sec
    par_sec["workers"] = per_shard
    return shard_toml


def _batch_calibration_per_shard_estimates(
    base_toml: dict[str, Any],
    chunks: list[list[int]],
    *,
    csv_path: Path,
    report_path: Path,
    per_shard: int,
    cal_n: int,
    time_estimate: bool,
    progress: Any,
) -> tuple[list[float | None], float | None]:
    """
    Sequential in-process calibration per shard. Returns ``(est_per_shard, eta_wall)``
    where each estimate is scaled full-shard wall time; ``eta_wall`` is ``max`` for parallel batch.
    """
    n = len(chunks)
    est: list[float | None] = [None] * n
    if not time_estimate or cal_n < 1:
        return est, None
    prog = progress if progress is not None else sys.stdout
    print("", file=prog, flush=True)
    print("--- Batch calibration (timing only; full shard runs follow) ---", file=prog, flush=True)
    for i, chunk in enumerate(chunks):
        L = len(chunk)
        cal_i = min(cal_n, L)
        if cal_i < 1 or cal_i >= L:
            print(
                f"  Shard {i + 1}/{n}: skip ETA (need 0 < cal seeds < shard size; "
                f"shard has {L} seed(s), cal uses {cal_i}).",
                file=prog,
                flush=True,
            )
            continue
        shard_toml = _build_shard_eval_toml(
            base_toml,
            chunk=chunk,
            csv_path=csv_path,
            report_path=report_path,
            shard_index=i,
            per_shard=per_shard,
        )
        with open(os.devnull, "w", encoding="utf-8") as dev_out:
            t0 = time.perf_counter()
            run_sweep(
                shard_toml,
                seeds=chunk[:cal_i],
                workers=per_shard,
                progress=dev_out,
            )
            t_cal = time.perf_counter() - t0
        est_i = t_cal * (float(L) / float(cal_i))
        est[i] = est_i
        print(
            f"  Shard {i + 1}/{n}: cal {t_cal:.1f}s  →  est. full shard ~{est_i:.1f}s "
            f"({cal_i} cal seeds × {L} total)",
            file=prog,
            flush=True,
        )
    finite = [x for x in est if x is not None]
    eta_wall = max(finite) if finite else None
    if eta_wall is None:
        print("  No per-shard ETA (calibration skipped or unavailable).", file=prog, flush=True)
    return est, eta_wall


def _emit_batch_supervisor_progress_block(
    prog: Any,
    *,
    aggregate_done: int,
    total_games: int,
    elapsed_s: float,
    eta_wall: float | None,
    tty: bool,
    shard_done: list[int],
    shard_total: list[int],
    est_per_shard: list[float | None],
    n_shards: int,
    tty_state: dict[str, bool],
) -> None:
    """
    Aggregate line plus one indented line per shard (same bar / counts / % / remaining pattern).
    On TTY, redraws a block of ``1 + n_shards`` lines using cursor-up so scrollback stays clean.
    """
    if total_games <= 0:
        return
    lines: list[str] = []
    bar = _progress_bar(aggregate_done, total_games)
    pct = 100.0 * float(aggregate_done) / float(total_games)
    extra_agg = ""
    if eta_wall is not None and elapsed_s >= 0:
        rem = max(0.0, eta_wall - elapsed_s)
        extra_agg = f"  ~{rem:.0f}s remaining (rough; parallel shards)"
    lines.append(
        f"  {bar}  {aggregate_done}/{total_games} games  ({pct:.0f}%){extra_agg}"
    )
    for i in range(n_shards):
        st = shard_total[i] if i < len(shard_total) else 0
        sd = shard_done[i] if i < len(shard_done) else 0
        est_i = est_per_shard[i] if i < len(est_per_shard) else None
        if st > 0:
            bar_s = _progress_bar(sd, st)
            pct_s = 100.0 * float(sd) / float(st)
        else:
            bar_s = "[?]"
            pct_s = 0.0
        extra_s = ""
        if est_i is not None and st > 0:
            rem_s = max(0.0, est_i * (1.0 - float(sd) / float(st)))
            extra_s = f"  ~{rem_s:.0f}s remaining (rough; shard)"
        lines.append(
            f"    shard {i + 1}/{n_shards}: {bar_s}  {sd}/{st} games  ({pct_s:.0f}%){extra_s}"
        )
    n_lines = len(lines)
    if tty:
        # ANSI: move cursor up to block start, rewrite each line, clear to EOL (no scrollback spam).
        if tty_state.get("block_started"):
            prog.write(f"\033[{n_lines}A")
        for line in lines:
            prog.write("\r" + line + "\033[K\n")
        tty_state["block_started"] = True
        prog.flush()
    else:
        for line in lines:
            prog.write(line + "\n")
        prog.flush()


def run_sweep_batch(
    toml: dict[str, Any],
    *,
    progress: Any = None,
    workers_override: int | None = None,
    batch_shards: int = 4,
    batch_terminals: bool = False,
    batch_log_dir: Path | None = None,
    batch_yes: bool = False,
) -> tuple[Path, Path]:
    """
    Split ``[run]`` seeds across up to ``batch_shards`` disjoint ranges; run that many
    ``eval_sweep`` subprocesses writing ``*_1..*_N`` CSV/Markdown, then merge tallies into
    ``*_master`` outputs.
    """
    if batch_shards < 1:
        raise SweepConfigError("batch_shards must be >= 1")
    runp = load_run_section(toml)
    csv_path, report_path, top_n, time_estimate = load_output_section(toml)
    cal_n = load_calibration_seed_count(toml)
    variants = load_variants(toml)
    prog = progress if progress is not None else sys.stdout
    n_variants = len(variants)
    seeds = list(runp.seeds)
    n_shards = min(batch_shards, len(seeds))
    chunks = _split_seeds(seeds, n_shards)
    workers_eff = workers_override if workers_override is not None else load_parallel_workers(toml)
    resolved = resolve_worker_count(workers_eff, n_variants)
    per_shard = max(1, resolved // n_shards)
    games_per_shard = [4 * len(chunks[i]) * n_variants for i in range(n_shards)]
    total_games = sum(games_per_shard)
    log_dir = batch_log_dir if batch_log_dir is not None else _default_batch_log_dir(csv_path)
    tty_prog = _is_tty(prog)

    print("", file=prog, flush=True)
    print("=== Sweep batch (supervisor) ===", file=prog, flush=True)
    print(
        f"  Shards: {n_shards}  |  In-process workers per shard: {per_shard}  "
        f"(effective pool size {resolved})  |  Variants: {n_variants}",
        file=prog,
        flush=True,
    )
    print(
        f"  Full batch work: ~{total_games} games total across shards "
        f"(4 × seeds × variants per shard).",
        file=prog,
        flush=True,
    )

    est_per_shard, eta_wall = _batch_calibration_per_shard_estimates(
        toml,
        chunks,
        csv_path=csv_path,
        report_path=report_path,
        per_shard=per_shard,
        cal_n=cal_n,
        time_estimate=time_estimate,
        progress=prog,
    )

    print("", file=prog, flush=True)
    if eta_wall is not None:
        print(
            f"  Aggregated batch wall time (max over shards, parallel): ~{eta_wall:.1f}s "
            f"({eta_wall / 60.0:.1f} min)",
            file=prog,
            flush=True,
        )

    print("", file=prog, flush=True)
    if batch_yes:
        print("  (--batch-yes) Skipping confirmation prompt.", file=prog, flush=True)
    elif not sys.stdin.isatty():
        print("Non-interactive stdin: proceeding without prompt.", file=prog, flush=True)
    else:
        try:
            ans = input("Proceed with full batch? [Y/n] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("n", "no"):
            raise BatchSweepAborted()

    log_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [log_dir / f"shard_{i + 1}.log" for i in range(n_shards)]
    for lp in log_paths:
        lp.write_text("", encoding="utf-8")

    print(f"  Per-shard logs: {log_dir.resolve()}", file=prog, flush=True)
    if batch_terminals:
        if sys.platform == "darwin":
            print("  Opening Terminal windows: tail -f each shard log …", file=prog, flush=True)
            _open_macos_tail_windows(log_paths, progress=prog)
        else:
            print(
                "  Note: --batch-terminals only auto-opens Terminal.app on macOS; "
                "use tail -f on the paths above on this platform.",
                file=prog,
                flush=True,
            )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    env = _batch_subprocess_env()

    def run_one_shard(shard_index: int, chunk: list[int], tmp_dir: Path, log_path: Path) -> None:
        shard_toml = _build_shard_eval_toml(
            toml,
            chunk=chunk,
            csv_path=csv_path,
            report_path=report_path,
            shard_index=shard_index,
            per_shard=per_shard,
        )
        cfg_file = tmp_dir / f"shard_{shard_index}.toml"
        cfg_file.write_text(tomli_w.dumps(shard_toml), encoding="utf-8")
        with log_path.open("w", encoding="utf-8") as lf:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "motley_crews_play.eval_sweep",
                    "--config",
                    str(cfg_file),
                    "--workers",
                    str(per_shard),
                ],
                cwd=str(_ROOT),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                check=True,
            )

    print("", file=prog, flush=True)
    print(f"Starting {n_shards} shard subprocess(es)…", file=prog, flush=True)
    t_batch = time.perf_counter()
    est_list = [est_per_shard[i] if i < len(est_per_shard) else None for i in range(n_shards)]

    tty_block_state: dict[str, bool] = {"block_started": False}
    emit_lock = threading.Lock()
    shard_completed = [False] * n_shards
    stop_poller = threading.Event()

    def _read_shard_games_done(shard_index: int) -> int:
        gp_var = 4 * len(chunks[shard_index])
        try:
            text = log_paths[shard_index].read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        return _games_done_from_shard_log(
            text,
            n_variants=n_variants,
            games_per_variant=gp_var,
        )

    def _emit_supervisor_from_logs() -> None:
        shard_done_counts: list[int] = []
        for i in range(n_shards):
            if shard_completed[i]:
                shard_done_counts.append(games_per_shard[i])
            else:
                shard_done_counts.append(_read_shard_games_done(i))
        agg = sum(shard_done_counts)
        elapsed = time.perf_counter() - t_batch
        _emit_batch_supervisor_progress_block(
            prog,
            aggregate_done=agg,
            total_games=total_games,
            elapsed_s=elapsed,
            eta_wall=eta_wall,
            tty=tty_prog,
            shard_done=shard_done_counts,
            shard_total=games_per_shard,
            est_per_shard=est_list,
            n_shards=n_shards,
            tty_state=tty_block_state,
        )

    _emit_supervisor_from_logs()

    def _poller_loop() -> None:
        while True:
            if stop_poller.wait(timeout=BATCH_SUPERVISOR_POLL_INTERVAL_S):
                break
            with emit_lock:
                _emit_supervisor_from_logs()

    with tempfile.TemporaryDirectory(dir=csv_path.parent) as tmpd:
        tmp_path = Path(tmpd)
        with ThreadPoolExecutor(max_workers=n_shards) as pool:
            futs = {
                pool.submit(run_one_shard, i, chunks[i], tmp_path, log_paths[i]): i
                for i in range(n_shards)
            }
            poller = threading.Thread(target=_poller_loop, name="batch-supervisor-poll", daemon=True)
            poller.start()
            try:
                for fut in as_completed(futs):
                    shard_i = futs[fut]
                    fut.result()
                    with emit_lock:
                        shard_completed[shard_i] = True
                        _emit_supervisor_from_logs()
            finally:
                stop_poller.set()
                poller.join(timeout=5.0)
    if tty_prog:
        prog.write("\n")
        prog.flush()

    print("", file=prog, flush=True)
    print("Merging shard CSVs and writing master outputs…", file=prog, flush=True)
    shard_csvs = [_numbered_path(csv_path, i + 1) for i in range(n_shards)]
    parsed = [variant_results_from_csv(p) for p in shard_csvs]
    merged = merge_variant_results(parsed)
    master_csv = _master_path(csv_path)
    master_report = _master_path(report_path)
    write_csv(master_csv, merged)
    batch_note = (
        f"Aggregated from {n_shards} batch shard(s); disjoint seed ranges pooled into one tally."
    )
    write_report(
        master_report,
        merged,
        seed_lo=int(seeds[0]),
        seed_hi=int(seeds[-1]),
        top_n=top_n,
        batch_note=batch_note,
    )
    print("", file=prog, flush=True)
    print("=== Batch complete ===", file=prog, flush=True)
    for i in range(n_shards):
        print(f"  Shard outputs: {_numbered_path(csv_path, i + 1)}", file=prog, flush=True)
        print(f"                 {_numbered_path(report_path, i + 1)}", file=prog, flush=True)
        print(f"  Shard log:     {log_paths[i]}", file=prog, flush=True)
    print(f"  Master CSV:    {master_csv}", file=prog, flush=True)
    print(f"  Master report: {master_report}", file=prog, flush=True)
    return master_csv, master_report


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
    p.add_argument(
        "--batch",
        action="store_true",
        help="Split seeds across parallel sweep subprocesses, then write merged *_master CSV/report.",
    )
    p.add_argument(
        "--batch-shards",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel subprocesses in --batch mode (default: 4). Capped by seed_count.",
    )
    p.add_argument(
        "--batch-terminals",
        action="store_true",
        help="In --batch mode (macOS): open Terminal.app windows with tail -f on each shard log.",
    )
    p.add_argument(
        "--batch-log-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="In --batch mode: directory for shard_1.log … (default: next to CSV, e.g. results_batch_logs).",
    )
    p.add_argument(
        "--batch-yes",
        "--yes",
        action="store_true",
        dest="batch_yes",
        help="In --batch mode: skip Proceed? [Y/n] after calibration (non-interactive runs also skip).",
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
        if args.batch:
            csv_p, report_p = run_sweep_batch(
                data,
                workers_override=args.workers,
                batch_shards=args.batch_shards,
                batch_terminals=args.batch_terminals,
                batch_log_dir=args.batch_log_dir,
                batch_yes=args.batch_yes,
            )
        else:
            csv_p, report_p = run_from_toml(data, workers_override=args.workers)
    except BatchSweepAborted:
        print("Batch sweep aborted.", flush=True)
        raise SystemExit(0)
    except SweepConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        raise SystemExit(1) from e
    print(f"Wrote {csv_p}", flush=True)
    print(f"Wrote {report_p}", flush=True)


if __name__ == "__main__":
    main()

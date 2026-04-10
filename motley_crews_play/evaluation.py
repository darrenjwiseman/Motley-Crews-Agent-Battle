"""
Batch evaluation: paired seeds, win-rate statistics, Wilson intervals, round-robin, Elo.

Headless only — uses :func:`motley_crews_play.match.run_match`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from motley_crews_play.match import MatchLogger, MatchResult, run_match
from motley_crews_play.policies import Policy

Outcome = Literal["win", "loss", "draw", "timeout"]


def wilson_score_interval(successes: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson score 95% interval for binomial proportion (clamped to [0, 1]).
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4 * n)) / n)
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def perspective_outcome(result: MatchResult, player: int) -> Outcome:
    """
    Result from the viewpoint of the player index (0 or 1) in that match's ``run_match`` call.
    """
    if result.stopped_early or not result.final_state.done:
        return "timeout"
    w = result.final_state.winner
    if w is None:
        return "draw"
    if w == player:
        return "win"
    return "loss"


@dataclass
class PairwiseRecord:
    """Aggregated results for ``focus`` vs ``opponent`` with paired seeds (both seatings)."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    timeouts: int = 0

    @property
    def games(self) -> int:
        return self.wins + self.losses + self.draws + self.timeouts

    def add(self, o: Outcome) -> None:
        if o == "win":
            self.wins += 1
        elif o == "loss":
            self.losses += 1
        elif o == "draw":
            self.draws += 1
        else:
            self.timeouts += 1


def evaluate_pair_swapped(
    focus: Policy,
    opponent: Policy,
    seeds: Sequence[int],
    *,
    max_plies: int = 5000,
) -> PairwiseRecord:
    """
    For each seed, play ``focus`` as player 0 then as player 1 (same seed for setup RNG),
    and aggregate outcomes from ``focus``'s perspective.
    """
    rec = PairwiseRecord()
    for seed in seeds:
        r0 = run_match(focus, opponent, seed=seed, max_plies=max_plies)
        rec.add(perspective_outcome(r0, 0))
        r1 = run_match(opponent, focus, seed=seed, max_plies=max_plies)
        rec.add(perspective_outcome(r1, 1))
    return rec


def win_rate_point_estimate(record: PairwiseRecord) -> float:
    """Wins / games that ended in win or loss (excludes draws and timeouts)."""
    decided = record.wins + record.losses
    if decided == 0:
        return 0.0
    return record.wins / decided


def win_rate_with_wilson(record: PairwiseRecord) -> tuple[float, float, float]:
    """
    Point estimate of win rate among decided games, and Wilson interval on wins vs decided.
    Returns ``(p_hat, lo, hi)``. If ``decided == 0``, returns ``(0.0, 0.0, 0.0)``.
    """
    decided = record.wins + record.losses
    if decided == 0:
        return (0.0, 0.0, 0.0)
    p = record.wins / decided
    lo, hi = wilson_score_interval(record.wins, decided)
    return (p, lo, hi)


@dataclass
class RoundRobinRow:
    name: str
    policy: Policy
    vs: dict[str, PairwiseRecord] = field(default_factory=dict)


@dataclass
class RoundRobinResult:
    """Symmetric pairwise table: ``rows[i].vs[name_j]`` is policy ``i`` vs ``j`` from i's perspective."""

    names: list[str]
    policies: list[Policy]
    rows: list[RoundRobinRow]


def round_robin(
    entries: Sequence[tuple[str, Policy]],
    seeds: Sequence[int],
    *,
    max_plies: int = 5000,
) -> RoundRobinResult:
    """
    For each unordered pair of distinct policies, run :func:`evaluate_pair_swapped`.

    ``rows[i].vs[names[j]]`` holds stats for policy ``i`` against ``j`` (from ``i``'s view);
    diagonal entries are empty ``PairwiseRecord``s.
    """
    names = [e[0] for e in entries]
    policies = [e[1] for e in entries]
    n = len(names)
    rows: list[RoundRobinRow] = [RoundRobinRow(name=names[i], policy=policies[i]) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                rows[i].vs[names[j]] = PairwiseRecord()
            elif j > i:
                rec_ij = evaluate_pair_swapped(policies[i], policies[j], seeds, max_plies=max_plies)
                rows[i].vs[names[j]] = rec_ij
                rec_ji = PairwiseRecord(
                    wins=rec_ij.losses,
                    losses=rec_ij.wins,
                    draws=rec_ij.draws,
                    timeouts=rec_ij.timeouts,
                )
                rows[j].vs[names[i]] = rec_ji
            else:
                pass  # filled when (j, i) with j > i
    return RoundRobinResult(names=names, policies=policies, rows=rows)


@dataclass
class EloState:
    ratings: dict[str, float]


class EloTracker:
    """
    Incremental Elo with optional draw handling (counts as half score for each).
    """

    def __init__(self, *, k: float = 32.0, base: float = 1500.0) -> None:
        self.k = k
        self.base = base

    def expected(self, r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))

    def update_one(
        self,
        r_a: float,
        r_b: float,
        score_a: float,
    ) -> tuple[float, float]:
        """``score_a`` in ``{0.0, 0.5, 1.0}``. Returns new ``(r_a, r_b)``."""
        ea = self.expected(r_a, r_b)
        delta = self.k * (score_a - ea)
        return (r_a + delta, r_b - delta)

    def ratings_from_round_robin(self, rr: RoundRobinResult) -> EloState:
        """
        Walk all unordered pairs and apply updates from each game outcome in the records.
        """
        ratings: dict[str, float] = {name: self.base for name in rr.names}
        for i, name_i in enumerate(rr.names):
            for j in range(i + 1, len(rr.names)):
                name_j = rr.names[j]
                rec = rr.rows[i].vs[name_j]
                # Each evaluate_pair_swapped produces 2 games per seed; treat each outcome as one update
                _apply_outcomes(self, ratings, name_i, name_j, rec)
        return EloState(ratings=ratings)


def _apply_outcomes(
    tracker: EloTracker,
    ratings: dict[str, float],
    name_a: str,
    name_b: str,
    rec: PairwiseRecord,
) -> None:
    for _ in range(rec.wins):
        ra, rb = tracker.update_one(ratings[name_a], ratings[name_b], 1.0)
        ratings[name_a], ratings[name_b] = ra, rb
    for _ in range(rec.losses):
        ra, rb = tracker.update_one(ratings[name_a], ratings[name_b], 0.0)
        ratings[name_a], ratings[name_b] = ra, rb
    for _ in range(rec.draws):
        ra, rb = tracker.update_one(ratings[name_a], ratings[name_b], 0.5)
        ratings[name_a], ratings[name_b] = ra, rb
    # Timeouts: skip to avoid distorting ratings


def action_tuple_has_special(action_tuple: tuple[Any, ...]) -> bool:
    """True if the serialized turn includes a special action (tag ``\"sp\"``)."""
    if len(action_tuple) < 2:
        return False
    ap = action_tuple[1]
    return isinstance(ap, tuple) and len(ap) > 0 and ap[0] == "sp"


def action_tuple_has_move(action_tuple: tuple[Any, ...]) -> bool:
    """True if the serialized turn includes a move (tag ``\"m\"``)."""
    mp = action_tuple[0]
    return mp is not None


@dataclass
class MatchBehaviorStats:
    plies: int
    special_actions: int
    turns_with_move: int

    @property
    def special_rate(self) -> float:
        if self.plies <= 0:
            return 0.0
        return self.special_actions / self.plies

    @property
    def move_rate(self) -> float:
        if self.plies <= 0:
            return 0.0
        return self.turns_with_move / self.plies


def behavior_stats_from_log(log: MatchLogger) -> MatchBehaviorStats:
    """
    Cheap diversity-oriented tags: ply count, specials used, moves taken.
    """
    specials = 0
    moves = 0
    for e in log.entries:
        t = e.get("action_tuple")
        if not isinstance(t, tuple):
            continue
        if action_tuple_has_special(t):
            specials += 1
        if action_tuple_has_move(t):
            moves += 1
    return MatchBehaviorStats(
        plies=len(log.entries),
        special_actions=specials,
        turns_with_move=moves,
    )


def evaluate_pair_with_logs(
    focus: Policy,
    opponent: Policy,
    seeds: Sequence[int],
    *,
    max_plies: int = 5000,
) -> tuple[PairwiseRecord, list[MatchBehaviorStats]]:
    """
    Like :func:`evaluate_pair_swapped` but also returns :class:`MatchBehaviorStats` per game played.
    """
    rec = PairwiseRecord()
    behaviors: list[MatchBehaviorStats] = []
    for seed in seeds:
        for a, b in ((focus, opponent), (opponent, focus)):
            log = MatchLogger()
            r = run_match(a, b, seed=seed, max_plies=max_plies, log=log)
            focus_is_p0 = a is focus
            o = perspective_outcome(r, 0 if focus_is_p0 else 1)
            rec.add(o)
            behaviors.append(behavior_stats_from_log(log))
    return rec, behaviors

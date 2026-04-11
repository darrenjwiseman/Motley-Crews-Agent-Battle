"""
Pluggable policies: random, scripted CPU, human queue, and future agent stub.

All policies implement ``choose(state, legal, rng) -> TurnAction`` and must return
one of the provided ``legal`` actions.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from motley_crews_env.constants import (
    BOARD_SIZE,
    DEPLOY_ROWS_PLAYER_A,
    DEPLOY_ROWS_PLAYER_B,
    FIGURES_PER_SIDE,
    NUM_CLASSES,
    TEAM_PLAYER_A,
    TEAM_PLAYER_B,
)
from motley_crews_env.engine import step
from motley_crews_env.state import GameState, IllegalActionError, StepResult, slot_unit, unit_at
from motley_crews_env.types import SetupPlacement, TurnAction


@runtime_checkable
class Policy(Protocol):
    def choose(
        self,
        state: GameState,
        legal: Sequence[TurnAction],
        rng: random.Random,
    ) -> TurnAction: ...


class HumanInputPendingError(RuntimeError):
    """Raised when HumanPolicy.choose is called before submit."""


class HumanPolicy:
    """
    Policy fed by the UI: call ``submit`` with a legal ``TurnAction``, then ``choose``
    consumes it. For synchronous drivers (CLI); Pygame may call ``step`` directly instead.
    """

    def __init__(self) -> None:
        self._queue: deque[TurnAction] = deque()

    def submit(self, action: TurnAction) -> None:
        self._queue.append(action)

    def clear(self) -> None:
        self._queue.clear()

    def choose(
        self,
        state: GameState,
        legal: Sequence[TurnAction],
        rng: random.Random,
    ) -> TurnAction:
        del state, rng
        if not self._queue:
            raise HumanInputPendingError("submit a TurnAction before choose")
        action = self._queue.popleft()
        legal_list = list(legal)
        if action not in legal_list:
            raise ValueError("submitted action is not in legal_actions for this state")
        return action


class RandomPolicy:
    def choose(
        self,
        state: GameState,
        legal: Sequence[TurnAction],
        rng: random.Random,
    ) -> TurnAction:
        del state
        if not legal:
            raise RuntimeError("no legal actions")
        return rng.choice(list(legal))

    def choose_setup(
        self,
        state: GameState,
        legal: Sequence[SetupPlacement],
        rng: random.Random,
    ) -> SetupPlacement:
        del state
        opts = list(legal)
        if not opts:
            raise RuntimeError("no legal setup actions")
        return rng.choice(opts)


def _enemy_hp_sum(state: GameState, me: int) -> int:
    opp = 1 - me
    total = 0
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(state, t, sl)
            if u is not None and u.alive and u.controller == opp:
                total += u.hp
    return total


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    """
    One-ply greedy scoring: larger primary score is better.

    Defaults match the original fixed mix: VP change scaled by 10_000 plus raw enemy HP damage,
    with large terminal bonuses.

    Per-class and group multipliers scale the primary score when the acting figure(s) are known.
    `w_class` is indexed by ``ClassId`` (knight, barbarian, white_mage, black_mage, arbalist).
    Group multipliers stack on top: melee → knight+barbarian; mage → both mages; arbalist → arbalist.

    Deployment scoring uses the same effective class weight times a simple geometry term controlled
    by ``deploy_forward`` and ``deploy_center`` (both 0 by default = class weights only).
    """

    vp_scale: float = 10_000.0
    damage_scale: float = 1.0
    win_bonus: float = 10_000_000.0
    w_class: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    group_melee: float = 1.0
    group_mage: float = 1.0
    group_arbalist: float = 1.0
    deploy_forward: float = 0.0
    deploy_center: float = 0.0


def effective_actor_weight(w: HeuristicWeights, class_id: int) -> float:
    """Product of per-class `w_class` and applicable group multipliers."""
    if class_id < 0 or class_id >= NUM_CLASSES:
        return 1.0
    base = w.w_class[class_id]
    g = 1.0
    if class_id in (0, 1):
        g *= w.group_melee
    if class_id in (2, 3):
        g *= w.group_mage
    if class_id == 4:
        g *= w.group_arbalist
    return base * g


def _turn_actor_factor(state: GameState, me: int, turn: TurnAction, w: HeuristicWeights) -> float:
    """
    Geometric mean of effective actor weights for figures involved in move/action (same slot twice
    counts twice). Pass-only turns and unknown actors yield 1.0.
    """
    slots: list[int] = []
    if turn.move:
        slots.append(turn.move.actor_slot)
    if turn.action:
        slots.append(turn.action.actor_slot)
    if not slots:
        return 1.0
    from math import prod

    weights: list[float] = []
    for sl in slots:
        u = slot_unit(state, me, sl)
        if u is None or not u.alive:
            continue
        weights.append(effective_actor_weight(w, u.class_id))
    if not weights:
        return 1.0
    p = prod(weights)
    return p ** (1.0 / len(weights))


def _parse_w_class(spec: Mapping[str, Any]) -> tuple[float, float, float, float, float]:
    if "w_class" in spec:
        raw = spec["w_class"]
        if isinstance(raw, (list, tuple)) and len(raw) == NUM_CLASSES:
            return tuple(float(x) for x in raw)  # type: ignore[return-value]
        raise ValueError("w_class must be a list of 5 floats")
    keys = ("w_knight", "w_barbarian", "w_white_mage", "w_black_mage", "w_arbalist")
    if any(k in spec for k in keys):
        return tuple(float(spec.get(k, 1.0)) for k in keys)
    return (1.0, 1.0, 1.0, 1.0, 1.0)


def heuristic_weights_from_spec(spec: Mapping[str, Any]) -> HeuristicWeights:
    """Build :class:`HeuristicWeights` from a flat mapping (e.g. TOML variant row)."""
    vp = float(spec.get("vp_scale", 10_000.0))
    dmg = float(spec.get("damage_scale", 1.0))
    win = float(spec.get("win_bonus", 10_000_000.0))
    w_class = _parse_w_class(spec)
    gm = float(spec.get("group_melee", 1.0))
    gmg = float(spec.get("group_mage", 1.0))
    ga = float(spec.get("group_arbalist", 1.0))
    df = float(spec.get("deploy_forward", 0.0))
    dc = float(spec.get("deploy_center", 0.0))
    return HeuristicWeights(
        vp_scale=vp,
        damage_scale=dmg,
        win_bonus=win,
        w_class=w_class,
        group_melee=gm,
        group_mage=gmg,
        group_arbalist=ga,
        deploy_forward=df,
        deploy_center=dc,
    )


def _deploy_forwardness(team: int, row: int) -> float:
    rows = DEPLOY_ROWS_PLAYER_A if team == TEAM_PLAYER_A else DEPLOY_ROWS_PLAYER_B
    lo, hi = min(rows), max(rows)
    if hi == lo:
        return 1.0
    if team == TEAM_PLAYER_A:
        return (hi - row) / (hi - lo)
    return (row - lo) / (hi - lo)


def _deploy_center_col(col: int) -> float:
    mid = (BOARD_SIZE - 1) * 0.5
    if mid <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(col - mid) / mid)


def score_setup_placement(
    state: GameState,
    team: int,
    placement: SetupPlacement,
    w: HeuristicWeights,
) -> float:
    """Higher is better. Uses staged unit class for ``actor_slot`` and optional board geometry."""
    u = slot_unit(state, team, placement.actor_slot)
    if u is None or not u.alive:
        return float("-inf")
    eff = effective_actor_weight(w, u.class_id)
    r, c = placement.destination
    fwd = _deploy_forwardness(team, r)
    ctr = _deploy_center_col(c)
    geom = 1.0 + w.deploy_forward * fwd + w.deploy_center * ctr
    return eff * geom


def _score_transition_weighted(
    old: GameState, sr: StepResult, me: int, w: HeuristicWeights
) -> tuple[float, int, int]:
    """Higher is better for ``me``. Lexicographic tuple; first field is the weighted heuristic."""
    new = sr.state
    opp = 1 - me
    if sr.done:
        if sr.winner == me:
            return (w.win_bonus, 0, 0)
        if sr.winner == opp:
            return (-w.win_bonus, 0, 0)
        return (0.0, 0, 0)

    d_score = new.score[me] - old.score[me]
    ehp_old = _enemy_hp_sum(old, me)
    ehp_new = _enemy_hp_sum(new, me)
    dmg = ehp_old - ehp_new
    primary = w.vp_scale * float(d_score) + w.damage_scale * float(dmg)
    return (primary, d_score, dmg)


class ParameterizedHeuristicPolicy:
    """
    Same as :class:`ScriptedHeuristicPolicy` but with tunable :class:`HeuristicWeights`
    for search, tournaments, and style experiments.
    """

    def __init__(self, weights: HeuristicWeights | None = None) -> None:
        self.weights = weights if weights is not None else HeuristicWeights()

    def choose_setup(
        self,
        state: GameState,
        legal: Sequence[SetupPlacement],
        rng: random.Random,
    ) -> SetupPlacement:
        opts = list(legal)
        if not opts:
            raise RuntimeError("no legal setup actions")
        w = self.weights
        best: list[SetupPlacement] = []
        best_s: float | None = None
        pl = state.setup_current_player
        for p in opts:
            s = score_setup_placement(state, pl, p, w)
            if best_s is None or s > best_s:
                best_s = s
                best = [p]
            elif s == best_s:
                best.append(p)
        return rng.choice(best)

    def choose(
        self,
        state: GameState,
        legal: Sequence[TurnAction],
        rng: random.Random,
    ) -> TurnAction:
        if not legal:
            raise RuntimeError("no legal actions")
        legal_list = list(legal)
        best: list[TurnAction] = []
        best_key: tuple[float, int, int] | None = None
        me = state.current_player
        w = self.weights
        for a in legal_list:
            try:
                sr = step(state, a)
            except IllegalActionError:
                continue
            base = _score_transition_weighted(state, sr, me, w)
            factor = _turn_actor_factor(state, me, a, w)
            key = (base[0] * factor, base[1], base[2])
            if best_key is None or key > best_key:
                best_key = key
                best = [a]
            elif key == best_key:
                best.append(a)
        if not best:
            return rng.choice(legal_list)
        return rng.choice(best)


class ScriptedHeuristicPolicy(ParameterizedHeuristicPolicy):
    """
    Greedy one-ply evaluation: simulate each legal full turn and prefer outcomes
    with higher score gain and enemy HP reduction. Tie-break is deterministic
    (lexicographic on action tuple), then ``rng`` among true ties.

    Replaceable later by :class:`AgentPolicy` without changing the match runner.
    """

    def __init__(self) -> None:
        super().__init__(HeuristicWeights())


class AgentPolicy(ABC):
    """
    Placeholder for Phase 2 baselines / RL.

    Subclass and implement :meth:`choose` using ``motley_crews_env.encoding.encode_observation``
    (or structured tensors) and the legal ``TurnAction`` list as an action mask /
    re-scored discrete index over ``legal``.
    """

    @abstractmethod
    def choose(
        self,
        state: GameState,
        legal: Sequence[TurnAction],
        rng: random.Random,
    ) -> TurnAction: ...

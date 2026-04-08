"""
Pluggable policies: random, scripted CPU, human queue, and future agent stub.

All policies implement ``choose(state, legal, rng) -> TurnAction`` and must return
one of the provided ``legal`` actions.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from typing import Protocol, Sequence, runtime_checkable

from motley_crews_env.constants import FIGURES_PER_SIDE, TEAM_PLAYER_A, TEAM_PLAYER_B
from motley_crews_env.engine import step
from motley_crews_env.state import GameState, IllegalActionError, StepResult, unit_at
from motley_crews_env.types import TurnAction


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


def _enemy_hp_sum(state: GameState, me: int) -> int:
    opp = 1 - me
    total = 0
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(state, t, sl)
            if u is not None and u.alive and u.controller == opp:
                total += u.hp
    return total


def _score_transition(old: GameState, sr: StepResult, me: int) -> tuple[int, int, int]:
    """Higher is better for ``me``. Returns a sortable tuple."""
    new = sr.state
    opp = 1 - me
    if sr.done:
        if sr.winner == me:
            return (10_000_000, 0, 0)
        if sr.winner == opp:
            return (-10_000_000, 0, 0)
        return (0, 0, 0)

    d_score = new.score[me] - old.score[me]
    ehp_old = _enemy_hp_sum(old, me)
    ehp_new = _enemy_hp_sum(new, me)
    dmg = ehp_old - ehp_new
    return (d_score * 10_000 + dmg, d_score, dmg)


class ScriptedHeuristicPolicy:
    """
    Greedy one-ply evaluation: simulate each legal full turn and prefer outcomes
    with higher score gain and enemy HP reduction. Tie-break is deterministic
    (lexicographic on action tuple), then ``rng`` among true ties.

    Replaceable later by :class:`AgentPolicy` without changing the match runner.
    """

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
        best_key: tuple[int, int, int] | None = None
        me = state.current_player
        for a in legal_list:
            try:
                sr = step(state, a)
            except IllegalActionError:
                continue
            key = _score_transition(state, sr, me)
            if best_key is None or key > best_key:
                best_key = key
                best = [a]
            elif key == best_key:
                best.append(a)
        if not best:
            return rng.choice(legal_list)
        return rng.choice(best)


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

"""Headless match runner with optional ply logging (for replay / debugging)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from motley_crews_env.engine import initial_state, legal_actions, step
from motley_crews_env.serialization import turn_action_to_tuple
from motley_crews_env.state import GameState
from motley_crews_env.types import TurnAction

from motley_crews_play.policies import Policy


@dataclass
class MatchResult:
    """Outcome of ``run_match``."""

    final_state: GameState
    plies: int
    stopped_early: bool  # True if max_plies reached before terminal


@dataclass
class MatchLogger:
    """Append-only log of plies for replay."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        ply_index: int,
        current_player: int,
        action: TurnAction,
        state_after: GameState,
    ) -> None:
        self.entries.append(
            {
                "ply_index": ply_index,
                "current_player": current_player,
                "action_tuple": turn_action_to_tuple(action),
                "score": tuple(state_after.score),
                "done": state_after.done,
                "winner": state_after.winner,
            }
        )


def run_match(
    policy_a: Policy,
    policy_b: Policy,
    *,
    seed: int,
    max_plies: int = 5000,
    log: Optional[MatchLogger] = None,
    initial: Optional[GameState] = None,
) -> MatchResult:
    """
    Play until terminal or ``max_plies`` half-turns. Player A moves first (engine default).

    Policies are selected by ``state.current_player`` (0 -> policy_a, 1 -> policy_b).
    """
    rng = random.Random(seed)
    state = initial_state() if initial is None else initial
    policies: tuple[Policy, Policy] = (policy_a, policy_b)
    n = 0
    while not state.done and n < max_plies:
        legal = legal_actions(state)
        if not legal:
            break
        pl = state.current_player
        action = policies[pl].choose(state, legal, rng)
        sr = step(state, action)
        state = sr.state
        if log is not None:
            log.record(
                ply_index=n,
                current_player=pl,
                action=action,
                state_after=state,
            )
        n += 1
    return MatchResult(final_state=state, plies=n, stopped_early=not state.done and n >= max_plies)

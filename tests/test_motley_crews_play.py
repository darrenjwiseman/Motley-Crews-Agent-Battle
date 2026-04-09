"""Policies and match runner smoke tests."""

from __future__ import annotations

import random

import pytest

from motley_crews_env.engine import initial_play_state, legal_actions, step
from motley_crews_play.match import MatchLogger, run_match
from motley_crews_play.policies import (
    HumanInputPendingError,
    HumanPolicy,
    RandomPolicy,
    ScriptedHeuristicPolicy,
)


def test_random_policy_always_legal() -> None:
    rng = random.Random(42)
    pol = RandomPolicy()
    state = initial_play_state()
    for _ in range(30):
        if state.done:
            break
        legal = legal_actions(state)
        assert legal
        a = pol.choose(state, legal, rng)
        assert a in legal
        state = step(state, a).state


def test_scripted_heuristic_always_legal() -> None:
    rng = random.Random(7)
    pol = ScriptedHeuristicPolicy()
    state = initial_play_state()
    for _ in range(40):
        if state.done:
            break
        legal = legal_actions(state)
        assert legal
        a = pol.choose(state, legal, rng)
        assert a in legal
        state = step(state, a).state


def test_run_match_heuristic_terminates() -> None:
    result = run_match(
        ScriptedHeuristicPolicy(),
        ScriptedHeuristicPolicy(),
        seed=123,
        max_plies=8000,
    )
    assert result.final_state.done
    assert result.plies < 8000


def test_run_match_with_log() -> None:
    log = MatchLogger()
    result = run_match(
        RandomPolicy(),
        RandomPolicy(),
        seed=1,
        max_plies=200,
        log=log,
    )
    assert result.plies == len(log.entries)


def test_human_policy_submit() -> None:
    rng = random.Random(0)
    state = initial_play_state()
    legal = legal_actions(state)
    pol = HumanPolicy()
    pol.submit(legal[0])
    assert pol.choose(state, legal, rng) == legal[0]


def test_human_policy_empty_raises() -> None:
    state = initial_play_state()
    legal = legal_actions(state)
    pol = HumanPolicy()
    with pytest.raises(HumanInputPendingError):
        pol.choose(state, legal, random.Random())

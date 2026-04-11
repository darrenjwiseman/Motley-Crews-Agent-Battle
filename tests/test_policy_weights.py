"""Piece/group heuristic weights and policy-driven setup."""

from __future__ import annotations

import random

import pytest

from motley_crews_env.engine import begin_setup, initial_state, legal_setup_actions
from motley_crews_env.types import MatchPhase, SetupPlacement
from motley_crews_play.match import run_match
from motley_crews_play.policies import (
    HeuristicWeights,
    ParameterizedHeuristicPolicy,
    RandomPolicy,
    ScriptedHeuristicPolicy,
    effective_actor_weight,
    heuristic_weights_from_spec,
    score_setup_placement,
)


def test_effective_actor_weight_defaults() -> None:
    w = HeuristicWeights()
    for cid in range(5):
        assert effective_actor_weight(w, cid) == 1.0


def test_effective_actor_weight_class_and_groups() -> None:
    w = HeuristicWeights(
        w_class=(2.0, 1.0, 1.0, 1.0, 1.0),
        group_melee=3.0,
        group_mage=5.0,
        group_arbalist=7.0,
    )
    assert effective_actor_weight(w, 0) == pytest.approx(2.0 * 3.0)  # knight
    assert effective_actor_weight(w, 2) == pytest.approx(1.0 * 5.0)  # white mage
    assert effective_actor_weight(w, 4) == pytest.approx(1.0 * 7.0)  # arbalist


def test_heuristic_weights_from_spec_w_class_list() -> None:
    w = heuristic_weights_from_spec(
        {
            "vp_scale": 1.0,
            "damage_scale": 2.0,
            "win_bonus": 3.0,
            "w_class": [0.5, 1.5, 1.0, 1.0, 2.0],
        }
    )
    assert w.w_class == (0.5, 1.5, 1.0, 1.0, 2.0)


def test_heuristic_weights_from_spec_individual_w_knight() -> None:
    w = heuristic_weights_from_spec(
        {
            "vp_scale": 10000.0,
            "damage_scale": 1.0,
            "win_bonus": 1e7,
            "w_knight": 9.0,
        }
    )
    assert w.w_class[0] == 9.0
    assert w.w_class[1] == 1.0


def test_run_match_random_setup_same_as_legacy() -> None:
    """RandomPolicy + setup_random True should match old complete_setup_random RNG usage."""
    a = run_match(RandomPolicy(), RandomPolicy(), seed=99991, max_plies=400, setup_random=True)
    b = run_match(RandomPolicy(), RandomPolicy(), seed=99991, max_plies=400, setup_random=False)
    assert a.plies == b.plies
    assert a.final_state.board.tolist() == b.final_state.board.tolist()


def test_setup_scoring_prefers_weighted_class() -> None:
    state = initial_state()
    rng = random.Random(0)
    state = begin_setup(state, coin_flip_winner=0, winner_chooses_first_setup=True)
    assert state.match_phase == int(MatchPhase.SETUP)
    legal = legal_setup_actions(state)
    assert legal
    w_hi = HeuristicWeights(w_class=(100.0, 1.0, 1.0, 1.0, 1.0))
    w_lo = HeuristicWeights()
    # Pick two legal placements with different actor_slot if possible
    by_slot: dict[int, SetupPlacement] = {}
    for p in legal:
        by_slot.setdefault(p.actor_slot, p)
    if len(by_slot) < 2:
        pytest.skip("need two distinct actor slots in first setup batch")
    s0 = min(by_slot)
    s1 = max(by_slot)
    p0 = by_slot[s0]
    p1 = by_slot[s1]
    # class_id follows slot index in this env
    if p0.actor_slot == 0 and p1.actor_slot != 0:
        assert score_setup_placement(state, state.setup_current_player, p0, w_hi) > score_setup_placement(
            state, state.setup_current_player, p1, w_hi
        )


def test_scripted_heuristic_terminates_with_policy_setup() -> None:
    result = run_match(
        ScriptedHeuristicPolicy(),
        ScriptedHeuristicPolicy(),
        seed=123,
        max_plies=8000,
    )
    assert result.final_state.done
    assert result.plies < 8000

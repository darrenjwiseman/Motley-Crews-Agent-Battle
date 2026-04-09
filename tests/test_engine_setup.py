"""Coin flip, first/second setup choice, and alternating staging placement."""

from __future__ import annotations

import random

import pytest

from motley_crews_env.constants import DEPLOY_ROWS_PLAYER_A, DEPLOY_ROWS_PLAYER_B, FIGURES_PER_SIDE
from motley_crews_env.engine import (
    begin_setup,
    complete_setup_random,
    initial_state,
    legal_actions,
    legal_setup_actions,
    setup_step,
)
from motley_crews_env.state import IllegalActionError, unit_at
from motley_crews_env.types import MatchPhase, SetupPlacement, TeamId


def test_initial_state_is_staging_only() -> None:
    s = initial_state()
    assert s.match_phase == int(MatchPhase.PENDING_SETUP)
    assert sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, 0, sl)) == 0
    assert sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, 1, sl)) == 0


def test_begin_setup_pairs_first_setup_with_first_turn() -> None:
    s = initial_state()
    s2 = begin_setup(s, coin_flip_winner=int(TeamId.PLAYER_B), winner_chooses_first_setup=True)
    assert s2.setup_current_player == int(TeamId.PLAYER_B)
    assert s2.first_player == int(TeamId.PLAYER_B)


def test_begin_setup_winner_can_take_second_setup_and_second_turn() -> None:
    s = initial_state()
    s2 = begin_setup(s, coin_flip_winner=int(TeamId.PLAYER_A), winner_chooses_first_setup=False)
    assert s2.setup_current_player == int(TeamId.PLAYER_B)
    assert s2.first_player == int(TeamId.PLAYER_B)


def test_placements_only_in_home_two_rows() -> None:
    rng = random.Random(0)
    s = initial_state()
    s = begin_setup(s, coin_flip_winner=0, winner_chooses_first_setup=True)
    s = complete_setup_random(s, rng)
    assert s.match_phase == int(MatchPhase.PLAY)
    for pl in (0, 1):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, pl, sl)
            assert u is not None
            r = u.row
            if pl == 0:
                assert r in DEPLOY_ROWS_PLAYER_A
            else:
                assert r in DEPLOY_ROWS_PLAYER_B


def test_middle_rows_empty_after_random_setup() -> None:
    rng = random.Random(2)
    s = initial_state()
    s = begin_setup(s, coin_flip_winner=1, winner_chooses_first_setup=False)
    s = complete_setup_random(s, rng)
    for r in (2, 3, 4, 5):
        for c in range(8):
            assert s.board[r, c] < 0


def test_manual_setup_ten_plies_then_play() -> None:
    s = initial_state()
    s = begin_setup(s, coin_flip_winner=int(TeamId.PLAYER_A), winner_chooses_first_setup=True)
    for _ in range(10):
        assert s.match_phase == int(MatchPhase.SETUP)
        opts = legal_setup_actions(s)
        assert opts
        s = setup_step(s, opts[0]).state
    assert s.match_phase == int(MatchPhase.PLAY)
    assert legal_actions(s)


def test_illegal_setup_square_raises() -> None:
    s = initial_state()
    s = begin_setup(s, coin_flip_winner=0, winner_chooses_first_setup=True)
    bad = SetupPlacement(actor_slot=0, destination=(3, 3))
    with pytest.raises(IllegalActionError):
        setup_step(s, bad)

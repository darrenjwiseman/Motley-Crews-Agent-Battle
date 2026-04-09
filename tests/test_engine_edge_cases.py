"""Edge cases: pass play, encoding bridge, random termination."""

from __future__ import annotations

import random

import numpy as np
import pytest

from motley_crews_env.constants import BOARD_SIZE, TERRAIN_OPEN
from motley_crews_env.encoding import encode_observation
from motley_crews_env.engine import initial_play_state, legal_actions, step, to_structured_observation
from motley_crews_env.types import TeamId, TurnAction


def test_pass_pass_is_legal() -> None:
    s = initial_play_state()
    ta = TurnAction(move=None, action=None)
    s2 = step(s, ta).state
    assert s2.current_player == int(TeamId.PLAYER_B)


def test_observation_encode_roundtrip_shapes() -> None:
    s = initial_play_state()
    obs = to_structured_observation(s)
    enc = encode_observation(obs)
    assert enc["spatial"].shape == (BOARD_SIZE, BOARD_SIZE, enc["spatial"].shape[2])
    assert enc["global"].shape[0] >= 8


def test_random_play_terminates() -> None:
    rng = random.Random(42)
    s = initial_play_state()
    for _ in range(2000):
        acts = legal_actions(s)
        assert acts
        s = step(s, rng.choice(acts)).state
        if s.done:
            assert s.winner in (0, 1)
            return
    pytest.fail("game did not end")

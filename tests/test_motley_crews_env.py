"""Tests for observation shapes and action serialization (step 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from motley_crews_env.constants import BOARD_SIZE
from motley_crews_env.encoding import (
    GLOBAL_DIM,
    SPATIAL_CHANNELS,
    SPEC,
    encode_observation,
    structured_observation_to_tensor,
    tensor_shapes,
)
from motley_crews_env.serialization import turn_action_from_tuple, turn_action_to_tuple
from motley_crews_env.types import (
    ActionBasicAttack,
    ActionSpecial,
    MoveIntent,
    SpecialId,
    StructuredObservation,
    TeamId,
    TurnAction,
)


def _empty_board_obs() -> StructuredObservation:
    h = w = BOARD_SIZE
    return StructuredObservation(
        terrain=np.zeros((h, w), dtype=np.int8),
        occupancy=np.zeros((h, w), dtype=np.float32),
        team=np.full((h, w), -1, dtype=np.int8),
        unit_class=np.full((h, w), -1, dtype=np.int8),
        hp_normalized=np.zeros((h, w), dtype=np.float32),
        containment=np.zeros((h, w), dtype=np.float32),
        reserved_status_a=np.zeros((h, w), dtype=np.float32),
        reserved_status_b=np.zeros((h, w), dtype=np.float32),
        score_player_a=0.0,
        score_player_b=0.0,
        turn_index=0.0,
        current_player=int(TeamId.PLAYER_A),
        phase=0,
        points_to_win=4.0,
    )


def test_tensor_shapes_match_spec() -> None:
    shapes = tensor_shapes()
    assert shapes["spatial"] == (BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS)
    assert shapes["global"] == (GLOBAL_DIM,)
    assert SPEC.spatial_shape == shapes["spatial"]
    assert SPEC.global_dim == GLOBAL_DIM


def test_structured_observation_dtype_and_range() -> None:
    obs = _empty_board_obs()
    sp = structured_observation_to_tensor(obs)
    assert sp.shape == (BOARD_SIZE, BOARD_SIZE, SPATIAL_CHANNELS)
    assert sp.dtype == np.float32
    assert np.all(sp >= 0.0) and np.all(sp <= 1.0)


def test_encode_observation_keys() -> None:
    enc = encode_observation(_empty_board_obs())
    assert set(enc.keys()) == {"spatial", "global"}
    assert enc["global"].shape == (GLOBAL_DIM,)
    assert enc["global"].dtype == np.float32


def test_turn_action_round_trip_pass_pass() -> None:
    ta = TurnAction(move=None, action=None)
    assert turn_action_from_tuple(turn_action_to_tuple(ta)) == ta


def test_turn_action_round_trip_move_and_basic() -> None:
    ta = TurnAction(
        move=MoveIntent(actor_slot=2, destination=(3, 4)),
        action=ActionBasicAttack(actor_slot=1, target_square=(5, 5)),
    )
    assert turn_action_from_tuple(turn_action_to_tuple(ta)) == ta


def test_turn_action_round_trip_special_curse() -> None:
    ta = TurnAction(
        move=MoveIntent(actor_slot=0, destination=(0, 1)),
        action=ActionSpecial(
            actor_slot=3,
            special_id=SpecialId.CURSE,
            target_square=(4, 4),
            curse_x=2,
        ),
    )
    assert turn_action_from_tuple(turn_action_to_tuple(ta)) == ta


def test_turn_action_round_trip_animate_dead_no_square() -> None:
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=2,
            special_id=SpecialId.ANIMATE_DEAD,
            target_square=None,
            animate_dead_crew_slot=1,
        ),
    )
    assert turn_action_from_tuple(turn_action_to_tuple(ta)) == ta


def test_turn_action_invalid_tuple_raises() -> None:
    with pytest.raises(ValueError):
        turn_action_from_tuple((None, None, None))

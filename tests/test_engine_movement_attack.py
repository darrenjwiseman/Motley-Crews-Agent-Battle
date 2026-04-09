"""Per-class movement and basic attack patterns."""

from __future__ import annotations

import numpy as np

from motley_crews_env.constants import TERRAIN_BLOCKED, TERRAIN_OPEN
from motley_crews_env.engine import initial_play_state, legal_actions, scenario_from_placements, step
from motley_crews_env.state import unit_at
from motley_crews_env.types import ClassId, TeamId, TurnAction
from motley_crews_env.types import ActionBasicAttack


def test_opening_board_has_five_figures_per_side() -> None:
    s = initial_play_state()
    from motley_crews_env.constants import FIGURES_PER_SIDE, TEAM_PLAYER_A, TEAM_PLAYER_B

    assert sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, TEAM_PLAYER_A, sl)) == 5
    assert sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, TEAM_PLAYER_B, sl)) == 5


def test_knight_melee_adjacent_only() -> None:
    # A knight slot 0 vs B barbarian isolated
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 0, 3, 3, int(ClassId.KNIGHT)),
            (1, 1, 3, 4, int(ClassId.BARBARIAN)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(
        move=None,
        action=ActionBasicAttack(actor_slot=0, target_square=(3, 4)),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 1).hp == 6 - 3


def test_barbarian_fear_of_occult_extra_from_mage() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 0, 2, 2, int(ClassId.WHITE_MAGE)),
            (1, 1, 2, 4, int(ClassId.BARBARIAN)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(
        move=None,
        action=ActionBasicAttack(actor_slot=0, target_square=(2, 4)),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 1).hp == 6 - 2  # 1 + 1 fear


def test_terrain_blocks_movement() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    terrain[4, 4] = TERRAIN_BLOCKED
    s = scenario_from_placements(
        terrain=terrain,
        placements=[(0, 0, 3, 3, int(ClassId.KNIGHT))],
        current_player=int(TeamId.PLAYER_A),
    )
    # Knight cannot move through blocked at 4,4
    dest = (4, 4)
    moves = [m for m in legal_actions(s) if m.move and m.move.destination == dest]
    assert moves == []


def test_arbalist_diagonal_move_in_legal_actions() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[(0, 4, 4, 4, int(ClassId.ARBALIST))],
        current_player=int(TeamId.PLAYER_A),
    )
    acts = legal_actions(s)
    diag = (3, 3)
    assert any(a.move and a.move.destination == diag for a in acts)


def test_white_mage_ranged_orthogonal_line() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 2, 2, 2, int(ClassId.WHITE_MAGE)),
            (1, 0, 2, 4, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(move=None, action=ActionBasicAttack(actor_slot=2, target_square=(2, 4)))
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 0).hp == 7 - 1

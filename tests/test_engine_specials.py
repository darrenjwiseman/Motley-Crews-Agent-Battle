"""Spell and special-action behavior."""

from __future__ import annotations

import numpy as np

from motley_crews_env.constants import TERRAIN_OPEN
from motley_crews_env.engine import legal_actions, scenario_from_placements, step
from motley_crews_env.state import slot_unit, unit_at
from motley_crews_env.types import ActionSpecial, ClassId, SpecialId, TeamId, TurnAction


def test_curse_simultaneous_lethal() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 3, 2, 2, int(ClassId.BLACK_MAGE)),
            (1, 0, 2, 4, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    bm = unit_at(s, 0, 3)
    assert bm is not None
    k = unit_at(s, 1, 0)
    assert k is not None
    k.hp = 4
    x = bm.hp  # 4 self, 5 to target — both lethal
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=3,
            special_id=SpecialId.CURSE,
            target_square=(2, 4),
            curse_x=x,
        ),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 0, 3) is None
    assert unit_at(s2, 1, 0) is None


def test_convert_changes_controller() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 2, 2, 2, int(ClassId.WHITE_MAGE)),
            (1, 0, 2, 4, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    k = unit_at(s, 1, 0)
    assert k is not None
    k.hp = 2
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=2,
            special_id=SpecialId.CONVERT,
            target_square=(2, 4),
        ),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 0).controller == int(TeamId.PLAYER_A)


def test_heal_increases_hp() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 2, 2, 2, int(ClassId.WHITE_MAGE)),
            (0, 0, 2, 4, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    unit_at(s, 0, 0).hp = 1
    ta = TurnAction(
        move=None,
        action=ActionSpecial(actor_slot=2, special_id=SpecialId.HEAL, target_square=(2, 4)),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 0, 0).hp == min(7, 1 + 3)


def test_conjure_containment_blocks_movement_next_turn() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 2, 4, 2, int(ClassId.WHITE_MAGE)),
            (1, 0, 4, 4, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=2,
            special_id=SpecialId.CONJURE_CONTAINMENT,
            target_square=(4, 4),
        ),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 0).containment_ticks == 2
    # B's turn: knight should have no legal moves
    acts = legal_actions(s2)
    assert not any(a.move for a in acts if a.move and a.move.actor_slot == 0)


def test_magic_bomb_area_damage() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 3, 3, 3, int(ClassId.BLACK_MAGE)),
            (1, 0, 4, 3, int(ClassId.KNIGHT)),
            (1, 1, 2, 3, int(ClassId.BARBARIAN)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=3,
            special_id=SpecialId.MAGIC_BOMB,
            target_square=(3, 3),
        ),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 1, 0).hp == 7 - 2
    # Barbarian: 2 damage + 1 Fear from Black Mage bomb
    assert unit_at(s2, 1, 1).hp == 6 - 3


def test_charge_passage_damage_and_landing() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 0, 3, 3, int(ClassId.KNIGHT)),
            (0, 1, 3, 5, int(ClassId.KNIGHT)),
            (1, 0, 3, 4, int(ClassId.BARBARIAN)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=0,
            special_id=SpecialId.CHARGE,
            target_square=(3, 6),
        ),
    )
    s2 = step(s, ta).state
    assert unit_at(s2, 0, 0).row == 3 and unit_at(s2, 0, 0).col == 6
    assert unit_at(s2, 0, 1).hp == 7 - 2
    assert unit_at(s2, 1, 0).hp == 6 - 2  # passage through (3,4)


def test_animate_dead_revives_and_score_refund() -> None:
    terrain = np.full((8, 8), TERRAIN_OPEN, dtype=np.int8)
    s = scenario_from_placements(
        terrain=terrain,
        placements=[
            (0, 3, 4, 4, int(ClassId.BLACK_MAGE)),
            (0, 0, 5, 5, int(ClassId.KNIGHT)),
        ],
        current_player=int(TeamId.PLAYER_A),
    )
    k = unit_at(s, 0, 0)
    s.board[k.row, k.col] = -1
    k.alive = False
    k.death_point_recipient = 1  # team B received kill credit when knight died
    s.score = (0, 1)

    ta = TurnAction(
        move=None,
        action=ActionSpecial(
            actor_slot=3,
            special_id=SpecialId.ANIMATE_DEAD,
            target_square=None,
            animate_dead_crew_slot=0,
        ),
    )
    s2 = step(s, ta).state
    assert s2.pending_resurrect == (0, 0)
    k2 = slot_unit(s2, 0, 0)
    assert k2 is not None and k2.alive and k2.row < 0
    assert k2.hp == 2
    assert s2.score[1] == 0  # refunded B's point

    place = TurnAction(move=None, action=None, resurrect_place=(6, 0))
    s3 = step(s2, place).state
    assert s3.pending_resurrect is None
    u = unit_at(s3, 0, 0)
    assert u is not None
    assert u.row == 6 and u.col == 0
    assert u.hp == 2

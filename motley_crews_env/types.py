"""
Structured observation and turn actions for Motley Crews.

Move and action may use different friendly figures (rules_spec turn.structure.different_figures_allowed).
Legality is enforced by the engine in step 4.3; types here are the contract only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ClassId",
    "Phase",
    "MatchPhase",
    "SpecialId",
    "TeamId",
    "TerrainCode",
    "MoveIntent",
    "ActionBasicAttack",
    "ActionSpecial",
    "ActionIntent",
    "TurnAction",
    "StructuredObservation",
    "SetupPlacement",
]


class TeamId(IntEnum):
    PLAYER_A = 0
    PLAYER_B = 1


class ClassId(IntEnum):
    KNIGHT = 0
    BARBARIAN = 1
    WHITE_MAGE = 2
    BLACK_MAGE = 3
    ARBALIST = 4


class TerrainCode(IntEnum):
    """Indices match constants.TERRAIN_* and terrain channels order."""

    OPEN = 0
    BLOCKED = 1
    WATER = 2


class Phase(IntEnum):
    MOVE = 0
    ACTION = 1


class MatchPhase(IntEnum):
    """Pre-play: coin flip + alternating placement from staging; then PLAY."""

    PENDING_SETUP = 0  # awaiting begin_setup(...)
    SETUP = 1
    PLAY = 2


class SpecialId(IntEnum):
    """Stable indices match constants.SPECIAL_IDS order."""

    CHARGE = 0
    CONVERT = 1
    HEAL = 2
    CONJURE_CONTAINMENT = 3
    CURSE = 4
    MAGIC_BOMB = 5
    ANIMATE_DEAD = 6
    LONG_EYE = 7


Coord = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class SetupPlacement:
    """Place one figure from staging into an empty square in that side’s two home rows."""

    actor_slot: int
    destination: Coord


@dataclass(frozen=True, slots=True)
class MoveIntent:
    """Move one friendly figure from figure slot to an empty legal square (4.3 validates)."""

    actor_slot: int
    destination: Coord


@dataclass(frozen=True, slots=True)
class ActionBasicAttack:
    actor_slot: int
    target_square: Coord


@dataclass(frozen=True, slots=True)
class ActionSpecial:
    """
    Special ability. Fields depend on special_id (engine validates combinations).

    - target_square: primary target cell (charge landing, spell targets, bomb center, Long Eye hit cell, etc.).
    - curse_x: Black Mage Curse only — damage to self is X, to target X+1; validated by engine (positive int).
    - animate_dead_crew_slot: which dead own-crew figure (slot index 0..4); placement in start zone is engine-side.
    """

    actor_slot: int
    special_id: SpecialId
    target_square: Optional[Coord] = None
    curse_x: Optional[int] = None
    animate_dead_crew_slot: Optional[int] = None


ActionIntent = Union[ActionBasicAttack, ActionSpecial]


@dataclass(frozen=True, slots=True)
class TurnAction:
    """
    Full turn: optional move sub-step and optional action sub-step.

    None means pass for that sub-step. Knight Charge is ActionSpecial(CHARGE, ...), not MoveIntent.
    """

    move: Optional[MoveIntent]
    action: Optional[ActionIntent]


@dataclass
class StructuredObservation:
    """
    Board tensors use matrix indices (row, col), origin top-left (rules_spec board.coordinate_system).

    Empty cells: occupancy 0; team/unit_class/hp/containment ignored (encode as -1 or 0 per field below).
    """

    terrain: NDArray[np.integer]  # (H, W) TerrainCode values 0..2
    occupancy: NDArray[np.floating]  # (H, W) 0/1
    team: NDArray[np.integer]  # (H, W) -1 empty, else TeamId
    unit_class: NDArray[np.integer]  # (H, W) -1 empty, else ClassId 0..4
    hp_normalized: NDArray[np.floating]  # (H, W) in [0, 1], 0 if empty
    containment: NDArray[np.floating]  # (H, W) 0/1 presence of containment status
    reserved_status_a: NDArray[np.floating]  # (H, W) reserved for future statuses
    reserved_status_b: NDArray[np.floating]  # (H, W) reserved for future statuses
    score_player_a: float
    score_player_b: float
    turn_index: float
    current_player: int  # TeamId value
    phase: int  # Phase value
    points_to_win: float
    repetition_placeholder: float = 0.0
    turn_limit_placeholder: float = 1.0

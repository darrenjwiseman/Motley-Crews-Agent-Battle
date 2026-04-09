"""
Mutable game state for the Motley Crews engine.

Match flow: figures begin in staging; after the coin flip and first/second setup choice,
players alternate placing into their two home rows (see ``constants.DEPLOY_ROWS_*``).
Class order per slot i matches ``constants.CLASS_IDS[i]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from motley_crews_env.constants import (
    BOARD_SIZE,
    DEFAULT_POINTS_TO_WIN,
    DEFAULT_START_COLS,
    FIGURES_PER_SIDE,
    PLAYER_A_HOME_ROW,
    PLAYER_B_HOME_ROW,
    STAGING_COORD,
    STARTING_HP_BY_CLASS,
    TEAM_PLAYER_A,
    TEAM_PLAYER_B,
    TERRAIN_OPEN,
)
from motley_crews_env.types import ClassId, MatchPhase


class IllegalActionError(ValueError):
    """Raised when ``step`` receives a turn that is not legal in the current state."""


@dataclass
class UnitState:
    """One crew slot (original team + slot index)."""

    class_id: int
    hp: int
    max_hp: int
    row: int
    col: int
    controller: int  # current controlling team
    alive: bool = True
    containment_ticks: int = 0  # remaining enemy-turn ticks; 0 = not contained
    containment_clock_team: int = 0  # team whose completed turn decrements ticks (enemy of victim)
    containment_skip_next_tick: bool = False  # skip first clock tick (cast turn just ended)
    used_conjure_containment: bool = False  # WM once per game
    used_magic_bomb: bool = False  # BM once per game
    moved_this_turn: bool = False  # set during a turn for careful aim / charge
    death_point_recipient: Optional[int] = None  # team that received score for this unit's last death


@dataclass
class GameState:
    """
    Full match state. ``board[r,c]`` holds linear index ``team * FIGURES_PER_SIDE + slot`` or -1.
    """

    terrain: NDArray[np.int8]
    board: NDArray[np.int8]  # -1 empty, else team*5+slot
    units: list[Optional[UnitState]]  # index: team*FIGURES_PER_SIDE + slot
    score: tuple[int, int]
    turn_index: int
    current_player: int  # TEAM_PLAYER_A or TEAM_PLAYER_B
    points_to_win: int
    done: bool = False
    winner: Optional[int] = None
    # Per-turn: any friendly figure has used a move (blocks Knight Charge)
    any_friendly_moved_this_turn: bool = False
    match_phase: int = int(MatchPhase.PLAY)
    setup_current_player: int = TEAM_PLAYER_A
    first_player: int = TEAM_PLAYER_A
    coin_flip_winner: Optional[int] = None
    # (team, slot) — revived unit must be placed in deploy rows before the turn advances
    pending_resurrect: Optional[Tuple[int, int]] = None

    def clone(self) -> GameState:
        u = [None if x is None else _copy_unit(x) for x in self.units]
        return GameState(
            terrain=self.terrain.copy(),
            board=self.board.copy(),
            units=u,
            score=(self.score[0], self.score[1]),
            turn_index=self.turn_index,
            current_player=self.current_player,
            points_to_win=self.points_to_win,
            done=self.done,
            winner=self.winner,
            any_friendly_moved_this_turn=self.any_friendly_moved_this_turn,
            match_phase=self.match_phase,
            setup_current_player=self.setup_current_player,
            first_player=self.first_player,
            coin_flip_winner=self.coin_flip_winner,
            pending_resurrect=self.pending_resurrect,
        )


def _copy_unit(u: UnitState) -> UnitState:
    return UnitState(
        class_id=u.class_id,
        hp=u.hp,
        max_hp=u.max_hp,
        row=u.row,
        col=u.col,
        controller=u.controller,
        alive=u.alive,
        containment_ticks=u.containment_ticks,
        containment_clock_team=u.containment_clock_team,
        used_conjure_containment=u.used_conjure_containment,
        used_magic_bomb=u.used_magic_bomb,
        moved_this_turn=u.moved_this_turn,
        death_point_recipient=u.death_point_recipient,
        containment_skip_next_tick=u.containment_skip_next_tick,
    )


def _idx(team: int, slot: int) -> int:
    return team * FIGURES_PER_SIDE + slot


def slot_unit(state: GameState, team: int, slot: int) -> Optional[UnitState]:
    """Crew slot including dead figures (for Animate Dead)."""
    return state.units[_idx(team, slot)]


def unit_at(state: GameState, team: int, slot: int) -> Optional[UnitState]:
    u = slot_unit(state, team, slot)
    if u is None or not u.alive:
        return None
    if u.row < 0:
        return None
    return u


def linear_to_team_slot(lin: int) -> tuple[int, int]:
    return lin // FIGURES_PER_SIDE, lin % FIGURES_PER_SIDE


def initial_state(
    *,
    points_to_win: int = DEFAULT_POINTS_TO_WIN,
    terrain: Optional[NDArray[np.int8]] = None,
) -> GameState:
    """
    Match start: empty board, all figures in staging (off-board).

    Call ``begin_setup`` after the coin flip so the winner can choose first vs second
    setup (which pairs with first vs second turn). Then alternate ``setup_step`` until
    play begins.
    """
    h, w = BOARD_SIZE, BOARD_SIZE
    if terrain is None:
        terrain = np.full((h, w), TERRAIN_OPEN, dtype=np.int8)
    board = np.full((h, w), -1, dtype=np.int8)
    units: list[Optional[UnitState]] = [None] * (2 * FIGURES_PER_SIDE)

    for slot in range(FIGURES_PER_SIDE):
        cid = slot  # ClassId matches slot index 0..4
        hp = STARTING_HP_BY_CLASS[cid]

        ua = UnitState(
            class_id=cid,
            hp=hp,
            max_hp=hp,
            row=STAGING_COORD,
            col=STAGING_COORD,
            controller=TEAM_PLAYER_A,
        )
        units[_idx(TEAM_PLAYER_A, slot)] = ua

        ub = UnitState(
            class_id=cid,
            hp=hp,
            max_hp=hp,
            row=STAGING_COORD,
            col=STAGING_COORD,
            controller=TEAM_PLAYER_B,
        )
        units[_idx(TEAM_PLAYER_B, slot)] = ub

    return GameState(
        terrain=terrain,
        board=board,
        units=units,
        score=(0, 0),
        turn_index=0,
        current_player=TEAM_PLAYER_A,
        points_to_win=points_to_win,
        match_phase=int(MatchPhase.PENDING_SETUP),
        setup_current_player=TEAM_PLAYER_A,
        first_player=TEAM_PLAYER_A,
        coin_flip_winner=None,
    )


def opponent(team: int) -> int:
    return 1 - team


def count_living_controlled_by(state: GameState, controller: int) -> int:
    n = 0
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for s in range(FIGURES_PER_SIDE):
            u = unit_at(state, t, s)
            if u is not None and u.controller == controller:
                n += 1
    return n


def class_move_value(class_id: int) -> int:
    return (4, 3, 2, 2, 2)[class_id]


def class_reach_basic(class_id: int) -> int:
    return (1, 1, 2, 2, 3)[class_id]


def class_basic_damage(class_id: int, *, careful_aim_not_moved: bool) -> int:
    if class_id == int(ClassId.ARBALIST) and careful_aim_not_moved:
        return 3
    return (3, 4, 1, 1, 2)[class_id]


def allows_diagonal_move(class_id: int) -> bool:
    return class_id == int(ClassId.ARBALIST)


def allows_diagonal_basic(class_id: int) -> bool:
    return class_id == int(ClassId.ARBALIST)


@dataclass(frozen=True, slots=True)
class DamageEvent:
    """One damage application at a board cell (for UI / logging)."""

    row: int
    col: int
    amount: int
    target_team: int
    target_slot: int


@dataclass
class StepResult:
    state: GameState
    done: bool
    winner: Optional[int] = None
    damage_events: Tuple[DamageEvent, ...] = ()

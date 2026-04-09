"""
Rules engine: ``legal_actions``, ``step``, win detection.

Implements core rules from ``rules/rules_spec.yaml`` (see module docstrings in ``state.py``).
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from motley_crews_env.constants import (
    BOARD_SIZE,
    DEFAULT_POINTS_TO_WIN,
    DEFAULT_START_COLS,
    DEPLOY_ROWS_PLAYER_A,
    DEPLOY_ROWS_PLAYER_B,
    FIGURES_PER_SIDE,
    MAX_CURSE_X,
    PLAYER_A_HOME_ROW,
    PLAYER_B_HOME_ROW,
    TEAM_PLAYER_A,
    TEAM_PLAYER_B,
    TERRAIN_BLOCKED,
    TERRAIN_OPEN,
)
from motley_crews_env.state import (
    GameState,
    IllegalActionError,
    StepResult,
    UnitState,
    allows_diagonal_move,
    class_basic_damage,
    class_move_value,
    class_reach_basic,
    count_living_controlled_by,
    initial_state,
    linear_to_team_slot,
    opponent,
    slot_unit,
    unit_at,
)
from motley_crews_env.state import _idx as lin_idx
from motley_crews_env.types import (
    ActionBasicAttack,
    ActionIntent,
    ActionSpecial,
    ClassId,
    MatchPhase,
    MoveIntent,
    Phase,
    SetupPlacement,
    SpecialId,
    StructuredObservation,
    TeamId,
    TerrainCode,
    TurnAction,
)

# --- Terrain -----------------------------------------------------------

def _blocks_movement(t: int) -> bool:
    return t != TERRAIN_OPEN


def _blocks_basic_los(t: int) -> bool:
    """Orthogonal/diagonal ray for basic attacks: blocked terrain stops; water does not."""
    return t == TERRAIN_BLOCKED


def _blocks_long_eye(t: int) -> bool:
    """Long Eye: blocked by impassable terrain; water optional — treat as open for LOS."""
    return t == TERRAIN_BLOCKED


def _blocks_charge_path(t: int) -> bool:
    return t != TERRAIN_OPEN


# --- Geometry ----------------------------------------------------------

ORTH_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAG_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
ALL8_DIRS = ORTH_DIRS + DIAG_DIRS


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _iter_reach_squares_orthogonal(r: int, c: int, reach: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ORTH_DIRS:
        for k in range(1, reach + 1):
            rr, cc = r + dr * k, c + dc * k
            if not _in_bounds(rr, cc):
                break
            out.append((rr, cc))
    return out


def _iter_reach_squares_arbalist(r: int, c: int, reach: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ALL8_DIRS:
        for k in range(1, reach + 1):
            rr, cc = r + dr * k, c + dc * k
            if not _in_bounds(rr, cc):
                break
            out.append((rr, cc))
    return out


def _manhattan_dist(r1: int, c1: int, r2: int, c2: int) -> int:
    return abs(r1 - r2) + abs(c1 - c2)


def _chebyshev_dist(r1: int, c1: int, r2: int, c2: int) -> int:
    return max(abs(r1 - r2), abs(c1 - c2))


# --- Reach / LOS -------------------------------------------------------

def _orthogonal_los_clear(state: GameState, r0: int, c0: int, r1: int, c1: int) -> bool:
    """Straight orthogonal line, figures and blocked terrain block (water OK)."""
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        for cc in range(c0 + step, c1, step):
            if not _in_bounds(r0, cc):
                return False
            if _blocks_basic_los(int(state.terrain[r0, cc])):
                return False
            if state.board[r0, cc] >= 0:
                return False
        return True
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        for rr in range(r0 + step, r1, step):
            if not _in_bounds(rr, c0):
                return False
            if _blocks_basic_los(int(state.terrain[rr, c0])):
                return False
            if state.board[rr, c0] >= 0:
                return False
        return True
    return False


def _in_orthogonal_reach_wm_bm(state: GameState, ar: int, ac: int, tr: int, tc: int, reach: int) -> bool:
    if _manhattan_dist(ar, ac, tr, tc) > reach or _manhattan_dist(ar, ac, tr, tc) == 0:
        return False
    if ar != tr and ac != tc:
        return False
    return _orthogonal_los_clear(state, ar, ac, tr, tc)


def _arbalist_basic_los(state: GameState, ar: int, ac: int, tr: int, tc: int, reach: int) -> bool:
    dist = _chebyshev_dist(ar, ac, tr, tc)
    if dist == 0 or dist > reach:
        return False
    dr = (tr - ar) // dist if tr != ar else 0
    dc = (tc - ac) // dist if tc != ac else 0
    if abs(dr) > 1 or abs(dc) > 1:
        return False
    rr, cc = ar, ac
    for _ in range(dist - 1):
        rr += dr
        cc += dc
        if _blocks_basic_los(int(state.terrain[rr, cc])):
            return False
        if state.board[rr, cc] >= 0:
            return False
    return True


# --- Movement enumeration ----------------------------------------------

def _legal_destinations_for_unit(state: GameState, team: int, slot: int) -> list[tuple[int, int]]:
    u = unit_at(state, team, slot)
    if u is None or u.controller != state.current_player:
        return []
    if u.containment_ticks > 0:
        return []
    mv = class_move_value(u.class_id)
    start = (u.row, u.col)
    dirs = list(ORTH_DIRS)
    if allows_diagonal_move(u.class_id):
        dirs = list(ALL8_DIRS)

    # BFS
    from collections import deque

    seen = {start: 0}
    dq: deque[tuple[int, int, int]] = deque([(start[0], start[1], 0)])
    out: list[tuple[int, int]] = []
    while dq:
        r, c, d = dq.popleft()
        if d >= mv:
            continue
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc):
                continue
            nd = d + 1
            if nd > mv:
                continue
            if _blocks_movement(int(state.terrain[nr, nc])):
                continue
            if state.board[nr, nc] >= 0:
                continue
            if (nr, nc) not in seen or seen[nr, nc] > nd:
                seen[(nr, nc)] = nd
                dq.append((nr, nc, nd))
                out.append((nr, nc))
    return list(dict.fromkeys(out))  # unique preserve order


def _apply_move_only(s: GameState, team: int, slot: int, dest: tuple[int, int]) -> None:
    u = unit_at(s, team, slot)
    if u is None:
        raise IllegalActionError("no unit")
    r, c = u.row, u.col
    dr, dc = dest
    s.board[r, c] = -1
    u.row, u.col = dr, dc
    s.board[dr, dc] = lin_idx(team, slot)
    u.moved_this_turn = True
    s.any_friendly_moved_this_turn = True


# --- Damage / death ----------------------------------------------------

def _is_mage_class(class_id: int) -> bool:
    return class_id in (int(ClassId.WHITE_MAGE), int(ClassId.BLACK_MAGE))


def _apply_damage_to_unit(
    s: GameState,
    target_team: int,
    target_slot: int,
    base: int,
    *,
    source_class_for_fear: Optional[int],
) -> None:
    u = unit_at(s, target_team, target_slot)
    if u is None:
        return
    dmg = base
    if source_class_for_fear is not None and u.class_id == int(ClassId.BARBARIAN):
        if _is_mage_class(source_class_for_fear):
            dmg += 1
    u.hp -= dmg
    if u.hp <= 0:
        _kill_unit(s, target_team, target_slot)


def _kill_unit(s: GameState, team: int, slot: int) -> None:
    u = s.units[lin_idx(team, slot)]
    if u is None or not u.alive:
        return
    r, c = u.row, u.col
    s.board[r, c] = -1
    u.alive = False
    recipient = opponent(u.controller)
    sc = list(s.score)
    sc[recipient] += 1
    s.score = (sc[0], sc[1])
    u.death_point_recipient = recipient


def _check_win(s: GameState) -> None:
    pa = count_living_controlled_by(s, TEAM_PLAYER_A)
    pb = count_living_controlled_by(s, TEAM_PLAYER_B)
    sa, sb = s.score
    thr = s.points_to_win
    if sa >= thr:
        s.done, s.winner = True, TEAM_PLAYER_A
    elif sb >= thr:
        s.done, s.winner = True, TEAM_PLAYER_B
    elif pb == 0:
        s.done, s.winner = True, TEAM_PLAYER_A
    elif pa == 0:
        s.done, s.winner = True, TEAM_PLAYER_B


def _end_turn_tick_containment(s: GameState, finished_team: int) -> None:
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, t, sl)
            if u is None or u.containment_ticks <= 0:
                continue
            if u.containment_clock_team != finished_team:
                continue
            if u.containment_skip_next_tick:
                u.containment_skip_next_tick = False
                continue
            u.containment_ticks -= 1


def _reset_turn_flags(s: GameState) -> None:
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, t, sl)
            if u is not None:
                u.moved_this_turn = False
    s.any_friendly_moved_this_turn = False


def _advance_turn(s: GameState) -> None:
    finished = s.current_player
    _end_turn_tick_containment(s, finished)
    _reset_turn_flags(s)
    s.current_player = opponent(finished)
    s.turn_index += 1


# --- Action legality helpers -------------------------------------------

def _actor_unit(state: GameState, team: int, slot: int) -> UnitState:
    u = unit_at(state, team, slot)
    if u is None or u.controller != state.current_player:
        raise IllegalActionError("bad actor")
    return u


def _legal_basic_attacks_after_move(s: GameState) -> list[ActionIntent]:
    out: list[ActionIntent] = []
    pl = s.current_player
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, t, sl)
            if u is None or u.controller != pl:
                continue
            if u.containment_ticks > 0:
                continue
            cid = u.class_id
            reach = class_reach_basic(cid)

            if cid in (int(ClassId.KNIGHT), int(ClassId.BARBARIAN)):
                for dr, dc in ORTH_DIRS:
                    tr, tc = u.row + dr, u.col + dc
                    if not _in_bounds(tr, tc):
                        continue
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller != pl:
                        out.append(ActionBasicAttack(actor_slot=sl, target_square=(tr, tc)))

            elif cid in (int(ClassId.WHITE_MAGE), int(ClassId.BLACK_MAGE)):
                for tr, tc in _iter_reach_squares_orthogonal(u.row, u.col, reach):
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, reach):
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller != pl:
                        out.append(ActionBasicAttack(actor_slot=sl, target_square=(tr, tc)))

            elif cid == int(ClassId.ARBALIST):
                targets = _iter_reach_squares_arbalist(u.row, u.col, reach)
                for tr, tc in targets:
                    if not _arbalist_basic_los(s, u.row, u.col, tr, tc, reach):
                        continue
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller != pl:
                        out.append(ActionBasicAttack(actor_slot=sl, target_square=(tr, tc)))
    return out


def _legal_long_eye(s: GameState) -> list[ActionIntent]:
    out: list[ActionIntent] = []
    pl = s.current_player
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, t, sl)
            if u is None or u.controller != pl:
                continue
            if u.class_id != int(ClassId.ARBALIST):
                continue
            if u.containment_ticks > 0:
                continue
            for dr, dc in ALL8_DIRS:
                cr, cc = u.row, u.col
                while True:
                    cr += dr
                    cc += dc
                    if not _in_bounds(cr, cc):
                        break
                    tt = int(s.terrain[cr, cc])
                    if _blocks_long_eye(tt):
                        break
                    tid = s.board[cr, cc]
                    if tid >= 0:
                        oteam, osl = linear_to_team_slot(tid)
                        ou = unit_at(s, oteam, osl)
                        if ou is not None and ou.controller != pl:
                            out.append(
                                ActionSpecial(
                                    actor_slot=sl,
                                    special_id=SpecialId.LONG_EYE,
                                    target_square=(cr, cc),
                                )
                            )
                        break
    return out


def _legal_specials(s: GameState) -> list[ActionIntent]:
    out: list[ActionIntent] = []
    pl = s.current_player
    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(s, t, sl)
            if u is None or u.controller != pl:
                continue
            cid = u.class_id

            if cid == int(ClassId.KNIGHT):
                if s.any_friendly_moved_this_turn:
                    continue
                for dr, dc in ORTH_DIRS:
                    for dist in range(1, 5):
                        tr, tc = u.row + dr * dist, u.col + dc * dist
                        if not _in_bounds(tr, tc):
                            break
                        if _blocks_charge_path(int(s.terrain[tr, tc])):
                            break
                        if s.board[tr, tc] >= 0:
                            continue
                        out.append(
                            ActionSpecial(
                                actor_slot=sl,
                                special_id=SpecialId.CHARGE,
                                target_square=(tr, tc),
                            )
                        )

            elif cid == int(ClassId.WHITE_MAGE):
                reach = 2
                if not u.used_conjure_containment:
                    for tr, tc in _iter_reach_squares_orthogonal(u.row, u.col, reach):
                        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, reach):
                            continue
                        tid = s.board[tr, tc]
                        if tid < 0:
                            continue
                        oteam, osl = linear_to_team_slot(tid)
                        ou = unit_at(s, oteam, osl)
                        if ou is not None and ou.controller != pl:
                            out.append(
                                ActionSpecial(
                                    actor_slot=sl,
                                    special_id=SpecialId.CONJURE_CONTAINMENT,
                                    target_square=(tr, tc),
                                )
                            )
                for tr, tc in _iter_reach_squares_orthogonal(u.row, u.col, reach):
                    if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, reach):
                        continue
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller != pl and ou.hp <= 2:
                        out.append(
                            ActionSpecial(actor_slot=sl, special_id=SpecialId.CONVERT, target_square=(tr, tc))
                        )
                for tr, tc in _iter_reach_squares_orthogonal(u.row, u.col, reach):
                    if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, reach):
                        continue
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller == pl and (oteam != t or osl != sl):
                        out.append(
                            ActionSpecial(actor_slot=sl, special_id=SpecialId.HEAL, target_square=(tr, tc))
                        )

            elif cid == int(ClassId.BLACK_MAGE):
                reach = 2
                for tr, tc in _iter_reach_squares_orthogonal(u.row, u.col, reach):
                    if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, reach):
                        continue
                    tid = s.board[tr, tc]
                    if tid < 0:
                        continue
                    oteam, osl = linear_to_team_slot(tid)
                    ou = unit_at(s, oteam, osl)
                    if ou is not None and ou.controller != pl:
                        for x in range(1, MAX_CURSE_X + 1):
                            out.append(
                                ActionSpecial(
                                    actor_slot=sl,
                                    special_id=SpecialId.CURSE,
                                    target_square=(tr, tc),
                                    curse_x=x,
                                )
                            )
                if not u.used_magic_bomb:
                    for tr in range(BOARD_SIZE):
                        for tc in range(BOARD_SIZE):
                            if _manhattan_dist(u.row, u.col, tr, tc) > reach:
                                continue
                            if _blocks_movement(int(s.terrain[tr, tc])):
                                continue
                            out.append(
                                ActionSpecial(
                                    actor_slot=sl,
                                    special_id=SpecialId.MAGIC_BOMB,
                                    target_square=(tr, tc),
                                )
                            )
                for osl in range(FIGURES_PER_SIDE):
                    du = slot_unit(s, t, osl)
                    if du is not None and (not du.alive):
                        out.append(
                            ActionSpecial(
                                actor_slot=sl,
                                special_id=SpecialId.ANIMATE_DEAD,
                                target_square=None,
                                animate_dead_crew_slot=osl,
                            )
                        )

    out.extend(_legal_long_eye(s))
    return out


def _resolve_basic_attack(s: GameState, a: ActionBasicAttack) -> None:
    pl = s.current_player
    u = _actor_unit(s, pl, a.actor_slot)
    if u.containment_ticks > 0:
        raise IllegalActionError("contained")
    tid = s.board[a.target_square[0], a.target_square[1]]
    if tid < 0:
        raise IllegalActionError("empty target")
    oteam, osl = linear_to_team_slot(tid)
    ou = unit_at(s, oteam, osl)
    if ou is None or ou.controller == pl:
        raise IllegalActionError("bad target")
    careful = not u.moved_this_turn
    dmg = class_basic_damage(u.class_id, careful_aim_not_moved=careful)
    if u.class_id in (int(ClassId.KNIGHT), int(ClassId.BARBARIAN)):
        if _manhattan_dist(u.row, u.col, a.target_square[0], a.target_square[1]) != 1:
            raise IllegalActionError("range")
    elif u.class_id in (int(ClassId.WHITE_MAGE), int(ClassId.BLACK_MAGE)):
        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, a.target_square[0], a.target_square[1], 2):
            raise IllegalActionError("los")
    elif u.class_id == int(ClassId.ARBALIST):
        if not _arbalist_basic_los(s, u.row, u.col, a.target_square[0], a.target_square[1], 3):
            raise IllegalActionError("los")
    _apply_damage_to_unit(s, oteam, osl, dmg, source_class_for_fear=u.class_id)


def _resolve_charge(s: GameState, actor_team: int, slot: int, dest: tuple[int, int]) -> None:
    u = _actor_unit(s, actor_team, slot)
    if u.class_id != int(ClassId.KNIGHT):
        raise IllegalActionError("class")
    if s.any_friendly_moved_this_turn:
        raise IllegalActionError("charge blocked")
    dr = dest[0] - u.row
    dc = dest[1] - u.col
    if dr != 0 and dc != 0:
        raise IllegalActionError("not straight")
    dist = abs(dr) + abs(dc)
    if not (1 <= dist <= 4):
        raise IllegalActionError("dist")
    step_r = dr // dist if dist else 0
    step_c = dc // dist if dist else 0
    for k in range(1, dist):
        rr, cc = u.row + step_r * k, u.col + step_c * k
        if _blocks_charge_path(int(s.terrain[rr, cc])):
            raise IllegalActionError("terrain")
        tid = s.board[rr, cc]
        if tid >= 0:
            oteam, osl = linear_to_team_slot(tid)
            _apply_damage_to_unit(s, oteam, osl, 2, source_class_for_fear=None)
    tr, tc = dest
    if _blocks_charge_path(int(s.terrain[tr, tc])) or s.board[tr, tc] >= 0:
        raise IllegalActionError("bad landing")
    s.board[u.row, u.col] = -1
    u.row, u.col = tr, tc
    s.board[tr, tc] = lin_idx(actor_team, slot)


def _resolve_special(s: GameState, sp: ActionSpecial) -> None:
    pl = s.current_player
    u = _actor_unit(s, pl, sp.actor_slot)
    sid = sp.special_id

    if sid == SpecialId.CHARGE:
        if sp.target_square is None:
            raise IllegalActionError("charge target")
        _resolve_charge(s, pl, sp.actor_slot, sp.target_square)
        return

    if sid == SpecialId.LONG_EYE:
        if u.class_id != int(ClassId.ARBALIST) or u.containment_ticks > 0:
            raise IllegalActionError("long eye")
        if sp.target_square is None:
            raise IllegalActionError("target")
        tr, tc = sp.target_square
        dr = tr - u.row
        dc = tc - u.col
        if dr == 0 and dc == 0:
            raise IllegalActionError("same cell")
        ur, uc = u.row, u.col
        if dr != 0 and dc != 0 and abs(dr) != abs(dc):
            raise IllegalActionError("not line")
        steps = max(abs(dr), abs(dc))
        sdr = dr // steps if steps else 0
        sdc = dc // steps if steps else 0
        if ur + sdr * steps != tr or uc + sdc * steps != tc:
            raise IllegalActionError("not line")
        cr, cc = ur, uc
        for _ in range(steps):
            cr += sdr
            cc += sdc
            if not _in_bounds(cr, cc):
                raise IllegalActionError("oob")
            if _blocks_long_eye(int(s.terrain[cr, cc])):
                raise IllegalActionError("terrain")
            if (cr, cc) != (tr, tc) and s.board[cr, cc] >= 0:
                raise IllegalActionError("blocked")
        tid = s.board[tr, tc]
        if tid < 0:
            raise IllegalActionError("no fig")
        oteam, osl = linear_to_team_slot(tid)
        ou = unit_at(s, oteam, osl)
        if ou is None or ou.controller == pl:
            raise IllegalActionError("target")
        _apply_damage_to_unit(s, oteam, osl, 1, source_class_for_fear=u.class_id)
        return

    if sid == SpecialId.CONVERT:
        if u.class_id != int(ClassId.WHITE_MAGE):
            raise IllegalActionError("class")
        if sp.target_square is None:
            raise IllegalActionError("target")
        tr, tc = sp.target_square
        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, 2):
            raise IllegalActionError("reach")
        tid = s.board[tr, tc]
        if tid < 0:
            raise IllegalActionError("empty")
        oteam, osl = linear_to_team_slot(tid)
        ou = unit_at(s, oteam, osl)
        if ou is None or ou.controller == pl or ou.hp > 2:
            raise IllegalActionError("convert")
        ou.controller = pl
        return

    if sid == SpecialId.HEAL:
        if u.class_id != int(ClassId.WHITE_MAGE):
            raise IllegalActionError("class")
        if sp.target_square is None:
            raise IllegalActionError("target")
        tr, tc = sp.target_square
        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, 2):
            raise IllegalActionError("reach")
        tid = s.board[tr, tc]
        if tid < 0:
            raise IllegalActionError("empty")
        oteam, osl = linear_to_team_slot(tid)
        ou = unit_at(s, oteam, osl)
        if ou is None or ou.controller != pl:
            raise IllegalActionError("heal target")
        # Exclude only self (same unit as the white mage), using identity — not
        # (oteam, osl) vs (pl, actor_slot), which mis-fires when roster indexing
        # does not match current_player slot conventions.
        if ou is u:
            raise IllegalActionError("heal target")
        ou.hp = min(ou.max_hp, ou.hp + 3)
        return

    if sid == SpecialId.CONJURE_CONTAINMENT:
        if u.class_id != int(ClassId.WHITE_MAGE) or u.used_conjure_containment:
            raise IllegalActionError("containment")
        if sp.target_square is None:
            raise IllegalActionError("target")
        tr, tc = sp.target_square
        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, 2):
            raise IllegalActionError("reach")
        tid = s.board[tr, tc]
        if tid < 0:
            raise IllegalActionError("empty")
        oteam, osl = linear_to_team_slot(tid)
        ou = unit_at(s, oteam, osl)
        if ou is None or ou.controller == pl:
            raise IllegalActionError("target")
        ou.containment_ticks = 2
        ou.containment_clock_team = pl
        ou.containment_skip_next_tick = True
        u.used_conjure_containment = True
        return

    if sid == SpecialId.CURSE:
        if u.class_id != int(ClassId.BLACK_MAGE):
            raise IllegalActionError("class")
        if sp.target_square is None or sp.curse_x is None:
            raise IllegalActionError("curse params")
        x = sp.curse_x
        if x < 1 or x > MAX_CURSE_X:
            raise IllegalActionError("x")
        tr, tc = sp.target_square
        if not _in_orthogonal_reach_wm_bm(s, u.row, u.col, tr, tc, 2):
            raise IllegalActionError("reach")
        tid = s.board[tr, tc]
        if tid < 0:
            raise IllegalActionError("empty")
        oteam, osl = linear_to_team_slot(tid)
        ou = unit_at(s, oteam, osl)
        if ou is None or ou.controller == pl:
            raise IllegalActionError("target")
        su = unit_at(s, pl, sp.actor_slot)
        if su is None:
            raise IllegalActionError("self")
        su.hp -= x
        tdmg = x + 1
        if ou.class_id == int(ClassId.BARBARIAN) and _is_mage_class(u.class_id):
            tdmg += 1
        ou.hp -= tdmg
        if su.hp <= 0:
            _kill_unit(s, pl, sp.actor_slot)
        if ou.hp <= 0:
            _kill_unit(s, oteam, osl)
        return

    if sid == SpecialId.MAGIC_BOMB:
        if u.class_id != int(ClassId.BLACK_MAGE) or u.used_magic_bomb:
            raise IllegalActionError("bomb")
        if sp.target_square is None:
            raise IllegalActionError("target")
        tr, tc = sp.target_square
        if _manhattan_dist(u.row, u.col, tr, tc) > 2:
            raise IllegalActionError("reach")
        if _blocks_movement(int(s.terrain[tr, tc])):
            raise IllegalActionError("terrain")
        cells = [(tr, tc), (tr + 1, tc), (tr - 1, tc), (tr, tc + 1), (tr, tc - 1)]
        u.used_magic_bomb = True
        for br, bc in cells:
            if not _in_bounds(br, bc):
                continue
            tid = s.board[br, bc]
            if tid < 0:
                continue
            oteam, osl = linear_to_team_slot(tid)
            _apply_damage_to_unit(s, oteam, osl, 2, source_class_for_fear=u.class_id)
        return

    if sid == SpecialId.ANIMATE_DEAD:
        if u.class_id != int(ClassId.BLACK_MAGE):
            raise IllegalActionError("class")
        if sp.animate_dead_crew_slot is None:
            raise IllegalActionError("slot")
        osl = sp.animate_dead_crew_slot
        du = slot_unit(s, pl, osl)
        if du is None or du.alive:
            raise IllegalActionError("not dead")
        _apply_damage_to_unit(s, pl, sp.actor_slot, 1, source_class_for_fear=None)
        if unit_at(s, pl, sp.actor_slot) is None:
            return
        dest = _find_empty_in_start_zone(s, pl)
        if dest is None:
            raise IllegalActionError("no space")
        dr, dc = dest
        rec = du.death_point_recipient
        du.alive = True
        du.hp = 2
        du.controller = pl
        du.row, du.col = dr, dc
        du.containment_ticks = 0
        du.death_point_recipient = None
        s.board[dr, dc] = lin_idx(pl, osl)
        if rec is not None:
            sc = list(s.score)
            if sc[rec] > 0:
                sc[rec] -= 1
            s.score = (sc[0], sc[1])
        return

    raise IllegalActionError("unknown special")


def _find_empty_in_start_zone(s: GameState, team: int) -> Optional[tuple[int, int]]:
    rows = DEPLOY_ROWS_PLAYER_A if team == TEAM_PLAYER_A else DEPLOY_ROWS_PLAYER_B
    for r in rows:
        for c in range(BOARD_SIZE):
            if s.board[r, c] < 0 and not _blocks_movement(int(s.terrain[r, c])):
                return (r, c)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if s.board[r, c] < 0 and not _blocks_movement(int(s.terrain[r, c])):
                return (r, c)
    return None


def begin_setup(
    state: GameState,
    *,
    coin_flip_winner: int,
    winner_chooses_first_setup: bool,
) -> GameState:
    """
    After a coin flip, ``coin_flip_winner`` chooses either first setup and first turn,
    or second setup and second turn (paired).
    """
    s = state.clone()
    if s.match_phase != int(MatchPhase.PENDING_SETUP):
        raise IllegalActionError("setup not pending")
    if coin_flip_winner not in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        raise ValueError("coin_flip_winner must be a team id")
    first = coin_flip_winner if winner_chooses_first_setup else opponent(coin_flip_winner)
    s.coin_flip_winner = coin_flip_winner
    s.first_player = first
    s.setup_current_player = first
    s.match_phase = int(MatchPhase.SETUP)
    return s


def legal_setup_actions(state: GameState) -> list[SetupPlacement]:
    """Legal single-piece placements for ``state.setup_current_player`` during setup."""
    if state.match_phase != int(MatchPhase.SETUP):
        return []
    pl = state.setup_current_player
    rows = DEPLOY_ROWS_PLAYER_A if pl == TEAM_PLAYER_A else DEPLOY_ROWS_PLAYER_B
    out: list[SetupPlacement] = []
    for sl in range(FIGURES_PER_SIDE):
        su = slot_unit(state, pl, sl)
        if su is None or not su.alive or su.row >= 0:
            continue
        for r in rows:
            for c in range(BOARD_SIZE):
                if state.board[r, c] >= 0:
                    continue
                if _blocks_movement(int(state.terrain[r, c])):
                    continue
                out.append(SetupPlacement(actor_slot=sl, destination=(r, c)))
    return out


def setup_step(state: GameState, placement: SetupPlacement) -> StepResult:
    """Place one staged figure; alternates players until both sides have five on the board."""
    s = state.clone()
    if s.match_phase != int(MatchPhase.SETUP):
        raise IllegalActionError("not in setup")
    legal = legal_setup_actions(s)
    if placement not in legal:
        raise IllegalActionError("illegal setup placement")
    pl = s.setup_current_player
    u = slot_unit(s, pl, placement.actor_slot)
    if u is None or not u.alive or u.row >= 0:
        raise IllegalActionError("bad slot")
    dr, dc = placement.destination
    if s.board[dr, dc] >= 0:
        raise IllegalActionError("occupied")
    u.row, u.col = dr, dc
    s.board[dr, dc] = lin_idx(pl, placement.actor_slot)

    na = sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, TEAM_PLAYER_A, sl))
    nb = sum(1 for sl in range(FIGURES_PER_SIDE) if unit_at(s, TEAM_PLAYER_B, sl))
    if na == FIGURES_PER_SIDE and nb == FIGURES_PER_SIDE:
        s.match_phase = int(MatchPhase.PLAY)
        s.current_player = s.first_player
        s.turn_index = 0
    else:
        s.setup_current_player = opponent(pl)

    return StepResult(state=s, done=False, winner=None)


def complete_setup_random(state: GameState, rng: random.Random) -> GameState:
    """Finish alternating placement using uniform random legal choices (for headless runs)."""
    s = state
    while s.match_phase == int(MatchPhase.SETUP):
        opts = legal_setup_actions(s)
        if not opts:
            raise RuntimeError("no legal setup actions")
        s = setup_step(s, rng.choice(opts)).state
    return s


def initial_play_state(
    *,
    points_to_win: int = DEFAULT_POINTS_TO_WIN,
    terrain: Optional[np.ndarray] = None,
    current_player: int = TEAM_PLAYER_A,
) -> GameState:
    """Deterministic fully placed board (classic one row per side, cols 0–4) for tests / baselines."""
    placements: list[tuple[int, int, int, int, int]] = []
    for slot in range(FIGURES_PER_SIDE):
        placements.append((TEAM_PLAYER_A, slot, PLAYER_A_HOME_ROW, DEFAULT_START_COLS[slot], slot))
        placements.append((TEAM_PLAYER_B, slot, PLAYER_B_HOME_ROW, DEFAULT_START_COLS[slot], slot))
    return scenario_from_placements(
        terrain=terrain,
        placements=placements,
        current_player=current_player,
        points_to_win=points_to_win,
    )


def _apply_action(s: GameState, act: Optional[ActionIntent]) -> None:
    if act is None:
        return
    if isinstance(act, ActionBasicAttack):
        _resolve_basic_attack(s, act)
    else:
        _resolve_special(s, act)


def _clone_apply_move(state: GameState, move: Optional[MoveIntent]) -> GameState:
    s = state.clone()
    if move is None:
        return s
    pl = s.current_player
    u = unit_at(s, pl, move.actor_slot)
    if u is None or u.controller != pl:
        raise IllegalActionError("move actor")
    if u.containment_ticks > 0:
        raise IllegalActionError("contained move")
    dest = move.destination
    if dest not in _legal_destinations_for_unit(s, pl, move.actor_slot):
        raise IllegalActionError("illegal dest")
    _apply_move_only(s, pl, move.actor_slot, dest)
    return s


def legal_actions(state: GameState) -> list[TurnAction]:
    """All legal full turns for ``state.current_player``."""
    if state.done:
        return []
    if state.match_phase != int(MatchPhase.PLAY):
        return []
    out: list[TurnAction] = []
    pl = state.current_player
    moves: list[Optional[MoveIntent]] = [None]
    for sl in range(FIGURES_PER_SIDE):
        for dest in _legal_destinations_for_unit(state, pl, sl):
            moves.append(MoveIntent(actor_slot=sl, destination=dest))

    for mv in moves:
        try:
            mid = _clone_apply_move(state, mv)
        except IllegalActionError:
            continue
        acts: list[Optional[ActionIntent]] = [None]
        acts.extend(_legal_basic_attacks_after_move(mid))
        acts.extend(_legal_specials(mid))
        seen: set[tuple[object, ...]] = set()
        for ac in acts:
            key = [
                type(ac).__name__,
                getattr(ac, "actor_slot", None),
                getattr(ac, "target_square", None),
            ]
            if isinstance(ac, ActionSpecial):
                key.extend([ac.special_id, ac.curse_x, ac.animate_dead_crew_slot])
            key_t = tuple(key)
            if key_t in seen:
                continue
            seen.add(key_t)
            try:
                test = mid.clone()
                _apply_action(test, ac)
            except IllegalActionError:
                continue
            out.append(TurnAction(move=mv, action=ac))
    return out


def step(state: GameState, turn: TurnAction) -> StepResult:
    """Apply a full turn (move then action). Raises ``IllegalActionError`` if illegal."""
    if state.done:
        raise IllegalActionError("game over")
    if state.match_phase != int(MatchPhase.PLAY):
        raise IllegalActionError("match not in play phase")
    s = state.clone()
    pl = s.current_player
    try:
        if turn.move is not None:
            u = unit_at(s, pl, turn.move.actor_slot)
            if u is None or u.controller != pl:
                raise IllegalActionError("move")
            if u.containment_ticks > 0:
                raise IllegalActionError("contained")
            dest = turn.move.destination
            if dest not in _legal_destinations_for_unit(s, pl, turn.move.actor_slot):
                raise IllegalActionError("dest")
            _apply_move_only(s, pl, turn.move.actor_slot, dest)
        _apply_action(s, turn.action)
    except IllegalActionError:
        raise
    _check_win(s)
    if not s.done:
        _advance_turn(s)
    return StepResult(state=s, done=s.done, winner=s.winner)


def to_structured_observation(state: GameState) -> StructuredObservation:
    """Build 4.2 observation from engine state (between turns: next player to act is ``current_player``)."""
    h, w = BOARD_SIZE, BOARD_SIZE
    terrain = np.zeros((h, w), dtype=np.int8)
    occupancy = np.zeros((h, w), dtype=np.float32)
    team = np.full((h, w), -1, dtype=np.int8)
    unit_class = np.full((h, w), -1, dtype=np.int8)
    hp_norm = np.zeros((h, w), dtype=np.float32)
    containment = np.zeros((h, w), dtype=np.float32)

    terrain[:, :] = state.terrain

    for t in (TEAM_PLAYER_A, TEAM_PLAYER_B):
        for sl in range(FIGURES_PER_SIDE):
            u = unit_at(state, t, sl)
            if u is None:
                continue
            r, c = u.row, u.col
            occupancy[r, c] = 1.0
            team[r, c] = u.controller
            unit_class[r, c] = u.class_id
            denom = max(u.max_hp, 1)
            hp_norm[r, c] = float(u.hp) / float(denom)
            if u.containment_ticks > 0:
                containment[r, c] = 1.0

    return StructuredObservation(
        terrain=terrain,
        occupancy=occupancy,
        team=team,
        unit_class=unit_class,
        hp_normalized=hp_norm,
        containment=containment,
        reserved_status_a=np.zeros((h, w), dtype=np.float32),
        reserved_status_b=np.zeros((h, w), dtype=np.float32),
        score_player_a=float(state.score[0]),
        score_player_b=float(state.score[1]),
        turn_index=float(state.turn_index),
        current_player=int(state.current_player),
        phase=int(Phase.MOVE),
        points_to_win=float(state.points_to_win),
    )


# Re-export helpers for tests
def scenario_from_placements(
    *,
    terrain: Optional[np.ndarray] = None,
    placements: list[tuple[int, int, int, int, int]],
    current_player: int = TEAM_PLAYER_A,
    points_to_win: int = 4,
) -> GameState:
    """
    Build a state with custom unit positions.

    Each placement: ``(team, slot, row, col, class_id)`` with HP from ``STARTING_HP_BY_CLASS``.
    """
    from motley_crews_env.constants import STARTING_HP_BY_CLASS, TERRAIN_OPEN

    h, w = BOARD_SIZE, BOARD_SIZE
    t = terrain if terrain is not None else np.full((h, w), TERRAIN_OPEN, dtype=np.int8)
    board = np.full((h, w), -1, dtype=np.int8)
    units: list[Optional[UnitState]] = [None] * (2 * FIGURES_PER_SIDE)

    for team, slot, row, col, cid in placements:
        hp = STARTING_HP_BY_CLASS[cid]
        u = UnitState(
            class_id=cid,
            hp=hp,
            max_hp=hp,
            row=row,
            col=col,
            controller=team,
        )
        units[lin_idx(team, slot)] = u
        if board[row, col] >= 0:
            raise ValueError("overlap")
        board[row, col] = lin_idx(team, slot)

    return GameState(
        terrain=t,
        board=board,
        units=units,
        score=(0, 0),
        turn_index=0,
        current_player=current_player,
        points_to_win=points_to_win,
        match_phase=int(MatchPhase.PLAY),
        setup_current_player=current_player,
        first_player=current_player,
        coin_flip_winner=None,
    )


__all__ = [
    "legal_actions",
    "step",
    "to_structured_observation",
    "scenario_from_placements",
    "initial_state",
    "initial_play_state",
    "begin_setup",
    "legal_setup_actions",
    "setup_step",
    "complete_setup_random",
    "IllegalActionError",
    "StepResult",
]

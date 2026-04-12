"""
Pure geometry for board highlight paths (UI only; mirrors movement / LOS shapes from the engine).
"""

from __future__ import annotations

from typing import Optional

from motley_crews_env.constants import BOARD_SIZE
from motley_crews_env.engine import preview_after_move
from motley_crews_env.state import GameState, unit_at
from motley_crews_env.types import ActionBasicAttack, ActionSpecial, ClassId, SpecialId, TurnAction


def orthogonal_straight_segment(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Inclusive cells on a straight horizontal or vertical segment; empty if not aligned."""
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 > r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    return []


def l_shaped_path_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    """Horizontal then vertical legs (inclusive), for non-collinear move visualization."""
    out: set[tuple[int, int]] = set()
    if r0 == r1 and c0 == c1:
        return {(r0, c0)}
    step_c = 1 if c1 >= c0 else -1
    for c in range(c0, c1 + step_c, step_c):
        out.add((r0, c))
    step_r = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + step_r, step_r):
        out.add((r, c1))
    return out


def move_path_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    """Path from start to destination: straight segment if aligned, else L-shaped."""
    seg = orthogonal_straight_segment(r0, c0, r1, c1)
    if seg:
        return set(seg)
    return l_shaped_path_cells(r0, c0, r1, c1)


def charge_path_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    """Knight charge: straight ray from start through landing (inclusive)."""
    dr = r1 - r0
    dc = c1 - c0
    if dr != 0 and dc != 0:
        return set()
    dist = abs(dr) + abs(dc)
    if dist < 1 or dist > 4:
        return set()
    step_r = dr // dist if dist else 0
    step_c = dc // dist if dist else 0
    return {(r0 + step_r * k, c0 + step_c * k) for k in range(dist + 1)}


def cells_along_orthogonal_ray(ar: int, ac: int, tr: int, tc: int) -> set[tuple[int, int]]:
    """All cells on the orthogonal segment from actor to target (inclusive)."""
    return set(orthogonal_straight_segment(ar, ac, tr, tc))


def cells_along_arbalist_ray(
    state: GameState, ar: int, ac: int, tr: int, tc: int
) -> set[tuple[int, int]]:
    """Cells along the arbalist shot line; bounded so bad inputs cannot spin forever."""
    del state
    dist = max(abs(tr - ar), abs(tc - ac))
    if dist == 0:
        return {(ar, ac)}
    dr = (tr - ar) // dist if tr != ar else 0
    dc = (tc - ac) // dist if tc != ac else 0
    if abs(dr) > 1 or abs(dc) > 1:
        return {(ar, ac), (tr, tc)}
    out: set[tuple[int, int]] = set()
    rr, cc = ar, ac
    out.add((rr, cc))
    for _ in range(dist):
        rr += dr
        cc += dc
        out.add((rr, cc))
        if (rr, cc) == (tr, tc):
            return out
        if not (0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE):
            return out | {(tr, tc)}
    return out | {(tr, tc)}


def cells_along_long_eye_ray(state: GameState, ar: int, ac: int, tr: int, tc: int) -> set[tuple[int, int]]:
    """Diagonal or orthogonal line for Long Eye (same stepping as engine)."""
    del state
    dr = tr - ar
    dc = tc - ac
    if dr == 0 and dc == 0:
        return {(ar, ac)}
    if dr != 0 and dc != 0 and abs(dr) != abs(dc):
        return {(ar, ac), (tr, tc)}
    steps = max(abs(dr), abs(dc))
    sdr = dr // steps if steps else 0
    sdc = dc // steps if steps else 0
    out: set[tuple[int, int]] = set()
    rr, cc = ar, ac
    out.add((rr, cc))
    for _ in range(steps):
        rr += sdr
        cc += sdc
        out.add((rr, cc))
    return out


def _roster_team_for_actor(pl: int, actor_team: Optional[int]) -> int:
    return pl if actor_team is None else actor_team


def path_cells_for_basic_attack(
    state: GameState,
    pl: int,
    actor_slot: int,
    target: tuple[int, int],
    *,
    actor_team: Optional[int] = None,
) -> set[tuple[int, int]]:
    team = _roster_team_for_actor(pl, actor_team)
    u = unit_at(state, team, actor_slot)
    if u is None:
        return {target}
    ar, ac = u.row, u.col
    tr, tc = target
    cid = u.class_id
    if cid in (int(ClassId.KNIGHT), int(ClassId.BARBARIAN)):
        return {(ar, ac), (tr, tc)}
    if cid in (int(ClassId.WHITE_MAGE), int(ClassId.BLACK_MAGE)):
        seg = cells_along_orthogonal_ray(ar, ac, tr, tc)
        return seg if seg else {(ar, ac), (tr, tc)}
    if cid == int(ClassId.ARBALIST):
        return cells_along_arbalist_ray(state, ar, ac, tr, tc)
    return {(ar, ac), (tr, tc)}


def path_cells_for_special(
    state: GameState, pl: int, sp: ActionSpecial
) -> set[tuple[int, int]]:
    team = _roster_team_for_actor(pl, sp.actor_team)
    u = unit_at(state, team, sp.actor_slot)
    if u is None:
        return set()
    ar, ac = u.row, u.col
    sid = int(sp.special_id)
    if sid == int(SpecialId.CHARGE) and sp.target_square is not None:
        tr, tc = sp.target_square
        return charge_path_cells(ar, ac, tr, tc)
    if sid == int(SpecialId.LONG_EYE) and sp.target_square is not None:
        tr, tc = sp.target_square
        return cells_along_long_eye_ray(state, ar, ac, tr, tc)
    if sid == int(SpecialId.CURSE) and sp.target_square is not None:
        tr, tc = sp.target_square
        return {(ar, ac), (tr, tc)}
    if sid == int(SpecialId.MAGIC_BOMB) and sp.target_square is not None:
        tr, tc = sp.target_square
        cells: set[tuple[int, int]] = set()
        for br, bc in ((tr, tc), (tr + 1, tc), (tr - 1, tc), (tr, tc + 1), (tr, tc - 1)):
            if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                cells.add((br, bc))
        return cells
    if sp.target_square is not None:
        tr, tc = sp.target_square
        return {(ar, ac), (tr, tc)}
    return {(ar, ac)}


def path_cells_for_turn(state: GameState, pl: int, ta: TurnAction) -> set[tuple[int, int]]:
    """Highlight cells for the chosen path / line-of-fire for ``ta``."""
    # Move+action: phase B shows attack/special line only (move path already shown in move-only turns)
    if ta.move is not None and ta.action is None:
        mi = ta.move
        team = _roster_team_for_actor(pl, mi.actor_team)
        u = unit_at(state, team, mi.actor_slot)
        if u is None:
            return set()
        r0, c0 = u.row, u.col
        r1, c1 = ta.move.destination
        return move_path_cells(r0, c0, r1, c1)
    if ta.action is None:
        return set()
    # Use post-move positions for attack/special lines when the turn includes a move.
    base = preview_after_move(state, ta.move) if ta.move is not None else state
    if isinstance(ta.action, ActionBasicAttack):
        ba = ta.action
        return path_cells_for_basic_attack(
            base, pl, ba.actor_slot, ba.target_square, actor_team=ba.actor_team
        )
    assert isinstance(ta.action, ActionSpecial)
    return path_cells_for_special(base, pl, ta.action)


def preview_emphasis_cells(state: GameState, pl: int, ta: TurnAction) -> set[tuple[int, int]]:
    """Cells to emphasize in phase C (targets / landing)."""
    out: set[tuple[int, int]] = set()
    if ta.move is not None:
        out.add(ta.move.destination)
    if ta.action is None:
        return out
    if isinstance(ta.action, ActionBasicAttack):
        out.add(ta.action.target_square)
        return out
    sp = ta.action
    sid = int(sp.special_id)
    if sid == int(SpecialId.MAGIC_BOMB) and sp.target_square is not None:
        tr, tc = sp.target_square
        for br, bc in ((tr, tc), (tr + 1, tc), (tr - 1, tc), (tr, tc + 1), (tr, tc - 1)):
            if 0 <= br < BOARD_SIZE and 0 <= bc < BOARD_SIZE:
                out.add((br, bc))
        return out
    if sp.target_square is not None:
        out.add(sp.target_square)
    return out


# --- Test helpers (no GameState) ---


def orthogonal_straight_segment_exposed(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    return orthogonal_straight_segment(r0, c0, r1, c1)


def charge_path_cells_exposed(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return charge_path_cells(r0, c0, r1, c1)

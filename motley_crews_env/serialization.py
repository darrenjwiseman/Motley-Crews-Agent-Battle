"""Canonical tuple serialization for ``TurnAction`` (logging, tests, future replay buffers)."""

from __future__ import annotations

from typing import Any, Optional

from motley_crews_env.types import (
    ActionBasicAttack,
    ActionSpecial,
    MoveIntent,
    SpecialId,
    TurnAction,
)

# Tuple tags
_TAG_MOVE = "m"
_TAG_BASIC = "ba"
_TAG_SPECIAL = "sp"


def turn_action_to_tuple(action: TurnAction) -> tuple[Any, ...]:
    """
    Nested tuple form::

        (move_part, action_part)

    move_part: None | ("m", actor_slot, row, col) | ("m", actor_slot, row, col, roster_team)

    action_part: None | ("ba", actor_slot, row, col) | ("ba", actor_slot, row, col, roster_team)
        | ("sp", spec_idx, actor_slot, tr, tc, curse_x, anim_slot)
        | ("sp", spec_idx, actor_slot, tr, tc, curse_x, anim_slot, roster_team)

    ``roster_team`` is -1 when ``actor_team`` is None (figure on current player's roster).

    Use -1 for missing tr/tc (no square), curse_x, or anim_slot when not applicable.
    """
    move_part: Any
    if action.move is None:
        move_part = None
    else:
        m = action.move
        rt = -1 if m.actor_team is None else int(m.actor_team)
        move_part = (_TAG_MOVE, m.actor_slot, m.destination[0], m.destination[1], rt)

    action_part: Any
    if action.action is None:
        action_part = None
    elif isinstance(action.action, ActionBasicAttack):
        a = action.action
        rt = -1 if a.actor_team is None else int(a.actor_team)
        action_part = (_TAG_BASIC, a.actor_slot, a.target_square[0], a.target_square[1], rt)
    else:
        a = action.action
        assert isinstance(a, ActionSpecial)
        tr = a.target_square[0] if a.target_square is not None else -1
        tc = a.target_square[1] if a.target_square is not None else -1
        cx = a.curse_x if a.curse_x is not None else -1
        an = a.animate_dead_crew_slot if a.animate_dead_crew_slot is not None else -1
        rt = -1 if a.actor_team is None else int(a.actor_team)
        action_part = (
            _TAG_SPECIAL,
            int(a.special_id),
            a.actor_slot,
            tr,
            tc,
            cx,
            an,
            rt,
        )

    if action.resurrect_place is not None:
        rr, rc = action.resurrect_place
        return (move_part, action_part, ("rp", rr, rc))
    return (move_part, action_part)


_TAG_RP = "rp"


def turn_action_from_tuple(data: tuple[Any, ...]) -> TurnAction:
    if len(data) == 2:
        move_part, action_part = data
        resurrect_place = None
    elif len(data) == 3:
        move_part, action_part, rp_part = data
        if (
            not isinstance(rp_part, tuple)
            or len(rp_part) != 3
            or rp_part[0] != _TAG_RP
        ):
            raise ValueError(f"Invalid resurrect_part: {rp_part!r}")
        resurrect_place = (int(rp_part[1]), int(rp_part[2]))
    else:
        raise ValueError("Expected (move_part, action_part) or with resurrect tuple")

    move: Optional[MoveIntent] = None
    if move_part is not None:
        if not isinstance(move_part, tuple) or move_part[0] != _TAG_MOVE:
            raise ValueError(f"Invalid move_part: {move_part!r}")
        if len(move_part) == 4:
            move = MoveIntent(actor_slot=int(move_part[1]), destination=(int(move_part[2]), int(move_part[3])))
        elif len(move_part) == 5:
            rt = int(move_part[4])
            move = MoveIntent(
                actor_slot=int(move_part[1]),
                destination=(int(move_part[2]), int(move_part[3])),
                actor_team=None if rt < 0 else rt,
            )
        else:
            raise ValueError(f"Invalid move_part length: {move_part!r}")

    action: Optional[ActionBasicAttack | ActionSpecial] = None
    if action_part is not None:
        if not isinstance(action_part, tuple) or len(action_part) < 2:
            raise ValueError(f"Invalid action_part: {action_part!r}")
        tag = action_part[0]
        if tag == _TAG_BASIC:
            if len(action_part) == 4:
                action = ActionBasicAttack(
                    actor_slot=int(action_part[1]),
                    target_square=(int(action_part[2]), int(action_part[3])),
                )
            elif len(action_part) == 5:
                rt = int(action_part[4])
                action = ActionBasicAttack(
                    actor_slot=int(action_part[1]),
                    target_square=(int(action_part[2]), int(action_part[3])),
                    actor_team=None if rt < 0 else rt,
                )
            else:
                raise ValueError("basic attack tuple length")
        elif tag == _TAG_SPECIAL:
            if len(action_part) not in (7, 8):
                raise ValueError("special tuple length")
            _, spec_idx, actor, tr, tc, cx, an = action_part[:7]
            rt_sp: Optional[int] = None
            if len(action_part) == 8:
                rt_raw = int(action_part[7])
                rt_sp = None if rt_raw < 0 else rt_raw
            ts: Optional[tuple[int, int]] = None
            if int(tr) >= 0 and int(tc) >= 0:
                ts = (int(tr), int(tc))
            action = ActionSpecial(
                actor_slot=int(actor),
                special_id=SpecialId(int(spec_idx)),
                target_square=ts,
                curse_x=int(cx) if int(cx) >= 0 else None,
                animate_dead_crew_slot=int(an) if int(an) >= 0 else None,
                actor_team=rt_sp,
            )
        else:
            raise ValueError(f"Unknown action tag {tag!r}")

    return TurnAction(move=move, action=action, resurrect_place=resurrect_place)

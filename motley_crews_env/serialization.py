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

    move_part: None | ("m", actor_slot, row, col)

    action_part: None | ("ba", actor_slot, row, col) | ("sp", spec_idx, actor_slot, tr, tc, curse_x, anim_slot)

    Use -1 for missing tr/tc (no square), curse_x, or anim_slot when not applicable.
    """
    move_part: Any
    if action.move is None:
        move_part = None
    else:
        m = action.move
        move_part = (_TAG_MOVE, m.actor_slot, m.destination[0], m.destination[1])

    action_part: Any
    if action.action is None:
        action_part = None
    elif isinstance(action.action, ActionBasicAttack):
        a = action.action
        action_part = (_TAG_BASIC, a.actor_slot, a.target_square[0], a.target_square[1])
    else:
        a = action.action
        assert isinstance(a, ActionSpecial)
        tr = a.target_square[0] if a.target_square is not None else -1
        tc = a.target_square[1] if a.target_square is not None else -1
        cx = a.curse_x if a.curse_x is not None else -1
        an = a.animate_dead_crew_slot if a.animate_dead_crew_slot is not None else -1
        action_part = (
            _TAG_SPECIAL,
            int(a.special_id),
            a.actor_slot,
            tr,
            tc,
            cx,
            an,
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
        if not isinstance(move_part, tuple) or len(move_part) != 4 or move_part[0] != _TAG_MOVE:
            raise ValueError(f"Invalid move_part: {move_part!r}")
        move = MoveIntent(actor_slot=int(move_part[1]), destination=(int(move_part[2]), int(move_part[3])))

    action: Optional[ActionBasicAttack | ActionSpecial] = None
    if action_part is not None:
        if not isinstance(action_part, tuple) or len(action_part) < 2:
            raise ValueError(f"Invalid action_part: {action_part!r}")
        tag = action_part[0]
        if tag == _TAG_BASIC:
            if len(action_part) != 4:
                raise ValueError("basic attack tuple length")
            action = ActionBasicAttack(
                actor_slot=int(action_part[1]),
                target_square=(int(action_part[2]), int(action_part[3])),
            )
        elif tag == _TAG_SPECIAL:
            if len(action_part) != 7:
                raise ValueError("special tuple length")
            _, spec_idx, actor, tr, tc, cx, an = action_part
            ts: Optional[tuple[int, int]] = None
            if int(tr) >= 0 and int(tc) >= 0:
                ts = (int(tr), int(tc))
            action = ActionSpecial(
                actor_slot=int(actor),
                special_id=SpecialId(int(spec_idx)),
                target_square=ts,
                curse_x=int(cx) if int(cx) >= 0 else None,
                animate_dead_crew_slot=int(an) if int(an) >= 0 else None,
            )
        else:
            raise ValueError(f"Unknown action tag {tag!r}")

    return TurnAction(move=move, action=action, resurrect_place=resurrect_place)

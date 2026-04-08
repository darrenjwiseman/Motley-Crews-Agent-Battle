"""Formatting helpers."""

from __future__ import annotations

from motley_crews_env.types import ActionBasicAttack, TurnAction
from motley_crews_play.formatting import format_turn_action


def test_format_turn_action_basic() -> None:
    a = TurnAction(
        move=None,
        action=ActionBasicAttack(actor_slot=0, target_square=(3, 4)),
    )
    s = format_turn_action(a)
    assert "atk" in s
    assert "(3,4)" in s.replace(" ", "")

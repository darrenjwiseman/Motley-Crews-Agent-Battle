"""Human-readable formatting for ``TurnAction`` (UI lists, debugging)."""

from __future__ import annotations

from motley_crews_env.constants import CLASS_IDS, SPECIAL_IDS, TEAM_PLAYER_A
from motley_crews_env.types import ActionBasicAttack, ActionSpecial, TurnAction


def player_label(team: int) -> str:
    return "Player A" if team == TEAM_PLAYER_A else "Player B"


def format_play_log_line(player: int, a: TurnAction) -> str:
    return f"{player_label(player)}: {format_turn_action(a)}"


def format_turn_action(a: TurnAction) -> str:
    parts: list[str] = []
    if a.move is None:
        parts.append("move: —")
    else:
        m = a.move
        parts.append(f"move: {CLASS_IDS[m.actor_slot][:3]} → ({m.destination[0]},{m.destination[1]})")
    if a.action is None:
        parts.append("act: —")
    elif isinstance(a.action, ActionBasicAttack):
        ba = a.action
        parts.append(f"act: {CLASS_IDS[ba.actor_slot][:3]} atk ({ba.target_square[0]},{ba.target_square[1]})")
    else:
        sp = a.action
        assert isinstance(sp, ActionSpecial)
        name = SPECIAL_IDS[int(sp.special_id)]
        extra = ""
        if sp.target_square is not None:
            extra = f" @ ({sp.target_square[0]},{sp.target_square[1]})"
        if sp.curse_x is not None:
            extra += f" curse_x={sp.curse_x}"
        if sp.animate_dead_crew_slot is not None:
            extra += f" anim_slot={sp.animate_dead_crew_slot}"
        parts.append(f"act: {CLASS_IDS[sp.actor_slot][:3]} {name}{extra}")
    return " | ".join(parts)

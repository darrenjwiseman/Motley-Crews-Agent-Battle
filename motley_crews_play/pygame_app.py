"""
Pygame front-end: menu, top-down / isometric board, legal-action list for humans.

Run: ``python -m motley_crews_play --ui`` (requires pygame; see requirements-play.txt).
"""

from __future__ import annotations

import random
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pygame

from motley_crews_env.constants import (
    BOARD_SIZE,
    CLASS_IDS,
    DEPLOY_ROWS_PLAYER_A,
    DEPLOY_ROWS_PLAYER_B,
    FIGURES_PER_SIDE,
    SPECIAL_IDS,
    TEAM_PLAYER_A,
    TEAM_PLAYER_B,
    TERRAIN_BLOCKED,
    TERRAIN_OPEN,
    TERRAIN_WATER,
)
from motley_crews_env.engine import (
    begin_setup,
    complete_setup_random,
    initial_state,
    legal_actions,
    legal_setup_actions,
    setup_step,
    step,
    to_structured_observation,
)
from motley_crews_env.types import (
    ActionBasicAttack,
    ActionSpecial,
    MatchPhase,
    MoveIntent,
    SetupPlacement,
    SpecialId,
    TurnAction,
)
from motley_crews_env.state import (
    GameState,
    UnitState,
    class_basic_damage,
    class_move_value,
    class_reach_basic,
    slot_unit,
    unit_at,
)
from motley_crews_play.formatting import (
    format_play_log_line,
    format_step_outcome,
    format_turn_action,
    player_label,
)
from motley_crews_play.policies import ScriptedHeuristicPolicy


class ViewMode(IntEnum):
    TOP_DOWN = 0
    ISOMETRIC = 1


class PlayMode(IntEnum):
    CPU_CPU = 0
    HUMAN_CPU_A = 1  # human = player A (0)
    HUMAN_CPU_B = 2  # human = player B (1)
    HUMAN_HUMAN = 3


# --- projection & layout --------------------------------------------------------

WINDOW_W = 1200
WINDOW_H = 860
BOARD_OX = 32
CELL_TOP = 56
TOP_MARGIN = 8
PLAY_HUD_H = 26
PANEL_B_H = 96
PANEL_A_H = 96
# Setup / menu: original board vertical position (unchanged for setup screens)
BOARD_OY_SETUP = 96
# Play: flanking roster bands + compact HUD above board
BOARD_OY_PLAY = TOP_MARGIN + PLAY_HUD_H + PANEL_B_H
BOARD_GRID_PX = BOARD_SIZE * CELL_TOP
TW_ISO = 52
TH_ISO = 28
ISO_OX = 120
ISO_OY_SETUP = 200
ISO_OY_PLAY = ISO_OY_SETUP + (BOARD_OY_PLAY - BOARD_OY_SETUP)

LOG_TOP = 62
LOG_LINE_H = 18

# Per-class combat stats and special names (roster display)
CLASS_SPECIALS_DISPLAY: tuple[str, ...] = (
    "Charge",
    "—",
    "Conjure containment, Convert, Heal",
    "Curse, Magic bomb, Animate dead",
    "Long eye",
)


def _wrap_text_to_width(font: pygame.font.Font, text: str, max_width: int) -> list[str]:
    if not text.strip():
        return [""]
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = " ".join(cur + [w]) if cur else w
        if font.size(trial)[0] <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines if lines else [""]


def board_origin_top(*, play: bool = False) -> tuple[int, int]:
    return BOARD_OX, BOARD_OY_PLAY if play else BOARD_OY_SETUP


def board_origin_iso(*, play: bool = False) -> tuple[int, int]:
    return ISO_OX, ISO_OY_PLAY if play else ISO_OY_SETUP


def cell_center_top(row: int, col: int, *, play: bool = False) -> tuple[int, int]:
    ox, oy = board_origin_top(play=play)
    cx = ox + col * CELL_TOP + CELL_TOP // 2
    cy = oy + row * CELL_TOP + CELL_TOP // 2
    return cx, cy


def cell_center_iso(row: int, col: int, *, play: bool = False) -> tuple[int, int]:
    ox, oy = board_origin_iso(play=play)
    cx = ox + (col - row) * (TW_ISO // 2)
    cy = oy + (col + row) * (TH_ISO // 2)
    return cx, cy


def _checkerboard_tint(
    base: tuple[int, int, int], r: int, c: int, terrain_kind: int
) -> tuple[int, int, int]:
    if terrain_kind != TERRAIN_OPEN:
        return base
    delta = 12 if (r + c) % 2 == 1 else -12
    return (
        max(0, min(255, base[0] + delta)),
        max(0, min(255, base[1] + delta)),
        max(0, min(255, base[2] + delta)),
    )


def _draw_team_marker(
    screen: pygame.Surface,
    cx: int,
    cy_token: int,
    team: int,
    *,
    view: ViewMode,
    radius: int,
) -> None:
    """Player A (south): ▲ above token; Player B (north): ▼ below."""
    if team == TEAM_PLAYER_A:
        col = (130, 170, 255)
        if view == ViewMode.TOP_DOWN:
            tip_y = cy_token - radius - 2
            pygame.draw.polygon(screen, col, [(cx, tip_y), (cx - 6, tip_y + 8), (cx + 6, tip_y + 8)])
        else:
            tip_y = cy_token - radius - 2
            pygame.draw.polygon(screen, col, [(cx, tip_y), (cx - 5, tip_y + 7), (cx + 5, tip_y + 7)])
    else:
        col = (255, 170, 130)
        if view == ViewMode.TOP_DOWN:
            tip_y = cy_token + radius + 2
            pygame.draw.polygon(screen, col, [(cx, tip_y), (cx - 6, tip_y - 8), (cx + 6, tip_y - 8)])
        else:
            tip_y = cy_token + radius + 2
            pygame.draw.polygon(screen, col, [(cx, tip_y), (cx - 5, tip_y - 7), (cx + 5, tip_y - 7)])


# --- palette --------------------------------------------------------------------

CLASS_COLORS = [
    (200, 200, 220),  # knight
    (220, 140, 100),  # barbarian
    (240, 240, 255),  # white mage
    (80, 60, 120),  # black mage
    (160, 180, 200),  # arbalist
]
TERRAIN_COLOR = {
    TERRAIN_OPEN: (245, 242, 235),
    TERRAIN_BLOCKED: (90, 90, 95),
    TERRAIN_WATER: (140, 170, 220),
}

HL_MOVE = (80, 180, 255, 72)
HL_ATTACK = (255, 140, 90, 72)
HL_SPECIAL = (200, 130, 255, 72)
HL_ACTION_FIGURE = (100, 210, 170, 68)


def _actor_slot_at_cell(gs: GameState, r: int, c: int) -> Optional[int]:
    for sl in range(FIGURES_PER_SIDE):
        u = unit_at(gs, gs.current_player, sl)
        if u is not None and u.alive and u.row == r and u.col == c:
            return sl
    return None


def _pass_turn_in_legal(legal: list[TurnAction]) -> bool:
    return TurnAction(move=None, action=None) in legal


def _pass_action_after_move_available(candidates: list[TurnAction]) -> bool:
    return any(ta.action is None for ta in candidates)


def _move_destinations_for_slot(legal: list[TurnAction], slot: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in legal:
        if ta.move is not None and ta.move.actor_slot == slot:
            out.add(ta.move.destination)
    return out


def _basic_targets_for_slot(legal: list[TurnAction], slot: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in legal:
        if ta.action is not None and isinstance(ta.action, ActionBasicAttack) and ta.action.actor_slot == slot:
            out.add(ta.action.target_square)
    return out


def _special_ids_for_slot(legal: list[TurnAction], slot: int) -> list[int]:
    seen: set[int] = set()
    for ta in legal:
        if ta.action is not None and isinstance(ta.action, ActionSpecial) and ta.action.actor_slot == slot:
            seen.add(int(ta.action.special_id))
    return sorted(seen)


def _turns_for_move_dest(legal: list[TurnAction], slot: int, dest: tuple[int, int]) -> list[TurnAction]:
    return [ta for ta in legal if ta.move is not None and ta.move.actor_slot == slot and ta.move.destination == dest]


def _turns_for_basic_target(
    legal: list[TurnAction], slot: int, target: tuple[int, int]
) -> list[TurnAction]:
    return [
        ta
        for ta in legal
        if ta.action is not None
        and isinstance(ta.action, ActionBasicAttack)
        and ta.action.actor_slot == slot
        and ta.action.target_square == target
    ]


def _turns_for_special_menu(
    legal: list[TurnAction], slot: int, special_id: int
) -> list[TurnAction]:
    return [
        ta
        for ta in legal
        if ta.action is not None
        and isinstance(ta.action, ActionSpecial)
        and ta.action.actor_slot == slot
        and int(ta.action.special_id) == special_id
    ]


def _special_target_cells(
    legal: list[TurnAction], slot: int, special_id: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in _turns_for_special_menu(legal, slot, special_id):
        assert ta.action is not None and isinstance(ta.action, ActionSpecial)
        sp = ta.action
        if int(sp.special_id) == int(SpecialId.ANIMATE_DEAD):
            continue
        if sp.target_square is not None:
            out.add(sp.target_square)
    return out


def _curse_x_options(
    legal: list[TurnAction], slot: int, target: tuple[int, int]
) -> list[int]:
    xs: set[int] = set()
    for ta in legal:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.CURSE):
            continue
        if sp.actor_slot != slot or sp.target_square != target or sp.curse_x is None:
            continue
        xs.add(int(sp.curse_x))
    return sorted(xs)


def _animate_dead_slots(legal: list[TurnAction], slot: int) -> list[int]:
    slots: set[int] = set()
    for ta in legal:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.ANIMATE_DEAD) or sp.actor_slot != slot:
            continue
        if sp.animate_dead_crew_slot is not None:
            slots.add(int(sp.animate_dead_crew_slot))
    return sorted(slots)


def _basic_targets_from_candidates(candidates: list[TurnAction]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in candidates:
        if ta.action is not None and isinstance(ta.action, ActionBasicAttack):
            out.add(ta.action.target_square)
    return out


def _special_ids_from_candidates(candidates: list[TurnAction]) -> list[int]:
    seen: set[int] = set()
    for ta in candidates:
        if ta.action is not None and isinstance(ta.action, ActionSpecial):
            seen.add(int(ta.action.special_id))
    return sorted(seen)


def _special_target_cells_from_candidates(
    candidates: list[TurnAction], special_id: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(special_id):
            continue
        if int(sp.special_id) == int(SpecialId.ANIMATE_DEAD):
            continue
        if sp.target_square is not None:
            out.add(sp.target_square)
    return out


def _turns_for_special_menu_candidates(
    candidates: list[TurnAction], special_id: int
) -> list[TurnAction]:
    return [
        ta
        for ta in candidates
        if ta.action is not None
        and isinstance(ta.action, ActionSpecial)
        and int(ta.action.special_id) == int(special_id)
    ]


def _curse_x_options_candidates(
    candidates: list[TurnAction], target: tuple[int, int]
) -> list[int]:
    """Curse X values for this target (any caster), within the given candidate turns."""
    xs: set[int] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.CURSE):
            continue
        if sp.target_square != target or sp.curse_x is None:
            continue
        xs.add(int(sp.curse_x))
    return sorted(xs)


def _animate_dead_slots_candidates(candidates: list[TurnAction]) -> list[int]:
    slots: set[int] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.ANIMATE_DEAD):
            continue
        if sp.animate_dead_crew_slot is not None:
            slots.add(int(sp.animate_dead_crew_slot))
    return sorted(slots)


def _slot_can_act_after_move(candidates: list[TurnAction], slot: int) -> bool:
    for ta in candidates:
        if ta.action is None:
            continue
        if ta.action.actor_slot == slot:
            return True
    return False


def _cells_with_actionable_figures(gs: GameState, candidates: list[TurnAction]) -> set[tuple[int, int]]:
    """Board cells of friendly figures that may take the action sub-step for this pending move."""
    slots: set[int] = set()
    for ta in candidates:
        if ta.action is not None:
            slots.add(ta.action.actor_slot)
    out: set[tuple[int, int]] = set()
    for sl in slots:
        u = unit_at(gs, gs.current_player, sl)
        if u is not None and u.alive and u.row >= 0:
            out.add((u.row, u.col))
    return out


def _basic_targets_for_slot_in_candidates(
    candidates: list[TurnAction], slot: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in candidates:
        if ta.action is not None and isinstance(ta.action, ActionBasicAttack) and ta.action.actor_slot == slot:
            out.add(ta.action.target_square)
    return out


def _special_ids_for_slot_candidates(candidates: list[TurnAction], slot: int) -> list[int]:
    seen: set[int] = set()
    for ta in candidates:
        if ta.action is not None and isinstance(ta.action, ActionSpecial) and ta.action.actor_slot == slot:
            seen.add(int(ta.action.special_id))
    return sorted(seen)


def _special_target_cells_for_slot_candidates(
    candidates: list[TurnAction], slot: int, special_id: int
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if sp.actor_slot != slot or int(sp.special_id) != int(special_id):
            continue
        if int(sp.special_id) == int(SpecialId.ANIMATE_DEAD):
            continue
        if sp.target_square is not None:
            out.add(sp.target_square)
    return out


def _turns_for_special_slot_candidates(
    candidates: list[TurnAction], slot: int, special_id: int
) -> list[TurnAction]:
    return [
        ta
        for ta in candidates
        if ta.action is not None
        and isinstance(ta.action, ActionSpecial)
        and ta.action.actor_slot == slot
        and int(ta.action.special_id) == int(special_id)
    ]


def _curse_x_options_for_slot_candidates(
    candidates: list[TurnAction], slot: int, target: tuple[int, int]
) -> list[int]:
    xs: set[int] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.CURSE):
            continue
        if sp.actor_slot != slot or sp.target_square != target or sp.curse_x is None:
            continue
        xs.add(int(sp.curse_x))
    return sorted(xs)


def _animate_dead_slots_for_slot_candidates(
    candidates: list[TurnAction], slot: int
) -> list[int]:
    slots: set[int] = set()
    for ta in candidates:
        if ta.action is None or not isinstance(ta.action, ActionSpecial):
            continue
        sp = ta.action
        if int(sp.special_id) != int(SpecialId.ANIMATE_DEAD) or sp.actor_slot != slot:
            continue
        if sp.animate_dead_crew_slot is not None:
            slots.add(int(sp.animate_dead_crew_slot))
    return sorted(slots)


class MotleyCrewsUI:
    def __init__(self, seed: int = 0) -> None:
        pygame.init()
        self.win_w = WINDOW_W
        self.win_h = WINDOW_H
        self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
        pygame.display.set_caption("Motley Crews")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo", 18) if sys.platform == "darwin" else pygame.font.SysFont("Consolas", 18)
        self.font_small = pygame.font.SysFont("Menlo", 15) if sys.platform == "darwin" else pygame.font.SysFont("Consolas", 15)
        self.seed = seed
        self.rng_cpu = random.Random(seed)

        self.phase = "menu"  # menu | play | over
        self.play_mode = PlayMode.CPU_CPU
        self.view_mode = ViewMode.TOP_DOWN
        self.game_state: Optional[GameState] = None
        self.cpu = ScriptedHeuristicPolicy()

        self.legal_list: list[TurnAction] = []
        self.selected_idx: int = 0
        self.action_scroll = 0
        self.cpu_timer = 0
        self.cpu_delay_ms = 180

        # Interactive setup (human modes): coin → choice → drag-drop placement
        self._coin_flip_winner: Optional[int] = None
        self._drag: Optional[tuple[int, int]] = None  # (team, slot) while dragging from staging
        self._drag_pos: Optional[tuple[int, int]] = None  # mouse position during drag

        self.btn_setup_first: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self.btn_setup_second: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self.btn_flip_coin: pygame.Rect = pygame.Rect(0, 0, 1, 1)

        # Play phase: board UI (piece menu → move / attack / special → highlights)
        self._play_ui_mode: str = "idle"
        self._play_slot: Optional[int] = None
        self._play_sp_id: Optional[int] = None
        self._play_menu_items: list[tuple[pygame.Rect, str, Any]] = []
        self._play_popup_anchor: tuple[int, int] = (0, 0)
        self._play_curse_target: Optional[tuple[int, int]] = None
        self._play_move_drag_slot: Optional[int] = None
        self._play_drag_xy: Optional[tuple[int, int]] = None
        self._play_ambiguous: list[TurnAction] = []
        self._play_show_fallback_list: bool = False
        self._pending_move: Optional[MoveIntent] = None
        self._turn_candidates: list[TurnAction] = []
        self.play_log: list[str] = []
        self.log_scroll: int = 0
        self._log_max_lines = 400
        # (row, col, total_damage, remaining_ms) for floating hit numbers
        self._damage_fx: list[tuple[int, int, int, float]] = []
        self._menu_hover_key: Optional[str] = None
        self.btn_return_menu: pygame.Rect = pygame.Rect(0, 0, 1, 1)
        self._recompute_layout()

    def _recompute_layout(self) -> None:
        w, h = self.win_w, self.win_h
        self.sidebar_x = max(520, int(w * 0.56))
        self.sidebar_w = w - self.sidebar_x
        self.board_ox = 12
        avail = self.sidebar_x - self.board_ox - 8
        self.cell_top = max(40, min(CELL_TOP, avail // BOARD_SIZE))
        self.board_grid_px = self.cell_top * BOARD_SIZE
        self.board_oy_play = TOP_MARGIN + PLAY_HUD_H + PANEL_B_H
        self.log_bottom = h - 12
        self.legal_list_top = min(int(h * 0.48), max(120, h - 280))
        scale = self.cell_top / float(CELL_TOP)
        self.tw_iso = max(36, int(TW_ISO * scale))
        self.th_iso = max(20, int(TH_ISO * scale))
        # Center isometric grid in the board panel
        iso_approx_w = (BOARD_SIZE - 1) * (self.tw_iso // 2) + self.tw_iso
        self.iso_ox = max(24, (self.sidebar_x - iso_approx_w) // 2)
        self.iso_oy_play = ISO_OY_SETUP + (self.board_oy_play - BOARD_OY_SETUP)
        self._token_r_top = max(12, int(18 * scale))
        self._token_r_iso = max(10, int(14 * scale))
        self.btn_pass_turn = pygame.Rect(self.sidebar_x + 8, TOP_MARGIN + 2, 150, 26)
        self.btn_pass_action_after_move = pygame.Rect(self.sidebar_x + 8, TOP_MARGIN + 30, 190, 26)
        self.log_scroll = min(self.log_scroll, self._max_log_scroll())
        self.btn_return_menu = pygame.Rect(w // 2 - 120, int(h * 0.70), 240, 48)
        self._layout_menu_buttons()
        self._layout_setup_buttons()

    def _board_origin_top(self, *, play: bool = False) -> tuple[int, int]:
        if not play:
            return BOARD_OX, BOARD_OY_SETUP
        return self.board_ox, self.board_oy_play

    def _cell_center_top(self, row: int, col: int, *, play: bool = False) -> tuple[int, int]:
        ox, oy = self._board_origin_top(play=play)
        ct = self.cell_top if play else CELL_TOP
        cx = ox + col * ct + ct // 2
        cy = oy + row * ct + ct // 2
        return cx, cy

    def _cell_center_iso(self, row: int, col: int, *, play: bool = False) -> tuple[int, int]:
        ox = self.iso_ox
        oy = self.iso_oy_play if play else ISO_OY_SETUP
        tw, th = (self.tw_iso, self.th_iso) if play else (TW_ISO, TH_ISO)
        cx = ox + (col - row) * (tw // 2)
        cy = oy + (col + row) * (th // 2)
        return cx, cy

    def _layout_menu_buttons(self) -> None:
        w, h = self.win_w, self.win_h
        left = max(48, int(w * 0.05))
        bw = min(420, int(w * 0.42))
        gap = max(36, int(h * 0.048))
        y0 = int(h * 0.14)
        self.btn_cpu_cpu = pygame.Rect(left, y0, bw, 40)
        self.btn_h_a = pygame.Rect(left, y0 + gap, bw, 40)
        self.btn_h_b = pygame.Rect(left, y0 + gap * 2, bw, 40)
        self.btn_h_h = pygame.Rect(left, y0 + gap * 3, bw, 40)
        self.btn_view = pygame.Rect(left, y0 + gap * 4 + 24, bw, 36)
        self.btn_start = pygame.Rect(left, y0 + gap * 5 + 40, min(220, bw), 44)

    def _layout_setup_buttons(self) -> None:
        w, h = self.win_w, self.win_h
        cx = w // 2
        self.btn_flip_coin = pygame.Rect(cx - 140, int(h * 0.36), 280, 48)
        self.btn_setup_first = pygame.Rect(cx - 300, int(h * 0.44), 280, 44)
        self.btn_setup_second = pygame.Rect(cx + 20, int(h * 0.44), 280, 44)

    def reset_match(self) -> None:
        self.rng_cpu = random.Random(self.seed)
        self._drag = None
        self._drag_pos = None
        self._coin_flip_winner = None
        self.game_state = initial_state()
        if self.play_mode == PlayMode.CPU_CPU:
            if self.game_state.match_phase == int(MatchPhase.PENDING_SETUP):
                cw = self.rng_cpu.randint(0, 1)
                wf = self.rng_cpu.choice([True, False])
                self.game_state = begin_setup(
                    self.game_state, coin_flip_winner=cw, winner_chooses_first_setup=wf
                )
            if self.game_state.match_phase == int(MatchPhase.SETUP):
                self.game_state = complete_setup_random(self.game_state, self.rng_cpu)
            self.phase = "play"
            self._refresh_legal()
        else:
            self.phase = "setup_coin"
        self.cpu_timer = 0
        self.play_log = []
        self.log_scroll = 0
        self._damage_fx = []

    def _refresh_legal(self) -> None:
        if self.game_state is None or self.game_state.done:
            self.legal_list = []
            return
        self.legal_list = legal_actions(self.game_state)
        self.selected_idx = 0
        self.action_scroll = 0
        self.cpu_timer = 0
        self._reset_play_interaction()

    def _reset_play_interaction(self) -> None:
        self._play_ui_mode = "idle"
        self._play_slot = None
        self._play_sp_id = None
        self._play_menu_items = []
        self._play_popup_anchor = (0, 0)
        self._play_curse_target = None
        self._play_move_drag_slot = None
        self._play_drag_xy = None
        self._play_ambiguous = []
        self._pending_move = None
        self._turn_candidates = []

    def _commit_turn(self, ta: TurnAction) -> None:
        if not self.game_state:
            return
        actor = self.game_state.current_player
        line = format_play_log_line(actor, ta)
        sr = step(self.game_state, ta)
        outcome = format_step_outcome(sr)
        if outcome:
            line = line + outcome
        self.play_log.append(line)
        if len(self.play_log) > self._log_max_lines:
            self.play_log = self.play_log[-self._log_max_lines :]
        self.game_state = sr.state
        # Aggregate damage markers per cell for this step
        by_cell: dict[tuple[int, int], int] = {}
        for ev in sr.damage_events:
            k = (ev.row, ev.col)
            by_cell[k] = by_cell.get(k, 0) + ev.amount
        for (r, c), amt in by_cell.items():
            self._damage_fx.append((r, c, amt, 1000.0))
        self.cpu_timer = 0
        self._after_step()

    def _screen_to_cell(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        play = self.phase == "play"
        if self.view_mode == ViewMode.TOP_DOWN:
            return self._screen_to_cell_top(pos, play=play)
        best: Optional[tuple[int, int]] = None
        best_d = 1e12
        tr = self._token_r_iso if play else 14
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cx, cy = self._cell_center_iso(r, c, play=play)
                d = (pos[0] - cx) ** 2 + (pos[1] - (cy - 4)) ** 2
                if d < best_d and d <= (tr + 20) ** 2:
                    best_d = d
                    best = (r, c)
        return best

    def _play_highlight(self) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]]:
        """Cells to tint on the board for the current play sub-mode."""
        empty: set[tuple[int, int]] = set()
        if (
            self.phase != "play"
            or not self.game_state
            or not self._is_human_turn()
            or not self.legal_list
        ):
            return empty, HL_MOVE
        gs = self.game_state
        if gs.pending_resurrect is not None:
            cells: set[tuple[int, int]] = set()
            for ta in self.legal_list:
                if ta.resurrect_place is not None:
                    cells.add(ta.resurrect_place)
            return cells, (255, 220, 80, 88)
        slot = self._play_slot
        if self._play_ui_mode == "action_after_move" and self.game_state is not None:
            if self._pending_move is not None:
                return (
                    _cells_with_actionable_figures(self.game_state, self._turn_candidates),
                    HL_ACTION_FIGURE,
                )
            return empty, HL_MOVE
        if self._play_ui_mode == "move_pick" and slot is not None:
            return _move_destinations_for_slot(self.legal_list, slot), HL_MOVE
        if self._play_ui_mode == "basic_pick":
            if self._pending_move is not None and slot is not None:
                return (
                    _basic_targets_for_slot_in_candidates(self._turn_candidates, slot),
                    HL_ATTACK,
                )
            if slot is not None:
                return _basic_targets_for_slot(self.legal_list, slot), HL_ATTACK
            return empty, HL_MOVE
        if self._play_ui_mode == "special_pick" and self._play_sp_id is not None:
            if self._pending_move is not None and slot is not None:
                return (
                    _special_target_cells_for_slot_candidates(
                        self._turn_candidates, slot, self._play_sp_id
                    ),
                    HL_SPECIAL,
                )
            if slot is not None:
                return (
                    _special_target_cells(self.legal_list, slot, self._play_sp_id),
                    HL_SPECIAL,
                )
            return empty, HL_MOVE
        if slot is None:
            return empty, HL_MOVE
        return empty, HL_MOVE

    def _open_piece_menu(self, mx: int, my: int, slot: int) -> None:
        legal = self.legal_list
        x = min(max(8, mx - 80), 860)
        y = min(max(72, my - 20), 480)
        self._play_popup_anchor = (x, y)
        self._play_slot = slot
        self._play_ui_mode = "piece_menu"
        h, w, gap = 28, 220, 3
        yy = y
        items: list[tuple[pygame.Rect, str, Any]] = []
        if _move_destinations_for_slot(legal, slot):
            items.append((pygame.Rect(x, yy, w, h), "move", None))
            yy += h + gap
        if _basic_targets_for_slot(legal, slot):
            items.append((pygame.Rect(x, yy, w, h), "basic", None))
            yy += h + gap
        if _special_ids_for_slot(legal, slot):
            items.append((pygame.Rect(x, yy, w, h), "special_menu", None))
            yy += h + gap
        items.append((pygame.Rect(x, yy, w, h), "cancel", None))
        self._play_menu_items = items

    def _enter_action_phase_after_move(self) -> None:
        """Move destination is fixed; user picks which figure acts (or passes) as separate steps."""
        self._play_ui_mode = "action_after_move"
        self._play_slot = None
        self._play_sp_id = None
        self._play_menu_items = []
        self._play_curse_target = None
        self._play_ambiguous = []

    def _open_action_piece_menu(self, mx: int, my: int, slot: int) -> None:
        """Action sub-step only: basic / special for this figure (after a move is already chosen)."""
        c = self._turn_candidates
        x = min(max(8, mx - 80), 860)
        y = min(max(72, my - 20), 480)
        self._play_popup_anchor = (x, y)
        self._play_slot = slot
        self._play_ui_mode = "action_piece_menu"
        h, w, gap = 28, 220, 3
        yy = y
        items: list[tuple[pygame.Rect, str, Any]] = []
        if _basic_targets_for_slot_in_candidates(c, slot):
            items.append((pygame.Rect(x, yy, w, h), "basic", None))
            yy += h + gap
        if _special_ids_for_slot_candidates(c, slot):
            items.append((pygame.Rect(x, yy, w, h), "special_menu", None))
            yy += h + gap
        items.append((pygame.Rect(x, yy, w, h), "cancel", None))
        self._play_menu_items = items

    def _finish_move_destination_pick(self, cand: list[TurnAction]) -> None:
        if not cand:
            return
        if len(cand) == 1:
            self._commit_turn(cand[0])
            return
        mv = cand[0].move
        assert mv is not None
        self._pending_move = mv
        self._turn_candidates = cand
        self._enter_action_phase_after_move()

    def _open_special_submenu(self) -> None:
        assert self._play_slot is not None
        slot = self._play_slot
        x, y = self._play_popup_anchor
        h, w, gap = 26, 240, 2
        yy = y
        items: list[tuple[pygame.Rect, str, Any]] = []
        items.append((pygame.Rect(x, yy, w, h), "back", None))
        yy += h + gap
        spec_ids = (
            _special_ids_for_slot_candidates(self._turn_candidates, slot)
            if self._pending_move is not None
            else _special_ids_for_slot(self.legal_list, slot)
        )
        for sid in spec_ids:
            label = SPECIAL_IDS[sid]
            items.append((pygame.Rect(x, yy, w, h), f"special_pick:{sid}", None))
            yy += h + gap
        self._play_menu_items = items
        self._play_ui_mode = "special_submenu"

    def _open_curse_x_menu(self, target: tuple[int, int]) -> None:
        assert self._play_slot is not None
        self._play_curse_target = target
        opts = (
            _curse_x_options_for_slot_candidates(self._turn_candidates, self._play_slot, target)
            if self._pending_move is not None
            else _curse_x_options(self.legal_list, self._play_slot, target)
        )
        x, y = self._play_popup_anchor
        h, w, gap = 26, 200, 2
        yy = y + 120
        items: list[tuple[pygame.Rect, str, Any]] = []
        for xval in opts:
            items.append((pygame.Rect(x, yy, w, h), f"curse_x:{xval}", None))
            yy += h + gap
        items.append((pygame.Rect(x, yy, w, h), "cancel", None))
        self._play_menu_items = items
        self._play_ui_mode = "curse_pick"

    def _open_anim_dead_menu(self) -> None:
        assert self._play_slot is not None
        slot = self._play_slot
        slots = (
            _animate_dead_slots_for_slot_candidates(self._turn_candidates, slot)
            if self._pending_move is not None
            else _animate_dead_slots(self.legal_list, slot)
        )
        x, y = self._play_popup_anchor
        h, w, gap = 26, 240, 2
        yy = y + 80
        items: list[tuple[pygame.Rect, str, Any]] = []
        for ds in slots:
            items.append((pygame.Rect(x, yy, w, h), f"anim_dead:{ds}", None))
            yy += h + gap
        items.append((pygame.Rect(x, yy, w, h), "cancel", None))
        self._play_menu_items = items
        self._play_ui_mode = "anim_dead_pick"

    def _open_ambiguous_menu(self, candidates: list[TurnAction]) -> None:
        self._play_ambiguous = candidates
        x, y = self._play_popup_anchor
        h, w, gap = 22, 480, 2
        yy = y + 40
        items: list[tuple[pygame.Rect, str, Any]] = []
        for i, ta in enumerate(candidates[:14]):
            line = format_turn_action(ta)
            if len(line) > 52:
                line = line[:49] + "…"
            items.append((pygame.Rect(x, yy, w, h), f"pick_turn:{i}", None))
            yy += h + gap
        items.append((pygame.Rect(x, yy, w, 160), "cancel", None))
        self._play_menu_items = items
        self._play_ui_mode = "ambiguous"

    def _resolve_or_commit(self, candidates: list[TurnAction]) -> None:
        if not candidates:
            return
        if len(candidates) == 1:
            self._commit_turn(candidates[0])
        else:
            self._open_ambiguous_menu(candidates)

    def _handle_play_menu_click(self, mx: int, my: int) -> bool:
        for rect, tag, _extra in self._play_menu_items:
            if not rect.collidepoint(mx, my):
                continue
            if tag == "cancel" or tag == "back":
                if tag == "back" and self._play_slot is not None:
                    if self._pending_move is not None:
                        if self._play_ui_mode == "special_submenu":
                            ax, ay = self._play_popup_anchor
                            self._open_action_piece_menu(ax, ay, self._play_slot)
                        else:
                            self._enter_action_phase_after_move()
                    else:
                        ax, ay = self._play_popup_anchor
                        self._open_piece_menu(ax, ay, self._play_slot)
                    return True
                if tag == "cancel" and self._pending_move is not None and self._play_ui_mode == "action_piece_menu":
                    self._enter_action_phase_after_move()
                    return True
                self._reset_play_interaction()
                return True
            if tag == "move":
                self._play_ui_mode = "move_pick"
                self._play_menu_items = []
                return True
            if tag == "basic":
                self._play_ui_mode = "basic_pick"
                self._play_menu_items = []
                return True
            if tag == "special_menu":
                self._open_special_submenu()
                return True
            if tag.startswith("special_pick:"):
                sid = int(tag.split(":")[1])
                self._play_sp_id = sid
                slot = self._play_slot or 0
                use_c = self._pending_move is not None
                if sid == int(SpecialId.ANIMATE_DEAD):
                    ads = (
                        _animate_dead_slots_for_slot_candidates(self._turn_candidates, slot)
                        if use_c
                        else _animate_dead_slots(self.legal_list, slot)
                    )
                    if len(ads) == 1:
                        menu_turns = (
                            _turns_for_special_slot_candidates(self._turn_candidates, slot, sid)
                            if use_c
                            else _turns_for_special_menu(self.legal_list, slot, sid)
                        )
                        cand = [
                            ta
                            for ta in menu_turns
                            if ta.action is not None
                            and isinstance(ta.action, ActionSpecial)
                            and ta.action.animate_dead_crew_slot == ads[0]
                        ]
                        self._resolve_or_commit(cand)
                    elif len(ads) > 1:
                        self._open_anim_dead_menu()
                    return True
                cells = (
                    _special_target_cells_for_slot_candidates(self._turn_candidates, slot, sid)
                    if use_c
                    else _special_target_cells(self.legal_list, slot, sid)
                )
                if not cells:
                    return True
                self._play_ui_mode = "special_pick"
                self._play_menu_items = []
                return True
            if tag.startswith("curse_x:"):
                xval = int(tag.split(":")[1])
                assert self._play_curse_target is not None
                pool = (
                    self._turn_candidates
                    if self._pending_move is not None
                    else self.legal_list
                )
                cand = [
                    ta
                    for ta in pool
                    if ta.action is not None
                    and isinstance(ta.action, ActionSpecial)
                    and int(ta.action.special_id) == int(SpecialId.CURSE)
                    and ta.action.target_square == self._play_curse_target
                    and ta.action.curse_x == xval
                ]
                assert self._play_slot is not None
                cand = [ta for ta in cand if ta.action is not None and ta.action.actor_slot == self._play_slot]
                self._resolve_or_commit(cand)
                return True
            if tag.startswith("anim_dead:"):
                ds = int(tag.split(":")[1])
                slot_ad = self._play_slot or 0
                cand = [
                    ta
                    for ta in (
                        _turns_for_special_slot_candidates(
                            self._turn_candidates, slot_ad, int(SpecialId.ANIMATE_DEAD)
                        )
                        if self._pending_move is not None
                        else _turns_for_special_menu(
                            self.legal_list, slot_ad, int(SpecialId.ANIMATE_DEAD)
                        )
                    )
                    if ta.action is not None
                    and isinstance(ta.action, ActionSpecial)
                    and ta.action.animate_dead_crew_slot == ds
                ]
                self._resolve_or_commit(cand)
                return True
            if tag.startswith("pick_turn:"):
                idx = int(tag.split(":")[1])
                if 0 <= idx < len(self._play_ambiguous):
                    self._commit_turn(self._play_ambiguous[idx])
                return True
        return False

    def _handle_play_board_click(self, mx: int, my: int) -> bool:
        """Return True if the click was used for board-driven play."""
        gs = self.game_state
        legal = self.legal_list
        if not gs or not legal or not self._is_human_turn():
            return False
        if gs.pending_resurrect is not None:
            cell = self._screen_to_cell((mx, my))
            if cell is None:
                return False
            ta = TurnAction(move=None, action=None, resurrect_place=cell)
            if ta in legal:
                self._commit_turn(ta)
                return True
            return False
        cell = self._screen_to_cell((mx, my))
        if cell is None:
            return False
        r, c = cell

        if self._play_ui_mode in ("idle", "action_after_move"):
            slot = _actor_slot_at_cell(gs, r, c)
            if slot is None:
                return False
            if self._pending_move is not None:
                if not _slot_can_act_after_move(self._turn_candidates, slot):
                    return False
                self._open_action_piece_menu(mx, my, slot)
                return True
            if self._play_ui_mode == "idle":
                self._open_piece_menu(mx, my, slot)
                return True
            return False

        if self._play_ui_mode == "move_pick" and self._play_slot is not None:
            dest = (r, c)
            if dest not in _move_destinations_for_slot(legal, self._play_slot):
                return False
            cand = _turns_for_move_dest(legal, self._play_slot, dest)
            self._finish_move_destination_pick(cand)
            return True

        if self._play_ui_mode == "basic_pick":
            tgt = (r, c)
            if self._pending_move is not None:
                if self._play_slot is None:
                    return False
                if tgt not in _basic_targets_for_slot_in_candidates(self._turn_candidates, self._play_slot):
                    return False
                cand = [
                    ta
                    for ta in self._turn_candidates
                    if ta.action is not None
                    and isinstance(ta.action, ActionBasicAttack)
                    and ta.action.actor_slot == self._play_slot
                    and ta.action.target_square == tgt
                ]
                self._resolve_or_commit(cand)
                return True
            if self._play_slot is None:
                return False
            if tgt not in _basic_targets_for_slot(legal, self._play_slot):
                return False
            cand = _turns_for_basic_target(legal, self._play_slot, tgt)
            self._resolve_or_commit(cand)
            return True

        if self._play_ui_mode == "special_pick" and self._play_slot is not None and self._play_sp_id is not None:
            tgt = (r, c)
            use_c = self._pending_move is not None
            slot = self._play_slot
            allowed = (
                _special_target_cells_for_slot_candidates(
                    self._turn_candidates, slot, self._play_sp_id
                )
                if use_c
                else _special_target_cells(legal, slot, self._play_sp_id)
            )
            if int(self._play_sp_id) == int(SpecialId.ANIMATE_DEAD):
                return False
            if tgt not in allowed:
                return False
            menu_turns = (
                _turns_for_special_slot_candidates(self._turn_candidates, slot, self._play_sp_id)
                if use_c
                else _turns_for_special_menu(legal, slot, self._play_sp_id)
            )
            cand = [
                ta
                for ta in menu_turns
                if ta.action is not None
                and isinstance(ta.action, ActionSpecial)
                and ta.action.target_square == tgt
            ]
            if int(self._play_sp_id) == int(SpecialId.CURSE):
                xs = (
                    _curse_x_options_for_slot_candidates(self._turn_candidates, slot, tgt)
                    if use_c
                    else _curse_x_options(legal, slot, tgt)
                )
                if len(xs) > 1:
                    self._open_curse_x_menu(tgt)
                    return True
            self._resolve_or_commit(cand)
            return True

        return False

    def _point_in_play_menu(self, mx: int, my: int) -> bool:
        for rect, _t, _e in self._play_menu_items:
            if rect.collidepoint(mx, my):
                return True
        return False

    def _is_human_turn(self) -> bool:
        assert self.game_state is not None
        if self.play_mode == PlayMode.HUMAN_HUMAN:
            return True
        if self.play_mode == PlayMode.HUMAN_CPU_A:
            return self.game_state.current_player == TEAM_PLAYER_A
        if self.play_mode == PlayMode.HUMAN_CPU_B:
            return self.game_state.current_player != TEAM_PLAYER_A
        return False

    def _is_human_setup_turn(self) -> bool:
        assert self.game_state is not None
        if self.game_state.match_phase != int(MatchPhase.SETUP):
            return False
        pl = self.game_state.setup_current_player
        if self.play_mode == PlayMode.HUMAN_HUMAN:
            return True
        if self.play_mode == PlayMode.HUMAN_CPU_A:
            return pl == TEAM_PLAYER_A
        if self.play_mode == PlayMode.HUMAN_CPU_B:
            return pl == TEAM_PLAYER_B
        return False

    def _screen_to_cell_top(self, pos: tuple[int, int], *, play: bool = False) -> Optional[tuple[int, int]]:
        mx, my = pos
        ox, oy = self._board_origin_top(play=play)
        ct = self.cell_top if play else CELL_TOP
        if mx < ox or my < oy:
            return None
        c = (mx - ox) // ct
        r = (my - oy) // ct
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return (r, c)
        return None

    @staticmethod
    def _staging_slot_rect(team: int, slot: int) -> pygame.Rect:
        ox = board_origin_top(play=False)[0]
        gap = CELL_TOP
        x = ox + slot * gap + (CELL_TOP - 44) // 2
        if team == TEAM_PLAYER_B:
            return pygame.Rect(x, 48, 44, 44)
        return pygame.Rect(x, 548, 44, 44)

    def _staging_slot_rect_play(self, team: int, slot: int) -> pygame.Rect:
        ox = self.board_ox
        gap = self.cell_top
        w = max(36, int(self.cell_top * 0.78))
        x = ox + slot * gap + (gap - w) // 2
        if team == TEAM_PLAYER_B:
            y = self.board_oy_play - w - 10
        else:
            y = self.board_oy_play + self.board_grid_px + 10
        return pygame.Rect(x, y, w, w)

    def _try_setup_placement(self, team: int, slot: int, dest: tuple[int, int]) -> None:
        assert self.game_state is not None
        placement = SetupPlacement(actor_slot=slot, destination=dest)
        if placement not in legal_setup_actions(self.game_state):
            return
        self.game_state = setup_step(self.game_state, placement).state
        self.cpu_timer = 0
        if self.game_state.match_phase == int(MatchPhase.PLAY):
            self.phase = "play"
            self._refresh_legal()

    def _step_setup_cpu(self) -> None:
        assert self.game_state is not None
        opts = legal_setup_actions(self.game_state)
        if not opts:
            return
        pl = self.rng_cpu.choice(opts)
        self.game_state = setup_step(self.game_state, pl).state
        self.cpu_timer = 0
        if self.game_state.match_phase == int(MatchPhase.PLAY):
            self.phase = "play"
            self._refresh_legal()

    def _step_cpu(self) -> None:
        assert self.game_state is not None
        legal = legal_actions(self.game_state)
        if not legal:
            return
        a = self.cpu.choose(self.game_state, legal, self.rng_cpu)
        self._commit_turn(a)

    def _after_step(self) -> None:
        if self.game_state is None:
            return
        if self.game_state.done:
            self.phase = "over"
            self.legal_list = []
        else:
            self._refresh_legal()

    def _confirm_human_action(self) -> None:
        if not self.game_state or self.game_state.done:
            return
        if not self.legal_list:
            return
        i = max(0, min(self.selected_idx, len(self.legal_list) - 1))
        a = self.legal_list[i]
        if a not in self.legal_list:
            return
        self._commit_turn(a)

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    self.win_w = max(800, event.w)
                    self.win_h = max(600, event.h)
                    self.screen = pygame.display.set_mode((self.win_w, self.win_h), pygame.RESIZABLE)
                    self._recompute_layout()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._on_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._on_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self._on_mouse_motion(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if self.phase == "play":
                        mx, my = pygame.mouse.get_pos()
                        if (
                            self._is_human_turn()
                            and self._play_show_fallback_list
                            and pygame.Rect(
                                self.sidebar_x + 4,
                                self.legal_list_top,
                                self.sidebar_w - 8,
                                self.log_bottom - self.legal_list_top - 50,
                            ).collidepoint(mx, my)
                        ):
                            self.action_scroll = max(0, self.action_scroll - event.y * 24)
                        elif self._log_clip_rect().collidepoint(mx, my):
                            self.log_scroll = max(
                                0,
                                min(
                                    self._max_log_scroll(),
                                    self.log_scroll - event.y * 24,
                                ),
                            )

            if self._damage_fx:
                new_fx: list[tuple[int, int, int, float]] = []
                for r, c, amt, rem in self._damage_fx:
                    rem -= dt
                    if rem > 0:
                        new_fx.append((r, c, amt, rem))
                self._damage_fx = new_fx

            if self.phase == "setup_place" and self.game_state and self.game_state.match_phase == int(
                MatchPhase.SETUP
            ):
                if not self._is_human_setup_turn():
                    self.cpu_timer += dt
                    if self.cpu_timer >= self.cpu_delay_ms:
                        self._step_setup_cpu()

            if self.phase == "play" and self.game_state and not self.game_state.done:
                if not self._is_human_turn():
                    self.cpu_timer += dt
                    if self.cpu_timer >= self.cpu_delay_ms:
                        self.cpu_timer = 0
                        self._step_cpu()

            self._draw()
            pygame.display.flip()

        pygame.quit()

    def _on_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            if self.phase == "menu":
                pass
            elif self.phase == "over":
                self.phase = "menu"
            elif self.phase == "play" and self._play_ui_mode != "idle":
                self._reset_play_interaction()
            else:
                self.phase = "menu"
                self._drag = None
                self._drag_pos = None
                self._reset_play_interaction()
        elif key == pygame.K_TAB and self.phase == "play":
            self._play_show_fallback_list = not self._play_show_fallback_list
        elif key == pygame.K_v and self.phase == "play":
            self.view_mode = ViewMode.ISOMETRIC if self.view_mode == ViewMode.TOP_DOWN else ViewMode.TOP_DOWN
        elif (
            self.phase == "play"
            and self._is_human_turn()
            and self.legal_list
            and self._play_show_fallback_list
        ):
            if key in (pygame.K_UP, pygame.K_k):
                self.selected_idx = max(0, self.selected_idx - 1)
            elif key in (pygame.K_DOWN, pygame.K_j):
                self.selected_idx = min(len(self.legal_list) - 1, self.selected_idx + 1)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._confirm_human_action()

    def _on_mouse_down(self, pos: tuple[int, int], button: int) -> None:
        mx, my = pos
        if self.phase == "menu":
            if self.btn_cpu_cpu.collidepoint(mx, my):
                self.play_mode = PlayMode.CPU_CPU
            elif self.btn_h_a.collidepoint(mx, my):
                self.play_mode = PlayMode.HUMAN_CPU_A
            elif self.btn_h_b.collidepoint(mx, my):
                self.play_mode = PlayMode.HUMAN_CPU_B
            elif self.btn_h_h.collidepoint(mx, my):
                self.play_mode = PlayMode.HUMAN_HUMAN
            elif self.btn_view.collidepoint(mx, my):
                self.view_mode = ViewMode.ISOMETRIC if self.view_mode == ViewMode.TOP_DOWN else ViewMode.TOP_DOWN
            elif self.btn_start.collidepoint(mx, my):
                self.reset_match()
            return

        if self.phase == "setup_coin":
            if button == 1 and self.btn_flip_coin.collidepoint(mx, my):
                self._coin_flip_winner = self.rng_cpu.randint(0, 1)
                self.phase = "setup_choice"
            return

        if self.phase == "setup_choice":
            assert self.game_state is not None and self._coin_flip_winner is not None
            if button == 1 and self.btn_setup_first.collidepoint(mx, my):
                self.game_state = begin_setup(
                    self.game_state,
                    coin_flip_winner=self._coin_flip_winner,
                    winner_chooses_first_setup=True,
                )
                self.phase = "setup_place"
                self.cpu_timer = 0
            elif button == 1 and self.btn_setup_second.collidepoint(mx, my):
                self.game_state = begin_setup(
                    self.game_state,
                    coin_flip_winner=self._coin_flip_winner,
                    winner_chooses_first_setup=False,
                )
                self.phase = "setup_place"
                self.cpu_timer = 0
            return

        if self.phase == "setup_place" and self.game_state and button == 1 and self._is_human_setup_turn():
            team = self.game_state.setup_current_player
            for sl in range(FIGURES_PER_SIDE):
                rect = self._staging_slot_rect(team, sl)
                if rect.collidepoint(mx, my):
                    u = slot_unit(self.game_state, team, sl)
                    if u is not None and u.alive and u.row < 0:
                        self._drag = (team, sl)
                        self._drag_pos = (mx, my)
            return

        if self.phase == "over":
            if button == 1 and self.btn_return_menu.collidepoint(mx, my):
                self.phase = "menu"
            return

        if self.phase == "play" and self.game_state and self._is_human_turn() and button == 1:
            mx, my = pos
            if self.btn_pass_turn.collidepoint(mx, my) and _pass_turn_in_legal(self.legal_list):
                self._commit_turn(TurnAction(move=None, action=None))
                return
            if (
                self.btn_pass_action_after_move.collidepoint(mx, my)
                and self._pending_move is not None
                and _pass_action_after_move_available(self._turn_candidates)
            ):
                cand = [ta for ta in self._turn_candidates if ta.action is None]
                self._resolve_or_commit(cand)
                return
            if self._play_show_fallback_list and mx >= self.sidebar_x:
                list_top = self.legal_list_top + 32
                list_h = self.log_bottom - self.legal_list_top - 50
                line_h = 20
                if list_top <= my < list_top + list_h - 8:
                    idx_y = my - list_top + self.action_scroll
                    if idx_y >= 0:
                        clicked = int(idx_y // line_h)
                        if 0 <= clicked < len(self.legal_list):
                            self.selected_idx = clicked
                conf = pygame.Rect(self.sidebar_x + 40, self.log_bottom - 48, 200, 36)
                if conf.collidepoint(mx, my):
                    self._confirm_human_action()
                return
            if self._play_menu_items:
                if self._handle_play_menu_click(mx, my):
                    return
                if not self._point_in_play_menu(mx, my):
                    self._reset_play_interaction()
            if self._handle_play_board_click(mx, my):
                return
            if (
                self._play_ui_mode == "move_pick"
                and self._play_slot is not None
                and self.game_state
            ):
                cell = self._screen_to_cell(pos)
                if cell:
                    sr, sc = cell
                    if _actor_slot_at_cell(self.game_state, sr, sc) == self._play_slot:
                        self._play_move_drag_slot = self._play_slot
                        self._play_drag_xy = (mx, my)
            return

    def _on_mouse_motion(self, pos: tuple[int, int]) -> None:
        mx, my = pos
        if self.phase == "menu":
            self._menu_hover_key = None
            for key, rect in (
                ("cpu", self.btn_cpu_cpu),
                ("ha", self.btn_h_a),
                ("hb", self.btn_h_b),
                ("hh", self.btn_h_h),
                ("view", self.btn_view),
                ("start", self.btn_start),
            ):
                if rect.collidepoint(mx, my):
                    self._menu_hover_key = key
                    break
        if self._drag is not None:
            self._drag_pos = pos
        elif self.phase == "play" and self._play_move_drag_slot is not None:
            self._play_drag_xy = pos

    def _on_mouse_up(self, pos: tuple[int, int], button: int) -> None:
        if button != 1:
            return
        if self.phase == "setup_place" and self._drag is not None:
            team, slot = self._drag
            self._drag = None
            self._drag_pos = None
            if not self.game_state or not self._is_human_setup_turn():
                return
            if team != self.game_state.setup_current_player:
                return
            cell = self._screen_to_cell_top(pos)
            if cell is not None:
                self._try_setup_placement(team, slot, cell)
            return
        if self.phase == "play" and self._play_move_drag_slot is not None:
            gs = self.game_state
            if gs and self._play_ui_mode == "move_pick" and self._play_slot is not None:
                cell = self._screen_to_cell(pos)
                if cell and cell in _move_destinations_for_slot(self.legal_list, self._play_slot):
                    cand = _turns_for_move_dest(self.legal_list, self._play_slot, cell)
                    self._finish_move_destination_pick(cand)
            self._play_move_drag_slot = None
            self._play_drag_xy = None

    @staticmethod
    def _play_menu_button_caption(tag: str) -> str:
        if tag == "move":
            return "Move"
        if tag == "basic":
            return "Basic attack"
        if tag == "special_menu":
            return "Special…"
        if tag == "cancel":
            return "Cancel"
        if tag == "back":
            return "Back"
        if tag.startswith("special_pick:"):
            sid = int(tag.split(":")[1])
            return SPECIAL_IDS[sid].replace("_", " ").title()
        if tag.startswith("curse_x:"):
            return f"Curse X = {tag.split(':')[1]}"
        if tag.startswith("anim_dead:"):
            return f"Raise crew slot {tag.split(':')[1]}"
        if tag.startswith("pick_turn:"):
            return ""
        return tag

    def _draw_play_menus(self) -> None:
        for rect, tag, _e in self._play_menu_items:
            pygame.draw.rect(self.screen, (55, 58, 72), rect, border_radius=4)
            pygame.draw.rect(self.screen, (120, 125, 145), rect, 1, border_radius=4)
            cap = self._play_menu_button_caption(tag)
            if tag.startswith("pick_turn:"):
                idx = int(tag.split(":")[1])
                if 0 <= idx < len(self._play_ambiguous):
                    cap = format_turn_action(self._play_ambiguous[idx])
                    if len(cap) > 48:
                        cap = cap[:45] + "…"
            if cap:
                self.screen.blit(self.font_small.render(cap, True, (235, 235, 240)), (rect.x + 8, rect.y + 6))

    def _log_clip_rect(self) -> pygame.Rect:
        if self._play_show_fallback_list:
            h = max(40, self.legal_list_top - LOG_TOP - 6)
            return pygame.Rect(self.sidebar_x, LOG_TOP, self.sidebar_w, h)
        return pygame.Rect(self.sidebar_x, LOG_TOP, self.sidebar_w, self.log_bottom - LOG_TOP)

    def _log_display_lines(self) -> list[str]:
        clip = self._log_clip_rect()
        max_w = max(40, clip.width - 12)
        out: list[str] = []
        for entry in self.play_log:
            out.extend(_wrap_text_to_width(self.font_small, entry, max_w))
        return out

    def _max_log_scroll(self) -> int:
        clip = self._log_clip_rect()
        lines = self._log_display_lines()
        content_h = len(lines) * LOG_LINE_H + 28
        return max(0, content_h - clip.height)

    @staticmethod
    def _class_combat_summary(class_id: int) -> str:
        mv = class_move_value(class_id)
        rch = class_reach_basic(class_id)
        a_move = class_basic_damage(class_id, careful_aim_not_moved=False)
        a_aim = class_basic_damage(class_id, careful_aim_not_moved=True)
        atk = f"{a_aim}" if a_move == a_aim else f"{a_aim}/{a_move}"
        sp = CLASS_SPECIALS_DISPLAY[class_id] if 0 <= class_id < len(CLASS_SPECIALS_DISPLAY) else ""
        return f"m{mv} r{rch} atk{atk} · {sp}"

    @staticmethod
    def _format_roster_line(u: Optional[UnitState], slot: int) -> str:
        if u is None:
            return f"{slot} —"
        abbr = CLASS_IDS[u.class_id][:3] if 0 <= u.class_id < len(CLASS_IDS) else "?"
        stats = MotleyCrewsUI._class_combat_summary(u.class_id)
        if not u.alive:
            return f"{slot} {abbr} dead · {stats}"
        extras: list[str] = []
        if u.containment_ticks > 0:
            extras.append(f"box={u.containment_ticks}")
        if u.moved_this_turn:
            extras.append("moved")
        if u.used_conjure_containment:
            extras.append("conj")
        if u.used_magic_bomb:
            extras.append("bomb")
        ex = f" {' '.join(extras)}" if extras else ""
        stage = "off-board" if u.row < 0 else ""
        bits = [f"{slot} {abbr}", f"{u.hp}/{u.max_hp}", stats]
        if stage:
            bits.append(stage)
        return " ".join(bits) + ex

    def _draw_flanking_rosters(self, gs: GameState) -> None:
        y_b = TOP_MARGIN + PLAY_HUD_H
        panel_b = pygame.Rect(BOARD_OX, y_b, self.board_grid_px, PANEL_B_H)
        y_a = self.board_oy_play + self.board_grid_px
        panel_a = pygame.Rect(BOARD_OX, y_a, self.board_grid_px, PANEL_A_H)
        for rect, title, team, color in (
            (panel_b, "Player B (north)", TEAM_PLAYER_B, (255, 200, 170)),
            (panel_a, "Player A (south)", TEAM_PLAYER_A, (170, 190, 255)),
        ):
            pygame.draw.rect(self.screen, (40, 42, 52), rect, border_radius=4)
            pygame.draw.rect(self.screen, (65, 68, 82), rect, 1, border_radius=4)
            self.screen.blit(self.font_small.render(title, True, color), (rect.x + 8, rect.y + 4))
            yy = rect.y + 22
            for sl in range(FIGURES_PER_SIDE):
                u = slot_unit(gs, team, sl)
                line = self._format_roster_line(u, sl)
                if len(line) > 54:
                    line = line[:51] + "…"
                self.screen.blit(self.font_small.render(line, True, (200, 202, 215)), (rect.x + 8, yy))
                yy += 14

    def _draw_play_log(self) -> None:
        clip = self._log_clip_rect()
        prev = self.screen.get_clip()
        self.screen.set_clip(clip)
        pygame.draw.rect(self.screen, (28, 30, 38), clip, border_radius=4)
        pygame.draw.rect(self.screen, (55, 58, 70), clip, 1, border_radius=4)
        self.screen.blit(self.font_small.render("Event log", True, (160, 165, 185)), (clip.x + 8, clip.y + 4))
        y = clip.y + 22 - self.log_scroll
        for line in self._log_display_lines():
            if y + LOG_LINE_H >= clip.y and y <= clip.y + clip.height:
                surf = self.font_small.render(line, True, (175, 178, 195))
                self.screen.blit(surf, (clip.x + 6, y))
            y += LOG_LINE_H
        self.screen.set_clip(prev)

    def _draw(self) -> None:
        self.screen.fill((34, 36, 42))
        if self.phase == "menu":
            self._draw_menu()
        elif self.phase == "setup_coin":
            self._draw_setup_coin()
        elif self.phase == "setup_choice":
            self._draw_setup_choice()
        elif self.phase == "setup_place":
            self._draw_setup_place()
        elif self.phase == "play":
            self._draw_play()
        elif self.phase == "over":
            self._draw_play()
            self._draw_game_over_modal()

    def _draw_menu(self) -> None:
        w, h = self.win_w, self.win_h
        tx = max(48, int(w * 0.05))
        ty = int(h * 0.07)
        t = self.font.render("Motley Crews", True, (240, 240, 245))
        self.screen.blit(t, (tx, ty))
        menu_rows = [
            ("cpu", self.btn_cpu_cpu, "CPU vs CPU"),
            ("ha", self.btn_h_a, "1P vs CPU (you: Player A)"),
            ("hb", self.btn_h_b, "1P vs CPU (you: Player B)"),
            ("hh", self.btn_h_h, "1P vs 2P (hotseat)"),
        ]
        for key, rect, text in menu_rows:
            hover = self._menu_hover_key == key
            bg = (95, 100, 130) if hover else (70, 74, 88)
            pygame.draw.rect(self.screen, bg, rect, border_radius=6)
            if hover:
                pygame.draw.rect(self.screen, (160, 185, 240), rect, 2, border_radius=6)
            self.screen.blit(self.font.render(text, True, (230, 230, 235)), (rect.x + 12, rect.y + 10))
        vm = "Isometric" if self.view_mode == ViewMode.ISOMETRIC else "Top-down"
        vkey = "view"
        vhover = self._menu_hover_key == vkey
        vbg = (105, 110, 135) if vhover else (86, 90, 108)
        pygame.draw.rect(self.screen, vbg, self.btn_view, border_radius=6)
        if vhover:
            pygame.draw.rect(self.screen, (160, 185, 240), self.btn_view, 2, border_radius=6)
        self.screen.blit(self.font.render(f"Board view: {vm} (toggle)", True, (220, 220, 230)), (self.btn_view.x + 12, self.btn_view.y + 8))
        shover = self._menu_hover_key == "start"
        sbg = (130, 150, 210) if shover else (110, 130, 180)
        pygame.draw.rect(self.screen, sbg, self.btn_start, border_radius=6)
        if shover:
            pygame.draw.rect(self.screen, (200, 215, 255), self.btn_start, 2, border_radius=6)
        self.screen.blit(self.font.render("Start", True, (255, 255, 255)), (self.btn_start.x + 72, self.btn_start.y + 12))
        hint = self.font_small.render(
            "Human modes: coin toss → setup choice → drag crew onto zones. In-game: V toggles view. Esc = menu.",
            True,
            (160, 165, 180),
        )
        self.screen.blit(hint, (tx, int(h * 0.88)))

    def _draw_setup_coin(self) -> None:
        self.screen.blit(self.font.render("Coin toss", True, (240, 240, 245)), (80, 60))
        self.screen.blit(
            self.font_small.render(
                "Flip a coin. The winner chooses whether to set up first (and go first) or second (and go second).",
                True,
                (180, 185, 200),
            ),
            (80, 100),
        )
        pygame.draw.rect(self.screen, (110, 130, 180), self.btn_flip_coin, border_radius=8)
        self.screen.blit(
            self.font.render("Flip coin", True, (255, 255, 255)),
            (self.btn_flip_coin.x + 88, self.btn_flip_coin.y + 12),
        )
        self.screen.blit(self.font_small.render("Esc = menu", True, (140, 145, 160)), (80, 520))

    def _draw_setup_choice(self) -> None:
        assert self._coin_flip_winner is not None
        w = self._coin_flip_winner
        self.screen.blit(self.font.render(f"{player_label(w)} won the toss", True, (240, 240, 245)), (80, 60))
        self.screen.blit(
            self.font_small.render(
                "Choose how to pair setup order with the first move of the game:",
                True,
                (200, 200, 210),
            ),
            (80, 100),
        )
        pygame.draw.rect(self.screen, (86, 120, 100), self.btn_setup_first, border_radius=8)
        pygame.draw.rect(self.screen, (100, 86, 120), self.btn_setup_second, border_radius=8)
        self.screen.blit(
            self.font_small.render("Set up first & take the first turn", True, (255, 255, 255)),
            (self.btn_setup_first.x + 28, self.btn_setup_first.y + 14),
        )
        self.screen.blit(
            self.font_small.render("Set up second & take the second turn", True, (255, 255, 255)),
            (self.btn_setup_second.x + 20, self.btn_setup_second.y + 14),
        )
        self.screen.blit(self.font_small.render("Esc = menu", True, (140, 145, 160)), (80, 520))

    def _blit_token_top(
        self,
        cx: int,
        cy: int,
        class_id: int,
        team: int,
        radius: int = 18,
        *,
        view: ViewMode = ViewMode.TOP_DOWN,
        draw_marker: bool = True,
    ) -> None:
        col = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        pygame.draw.circle(self.screen, col, (cx, cy), radius)
        pygame.draw.circle(self.screen, (20, 20, 30), (cx, cy), radius, 2)
        label = CLASS_IDS[class_id][0].upper() if 0 <= class_id < len(CLASS_IDS) else "?"
        tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
        self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 8))
        ring = (120, 160, 255) if team == TEAM_PLAYER_A else (255, 160, 120)
        pygame.draw.circle(self.screen, ring, (cx, cy), radius + 3, 2)
        if draw_marker:
            _draw_team_marker(self.screen, cx, cy, team, view=view, radius=radius)

    def _draw_setup_place(self) -> None:
        assert self.game_state is not None
        gs = self.game_state
        obs = to_structured_observation(gs)

        self.screen.blit(self.font.render("Place your crew (drag from staging to your zone)", True, (230, 230, 235)), (32, 8))
        self.screen.blit(
            self.font_small.render("Top-down only during setup  ·  Rows 0–1 and 6–7 are deployment zones", True, (150, 155, 170)),
            (32, 32),
        )
        self.screen.blit(self.font_small.render("Player B — off board", True, (255, 200, 170)), (32, 52))
        self.screen.blit(self.font_small.render("Player A — off board", True, (170, 190, 255)), (32, 528))

        legal_cells: set[tuple[int, int]] = set()
        if self._is_human_setup_turn():
            for p in legal_setup_actions(gs):
                legal_cells.add(p.destination)

        ox, oy = board_origin_top()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tr = int(obs.terrain[r, c])
                color = list(TERRAIN_COLOR.get(tr, (200, 200, 200)))
                if r in DEPLOY_ROWS_PLAYER_B:
                    color = [min(255, x + 12) for x in color]
                elif r in DEPLOY_ROWS_PLAYER_A:
                    color = [min(255, x + 12) for x in color]
                rect = pygame.Rect(ox + c * CELL_TOP, oy + r * CELL_TOP, CELL_TOP, CELL_TOP)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (55, 58, 70), rect, 1)
                if (r, c) in legal_cells:
                    hl = pygame.Surface((CELL_TOP, CELL_TOP), pygame.SRCALPHA)
                    hl.fill((255, 220, 80, 55))
                    self.screen.blit(hl, (rect.x, rect.y))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if float(obs.occupancy[r, c]) < 0.5:
                    continue
                cx, cy = cell_center_top(r, c)
                cid = int(obs.unit_class[r, c])
                tid = int(obs.team[r, c])
                self._blit_token_top(cx, cy, cid, tid)

        for team in (TEAM_PLAYER_B, TEAM_PLAYER_A):
            for sl in range(FIGURES_PER_SIDE):
                u = slot_unit(gs, team, sl)
                if u is None or not u.alive or u.row >= 0:
                    continue
                rect = self._staging_slot_rect(team, sl)
                cx, cy = rect.centerx, rect.centery
                self._blit_token_top(cx, cy, u.class_id, team, radius=16)

        if self._drag is not None and self._drag_pos is not None:
            team_d, slot_d = self._drag
            u = slot_unit(gs, team_d, slot_d)
            if u is not None:
                self._blit_token_top(self._drag_pos[0], self._drag_pos[1], u.class_id, team_d, radius=16)

        pl = gs.setup_current_player
        hud = f"Place figures — {player_label(pl)}'s turn  |  seed {self.seed}"
        if self._is_human_setup_turn():
            hud += "  [drag a piece to a yellow square]"
        else:
            hud += "  [CPU placing…]"
        self.screen.blit(self.font.render(hud, True, (220, 220, 230)), (32, 600))

    def _draw_game_over_modal(self) -> None:
        assert self.game_state is not None
        s = self.game_state
        w = s.winner
        msg = f"Winner: {player_label(w)}" if w is not None else "Draw"
        overlay = pygame.Surface((self.win_w, self.win_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 125))
        self.screen.blit(overlay, (0, 0))
        cx = self.win_w // 2
        cy = int(self.win_h * 0.36)
        tw = self.font.size(msg)[0]
        self.screen.blit(self.font.render(msg, True, (255, 255, 255)), (cx - tw // 2, cy))
        score_t = f"Scores: {s.score[0]} — {s.score[1]}"
        sw = self.font.size(score_t)[0]
        self.screen.blit(self.font.render(score_t, True, (220, 220, 230)), (cx - sw // 2, cy + 42))
        pygame.draw.rect(self.screen, (90, 130, 200), self.btn_return_menu, border_radius=8)
        self.screen.blit(
            self.font.render("Return to menu", True, (255, 255, 255)),
            (self.btn_return_menu.x + 42, self.btn_return_menu.y + 12),
        )

    def _draw_damage_fx(self) -> None:
        for r, c, amt, rem in self._damage_fx:
            alpha = min(255, int(255 * (rem / 1000.0)))
            col = (255, min(255, 70 + (255 - alpha) // 4), 60)
            if self.view_mode == ViewMode.TOP_DOWN:
                cx, cy = self._cell_center_top(r, c, play=True)
            else:
                cx, cy = self._cell_center_iso(r, c, play=True)
                cy -= 4
            txt = self.font.render(f"−{amt}", True, col)
            txt.set_alpha(alpha)
            self.screen.blit(txt, (cx - txt.get_width() // 2, cy - 26))

    def _draw_play(self) -> None:
        assert self.game_state is not None
        gs = self.game_state
        obs = to_structured_observation(gs)
        hl_cells, hl_rgba = self._play_highlight()
        self._draw_flanking_rosters(gs)
        if self.view_mode == ViewMode.TOP_DOWN:
            self._draw_board_top(obs, highlight_cells=hl_cells, hl_rgba=hl_rgba)
        else:
            self._draw_board_iso(obs, highlight_cells=hl_cells, hl_rgba=hl_rgba)

        self._draw_damage_fx()

        if gs.pending_resurrect is not None:
            team, slot = gs.pending_resurrect
            u = slot_unit(gs, team, slot)
            if u is not None and u.alive and u.row < 0:
                rect = self._staging_slot_rect_play(team, slot)
                cx, cy = rect.centerx, rect.centery
                self._blit_token_top(
                    cx,
                    cy,
                    u.class_id,
                    team,
                    radius=max(12, self._token_r_top - 2),
                    view=ViewMode.TOP_DOWN,
                )
                hy = self.board_oy_play - 26 if team == TEAM_PLAYER_B else self.board_oy_play + self.board_grid_px + 4
                self.screen.blit(
                    self.font_small.render("Revived — place in a highlighted square in your start zone", True, (255, 215, 130)),
                    (self.board_ox, hy),
                )

        if self._is_human_turn() and self._play_slot is not None:
            u = unit_at(gs, gs.current_player, self._play_slot)
            if u is not None and u.alive:
                if self.view_mode == ViewMode.TOP_DOWN:
                    cx, cy = self._cell_center_top(u.row, u.col, play=True)
                    pr = max(18, int(24 * self.cell_top / max(1, CELL_TOP)))
                    pygame.draw.circle(self.screen, (255, 220, 60), (cx, cy), pr, 3)
                else:
                    cx, cy = self._cell_center_iso(u.row, u.col, play=True)
                    pr = max(16, int(20 * self._token_r_iso / max(1, 14)))
                    pygame.draw.circle(self.screen, (255, 220, 60), (cx, cy - 4), pr, 3)

        if (
            self._is_human_turn()
            and self._play_drag_xy is not None
            and self._play_move_drag_slot is not None
        ):
            u = unit_at(gs, gs.current_player, self._play_move_drag_slot)
            if u is not None:
                self._blit_token_top(
                    self._play_drag_xy[0],
                    self._play_drag_xy[1],
                    u.class_id,
                    gs.current_player,
                    radius=max(12, self._token_r_top - 2),
                    view=self.view_mode,
                )

        self._draw_play_menus()

        flip = (
            f"coin {player_label(gs.coin_flip_winner)}  "
            if gs.coin_flip_winner is not None
            else ""
        )
        hud = f"{player_label(gs.current_player)} turn  |  Score {gs.score[0]} — {gs.score[1]}  |  {flip}seed {self.seed}"
        if self._is_human_turn():
            hud += "  [your turn]"
        self.screen.blit(self.font.render(hud, True, (220, 220, 230)), (BOARD_OX, TOP_MARGIN))
        vm = "iso" if self.view_mode == ViewMode.ISOMETRIC else "top"
        self.screen.blit(
            self.font_small.render(f"{vm} view · V  ·  Tab = all legal turns", True, (150, 155, 170)),
            (self.sidebar_x + 8, TOP_MARGIN + 4),
        )

        self._draw_play_log()

        if gs.pending_resurrect is not None and self._is_human_turn():
            play_hint = "Click a highlighted square in your start zone to place the revived figure."
        elif self._pending_move is not None:
            play_hint = "Move set — teal = who may act. Pass or Esc cancels."
        else:
            play_hint = "Click piece → Move / Attack / Special. After move, pick who acts. Esc = cancel."
        self.screen.blit(self.font_small.render(play_hint, True, (130, 135, 155)), (BOARD_OX, TOP_MARGIN + 22))

        if self._is_human_turn() and _pass_turn_in_legal(self.legal_list):
            pygame.draw.rect(self.screen, (70, 90, 70), self.btn_pass_turn, border_radius=4)
            self.screen.blit(
                self.font_small.render("Pass turn (skip both)", True, (230, 245, 230)),
                (self.btn_pass_turn.x + 10, self.btn_pass_turn.y + 5),
            )

        if (
            self._is_human_turn()
            and self._pending_move is not None
            and _pass_action_after_move_available(self._turn_candidates)
        ):
            pygame.draw.rect(self.screen, (75, 95, 115), self.btn_pass_action_after_move, border_radius=4)
            self.screen.blit(
                self.font_small.render("Pass action (move only)", True, (220, 228, 238)),
                (self.btn_pass_action_after_move.x + 8, self.btn_pass_action_after_move.y + 5),
            )

        if self._is_human_turn() and not self.legal_list and not gs.done:
            self.screen.blit(
                self.font.render("No legal actions (stalemate)", True, (255, 160, 120)),
                (self.sidebar_x + 8, self.win_h // 2),
            )

        pygame.draw.line(self.screen, (60, 64, 78), (self.sidebar_x, 0), (self.sidebar_x, self.win_h), 2)

        if self._is_human_turn() and self.legal_list and self._play_show_fallback_list:
            x0 = self.sidebar_x
            list_top = self.legal_list_top
            list_h = self.log_bottom - list_top - 50
            list_rect = pygame.Rect(x0 + 4, list_top, self.sidebar_w - 8, list_h)
            pygame.draw.rect(self.screen, (32, 34, 44), list_rect, border_radius=4)
            pygame.draw.rect(self.screen, (75, 80, 100), list_rect, 1, border_radius=4)
            title = self.font.render("All legal turns (Tab to hide)", True, (200, 205, 220))
            self.screen.blit(title, (x0 + 12, list_top + 6))
            y = list_top + 32 - self.action_scroll
            line_h = 20
            for i, act in enumerate(self.legal_list):
                if y > list_top + list_h - 8:
                    break
                if y + line_h >= list_top + 28:
                    sel = i == self.selected_idx
                    col = (80, 120, 200) if sel else (200, 200, 210)
                    prefix = "› " if sel else "  "
                    line = prefix + format_turn_action(act)
                    if len(line) > 52:
                        line = line[:49] + "…"
                    surf = self.font_small.render(f"{i+1}. {line}", True, col)
                    self.screen.blit(surf, (x0 + 8, y))
                y += line_h
            conf = pygame.Rect(self.sidebar_x + 40, self.log_bottom - 48, 200, 36)
            pygame.draw.rect(self.screen, (100, 140, 200), conf, border_radius=4)
            self.screen.blit(self.font.render("Confirm (Enter)", True, (255, 255, 255)), (conf.x + 20, conf.y + 6))

    def _draw_board_top(
        self,
        obs,
        *,
        highlight_cells: Optional[set[tuple[int, int]]] = None,
        hl_rgba: tuple[int, int, int, int] = HL_MOVE,
    ) -> None:
        hl = highlight_cells or set()
        ct = self.cell_top
        tradius = self._token_r_top
        ox, oy = self._board_origin_top(play=True)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tr = int(obs.terrain[r, c])
                base = TERRAIN_COLOR.get(tr, (200, 200, 200))
                color = _checkerboard_tint(base, r, c, tr)
                rect = pygame.Rect(ox + c * ct, oy + r * ct, ct, ct)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (55, 58, 70), rect, 1)
                if (r, c) in hl:
                    hls = pygame.Surface((ct, ct), pygame.SRCALPHA)
                    hls.fill(hl_rgba)
                    self.screen.blit(hls, (rect.x, rect.y))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if float(obs.occupancy[r, c]) < 0.5:
                    continue
                tid = int(obs.team[r, c])
                cid = int(obs.unit_class[r, c])
                col = CLASS_COLORS[cid % len(CLASS_COLORS)] if tid >= 0 else (150, 150, 150)
                cx, cy = self._cell_center_top(r, c, play=True)
                pygame.draw.circle(self.screen, col, (cx, cy), tradius)
                pygame.draw.circle(self.screen, (20, 20, 30), (cx, cy), tradius, 2)
                label = CLASS_IDS[cid][0].upper() if 0 <= cid < len(CLASS_IDS) else "?"
                tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
                self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 8))
                ring = (120, 160, 255) if tid == TEAM_PLAYER_A else (255, 160, 120)
                pygame.draw.circle(self.screen, ring, (cx, cy), tradius + 3, 2)
                if tid >= 0:
                    _draw_team_marker(self.screen, cx, cy, tid, view=ViewMode.TOP_DOWN, radius=tradius)

    def _draw_board_iso(
        self,
        obs,
        *,
        highlight_cells: Optional[set[tuple[int, int]]] = None,
        hl_rgba: tuple[int, int, int, int] = HL_MOVE,
    ) -> None:
        hl = highlight_cells or set()
        # painter: lower screen rows first (higher r+c)
        cells: list[tuple[int, int, int, int]] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cells.append((r + c, c, r, int(obs.terrain[r, c])))
        cells.sort(key=lambda x: (x[0], x[2]))
        tw, th = self.tw_iso, self.th_iso
        ir = self._token_r_iso
        for _, c, r, tr in cells:
            cx, cy = self._cell_center_iso(r, c, play=True)
            base = TERRAIN_COLOR.get(tr, (200, 200, 200))
            color = _checkerboard_tint(base, r, c, tr)
            pts = [
                (cx, cy - th // 2),
                (cx + tw // 2, cy),
                (cx, cy + th // 2),
                (cx - tw // 2, cy),
            ]
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, (55, 58, 70), pts, 1)
            if (r, c) in hl:
                pygame.draw.circle(self.screen, hl_rgba[:3], (cx, cy - 2), ir + 8, 3)
        unit_cells: list[tuple[int, int]] = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if float(obs.occupancy[r, c]) >= 0.5:
                    unit_cells.append((r, c))
        unit_cells.sort(key=lambda rc: (rc[0] + rc[1], rc[0]))
        for r, c in unit_cells:
            tid = int(obs.team[r, c])
            cid = int(obs.unit_class[r, c])
            col = CLASS_COLORS[cid % len(CLASS_COLORS)] if tid >= 0 else (150, 150, 150)
            cx, cy = self._cell_center_iso(r, c, play=True)
            ty = cy - 4
            pygame.draw.circle(self.screen, col, (cx, ty), ir)
            pygame.draw.circle(self.screen, (20, 20, 30), (cx, ty), ir, 2)
            label = CLASS_IDS[cid][0].upper() if 0 <= cid < len(CLASS_IDS) else "?"
            tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
            self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 14))
            ring = (120, 160, 255) if tid == TEAM_PLAYER_A else (255, 160, 120)
            pygame.draw.circle(self.screen, ring, (cx, ty), ir + 3, 2)
            if tid >= 0:
                _draw_team_marker(self.screen, cx, ty, tid, view=ViewMode.ISOMETRIC, radius=ir)


def run(seed: int = 0) -> None:
    MotleyCrewsUI(seed=seed).run()


if __name__ == "__main__":
    run()

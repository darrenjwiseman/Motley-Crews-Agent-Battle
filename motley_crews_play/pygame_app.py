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
    SetupPlacement,
    SpecialId,
    TurnAction,
)
from motley_crews_env.state import GameState, slot_unit, unit_at
from motley_crews_play.formatting import format_turn_action
from motley_crews_play.policies import ScriptedHeuristicPolicy


class ViewMode(IntEnum):
    TOP_DOWN = 0
    ISOMETRIC = 1


class PlayMode(IntEnum):
    CPU_CPU = 0
    HUMAN_CPU_A = 1  # human = player A (0)
    HUMAN_CPU_B = 2  # human = player B (1)
    HUMAN_HUMAN = 3


# --- projection -----------------------------------------------------------------

CELL_TOP = 56
TW_ISO = 52
TH_ISO = 28


def board_origin_top() -> tuple[int, int]:
    return 32, 96


def board_origin_iso() -> tuple[int, int]:
    return 120, 200


def cell_center_top(row: int, col: int) -> tuple[int, int]:
    ox, oy = board_origin_top()
    cx = ox + col * CELL_TOP + CELL_TOP // 2
    cy = oy + row * CELL_TOP + CELL_TOP // 2
    return cx, cy


def cell_center_iso(row: int, col: int) -> tuple[int, int]:
    ox, oy = board_origin_iso()
    cx = ox + (col - row) * (TW_ISO // 2)
    cy = oy + (col + row) * (TH_ISO // 2)
    return cx, cy


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


def _actor_slot_at_cell(gs: GameState, r: int, c: int) -> Optional[int]:
    for sl in range(FIGURES_PER_SIDE):
        u = unit_at(gs, gs.current_player, sl)
        if u is not None and u.alive and u.row == r and u.col == c:
            return sl
    return None


def _pass_turn_in_legal(legal: list[TurnAction]) -> bool:
    return TurnAction(move=None, action=None) in legal


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


class MotleyCrewsUI:
    def __init__(self, seed: int = 0) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((1040, 720))
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
        self.btn_pass_turn = pygame.Rect(708, 26, 150, 26)

        # menu buttons: (rect, mode or "start" or view toggle)
        self._layout_menu_buttons()
        self._layout_setup_buttons()

    def _layout_menu_buttons(self) -> None:
        self.btn_cpu_cpu = pygame.Rect(80, 140, 280, 40)
        self.btn_h_a = pygame.Rect(80, 190, 280, 40)
        self.btn_h_b = pygame.Rect(80, 240, 280, 40)
        self.btn_h_h = pygame.Rect(80, 290, 280, 40)
        self.btn_view = pygame.Rect(80, 360, 280, 36)
        self.btn_start = pygame.Rect(80, 430, 200, 44)

    def _layout_setup_buttons(self) -> None:
        self.btn_flip_coin = pygame.Rect(380, 320, 280, 48)
        self.btn_setup_first = pygame.Rect(220, 380, 300, 44)
        self.btn_setup_second = pygame.Rect(540, 380, 300, 44)

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

    def _commit_turn(self, ta: TurnAction) -> None:
        if not self.game_state:
            return
        self.game_state = step(self.game_state, ta).state
        self.cpu_timer = 0
        self._after_step()

    def _screen_to_cell(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        if self.view_mode == ViewMode.TOP_DOWN:
            return self._screen_to_cell_top(pos)
        best: Optional[tuple[int, int]] = None
        best_d = 1e12
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cx, cy = cell_center_iso(r, c)
                d = (pos[0] - cx) ** 2 + (pos[1] - (cy - 4)) ** 2
                if d < best_d and d <= 34 * 34:
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
        slot = self._play_slot
        if slot is None:
            return empty, HL_MOVE
        if self._play_ui_mode == "move_pick":
            return _move_destinations_for_slot(self.legal_list, slot), HL_MOVE
        if self._play_ui_mode == "basic_pick":
            return _basic_targets_for_slot(self.legal_list, slot), HL_ATTACK
        if self._play_ui_mode == "special_pick" and self._play_sp_id is not None:
            return (
                _special_target_cells(self.legal_list, slot, self._play_sp_id),
                HL_SPECIAL,
            )
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

    def _open_special_submenu(self) -> None:
        assert self._play_slot is not None
        slot = self._play_slot
        x, y = self._play_popup_anchor
        h, w, gap = 26, 240, 2
        yy = y
        items: list[tuple[pygame.Rect, str, Any]] = []
        items.append((pygame.Rect(x, yy, w, h), "back", None))
        yy += h + gap
        for sid in _special_ids_for_slot(self.legal_list, slot):
            label = SPECIAL_IDS[sid]
            items.append((pygame.Rect(x, yy, w, h), f"special_pick:{sid}", None))
            yy += h + gap
        self._play_menu_items = items
        self._play_ui_mode = "special_submenu"

    def _open_curse_x_menu(self, target: tuple[int, int]) -> None:
        assert self._play_slot is not None
        self._play_curse_target = target
        opts = _curse_x_options(self.legal_list, self._play_slot, target)
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
        slots = _animate_dead_slots(self.legal_list, slot)
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
                    ax, ay = self._play_popup_anchor
                    self._open_piece_menu(ax, ay, self._play_slot)
                else:
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
                if sid == int(SpecialId.ANIMATE_DEAD):
                    ads = _animate_dead_slots(self.legal_list, self._play_slot or 0)
                    if len(ads) == 1:
                        cand = [
                            ta
                            for ta in _turns_for_special_menu(
                                self.legal_list, self._play_slot or 0, sid
                            )
                            if ta.action is not None
                            and isinstance(ta.action, ActionSpecial)
                            and ta.action.animate_dead_crew_slot == ads[0]
                        ]
                        self._resolve_or_commit(cand)
                    elif len(ads) > 1:
                        self._open_anim_dead_menu()
                    return True
                cells = _special_target_cells(self.legal_list, self._play_slot or 0, sid)
                if not cells:
                    return True
                self._play_ui_mode = "special_pick"
                self._play_menu_items = []
                return True
            if tag.startswith("curse_x:"):
                xval = int(tag.split(":")[1])
                assert self._play_slot is not None and self._play_curse_target is not None
                cand = [
                    ta
                    for ta in self.legal_list
                    if ta.action is not None
                    and isinstance(ta.action, ActionSpecial)
                    and int(ta.action.special_id) == int(SpecialId.CURSE)
                    and ta.action.actor_slot == self._play_slot
                    and ta.action.target_square == self._play_curse_target
                    and ta.action.curse_x == xval
                ]
                self._resolve_or_commit(cand)
                return True
            if tag.startswith("anim_dead:"):
                ds = int(tag.split(":")[1])
                cand = [
                    ta
                    for ta in _turns_for_special_menu(
                        self.legal_list, self._play_slot or 0, int(SpecialId.ANIMATE_DEAD)
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
        cell = self._screen_to_cell((mx, my))
        if cell is None:
            return False
        r, c = cell

        if self._play_ui_mode == "idle":
            slot = _actor_slot_at_cell(gs, r, c)
            if slot is not None:
                self._open_piece_menu(mx, my, slot)
                return True
            return False

        if self._play_ui_mode == "move_pick" and self._play_slot is not None:
            dest = (r, c)
            if dest not in _move_destinations_for_slot(legal, self._play_slot):
                return False
            cand = _turns_for_move_dest(legal, self._play_slot, dest)
            self._resolve_or_commit(cand)
            return True

        if self._play_ui_mode == "basic_pick" and self._play_slot is not None:
            tgt = (r, c)
            if tgt not in _basic_targets_for_slot(legal, self._play_slot):
                return False
            cand = _turns_for_basic_target(legal, self._play_slot, tgt)
            self._resolve_or_commit(cand)
            return True

        if self._play_ui_mode == "special_pick" and self._play_slot is not None and self._play_sp_id is not None:
            tgt = (r, c)
            allowed = _special_target_cells(legal, self._play_slot, self._play_sp_id)
            if int(self._play_sp_id) == int(SpecialId.ANIMATE_DEAD):
                return False
            if tgt not in allowed:
                return False
            cand = _turns_for_special_menu(legal, self._play_slot, self._play_sp_id)
            cand = [
                ta
                for ta in cand
                if ta.action is not None
                and isinstance(ta.action, ActionSpecial)
                and ta.action.target_square == tgt
            ]
            if int(self._play_sp_id) == int(SpecialId.CURSE):
                xs = _curse_x_options(legal, self._play_slot, tgt)
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

    @staticmethod
    def _screen_to_cell_top(pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        mx, my = pos
        ox, oy = board_origin_top()
        if mx < ox or my < oy:
            return None
        c = (mx - ox) // CELL_TOP
        r = (my - oy) // CELL_TOP
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            return (r, c)
        return None

    @staticmethod
    def _staging_slot_rect(team: int, slot: int) -> pygame.Rect:
        ox = board_origin_top()[0]
        gap = CELL_TOP
        x = ox + slot * gap + (CELL_TOP - 44) // 2
        if team == TEAM_PLAYER_B:
            return pygame.Rect(x, 48, 44, 44)
        return pygame.Rect(x, 548, 44, 44)

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
        self.game_state = step(self.game_state, a).state
        self.cpu_timer = 0
        self._after_step()

    def _after_step(self) -> None:
        if self.game_state is None:
            return
        if self.game_state.done:
            self.phase = "over"
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
        self.game_state = step(self.game_state, a).state
        self.cpu_timer = 0
        self._after_step()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._on_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._on_mouse_down(event.pos, event.button)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._on_mouse_up(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self._on_mouse_motion(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if (
                        self.phase == "play"
                        and self._is_human_turn()
                        and self._play_show_fallback_list
                    ):
                        self.action_scroll = max(0, self.action_scroll - event.y * 24)

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
            self.phase = "menu"
            return

        if self.phase == "play" and self.game_state and self._is_human_turn() and button == 1:
            mx, my = pos
            if self.btn_pass_turn.collidepoint(mx, my) and _pass_turn_in_legal(self.legal_list):
                self._commit_turn(TurnAction(move=None, action=None))
                return
            if self._play_show_fallback_list and mx >= 700:
                sidebar_top = 96
                line_h = 20
                idx_y = my - sidebar_top + self.action_scroll
                if idx_y >= 0:
                    clicked = idx_y // line_h
                    if 0 <= clicked < len(self.legal_list):
                        self.selected_idx = clicked
                if pygame.Rect(720, 640, 200, 36).collidepoint(mx, my):
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
                    self._resolve_or_commit(cand)
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
        else:
            self._draw_game_over()

    def _draw_menu(self) -> None:
        t = self.font.render("Motley Crews", True, (240, 240, 245))
        self.screen.blit(t, (80, 60))
        labels = [
            (self.btn_cpu_cpu, "CPU vs CPU"),
            (self.btn_h_a, "1P vs CPU (you: Player A)"),
            (self.btn_h_b, "1P vs CPU (you: Player B)"),
            (self.btn_h_h, "1P vs 2P (hotseat)"),
        ]
        for rect, text in labels:
            pygame.draw.rect(self.screen, (70, 74, 88), rect, border_radius=6)
            self.screen.blit(self.font.render(text, True, (230, 230, 235)), (rect.x + 12, rect.y + 10))
        vm = "Isometric" if self.view_mode == ViewMode.ISOMETRIC else "Top-down"
        pygame.draw.rect(self.screen, (86, 90, 108), self.btn_view, border_radius=6)
        self.screen.blit(self.font.render(f"Board view: {vm} (toggle)", True, (220, 220, 230)), (self.btn_view.x + 12, self.btn_view.y + 8))
        pygame.draw.rect(self.screen, (110, 130, 180), self.btn_start, border_radius=6)
        self.screen.blit(self.font.render("Start", True, (255, 255, 255)), (self.btn_start.x + 72, self.btn_start.y + 12))
        hint = self.font_small.render(
            "Human modes: coin toss → setup choice → drag crew onto zones. In-game: V toggles view. Esc = menu.",
            True,
            (160, 165, 180),
        )
        self.screen.blit(hint, (80, 500))

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
        self.screen.blit(self.font.render(f"Player {w} won the toss", True, (240, 240, 245)), (80, 60))
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

    def _blit_token_top(self, cx: int, cy: int, class_id: int, team: int, radius: int = 18) -> None:
        col = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        pygame.draw.circle(self.screen, col, (cx, cy), radius)
        pygame.draw.circle(self.screen, (20, 20, 30), (cx, cy), radius, 2)
        label = CLASS_IDS[class_id][0].upper() if 0 <= class_id < len(CLASS_IDS) else "?"
        tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
        self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 8))
        ring = (120, 160, 255) if team == TEAM_PLAYER_A else (255, 160, 120)
        pygame.draw.circle(self.screen, ring, (cx, cy), radius + 3, 2)

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
        hud = f"Place figures — Player {pl}'s turn  |  seed {self.seed}"
        if self._is_human_setup_turn():
            hud += "  [drag a piece to a yellow square]"
        else:
            hud += "  [CPU placing…]"
        self.screen.blit(self.font.render(hud, True, (220, 220, 230)), (32, 600))

    def _draw_game_over(self) -> None:
        assert self.game_state is not None
        s = self.game_state
        w = s.winner
        msg = f"Winner: Player {w}" if w is not None else "Draw"
        overlay = pygame.Surface((1040, 720), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))
        self.screen.blit(self.font.render(msg, True, (255, 255, 255)), (400, 300))
        self.screen.blit(self.font.render(f"Scores: {s.score[0]} — {s.score[1]}", True, (220, 220, 230)), (380, 340))
        self.screen.blit(self.font_small.render("Click anywhere for menu", True, (180, 185, 200)), (400, 400))

    def _draw_play(self) -> None:
        assert self.game_state is not None
        gs = self.game_state
        obs = to_structured_observation(gs)
        hl_cells, hl_rgba = self._play_highlight()
        if self.view_mode == ViewMode.TOP_DOWN:
            self._draw_board_top(obs, highlight_cells=hl_cells, hl_rgba=hl_rgba)
        else:
            self._draw_board_iso(obs, highlight_cells=hl_cells, hl_rgba=hl_rgba)

        if self._is_human_turn() and self._play_slot is not None:
            u = unit_at(gs, gs.current_player, self._play_slot)
            if u is not None and u.alive:
                if self.view_mode == ViewMode.TOP_DOWN:
                    cx, cy = cell_center_top(u.row, u.col)
                    pygame.draw.circle(self.screen, (255, 220, 60), (cx, cy), 24, 3)
                else:
                    cx, cy = cell_center_iso(u.row, u.col)
                    pygame.draw.circle(self.screen, (255, 220, 60), (cx, cy - 4), 20, 3)

        if (
            self._is_human_turn()
            and self._play_drag_xy is not None
            and self._play_move_drag_slot is not None
        ):
            u = unit_at(gs, gs.current_player, self._play_move_drag_slot)
            if u is not None:
                self._blit_token_top(
                    self._play_drag_xy[0], self._play_drag_xy[1], u.class_id, gs.current_player, radius=16
                )

        self._draw_play_menus()

        flip = (
            f"coin P{gs.coin_flip_winner}  "
            if gs.coin_flip_winner is not None
            else ""
        )
        hud = f"P{gs.current_player} turn  |  Score {gs.score[0]} — {gs.score[1]}  |  {flip}seed {self.seed}"
        if self._is_human_turn():
            hud += "  [your turn]"
        self.screen.blit(self.font.render(hud, True, (220, 220, 230)), (32, 32))
        vm = "iso" if self.view_mode == ViewMode.ISOMETRIC else "top"
        self.screen.blit(self.font_small.render(f"View: {vm} (V to toggle)", True, (150, 155, 170)), (32, 56))
        self.screen.blit(
            self.font_small.render(
                "Board: click your piece → Move / Attack / Special. Tab = full turn list. Esc = cancel menu.",
                True,
                (130, 135, 155),
            ),
            (32, 76),
        )

        if self._is_human_turn() and _pass_turn_in_legal(self.legal_list):
            pygame.draw.rect(self.screen, (70, 90, 70), self.btn_pass_turn, border_radius=4)
            self.screen.blit(
                self.font_small.render("Pass turn (skip both)", True, (230, 245, 230)),
                (self.btn_pass_turn.x + 10, self.btn_pass_turn.y + 5),
            )

        if self._is_human_turn() and not self.legal_list and not gs.done:
            self.screen.blit(
                self.font.render("No legal actions (stalemate)", True, (255, 160, 120)),
                (712, 400),
            )

        if self._is_human_turn() and self.legal_list and self._play_show_fallback_list:
            x0 = 700
            pygame.draw.line(self.screen, (60, 64, 78), (x0, 0), (x0, 720), 2)
            title = self.font.render("All legal turns (Tab to hide)", True, (200, 205, 220))
            self.screen.blit(title, (x0 + 12, 72))
            y = 96 - self.action_scroll
            for i, act in enumerate(self.legal_list):
                if y > 620:
                    break
                if y + 18 >= 96:
                    sel = i == self.selected_idx
                    col = (80, 120, 200) if sel else (200, 200, 210)
                    prefix = "› " if sel else "  "
                    line = prefix + format_turn_action(act)
                    if len(line) > 52:
                        line = line[:49] + "…"
                    surf = self.font_small.render(f"{i+1}. {line}", True, col)
                    self.screen.blit(surf, (x0 + 8, y))
                y += 20
            pygame.draw.rect(self.screen, (100, 140, 200), pygame.Rect(720, 640, 200, 36), border_radius=4)
            self.screen.blit(self.font.render("Confirm (Enter)", True, (255, 255, 255)), (740, 646))

    def _draw_board_top(
        self,
        obs,
        *,
        highlight_cells: Optional[set[tuple[int, int]]] = None,
        hl_rgba: tuple[int, int, int, int] = HL_MOVE,
    ) -> None:
        hl = highlight_cells or set()
        ox, oy = board_origin_top()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tr = int(obs.terrain[r, c])
                color = TERRAIN_COLOR.get(tr, (200, 200, 200))
                rect = pygame.Rect(ox + c * CELL_TOP, oy + r * CELL_TOP, CELL_TOP, CELL_TOP)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (55, 58, 70), rect, 1)
                if (r, c) in hl:
                    hls = pygame.Surface((CELL_TOP, CELL_TOP), pygame.SRCALPHA)
                    hls.fill(hl_rgba)
                    self.screen.blit(hls, (rect.x, rect.y))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if float(obs.occupancy[r, c]) < 0.5:
                    continue
                tid = int(obs.team[r, c])
                cid = int(obs.unit_class[r, c])
                col = CLASS_COLORS[cid % len(CLASS_COLORS)] if tid >= 0 else (150, 150, 150)
                cx, cy = cell_center_top(r, c)
                pygame.draw.circle(self.screen, col, (cx, cy), 18)
                pygame.draw.circle(self.screen, (20, 20, 30), (cx, cy), 18, 2)
                label = CLASS_IDS[cid][0].upper() if 0 <= cid < len(CLASS_IDS) else "?"
                tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
                self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 8))
                # team tint
                ring = (120, 160, 255) if tid == TEAM_PLAYER_A else (255, 160, 120)
                pygame.draw.circle(self.screen, ring, (cx, cy), 21, 2)

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
        for _, c, r, tr in cells:
            cx, cy = cell_center_iso(r, c)
            color = TERRAIN_COLOR.get(tr, (200, 200, 200))
            pts = [
                (cx, cy - TH_ISO // 2),
                (cx + TW_ISO // 2, cy),
                (cx, cy + TH_ISO // 2),
                (cx - TW_ISO // 2, cy),
            ]
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, (55, 58, 70), pts, 1)
            if (r, c) in hl:
                pygame.draw.circle(self.screen, hl_rgba[:3], (cx, cy - 2), 22, 3)
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
            cx, cy = cell_center_iso(r, c)
            pygame.draw.circle(self.screen, col, (cx, cy - 4), 14)
            pygame.draw.circle(self.screen, (20, 20, 30), (cx, cy - 4), 14, 2)
            label = CLASS_IDS[cid][0].upper() if 0 <= cid < len(CLASS_IDS) else "?"
            tcol = (20, 20, 25) if sum(col) > 400 else (250, 250, 255)
            self.screen.blit(self.font_small.render(label, True, tcol), (cx - 5, cy - 14))
            ring = (120, 160, 255) if tid == TEAM_PLAYER_A else (255, 160, 120)
            pygame.draw.circle(self.screen, ring, (cx, cy - 4), 17, 2)


def run(seed: int = 0) -> None:
    MotleyCrewsUI(seed=seed).run()


if __name__ == "__main__":
    run()

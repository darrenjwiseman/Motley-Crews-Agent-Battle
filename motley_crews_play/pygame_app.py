"""
Pygame front-end: menu, top-down / isometric board, legal-action list for humans.

Run: ``python -m motley_crews_play --ui`` (requires pygame; see requirements-play.txt).
"""

from __future__ import annotations

import random
import sys
from enum import IntEnum
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pygame

from motley_crews_env.constants import (
    BOARD_SIZE,
    CLASS_IDS,
    TEAM_PLAYER_A,
    TERRAIN_BLOCKED,
    TERRAIN_OPEN,
    TERRAIN_WATER,
)
from motley_crews_env.engine import initial_state, legal_actions, step, to_structured_observation
from motley_crews_env.state import GameState
from motley_crews_env.types import TurnAction
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

        # menu buttons: (rect, mode or "start" or view toggle)
        self._layout_menu_buttons()

    def _layout_menu_buttons(self) -> None:
        self.btn_cpu_cpu = pygame.Rect(80, 140, 280, 40)
        self.btn_h_a = pygame.Rect(80, 190, 280, 40)
        self.btn_h_b = pygame.Rect(80, 240, 280, 40)
        self.btn_h_h = pygame.Rect(80, 290, 280, 40)
        self.btn_view = pygame.Rect(80, 360, 280, 36)
        self.btn_start = pygame.Rect(80, 430, 200, 44)

    def reset_match(self) -> None:
        self.rng_cpu = random.Random(self.seed)
        self.game_state = initial_state()
        self._refresh_legal()
        self.phase = "play"
        self.cpu_timer = 0

    def _refresh_legal(self) -> None:
        if self.game_state is None or self.game_state.done:
            self.legal_list = []
            return
        self.legal_list = legal_actions(self.game_state)
        self.selected_idx = 0
        self.action_scroll = 0
        self.cpu_timer = 0

    def _is_human_turn(self) -> bool:
        assert self.game_state is not None
        if self.play_mode == PlayMode.HUMAN_HUMAN:
            return True
        if self.play_mode == PlayMode.HUMAN_CPU_A:
            return self.game_state.current_player == TEAM_PLAYER_A
        if self.play_mode == PlayMode.HUMAN_CPU_B:
            return self.game_state.current_player != TEAM_PLAYER_A
        return False

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
                    self._on_mouse(event.pos, event.button)
                elif event.type == pygame.MOUSEWHEEL:
                    if self.phase == "play" and self._is_human_turn():
                        self.action_scroll = max(0, self.action_scroll - event.y * 24)

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
            else:
                self.phase = "menu"
        elif key == pygame.K_v and self.phase == "play":
            self.view_mode = ViewMode.ISOMETRIC if self.view_mode == ViewMode.TOP_DOWN else ViewMode.TOP_DOWN
        elif self.phase == "play" and self._is_human_turn() and self.legal_list:
            if key in (pygame.K_UP, pygame.K_k):
                self.selected_idx = max(0, self.selected_idx - 1)
            elif key in (pygame.K_DOWN, pygame.K_j):
                self.selected_idx = min(len(self.legal_list) - 1, self.selected_idx + 1)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._confirm_human_action()

    def _on_mouse(self, pos: tuple[int, int], button: int) -> None:
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

        if self.phase == "over":
            # simple click anywhere returns to menu
            self.phase = "menu"
            return

        if self.phase == "play" and self._is_human_turn():
            # sidebar: 700..1040
            if mx >= 700 and button == 1:
                sidebar_top = 96
                line_h = 20
                idx_y = my - sidebar_top + self.action_scroll
                if idx_y >= 0:
                    clicked = idx_y // line_h
                    if 0 <= clicked < len(self.legal_list):
                        self.selected_idx = clicked
                # confirm button
                if pygame.Rect(720, 640, 200, 36).collidepoint(mx, my):
                    self._confirm_human_action()

    def _draw(self) -> None:
        self.screen.fill((34, 36, 42))
        if self.phase == "menu":
            self._draw_menu()
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
        hint = self.font_small.render("In-game: V toggles view. Esc = menu.", True, (160, 165, 180))
        self.screen.blit(hint, (80, 500))

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
        if self.view_mode == ViewMode.TOP_DOWN:
            self._draw_board_top(obs)
        else:
            self._draw_board_iso(obs)

        hud = f"P{gs.current_player} turn  |  Score {gs.score[0]} — {gs.score[1]}  |  seed {self.seed}"
        if self._is_human_turn():
            hud += "  [your turn]"
        self.screen.blit(self.font.render(hud, True, (220, 220, 230)), (32, 32))
        vm = "iso" if self.view_mode == ViewMode.ISOMETRIC else "top"
        self.screen.blit(self.font_small.render(f"View: {vm} (V to toggle)", True, (150, 155, 170)), (32, 56))

        if self._is_human_turn() and not self.legal_list and not gs.done:
            self.screen.blit(
                self.font.render("No legal actions (stalemate)", True, (255, 160, 120)),
                (712, 400),
            )

        # sidebar: legal actions for human
        if self._is_human_turn() and self.legal_list:
            x0 = 700
            pygame.draw.line(self.screen, (60, 64, 78), (x0, 0), (x0, 720), 2)
            title = self.font.render("Legal turns (click / arrows)", True, (200, 205, 220))
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

    def _draw_board_top(self, obs) -> None:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                tr = int(obs.terrain[r, c])
                color = TERRAIN_COLOR.get(tr, (200, 200, 200))
                ox, oy = board_origin_top()
                rect = pygame.Rect(ox + c * CELL_TOP, oy + r * CELL_TOP, CELL_TOP, CELL_TOP)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (55, 58, 70), rect, 1)
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

    def _draw_board_iso(self, obs) -> None:
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

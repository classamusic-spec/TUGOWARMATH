"""On-screen calculator panel for one team.

Each team owns a Calculator. It accepts both mouse clicks on its keypad and
keyboard events routed by the engine (left team = main keys, right team =
numpad). The calculator shows the current problem, an answer buffer, and a
'submit' button. Submitting fires a callback that the engine evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import pygame


KEY_LAYOUT: List[List[str]] = [
    ["7", "8", "9", "/"],
    ["4", "5", "6", "x"],
    ["1", "2", "3", "-"],
    ["0", ".", "C", "+"],
    ["<-", "=", "=", "="],  # backspace + wide submit
]


@dataclass
class Button:
    rect: pygame.Rect
    label: str


@dataclass
class Calculator:
    panel_rect: pygame.Rect
    team_color: Tuple[int, int, int]
    label: str  # 'RED' or 'BLUE'
    on_submit: Callable[[str], None]
    answer: str = ""
    problem_text: str = ""
    flash_timer: float = 0.0       # red flash on wrong
    flash_color: Tuple[int, int, int] = (255, 80, 80)
    lockout_timer: float = 0.0     # seconds remaining where input is ignored
    glow_timer: float = 0.0        # green glow on correct
    buttons: List[Button] = field(default_factory=list)

    # ----- layout -----
    def __post_init__(self) -> None:
        self._build_buttons()

    def _build_buttons(self) -> None:
        pad = 8
        # Calculator grid lives in the lower 60% of the panel
        grid_top = self.panel_rect.top + int(self.panel_rect.height * 0.42)
        grid = pygame.Rect(
            self.panel_rect.left + pad,
            grid_top,
            self.panel_rect.width - pad * 2,
            self.panel_rect.bottom - grid_top - pad,
        )
        rows = len(KEY_LAYOUT)
        cols = 4
        bw = grid.width // cols
        bh = grid.height // rows
        seen = set()
        for r, row in enumerate(KEY_LAYOUT):
            c = 0
            while c < cols:
                label = row[c]
                # Span repeated labels horizontally (the '=' row is one wide button)
                span = 1
                while c + span < cols and row[c + span] == label:
                    span += 1
                rect = pygame.Rect(
                    grid.left + c * bw,
                    grid.top + r * bh,
                    bw * span - 4,
                    bh - 4,
                )
                key = (r, c, label)
                if key not in seen:
                    self.buttons.append(Button(rect, label))
                    seen.add(key)
                c += span

    # ----- input -----
    def is_locked(self) -> bool:
        return self.lockout_timer > 0

    def handle_mouse(self, pos: Tuple[int, int]) -> Optional[str]:
        if self.is_locked():
            return None
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                self.press(b.label)
                return b.label
        return None

    def press(self, label: str) -> None:
        if self.is_locked():
            return
        if label == "C":
            self.answer = ""
        elif label == "<-":
            self.answer = self.answer[:-1]
        elif label == "=":
            if self.answer:
                self.on_submit(self.answer)
        elif label in "+-x/":
            self.answer += label
        elif label.isdigit() or label == ".":
            if len(self.answer) < 14:
                self.answer += label

    # ----- feedback -----
    def flash_wrong(self, lockout: float = 1.2) -> None:
        self.flash_timer = 0.45
        self.lockout_timer = lockout
        self.answer = ""

    def flash_right(self) -> None:
        self.glow_timer = 0.5
        self.answer = ""

    def update(self, dt: float) -> None:
        self.flash_timer = max(0.0, self.flash_timer - dt)
        self.glow_timer = max(0.0, self.glow_timer - dt)
        self.lockout_timer = max(0.0, self.lockout_timer - dt)

    # ----- drawing -----
    def draw(
        self,
        surface: pygame.Surface,
        font_big: pygame.font.Font,
        font_med: pygame.font.Font,
        font_sm: pygame.font.Font,
    ) -> None:
        # Panel background with team-tinted border and feedback flash
        bg = (245, 240, 230)
        if self.flash_timer > 0:
            mix = self.flash_timer / 0.45
            bg = tuple(int(bg[i] * (1 - mix) + self.flash_color[i] * mix) for i in range(3))
        if self.glow_timer > 0:
            mix = self.glow_timer / 0.5
            bg = tuple(int(bg[i] * (1 - mix) + (140, 230, 140)[i] * mix) for i in range(3))
        pygame.draw.rect(surface, bg, self.panel_rect, border_radius=14)
        pygame.draw.rect(surface, self.team_color, self.panel_rect, 4, border_radius=14)

        # Header: team label
        team_surf = font_med.render(f"{self.label} TEAM", True, self.team_color)
        surface.blit(
            team_surf,
            (self.panel_rect.left + 14, self.panel_rect.top + 8),
        )

        # Problem area
        prob_rect = pygame.Rect(
            self.panel_rect.left + 14,
            self.panel_rect.top + 38,
            self.panel_rect.width - 28,
            int(self.panel_rect.height * 0.22),
        )
        pygame.draw.rect(surface, (255, 255, 255), prob_rect, border_radius=8)
        pygame.draw.rect(surface, (60, 60, 60), prob_rect, 2, border_radius=8)
        # Multiline prompt
        line_y = prob_rect.top + 6
        for line in self.problem_text.splitlines():
            txt = font_big.render(line, True, (30, 30, 30))
            txt_rect = txt.get_rect(midtop=(prob_rect.centerx, line_y))
            surface.blit(txt, txt_rect)
            line_y += txt.get_height() + 2

        # Answer buffer / display
        disp_rect = pygame.Rect(
            self.panel_rect.left + 14,
            prob_rect.bottom + 8,
            self.panel_rect.width - 28,
            44,
        )
        pygame.draw.rect(surface, (20, 20, 20), disp_rect, border_radius=6)
        pygame.draw.rect(surface, self.team_color, disp_rect, 2, border_radius=6)
        ans_surf = font_med.render(self.answer or "_", True, (180, 255, 180))
        ans_rect = ans_surf.get_rect(midright=(disp_rect.right - 10, disp_rect.centery))
        surface.blit(ans_surf, ans_rect)

        if self.lockout_timer > 0:
            lock_surf = font_sm.render("LOCKED", True, (255, 100, 100))
            surface.blit(
                lock_surf,
                lock_surf.get_rect(midleft=(disp_rect.left + 8, disp_rect.centery)),
            )

        # Buttons
        for b in self.buttons:
            color = (255, 255, 255)
            text_color = (30, 30, 30)
            if b.label == "=":
                color = self.team_color
                text_color = (255, 255, 255)
            elif b.label in "+-x/":
                color = (250, 220, 180)
            elif b.label in ("C", "<-"):
                color = (240, 200, 200)
            pygame.draw.rect(surface, color, b.rect, border_radius=6)
            pygame.draw.rect(surface, (60, 60, 60), b.rect, 1, border_radius=6)
            label = "DEL" if b.label == "<-" else b.label
            t = font_med.render(label, True, text_color)
            surface.blit(t, t.get_rect(center=b.rect.center))

"""Tug-of-War Math - main entry point.

Two teams of cartoon kids, one rope, a stream of math problems each.
Solve faster than the other side, the rope (and the flag) shifts toward your
team. Wrong answers lock you out for a moment and let the other team gain
ground. First team to drag the flag across their victory line wins; if the
timer runs out first, whichever side has the flag on it wins.

Run with:  python main.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

from audio import SoundBank
from calculator import Calculator
from characters import Kid, draw_kid, make_team
from math_problems import GRADES, Problem, grade_label, make_problem


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCREEN_W, SCREEN_H = 1280, 800
FPS = 60

PAGE_BG = (231, 238, 250)
CARD_COLOR = (255, 255, 255)
CARD_RECT = pygame.Rect(20, 12, 1240, 776)

HEADER_H = 62
HEADER_TOP = CARD_RECT.top + 8

CALC_W = 344
CALC_TOP = CARD_RECT.top + HEADER_H + 16
CALC_BOTTOM = CARD_RECT.bottom - 20
LEFT_CALC_RECT = pygame.Rect(CARD_RECT.left + 20, CALC_TOP, CALC_W, CALC_BOTTOM - CALC_TOP)
RIGHT_CALC_RECT = pygame.Rect(CARD_RECT.right - 20 - CALC_W, CALC_TOP, CALC_W, CALC_BOTTOM - CALC_TOP)

CENTER_LEFT = LEFT_CALC_RECT.right + 16
CENTER_RIGHT = RIGHT_CALC_RECT.left - 16
CENTER_X = (CENTER_LEFT + CENTER_RIGHT) // 2

SCORE_CHIP_RECT = pygame.Rect(CENTER_LEFT + 4, CALC_TOP + 4, CENTER_RIGHT - CENTER_LEFT - 8, 112)

SCENE_TOP = SCORE_CHIP_RECT.bottom + 16
SCENE_BOTTOM = CALC_BOTTOM - 20
GROUND_Y = SCENE_BOTTOM - 40
ROPE_Y = GROUND_Y - 78

VICTORY_THRESHOLD = 210
ROUND_SECONDS = 90

PULL_PER_PROBLEM = 38.0
WRONG_KICKBACK = 18.0
WRONG_LOCKOUT = 1.4
IDEAL_SOLVE_TIME = 6.0

TEAM1_COLOR = (237, 91, 93)
TEAM2_COLOR = (53, 130, 237)

INK = (28, 42, 80)
INK_SOFT = (100, 110, 130)
CHIP_BG = (241, 245, 252)
CHIP_BORDER = (222, 228, 240)
DIVIDER = (180, 190, 210)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_answer(text: str) -> Optional[float]:
    if not text:
        return None
    t = text.strip().replace("x", "*")
    if all(c in "0123456789.+-*/ " for c in t):
        try:
            return float(eval(t, {"__builtins__": {}}, {}))  # nosec - sanitized
        except Exception:
            return None
    return None


def fmt_time(seconds: float) -> str:
    seconds = max(0, int(math.ceil(seconds)))
    m, s = divmod(seconds, 60)
    return f"{m}:{s:02d}"


def rrect(surf, color, rect, radius, outline=None, outline_w=0):
    pygame.draw.rect(surf, color, rect, border_radius=int(radius))
    if outline and outline_w > 0:
        pygame.draw.rect(surf, outline, rect, outline_w, border_radius=int(radius))


def drop_shadow(surf, rect, radius, alpha=32, offset=6):
    sh = pygame.Surface((rect.width + 16, rect.height + 16), pygame.SRCALPHA)
    pygame.draw.rect(sh, (15, 25, 60, alpha), sh.get_rect().inflate(-4, -4), border_radius=int(radius))
    surf.blit(sh, (rect.left - 8, rect.top - 8 + offset))


# --------------------------------------------------------------------------- #
# Particles
# --------------------------------------------------------------------------- #

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float


def spawn_burst(particles, x, y, color, count=18, speed=180):
    for _ in range(count):
        ang = random.uniform(0, math.tau)
        spd = random.uniform(speed * 0.3, speed)
        particles.append(Particle(
            x=x, y=y,
            vx=math.cos(ang) * spd,
            vy=math.sin(ang) * spd - 80,
            life=0.8, max_life=0.8,
            color=color, size=random.uniform(3, 6),
        ))


def spawn_dust(particles, x, y, count=12):
    for _ in range(count):
        ang = random.uniform(-math.pi, 0)
        spd = random.uniform(40, 140)
        particles.append(Particle(
            x=x, y=y,
            vx=math.cos(ang) * spd,
            vy=math.sin(ang) * spd,
            life=0.7, max_life=0.7,
            color=(190, 200, 220), size=random.uniform(3, 6),
        ))


def update_particles(particles, dt):
    survivors = []
    for p in particles:
        p.life -= dt
        if p.life <= 0:
            continue
        p.vy += 380 * dt
        p.x += p.vx * dt
        p.y += p.vy * dt
        survivors.append(p)
    particles[:] = survivors


def draw_particles(surface, particles):
    for p in particles:
        a = max(0.0, p.life / p.max_life)
        r = max(1, int(p.size * a))
        pygame.draw.circle(surface, p.color, (int(p.x), int(p.y)), r)


# --------------------------------------------------------------------------- #
# Team state
# --------------------------------------------------------------------------- #

@dataclass
class TeamState:
    name: str
    color: Tuple[int, int, int]
    sign: int
    kids: List[Kid]
    calc: Calculator
    problem: Problem
    problem_started: float
    score_correct: int = 0
    score_wrong: int = 0
    pull_pulse: float = 0.0

    def new_problem(self, grade_index, now):
        self.problem = make_problem(grade_index)
        self.problem_started = now
        self.calc.problem_text = self.problem.prompt


# --------------------------------------------------------------------------- #
# The game
# --------------------------------------------------------------------------- #

class Game:
    STATE_MENU = "menu"
    STATE_PLAY = "play"
    STATE_OVER = "over"

    def __init__(self):
        self.sound = SoundBank()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Tug-of-War Math")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 34, bold=True)
        self.font_huge = pygame.font.SysFont("arial", 64, bold=True)
        self.font_big = pygame.font.SysFont("arial", 30, bold=True)
        self.font_med = pygame.font.SysFont("arial", 22, bold=True)
        self.font_sm = pygame.font.SysFont("arial", 16)
        self.font_xs = pygame.font.SysFont("arial", 14)

        self.state = self.STATE_MENU
        self.grade_index = 2
        self.particles: List[Particle] = []

        self.left: Optional[TeamState] = None
        self.right: Optional[TeamState] = None
        self.rope_offset = 0.0
        self.rope_wobble_phase = 0.0
        self.round_time_left = ROUND_SECONDS
        self.last_tick_second: int = -1
        self.winner: Optional[str] = None
        self.shake_timer = 0.0
        self.t = 0.0

    # ---- lifecycle -----------------------------------------------------

    def start_round(self):
        now = time.time()
        left_x_range = (CENTER_LEFT + 60, CENTER_X - 90)
        right_x_range = (CENTER_X + 90, CENTER_RIGHT - 60)

        self.left = TeamState(
            name="Team 1",
            color=TEAM1_COLOR,
            sign=-1,
            kids=make_team("left", 2, GROUND_Y, left_x_range),
            calc=Calculator(LEFT_CALC_RECT, TEAM1_COLOR, "Team 1", lambda v: self._submit("left", v)),
            problem=make_problem(self.grade_index),
            problem_started=now,
        )
        self.right = TeamState(
            name="Team 2",
            color=TEAM2_COLOR,
            sign=+1,
            kids=make_team("right", 2, GROUND_Y, right_x_range),
            calc=Calculator(RIGHT_CALC_RECT, TEAM2_COLOR, "Team 2", lambda v: self._submit("right", v)),
            problem=make_problem(self.grade_index),
            problem_started=now,
        )
        self.left.calc.problem_text = self.left.problem.prompt
        self.right.calc.problem_text = self.right.problem.prompt

        self.rope_offset = 0.0
        self.round_time_left = ROUND_SECONDS
        self.last_tick_second = int(self.round_time_left)
        self.winner = None
        self.particles.clear()
        self.state = self.STATE_PLAY
        self.sound.play("whistle")

    def _submit(self, side, value_text):
        team = self.left if side == "left" else self.right
        opp = self.right if side == "left" else self.left
        assert team is not None and opp is not None
        if team.calc.is_locked():
            return
        value = parse_answer(value_text)
        if value is None:
            team.calc.flash_wrong(lockout=0.4)
            self.sound.play("wrong", 0.6)
            return

        now = time.time()
        elapsed = max(0.1, now - team.problem_started)

        if team.problem.check(value):
            speed_bonus = 1.0 + max(0.0, (IDEAL_SOLVE_TIME - elapsed)) / IDEAL_SOLVE_TIME * 0.6
            impulse = PULL_PER_PROBLEM * team.problem.weight * speed_bonus
            self.rope_offset += team.sign * impulse
            team.score_correct += 1
            team.pull_pulse = min(1.5, team.pull_pulse + 1.0)
            team.calc.flash_right()
            self.sound.play("right")
            self.sound.play("scream_left" if side == "left" else "scream_right", 0.7)
            self.sound.play("whoosh", 0.6)
            burst_x = CENTER_X + self.rope_offset
            spawn_burst(self.particles, burst_x, ROPE_Y, team.color)
            team.new_problem(self.grade_index, now)
            self.shake_timer = max(self.shake_timer, 0.15)
        else:
            self.rope_offset += opp.sign * WRONG_KICKBACK
            team.score_wrong += 1
            team.calc.flash_wrong(WRONG_LOCKOUT)
            self.sound.play("wrong")
            spawn_dust(self.particles, CENTER_X + self.rope_offset, ROPE_Y + 6)

        self._check_victory()

    def _check_victory(self):
        if self.rope_offset <= -VICTORY_THRESHOLD:
            self._end_round("Team 1")
        elif self.rope_offset >= VICTORY_THRESHOLD:
            self._end_round("Team 2")

    def _end_round(self, winner):
        self.winner = winner
        self.state = self.STATE_OVER
        self.sound.play("whistle")

    # ---- input ---------------------------------------------------------

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        if self.state == self.STATE_MENU:
            self._handle_menu_event(event)
        elif self.state == self.STATE_PLAY:
            self._handle_play_event(event)
        elif self.state == self.STATE_OVER:
            self._handle_over_event(event)

    def _handle_menu_event(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_LEFT, pygame.K_a):
            self.grade_index = (self.grade_index - 1) % len(GRADES)
            self.sound.play("click")
        elif event.key in (pygame.K_RIGHT, pygame.K_d):
            self.grade_index = (self.grade_index + 1) % len(GRADES)
            self.sound.play("click")
        elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
            self.start_round()
        elif event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit(0)

    def _handle_play_event(self, event):
        assert self.left and self.right
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = self.STATE_MENU
                return

            kp_digits = {
                pygame.K_KP0: "0", pygame.K_KP1: "1", pygame.K_KP2: "2",
                pygame.K_KP3: "3", pygame.K_KP4: "4", pygame.K_KP5: "5",
                pygame.K_KP6: "6", pygame.K_KP7: "7", pygame.K_KP8: "8",
                pygame.K_KP9: "9",
            }
            kp_ops = {
                pygame.K_KP_PLUS: "+", pygame.K_KP_MINUS: "-",
                pygame.K_KP_MULTIPLY: "x", pygame.K_KP_DIVIDE: "/",
                pygame.K_KP_PERIOD: ".",
            }
            if event.key in kp_digits:
                self.right.calc.press(kp_digits[event.key])
                self.sound.play("click", 0.3)
                return
            if event.key in kp_ops:
                self.right.calc.press(kp_ops[event.key])
                self.sound.play("click", 0.3)
                return
            if event.key == pygame.K_KP_ENTER:
                self.right.calc.press("=")
                return
            if event.key == pygame.K_DELETE:
                self.right.calc.press("C")
                return

            if event.key == pygame.K_RETURN:
                self.left.calc.press("=")
                return
            if event.key == pygame.K_BACKSPACE:
                self.left.calc.press("<-")
                return
            ch = event.unicode
            if ch:
                if ch.isdigit() or ch == ".":
                    self.left.calc.press(ch)
                    self.sound.play("click", 0.3)
                elif ch in "+-/":
                    self.left.calc.press(ch)
                    self.sound.play("click", 0.3)
                elif ch == "*":
                    self.left.calc.press("x")
                    self.sound.play("click", 0.3)
                elif ch.lower() == "c":
                    self.left.calc.press("C")

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.left.calc.handle_mouse(event.pos)
            self.right.calc.handle_mouse(event.pos)

    def _handle_over_event(self, event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
            self.start_round()
        elif event.key in (pygame.K_m, pygame.K_ESCAPE):
            self.state = self.STATE_MENU

    # ---- update --------------------------------------------------------

    def update(self, dt):
        self.t += dt
        self.rope_wobble_phase += dt * 6
        update_particles(self.particles, dt)
        self.shake_timer = max(0.0, self.shake_timer - dt)

        if self.state != self.STATE_PLAY:
            return
        assert self.left and self.right
        self.left.calc.update(dt)
        self.right.calc.update(dt)
        self.left.pull_pulse = max(0.0, self.left.pull_pulse - dt * 1.5)
        self.right.pull_pulse = max(0.0, self.right.pull_pulse - dt * 1.5)

        self.round_time_left -= dt
        cur_sec = int(math.ceil(self.round_time_left))
        if cur_sec != self.last_tick_second and 0 < cur_sec <= 5:
            self.sound.play("tick", 0.5)
        self.last_tick_second = cur_sec

        if self.round_time_left <= 0:
            if self.rope_offset < -10:
                self._end_round("Team 1")
            elif self.rope_offset > 10:
                self._end_round("Team 2")
            else:
                self._end_round(None)

    # ---- draw pieces ---------------------------------------------------

    def _draw_page(self, surf):
        surf.fill(PAGE_BG)
        drop_shadow(surf, CARD_RECT, radius=28, alpha=45, offset=8)
        rrect(surf, CARD_COLOR, CARD_RECT, radius=28)

    def _draw_header(self, surf):
        hx = CARD_RECT.left + 60
        hy = HEADER_TOP + HEADER_H // 2
        roof = [(hx - 12, hy + 2), (hx, hy - 12), (hx + 12, hy + 2)]
        pygame.draw.polygon(surf, INK, roof)
        body = pygame.Rect(0, 0, 20, 14)
        body.midtop = (hx, hy + 2)
        rrect(surf, INK, body, radius=2)
        home_text = self.font_med.render("Home", True, INK)
        surf.blit(home_text, home_text.get_rect(midleft=(hx + 22, hy)))

        title = self.font_title.render("TUG OF WAR: MATHEMATICS", True, INK)
        surf.blit(title, title.get_rect(midtop=(SCREEN_W // 2, HEADER_TOP + 12)))

        rx = CARD_RECT.right - 40
        ry = hy
        pygame.draw.polygon(surf, INK, [
            (rx - 6, ry - 10), (rx + 6, ry - 10),
            (rx + 6, ry + 10), (rx, ry + 4), (rx - 6, ry + 10),
        ])
        sx = rx - 34
        pygame.draw.polygon(surf, INK, [
            (sx - 8, ry - 4), (sx - 2, ry - 4),
            (sx + 4, ry - 10), (sx + 4, ry + 10),
            (sx - 2, ry + 4), (sx - 8, ry + 4),
        ])
        pygame.draw.arc(surf, INK, pygame.Rect(sx + 6, ry - 8, 10, 16), -0.9, 0.9, 2)

    def _draw_score_chip(self, surf):
        chip = SCORE_CHIP_RECT
        rrect(surf, CHIP_BG, chip, radius=22, outline=CHIP_BORDER, outline_w=1)
        section_w = chip.width // 3

        # Team 1
        lbl1 = self.font_sm.render("Team 1", True, INK_SOFT)
        surf.blit(lbl1, lbl1.get_rect(midtop=(chip.left + section_w // 2, chip.top + 18)))
        val1 = self.font_big.render(
            str(self.left.score_correct) if self.left else "0", True, INK,
        )
        surf.blit(val1, val1.get_rect(midbottom=(chip.left + section_w // 2, chip.bottom - 18)))

        # Center: clock icon + timer, both centered together horizontally
        time_str = fmt_time(self.round_time_left if self.state == self.STATE_PLAY else ROUND_SECONDS)
        time_surf = self.font_big.render(time_str, True, TEAM2_COLOR)
        cluster_w = 20 + 12 + time_surf.get_width()  # icon + gap + text
        c_cx = chip.left + section_w + section_w // 2
        cluster_left = c_cx - cluster_w // 2
        cy = chip.centery + 2
        clock_cx = cluster_left + 10
        pygame.draw.circle(surf, TEAM2_COLOR, (clock_cx, cy), 12, 3)
        pygame.draw.line(surf, TEAM2_COLOR, (clock_cx, cy), (clock_cx, cy - 7), 3)
        pygame.draw.line(surf, TEAM2_COLOR, (clock_cx, cy), (clock_cx + 5, cy), 3)
        surf.blit(time_surf, time_surf.get_rect(midleft=(clock_cx + 14, cy)))

        # Team 2
        r_cx = chip.right - section_w // 2
        lbl2 = self.font_sm.render("Team 2", True, INK_SOFT)
        surf.blit(lbl2, lbl2.get_rect(midtop=(r_cx, chip.top + 18)))
        val2 = self.font_big.render(
            str(self.right.score_correct) if self.right else "0", True, INK,
        )
        surf.blit(val2, val2.get_rect(midbottom=(r_cx, chip.bottom - 18)))

    def _draw_center_divider(self, surf):
        dash_h = 8
        gap = 8
        y = SCENE_TOP + 8
        while y < SCENE_BOTTOM:
            pygame.draw.line(surf, DIVIDER, (CENTER_X, y), (CENTER_X, y + dash_h), 2)
            y += dash_h + gap

    def _team_shift(self):
        return self.rope_offset * 0.22

    def _draw_kids(self, surf):
        assert self.left and self.right
        shift = self._team_shift()
        left_losing = max(0.0, min(1.0, self.rope_offset / VICTORY_THRESHOLD))
        right_losing = max(0.0, min(1.0, -self.rope_offset / VICTORY_THRESHOLD))

        for kid in self.left.kids:
            view = Kid(x=kid.x + shift, y=kid.y, facing=kid.facing, phase=kid.phase,
                       skin=kid.skin, shirt=kid.shirt, pants=kid.pants,
                       height=kid.height, shoe=kid.shoe, headband=kid.headband,
                       hair=kid.hair, pattern_seed=kid.pattern_seed)
            draw_kid(surf, view, self.t, self.left.pull_pulse, left_losing, ROPE_Y)
        for kid in self.right.kids:
            view = Kid(x=kid.x + shift, y=kid.y, facing=kid.facing, phase=kid.phase,
                       skin=kid.skin, shirt=kid.shirt, pants=kid.pants,
                       height=kid.height, shoe=kid.shoe, headband=kid.headband,
                       hair=kid.hair, pattern_seed=kid.pattern_seed)
            draw_kid(surf, view, self.t, self.right.pull_pulse, right_losing, ROPE_Y)

    def _draw_rope(self, surf):
        assert self.left and self.right
        shift = self._team_shift()
        inner_left_kid = max(self.left.kids, key=lambda k: k.x)
        inner_right_kid = min(self.right.kids, key=lambda k: k.x)
        left_end = inner_left_kid.x + shift + inner_left_kid.height * 0.30
        right_end = inner_right_kid.x + shift - inner_right_kid.height * 0.30
        knot_x = CENTER_X + self.rope_offset

        pygame.draw.line(surf, (40, 55, 90), (left_end, ROPE_Y), (right_end, ROPE_Y), 4)
        ribbon = [
            (knot_x - 7, ROPE_Y - 1),
            (knot_x + 7, ROPE_Y - 1),
            (knot_x + 6, ROPE_Y + 20),
            (knot_x, ROPE_Y + 14),
            (knot_x - 6, ROPE_Y + 20),
        ]
        pygame.draw.polygon(surf, TEAM1_COLOR, ribbon)
        pygame.draw.line(surf, (60, 20, 20), (knot_x - 7, ROPE_Y), (knot_x + 7, ROPE_Y), 2)

    def _draw_grade_pill(self, surf):
        # A subtle pill in the top-left of the header area, next to the Home button
        label = self.font_xs.render(f"Grade  {grade_label(self.grade_index)}", True, INK_SOFT)
        pad_x, pad_y = 10, 6
        rect = label.get_rect().inflate(pad_x * 2, pad_y * 2)
        rect.midleft = (CARD_RECT.left + 170, HEADER_TOP + HEADER_H // 2)
        rrect(surf, (247, 249, 253), rect, radius=rect.height // 2, outline=CHIP_BORDER, outline_w=1)
        surf.blit(label, label.get_rect(center=rect.center))

    def _draw_help_footer(self, surf):
        line = self.font_xs.render(
            "Left: number row  +  -  x  /  .   Enter=submit   Backspace=del   |   "
            "Right: numpad   KP-Enter=submit   Del=clear   |   Mouse works too",
            True, INK_SOFT,
        )
        surf.blit(line, line.get_rect(midbottom=(SCREEN_W // 2, CARD_RECT.bottom - 4)))

    # ---- state renderers -----------------------------------------------

    def _draw_play(self, surf):
        self._draw_page(surf)
        self._draw_header(surf)
        self._draw_score_chip(surf)
        self._draw_grade_pill(surf)
        self._draw_center_divider(surf)
        self._draw_kids(surf)
        self._draw_rope(surf)
        draw_particles(surf, self.particles)
        if self.left and self.right:
            self.left.calc.draw(surf, self.font_big, self.font_med, self.font_sm)
            self.right.calc.draw(surf, self.font_big, self.font_med, self.font_sm)
        self._draw_help_footer(surf)

    def _draw_menu(self, surf):
        self._draw_page(surf)
        self._draw_header(surf)

        title = self.font_huge.render("TUG OF WAR", True, INK)
        sub = self.font_big.render("Two teams. One rope. A blizzard of math.", True, INK_SOFT)
        surf.blit(title, title.get_rect(midtop=(SCREEN_W // 2, HEADER_TOP + HEADER_H + 40)))
        surf.blit(sub, sub.get_rect(midtop=(SCREEN_W // 2, HEADER_TOP + HEADER_H + 118)))

        picker_label = self.font_med.render("Pick a grade band", True, INK_SOFT)
        surf.blit(picker_label, picker_label.get_rect(midtop=(SCREEN_W // 2, 300)))

        pill_w, pill_h, gap = 110, 56, 12
        total_w = pill_w * len(GRADES) + gap * (len(GRADES) - 1)
        start_x = SCREEN_W // 2 - total_w // 2
        for i, g in enumerate(GRADES):
            r = pygame.Rect(start_x + i * (pill_w + gap), 340, pill_w, pill_h)
            if i == self.grade_index:
                drop_shadow(surf, r, radius=pill_h // 2, alpha=40, offset=4)
                rrect(surf, TEAM2_COLOR, r, radius=pill_h // 2)
                col = (255, 255, 255)
            else:
                rrect(surf, (247, 249, 253), r, radius=pill_h // 2, outline=CHIP_BORDER, outline_w=1)
                col = INK
            t = self.font_big.render(g, True, col)
            surf.blit(t, t.get_rect(center=r.center))

        btn = pygame.Rect(0, 0, 260, 68)
        btn.center = (SCREEN_W // 2, 460)
        drop_shadow(surf, btn, radius=34, alpha=50, offset=6)
        rrect(surf, TEAM1_COLOR, btn, radius=34)
        st = self.font_big.render("START", True, (255, 255, 255))
        surf.blit(st, st.get_rect(center=btn.center))
        hint = self.font_sm.render("Press SPACE  |  Left/Right to change grade", True, INK_SOFT)
        surf.blit(hint, hint.get_rect(midtop=(SCREEN_W // 2, 540)))

        lines = [
            "Left team uses the number row + Enter.  Right team uses the numpad + KP-Enter.",
            "Either team can also tap the on-screen calculator with the mouse.",
            "Wrong answer = locked out ~1.4s and the other team gains rope.",
            "First team to drag the flag to their side wins.  90-second round.",
        ]
        y = 600
        for line in lines:
            t = self.font_sm.render(line, True, INK_SOFT)
            surf.blit(t, t.get_rect(midtop=(SCREEN_W // 2, y)))
            y += 20

    def _draw_over(self, surf):
        self._draw_play(surf)
        dim = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        dim.fill((15, 25, 60, 110))
        surf.blit(dim, (0, 0))

        panel = pygame.Rect(0, 0, 560, 320)
        panel.center = (SCREEN_W // 2, SCREEN_H // 2 - 20)
        drop_shadow(surf, panel, radius=28, alpha=80, offset=8)
        rrect(surf, (255, 255, 255), panel, radius=28)

        if self.winner is None:
            msg = "IT'S A TIE!"
            color = INK
        else:
            msg = f"{self.winner.upper()} WINS!"
            color = TEAM1_COLOR if self.winner == "Team 1" else TEAM2_COLOR
        title = self.font_huge.render(msg, True, color)
        surf.blit(title, title.get_rect(midtop=(panel.centerx, panel.top + 28)))

        if self.left and self.right:
            row1 = self.font_med.render(
                f"Team 1     correct {self.left.score_correct}     missed {self.left.score_wrong}",
                True, INK,
            )
            row2 = self.font_med.render(
                f"Team 2     correct {self.right.score_correct}     missed {self.right.score_wrong}",
                True, INK,
            )
            surf.blit(row1, row1.get_rect(midtop=(panel.centerx, panel.top + 130)))
            surf.blit(row2, row2.get_rect(midtop=(panel.centerx, panel.top + 166)))

        b1 = pygame.Rect(0, 0, 180, 52)
        b1.midbottom = (panel.centerx - 100, panel.bottom - 22)
        b2 = pygame.Rect(0, 0, 180, 52)
        b2.midbottom = (panel.centerx + 100, panel.bottom - 22)
        rrect(surf, TEAM1_COLOR, b1, radius=26)
        rrect(surf, (247, 249, 253), b2, radius=26, outline=CHIP_BORDER, outline_w=1)
        r_txt = self.font_med.render("Rematch (R)", True, (255, 255, 255))
        m_txt = self.font_med.render("Menu (M)", True, INK)
        surf.blit(r_txt, r_txt.get_rect(center=b1.center))
        surf.blit(m_txt, m_txt.get_rect(center=b2.center))

    def draw(self):
        offset = (0, 0)
        if self.shake_timer > 0:
            mag = int(self.shake_timer * 20)
            offset = (random.randint(-mag, mag), random.randint(-mag, mag))

        scratch = pygame.Surface((SCREEN_W, SCREEN_H))
        if self.state == self.STATE_MENU:
            self._draw_menu(scratch)
        elif self.state == self.STATE_PLAY:
            self._draw_play(scratch)
        elif self.state == self.STATE_OVER:
            self._draw_over(scratch)

        self.screen.fill(PAGE_BG)
        self.screen.blit(scratch, offset)
        pygame.display.flip()

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw()


def main():
    Game().run()


if __name__ == "__main__":
    main()

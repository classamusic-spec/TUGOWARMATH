"""Tug-of-War Math - main entry point.

Two teams of stick-figure kids, one rope, a stream of math problems each.
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

TOP_BAR_H = 60
SCENE_BOTTOM = 470     # bottom of the play scene; calculators live below this
GROUND_Y = 430
ROPE_Y = 300

CENTER_X = SCREEN_W // 2
VICTORY_THRESHOLD = 260   # px of rope offset needed to win
ROUND_SECONDS = 90

PULL_PER_PROBLEM = 38.0   # base px the rope shifts on a correct answer
WRONG_KICKBACK = 18.0     # px the opponent gains on a wrong answer
WRONG_LOCKOUT = 1.4       # seconds you're locked out after a wrong answer
IDEAL_SOLVE_TIME = 6.0    # solving faster than this gives a speed bonus

LEFT_COLOR = (210, 60, 60)
RIGHT_COLOR = (60, 130, 230)
SKY_TOP = (135, 200, 240)
SKY_BOTTOM = (210, 235, 255)
GROUND_COLOR = (120, 180, 90)
DIRT_COLOR = (160, 120, 70)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def parse_answer(text: str) -> Optional[float]:
    """Convert the calculator buffer to a float; return None if invalid."""
    if not text:
        return None
    t = text.strip().replace("x", "*")
    # Allow simple expressions like "3+4" via eval on a sanitized string
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


def spawn_burst(particles: List[Particle], x: float, y: float, color, count=18, speed=180):
    for _ in range(count):
        ang = random.uniform(0, math.tau)
        spd = random.uniform(speed * 0.3, speed)
        particles.append(
            Particle(
                x=x,
                y=y,
                vx=math.cos(ang) * spd,
                vy=math.sin(ang) * spd - 80,
                life=0.8,
                max_life=0.8,
                color=color,
                size=random.uniform(3, 6),
            )
        )


def spawn_dust(particles: List[Particle], x: float, y: float, count=12):
    for _ in range(count):
        ang = random.uniform(-math.pi, 0)
        spd = random.uniform(40, 140)
        particles.append(
            Particle(
                x=x,
                y=y,
                vx=math.cos(ang) * spd,
                vy=math.sin(ang) * spd,
                life=0.7,
                max_life=0.7,
                color=(180, 160, 120),
                size=random.uniform(4, 8),
            )
        )


def update_particles(particles: List[Particle], dt: float) -> None:
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


def draw_particles(surface: pygame.Surface, particles: List[Particle]) -> None:
    for p in particles:
        a = max(0.0, p.life / p.max_life)
        r = max(1, int(p.size * a))
        pygame.draw.circle(surface, p.color, (int(p.x), int(p.y)), r)


# --------------------------------------------------------------------------- #
# Team state
# --------------------------------------------------------------------------- #

@dataclass
class TeamState:
    name: str             # 'RED' / 'BLUE'
    color: Tuple[int, int, int]
    sign: int             # -1 for left team, +1 for right team
    kids: List[Kid]
    calc: Calculator
    problem: Problem
    problem_started: float
    score_correct: int = 0
    score_wrong: int = 0
    pull_pulse: float = 0.0   # animation amplitude that decays over time

    def new_problem(self, grade_index: int, now: float) -> None:
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

    def __init__(self) -> None:
        # Audio first so mixer params take effect; survives if no audio device.
        self.sound = SoundBank()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Tug-of-War Math")
        self.clock = pygame.time.Clock()

        self.font_huge = pygame.font.SysFont("arial", 64, bold=True)
        self.font_big = pygame.font.SysFont("arial", 30, bold=True)
        self.font_med = pygame.font.SysFont("arial", 22, bold=True)
        self.font_sm = pygame.font.SysFont("arial", 16)

        self.state = self.STATE_MENU
        self.grade_index = 2  # default 1st grade
        self.particles: List[Particle] = []

        self.left: Optional[TeamState] = None
        self.right: Optional[TeamState] = None
        self.rope_offset = 0.0     # +ve = pulled toward right team
        self.rope_wobble_phase = 0.0
        self.round_time_left = ROUND_SECONDS
        self.last_tick_second: int = -1
        self.winner: Optional[str] = None
        self.shake_timer = 0.0
        self.t = 0.0

    # ---- lifecycle -----------------------------------------------------

    def start_round(self) -> None:
        now = time.time()
        # Build calculators
        calc_w, calc_h = 600, SCREEN_H - SCENE_BOTTOM - 16
        left_rect = pygame.Rect(16, SCENE_BOTTOM, calc_w, calc_h)
        right_rect = pygame.Rect(SCREEN_W - 16 - calc_w, SCENE_BOTTOM, calc_w, calc_h)

        self.left = TeamState(
            name="RED",
            color=LEFT_COLOR,
            sign=-1,
            kids=make_team("left", 4, GROUND_Y, (90, 380)),
            calc=Calculator(left_rect, LEFT_COLOR, "RED", lambda v: self._submit("left", v)),
            problem=make_problem(self.grade_index),
            problem_started=now,
        )
        self.right = TeamState(
            name="BLUE",
            color=RIGHT_COLOR,
            sign=+1,
            kids=make_team("right", 4, GROUND_Y, (SCREEN_W - 380, SCREEN_W - 90)),
            calc=Calculator(right_rect, RIGHT_COLOR, "BLUE", lambda v: self._submit("right", v)),
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

    def _submit(self, side: str, value_text: str) -> None:
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
            # Burst near the rope where this team is pulling
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
            # Don't replace the problem on a wrong answer - they need to retry

        self._check_victory()

    def _check_victory(self) -> None:
        if self.rope_offset <= -VICTORY_THRESHOLD:
            self._end_round("RED")
        elif self.rope_offset >= VICTORY_THRESHOLD:
            self._end_round("BLUE")

    def _end_round(self, winner: Optional[str]) -> None:
        self.winner = winner
        self.state = self.STATE_OVER
        self.sound.play("whistle")

    # ---- input ---------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if self.state == self.STATE_MENU:
            self._handle_menu_event(event)
        elif self.state == self.STATE_PLAY:
            self._handle_play_event(event)
        elif self.state == self.STATE_OVER:
            self._handle_over_event(event)

    def _handle_menu_event(self, event: pygame.event.Event) -> None:
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

    def _handle_play_event(self, event: pygame.event.Event) -> None:
        assert self.left and self.right
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.state = self.STATE_MENU
                return

            # ---- right team: numpad ----
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

            # ---- left team: main keyboard via unicode ----
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
            # Either calculator can be operated by mouse - useful for
            # younger kids and for testing.
            self.left.calc.handle_mouse(event.pos)
            self.right.calc.handle_mouse(event.pos)

    def _handle_over_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key in (pygame.K_r, pygame.K_RETURN, pygame.K_SPACE):
            self.start_round()
        elif event.key in (pygame.K_m, pygame.K_ESCAPE):
            self.state = self.STATE_MENU

    # ---- update --------------------------------------------------------

    def update(self, dt: float) -> None:
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
                self._end_round("RED")
            elif self.rope_offset > 10:
                self._end_round("BLUE")
            else:
                self._end_round(None)  # tie

    # ---- draw ----------------------------------------------------------

    def _draw_sky(self, surf: pygame.Surface) -> None:
        h = SCENE_BOTTOM
        for y in range(h):
            t = y / max(1, h - 1)
            r = int(SKY_TOP[0] * (1 - t) + SKY_BOTTOM[0] * t)
            g = int(SKY_TOP[1] * (1 - t) + SKY_BOTTOM[1] * t)
            b = int(SKY_TOP[2] * (1 - t) + SKY_BOTTOM[2] * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (SCREEN_W, y))

    def _draw_scene(self, surf: pygame.Surface) -> None:
        self._draw_sky(surf)

        # Ground
        pygame.draw.rect(
            surf, GROUND_COLOR,
            pygame.Rect(0, GROUND_Y, SCREEN_W, SCENE_BOTTOM - GROUND_Y),
        )
        # Mud pit / center marker
        pit = pygame.Rect(0, 0, 110, 22)
        pit.center = (CENTER_X, GROUND_Y + 8)
        pygame.draw.ellipse(surf, DIRT_COLOR, pit)
        # Center line on the ground (the "danger" line)
        pygame.draw.line(surf, (200, 60, 60), (CENTER_X, GROUND_Y - 3), (CENTER_X, GROUND_Y + 14), 3)
        # Victory threshold markers
        for sign, color in [(-1, LEFT_COLOR), (1, RIGHT_COLOR)]:
            x = CENTER_X + sign * VICTORY_THRESHOLD
            pygame.draw.line(surf, color, (x, GROUND_Y - 6), (x, GROUND_Y + 16), 2)

        # Few clouds
        for i, cx in enumerate((180, 540, 900, 1140)):
            cy = 80 + 14 * math.sin(self.t * 0.4 + i)
            for dx, dy, r in [(0, 0, 22), (24, 6, 18), (-22, 8, 18), (0, -8, 20)]:
                pygame.draw.circle(surf, (255, 255, 255), (int(cx + dx), int(cy + dy)), r)

    def _draw_top_bar(self, surf: pygame.Surface) -> None:
        bar = pygame.Rect(0, 0, SCREEN_W, TOP_BAR_H)
        pygame.draw.rect(surf, (30, 30, 50), bar)

        # Timer bar
        ratio = max(0.0, self.round_time_left / ROUND_SECONDS) if self.state == self.STATE_PLAY else 1.0
        tb = pygame.Rect(SCREEN_W // 2 - 200, 18, 400, 24)
        pygame.draw.rect(surf, (80, 80, 100), tb, border_radius=10)
        fill = tb.copy()
        fill.width = int(tb.width * ratio)
        col = (90, 220, 110) if ratio > 0.4 else (240, 220, 80) if ratio > 0.2 else (240, 90, 90)
        pygame.draw.rect(surf, col, fill, border_radius=10)
        pygame.draw.rect(surf, (255, 255, 255), tb, 2, border_radius=10)
        time_text = self.font_med.render(fmt_time(self.round_time_left), True, (255, 255, 255))
        surf.blit(time_text, time_text.get_rect(center=tb.center))

        # Scores
        if self.left and self.right:
            ls = self.font_med.render(
                f"RED  {self.left.score_correct}", True, LEFT_COLOR
            )
            rs = self.font_med.render(
                f"{self.right.score_correct}  BLUE", True, RIGHT_COLOR
            )
            surf.blit(ls, ls.get_rect(midleft=(20, TOP_BAR_H // 2)))
            surf.blit(rs, rs.get_rect(midright=(SCREEN_W - 20, TOP_BAR_H // 2)))

        # Grade
        grade = self.font_sm.render(
            f"Grade: {grade_label(self.grade_index)}", True, (210, 210, 230)
        )
        surf.blit(grade, grade.get_rect(midtop=(SCREEN_W // 2, 44)))

    def _team_x_shift(self) -> float:
        # Both teams visually shuffle in the direction the rope is going
        return self.rope_offset * 0.35

    def _draw_kids(self, surf: pygame.Surface) -> None:
        assert self.left and self.right
        shift = self._team_x_shift()
        # Losing factor 0..1 - mouths open wider when getting dragged
        left_losing = max(0.0, min(1.0, self.rope_offset / VICTORY_THRESHOLD))
        right_losing = max(0.0, min(1.0, -self.rope_offset / VICTORY_THRESHOLD))

        for kid in self.left.kids:
            kid_view = Kid(
                x=kid.x + shift,
                y=kid.y,
                facing=kid.facing,
                phase=kid.phase,
                skin=kid.skin,
                shirt=kid.shirt,
                pants=kid.pants,
                height=kid.height,
            )
            draw_kid(surf, kid_view, self.t, self.left.pull_pulse, left_losing, ROPE_Y)
        for kid in self.right.kids:
            kid_view = Kid(
                x=kid.x + shift,
                y=kid.y,
                facing=kid.facing,
                phase=kid.phase,
                skin=kid.skin,
                shirt=kid.shirt,
                pants=kid.pants,
                height=kid.height,
            )
            draw_kid(surf, kid_view, self.t, self.right.pull_pulse, right_losing, ROPE_Y)

    def _draw_rope(self, surf: pygame.Surface) -> None:
        # Rope spans across the screen at ROPE_Y. Sag with a sin wave; flag in the middle.
        shift = self._team_x_shift()
        left_x = 80 + shift
        right_x = SCREEN_W - 80 + shift
        knot_x = CENTER_X + self.rope_offset
        wobble_amp = 6 + 3 * (self.left.pull_pulse + self.right.pull_pulse) if (self.left and self.right) else 6

        # Draw rope as a series of segments with vertical wobble
        prev = (left_x, ROPE_Y)
        steps = 60
        for i in range(1, steps + 1):
            t = i / steps
            x = left_x + (right_x - left_x) * t
            sag = math.sin(t * math.pi) * 4
            wob = math.sin(t * 8 + self.rope_wobble_phase) * (wobble_amp * t * (1 - t) * 4)
            y = ROPE_Y + sag + wob
            pygame.draw.line(surf, (160, 110, 60), prev, (x, y), 6)
            prev = (x, y)

        # Knot / flag
        flag_h = 36
        flag_w = 28
        # Knot
        pygame.draw.circle(surf, (90, 50, 20), (int(knot_x), ROPE_Y), 10)
        pygame.draw.circle(surf, (60, 30, 10), (int(knot_x), ROPE_Y), 10, 2)
        # Pole
        pygame.draw.line(surf, (40, 40, 40), (knot_x, ROPE_Y), (knot_x, ROPE_Y - flag_h), 3)
        # Flag color depends on which side is winning
        if self.rope_offset < 0:
            flag_color = LEFT_COLOR
        elif self.rope_offset > 0:
            flag_color = RIGHT_COLOR
        else:
            flag_color = (240, 240, 240)
        pts = [
            (knot_x, ROPE_Y - flag_h),
            (knot_x + flag_w, ROPE_Y - flag_h + 8),
            (knot_x, ROPE_Y - flag_h + 16),
        ]
        pygame.draw.polygon(surf, flag_color, pts)

    def _draw_play(self, surf: pygame.Surface) -> None:
        self._draw_scene(surf)
        self._draw_kids(surf)
        self._draw_rope(surf)
        draw_particles(surf, self.particles)
        self._draw_top_bar(surf)
        # Calculators
        if self.left and self.right:
            self.left.calc.draw(surf, self.font_big, self.font_med, self.font_sm)
            self.right.calc.draw(surf, self.font_big, self.font_med, self.font_sm)
            # Help footer
            help_l = self.font_sm.render(
                "RED: number row + - * / . | Enter=submit  Backspace=del  C=clear",
                True, (230, 230, 230),
            )
            help_r = self.font_sm.render(
                "BLUE: numpad 0-9 + - * / . | KP-Enter=submit  Del=clear  (mouse for backspace)",
                True, (230, 230, 230),
            )
            surf.blit(help_l, help_l.get_rect(midleft=(20, SCENE_BOTTOM - 14)))
            surf.blit(help_r, help_r.get_rect(midright=(SCREEN_W - 20, SCENE_BOTTOM - 14)))

    def _draw_menu(self, surf: pygame.Surface) -> None:
        self._draw_scene(surf)
        # A pretend match still happening behind the menu
        if not self.left and not self.right:
            # Phantom kids for atmosphere
            tmp_left = make_team("left", 4, GROUND_Y, (90, 380))
            tmp_right = make_team("right", 4, GROUND_Y, (SCREEN_W - 380, SCREEN_W - 90))
            for k in tmp_left:
                draw_kid(surf, k, self.t, 0.4, 0.0, ROPE_Y)
            for k in tmp_right:
                draw_kid(surf, k, self.t, 0.4, 0.0, ROPE_Y)
            # rope across
            pygame.draw.line(surf, (160, 110, 60), (80, ROPE_Y), (SCREEN_W - 80, ROPE_Y), 6)
            pygame.draw.circle(surf, (90, 50, 20), (CENTER_X, ROPE_Y), 10)

        # Title card
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 110))
        surf.blit(overlay, (0, 0))

        title = self.font_huge.render("TUG-OF-WAR MATH", True, (255, 230, 90))
        sub = self.font_med.render(
            "Two teams. One rope. A blizzard of math problems.", True, (240, 240, 240),
        )
        surf.blit(title, title.get_rect(midtop=(SCREEN_W // 2, 90)))
        surf.blit(sub, sub.get_rect(midtop=(SCREEN_W // 2, 170)))

        # Grade picker
        picker_y = 260
        label = self.font_big.render("Pick a grade band:", True, (255, 255, 255))
        surf.blit(label, label.get_rect(midtop=(SCREEN_W // 2, picker_y)))
        # Render the grades horizontally; selected one highlighted
        x = SCREEN_W // 2 - (len(GRADES) - 1) * 90 // 2
        for i, g in enumerate(GRADES):
            color = (255, 255, 255)
            bg = None
            if i == self.grade_index:
                color = (0, 0, 0)
                bg = (255, 230, 90)
            text = self.font_big.render(g, True, color)
            rect = text.get_rect(center=(x + i * 90, picker_y + 60))
            box = rect.inflate(20, 14)
            if bg:
                pygame.draw.rect(surf, bg, box, border_radius=10)
            else:
                pygame.draw.rect(surf, (60, 60, 80), box, border_radius=10)
                pygame.draw.rect(surf, (180, 180, 200), box, 2, border_radius=10)
            surf.blit(text, rect)

        # Controls help
        lines = [
            "Use LEFT/RIGHT to change grade.  Press SPACE to start.",
            "",
            "RED TEAM (left side):  number row 0-9, + - * /, ., Enter to submit, Backspace to delete, C to clear",
            "BLUE TEAM (right side):  numpad 0-9, + - * /, ., KP-Enter to submit, Delete to clear",
            "Either team can also click the on-screen calculator buttons.",
            "",
            "Wrong answer = you get LOCKED for ~1.4 seconds and the other team gains rope.",
            "First team to drag the flag to their side wins.  90-second round.",
        ]
        y = 460
        for line in lines:
            t = self.font_sm.render(line, True, (230, 230, 240))
            surf.blit(t, t.get_rect(midtop=(SCREEN_W // 2, y)))
            y += 22

    def _draw_over(self, surf: pygame.Surface) -> None:
        self._draw_play(surf)
        overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surf.blit(overlay, (0, 0))
        if self.winner is None:
            msg = "TIE!"
            color = (255, 255, 255)
        else:
            msg = f"{self.winner} TEAM WINS!"
            color = LEFT_COLOR if self.winner == "RED" else RIGHT_COLOR
        title = self.font_huge.render(msg, True, color)
        surf.blit(title, title.get_rect(midtop=(SCREEN_W // 2, 200)))

        if self.left and self.right:
            stats = [
                f"RED   correct: {self.left.score_correct}   wrong: {self.left.score_wrong}",
                f"BLUE  correct: {self.right.score_correct}   wrong: {self.right.score_wrong}",
            ]
            y = 320
            for s in stats:
                t = self.font_med.render(s, True, (240, 240, 240))
                surf.blit(t, t.get_rect(midtop=(SCREEN_W // 2, y)))
                y += 36

        prompt = self.font_med.render(
            "Press R for rematch, M for menu, Esc to quit.",
            True, (255, 230, 90),
        )
        surf.blit(prompt, prompt.get_rect(midtop=(SCREEN_W // 2, 440)))

    def draw(self) -> None:
        # Screen shake
        offset = (0, 0)
        if self.shake_timer > 0:
            mag = int(self.shake_timer * 30)
            offset = (random.randint(-mag, mag), random.randint(-mag, mag))

        scratch = pygame.Surface((SCREEN_W, SCREEN_H))
        if self.state == self.STATE_MENU:
            self._draw_menu(scratch)
        elif self.state == self.STATE_PLAY:
            self._draw_play(scratch)
        elif self.state == self.STATE_OVER:
            self._draw_over(scratch)

        self.screen.fill((0, 0, 0))
        self.screen.blit(scratch, offset)
        pygame.display.flip()

    # ---- main loop -----------------------------------------------------

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(dt)
            self.draw()


def main() -> None:
    Game().run()


if __name__ == "__main__":
    main()

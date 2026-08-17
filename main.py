"""RopeRush - a math tug-of-war for Pre-K through 4th grade.

Two squads brace against one rope. Solve your problem faster than the other
side and your team heaves; miss it and you're locked out while they drag you
toward the line. Six modes, an adaptive curriculum, and a profile that tracks
which skills a child actually needs to practise.

Run with:  python main.py

Architecture
------------
`app.App` owns the window and renders everything into a fixed 1280x720 logical
canvas that letterbox-scales to any phone held horizontally. This module owns
the scenes and the match loop; the heavy lifting lives in dedicated modules:

    theme / render_utils   design tokens and cached drawing primitives
    backdrop / vfx         animated arena, particles, camera juice
    characters             the kids and their pull animation
    ui / icons             widget kit and vector icons
    manipulatives          ten-frames, number lines, clocks, coins...
    math_engine            curriculum problem generation
    modes                  per-mode rules, power-ups, AI opponent
    progression / content  save file, XP, unlocks, mode metadata
    audio                  synthesized SFX and music
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import pygame

import backdrop as BD
import characters as CH
import content as CO
import icons as IC
import manipulatives as MP
import math_engine as ME
import modes as MD
import progression as PR
import render_utils as R
import theme as T
import ui as U
import vfx as VFX
from app import App, Scene
from audio import AudioEngine


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

W, H = T.LOGICAL_W, T.LOGICAL_H

#: The arena occupies the upper band; the two answer panels flank it below.
PANEL_W = 336
PANEL_TOP = 296
PANEL_H = H - PANEL_TOP - T.SAFE_B

ARENA_TOP = 92
GROUND_Y = 282
ROPE_Y = GROUND_Y - 54

#: Rope travel in logical px from centre out to a team's victory line.
ROPE_TRAVEL = 300


def side_palette(side: str) -> T.TeamPalette:
    return T.EMBER if side == "left" else T.FROST


def _veil(surf, alpha: int) -> None:
    """Darken the arena so foreground UI stays legible over it."""
    layer = pygame.Surface((W, H), pygame.SRCALPHA)
    layer.fill((6, 8, 22, alpha))
    surf.blit(layer, (0, 0))


# --------------------------------------------------------------------------- #
# Answer input
# --------------------------------------------------------------------------- #

class Keypad:
    """A numeric pad, or a row of choice buttons, for one team.

    Each problem declares its own `input_mode`. Most are typed on the pad, but
    comparisons and clock readings arrive as `choice` with pre-formatted
    labels ("2:40") - asking a six-year-old to type 240 for twenty-to-three
    would be a UI problem masquerading as a maths one.
    """

    def __init__(self, rect: pygame.Rect, side: str, on_submit) -> None:
        self.rect = rect
        self.side = side
        self.palette = side_palette(side)
        self.on_submit = on_submit
        self.buffer = ""
        self.enabled = True
        self.mode = "number"
        self._widgets: List[U.Widget] = []
        self._build_numeric()

    # ---- construction ----

    def _build_numeric(self) -> None:
        self.mode = "number"
        self._widgets = []
        pad = 6
        cols, rows = 3, 4
        bw = (self.rect.width - pad * (cols + 1)) / cols
        bh = (self.rect.height - pad * (rows + 1)) / rows
        keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "del", "0", "ok"]
        for i, k in enumerate(keys):
            c, r = i % cols, i // cols
            rc = pygame.Rect(
                int(self.rect.left + pad + c * (bw + pad)),
                int(self.rect.top + pad + r * (bh + pad)),
                int(bw), int(bh),
            )
            if k == "del":
                # Squircle rather than circle: the pad reads as one grid.
                wgt = U.IconButton(rc, "backspace", variant="danger",
                                   on_click=self._backspace, shape="squircle")
            elif k == "ok":
                wgt = U.IconButton(rc, "check", variant="primary",
                                   palette=self.palette, on_click=self._submit,
                                   shape="squircle", glow=True)
            else:
                wgt = U.Button(rc, k, variant="secondary", font_size=T.T_HEAD,
                               radius=T.R_MD, on_click=(lambda d=k: self._digit(d)))
            self._widgets.append(wgt)

    def _build_choices(self, choices, labels) -> None:
        self.mode = "choice"
        vals = list(choices)
        text = list(labels or [f"{v:g}" for v in vals])
        self._widgets = []
        n = max(1, len(vals))
        pad = 8
        rows = 2 if n > 2 else n
        cols = math.ceil(n / rows)
        bw = (self.rect.width - pad * (cols + 1)) / cols
        bh = (self.rect.height - pad * (rows + 1)) / rows
        for i in range(n):
            c, r = i % cols, i // cols
            rc = pygame.Rect(
                int(self.rect.left + pad + c * (bw + pad)),
                int(self.rect.top + pad + r * (bh + pad)),
                int(bw), int(bh),
            )
            label = text[i] if i < len(text) else f"{vals[i]:g}"
            self._widgets.append(
                U.Button(rc, label, variant="secondary", font_size=T.T_HEAD,
                         radius=T.R_LG,
                         on_click=(lambda v=vals[i]: self.on_submit(v)))
            )

    def set_problem(self, problem: ME.Problem) -> None:
        self.buffer = ""
        wants_choice = (getattr(problem, "input_mode", "number") == "choice"
                        and problem.choices)
        if wants_choice:
            self._build_choices(problem.choices,
                                getattr(problem, "choice_labels", None))
        elif self.mode != "number":
            self._build_numeric()

    # ---- input ----

    def _digit(self, d: str) -> None:
        if self.enabled and len(self.buffer) < 6:
            self.buffer += d

    def _backspace(self) -> None:
        if self.enabled:
            self.buffer = self.buffer[:-1]

    def _submit(self) -> None:
        if self.enabled and self.buffer:
            try:
                value = float(self.buffer)
            except ValueError:
                self.buffer = ""
                return
            self.buffer = ""
            self.on_submit(value)

    def key(self, ch: str) -> None:
        """Route a physical keystroke, for desktop play."""
        if not self.enabled or self.mode != "number":
            return
        if ch == "\b":
            self._backspace()
        elif ch == "\r":
            self._submit()
        elif ch.isdigit():
            self._digit(ch)

    def handle_event(self, event) -> bool:
        if not self.enabled:
            return False
        return any(wgt.handle_event(event) for wgt in self._widgets)

    def update(self, dt: float) -> None:
        for wgt in self._widgets:
            wgt.enabled = self.enabled
            wgt.update(dt)

    def draw(self, surf) -> None:
        for wgt in self._widgets:
            wgt.draw(surf)


class AnswerPanel:
    """One team's half of the screen: problem, manipulative and input."""

    def __init__(self, side: str, rect: pygame.Rect, on_submit) -> None:
        self.side = side
        self.rect = rect
        self.palette = side_palette(side)
        self.problem: Optional[ME.Problem] = None
        self.flash = 0.0          # >0 just answered right, <0 just answered wrong
        self.lock = 0.0
        self.hint: Optional[str] = None
        self.ai = False           # driven by the AI, so no input is drawn

        pad = T.S3
        self.prompt_rect = pygame.Rect(rect.left + pad, rect.top + pad,
                                       rect.width - pad * 2, 128)
        keypad_rect = pygame.Rect(
            rect.left + pad, self.prompt_rect.bottom + T.S2,
            rect.width - pad * 2,
            rect.bottom - self.prompt_rect.bottom - T.S2 - pad)
        self.keypad = Keypad(keypad_rect, side, on_submit)

    def set_problem(self, problem: ME.Problem) -> None:
        self.problem = problem
        self.hint = None
        self.keypad.set_problem(problem)

    def update(self, dt: float) -> None:
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt * 2.2)
        else:
            self.flash = min(0.0, self.flash + dt * 2.2)
        self.lock = max(0.0, self.lock - dt)
        self.keypad.enabled = self.lock <= 0 and not self.ai
        self.keypad.update(dt)

    def handle_event(self, event) -> bool:
        return False if self.ai else self.keypad.handle_event(event)

    def draw(self, surf, t: float) -> None:
        tint = (255, 255, 255, 18)
        if self.flash > 0:
            k = self.flash
            tint = (110, 255, 170, int(18 + 40 * k))
        elif self.flash < 0:
            k = -self.flash
            tint = (255, 120, 120, int(18 + 40 * k))
        R.glass_panel(surf, self.rect, radius=T.R_XL, tint=tint)

        # Team stripe down the outer edge.
        stripe = pygame.Rect(self.rect.left + 10, self.rect.top + 10, 6,
                             self.rect.height - 20)
        if self.side == "right":
            stripe.right = self.rect.right - 10
        R.draw_rrect(surf, stripe, self.palette.core, T.R_PILL)

        if self.problem is None:
            return

        pr = self.prompt_rect
        lines = self.problem.prompt.splitlines()[:2]
        base = T.T_TITLE if (len(lines) == 1 and len(lines[0]) <= 13) else T.T_HEAD
        if self.problem.visual is not None:
            base = min(base, T.T_HEAD)
        y = pr.top + 4
        for line in lines:
            font = U.fit_font(line, pr.width - 16, base, bold=True)
            R.draw_text(surf, line, font, T.INK, midtop=(pr.centerx, y))
            y += font.get_height() + 2

        # Manipulative fills whatever room the prompt left behind.
        vis_bottom = pr.bottom - (44 if not self.ai and self.keypad.mode == "number" else 0)
        if self.problem.visual is not None and vis_bottom - y > 24:
            MP.draw_visual(surf, pygame.Rect(pr.left, y, pr.width, vis_bottom - y),
                           self.problem.visual, self.palette, t)

        if self.ai:
            self._draw_ai(surf, t)
            return

        if self.keypad.mode == "number":
            box = pygame.Rect(pr.left + 24, pr.bottom - 40, pr.width - 48, 42)
            R.draw_rrect(surf, box, (10, 12, 30, 200), T.R_MD)
            R.draw_rrect(surf, box, T.with_alpha(self.palette.core, 160), T.R_MD, width=2)
            shown = self.keypad.buffer or "_"
            R.draw_text(surf, shown, R.font_num(T.T_HEAD, bold=True),
                        T.INK if self.keypad.buffer else T.INK_FAINT,
                        center=box.center)

        self.keypad.draw(surf)
        if self.lock > 0:
            self._draw_lock(surf)
        if self.hint:
            for i, line in enumerate(U.wrap_text(self.hint, R.font_text(T.T_MICRO),
                                                 self.rect.width - 28)[:2]):
                R.draw_text(surf, line, R.font_text(T.T_MICRO), T.GOLD,
                            midbottom=(self.rect.centerx,
                                       self.rect.bottom - 8 - (1 - i) * 16))

    def _draw_ai(self, surf, t: float) -> None:
        cx, cy = self.rect.centerx, self.keypad.rect.centery
        R.add_glow(surf, (cx, cy), 96, self.palette.glow, 60)
        IC.draw_icon(surf, "users", pygame.Rect(cx - 34, cy - 58, 68, 68), T.INK_DIM)
        for i in range(3):
            a = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(t * 4 - i * 0.7))
            pygame.draw.circle(surf, T.lerp_color(T.INK_FAINT, self.palette.light, a),
                               (cx - 26 + i * 26, cy + 32), 7)

    def _draw_lock(self, surf) -> None:
        veil = pygame.Surface(self.keypad.rect.size, pygame.SRCALPHA)
        veil.fill((8, 10, 26, 180))
        surf.blit(veil, self.keypad.rect.topleft)
        cx, cy = self.keypad.rect.center
        IC.draw_icon(surf, "lock", pygame.Rect(cx - 26, cy - 46, 52, 52), T.BAD)
        R.draw_text(surf, f"{self.lock:0.1f}s", R.font_num(T.T_HEAD, bold=True),
                    T.BAD, center=(cx, cy + 24))


# --------------------------------------------------------------------------- #
# Match
# --------------------------------------------------------------------------- #

class MatchScene(Scene):
    """The game itself: two squads, one rope, two streams of problems."""

    music = "match"

    def on_enter(self, mode: str = "classic", grade: str = "1st",
                 versus: bool = False, **kw) -> None:
        app = self.app
        self.mode_key = mode
        self.grade = grade

        difficulty = CO.pick_difficulty(app.profile, grade)
        self.rules = MD.make_mode(mode, {"grade": grade, "difficulty": difficulty})
        self.state = self.rules.new_state()
        self.rules.on_start(self.state)
        self.player_side = getattr(self.rules, "player_side", "left")

        weak = app.profile.weakest_skills(3)
        self.streams = {
            "left": ME.ProblemStream(grade, difficulty, weak_skills=weak),
            "right": ME.ProblemStream(grade, difficulty),
        }

        # Solo modes (and versus-AI) hand the far side to a bot.
        self.ai: Optional[MD.AIOpponent] = None
        if self.rules.solo or not versus:
            self.ai = MD.AIOpponent(grade, 0.35 + 0.40 * difficulty)

        self.panels = {
            "left": AnswerPanel("left",
                                pygame.Rect(T.SAFE_L - 24, PANEL_TOP, PANEL_W, PANEL_H),
                                lambda v: self.submit("left", v)),
            "right": AnswerPanel("right",
                                 pygame.Rect(W - T.SAFE_R + 24 - PANEL_W, PANEL_TOP,
                                             PANEL_W, PANEL_H),
                                 lambda v: self.submit("right", v)),
        }
        self.panels["right"].ai = self.ai is not None

        self.asked: Dict[str, float] = {"left": 0.0, "right": 0.0}
        for side, panel in self.panels.items():
            panel.set_problem(self.streams[side].next())

        # Squads pull toward each other across the centre line.
        self.teams = {
            "left": CH.make_team(T.EMBER, "left", 3, GROUND_Y, (W * 0.28, W * 0.41)),
            "right": CH.make_team(T.FROST, "right", 3, GROUND_Y, (W * 0.59, W * 0.72)),
        }
        self.pull = {"left": 0.0, "right": 0.0}

        self.t = 0.0
        self.countdown = 3.0
        self.finished: Optional[MD.MatchResult] = None
        self.result_delay = 0.0
        self.rope_visual = 0.0
        self._tense = False
        self._last_tick = -1

        self.pause_btn = U.IconButton(
            pygame.Rect(T.SAFE_L, T.SAFE_T + 4, 52, 52), "pause",
            variant="ghost", on_click=self._quit_to_menu)

        app.vfx.clear()
        app.audio.play("whistle_start")

    # ---- flow ----

    def _quit_to_menu(self) -> None:
        self.app.audio.play("ui_back")
        self.app.goto("menu")

    def submit(self, side: str, value: float) -> None:
        """A team committed to an answer."""
        if self.finished or self.countdown > 0:
            return
        panel = self.panels[side]
        problem = panel.problem
        if problem is None or self.state.lockouts.get(side, 0.0) > 0:
            return

        app = self.app
        elapsed = max(0.05, self.t - self.asked.get(side, self.t))
        correct = bool(problem.check(value))

        if correct:
            res = self.rules.on_correct(self.state, side, problem, elapsed)
            panel.flash = 1.0
            self.pull[side] = min(1.5, self.pull[side] + 1.0)
            for kid in self.teams[side]:
                CH.trigger_heave(kid, 1.0)
            self._celebrate(side, res)
            panel.set_problem(self.streams[side].next())
            self.asked[side] = self.t
        else:
            self.rules.on_wrong(self.state, side, problem)
            panel.flash = -1.0
            panel.lock = self.state.lockouts.get(side, 0.0)
            app.audio.play("wrong")
            app.vfx.dust(self._rope_x(), ROPE_Y + 10,
                         1 if side == "left" else -1, 16)
            app.vfx.shake(5, 0.2)
            if self.mode_key == "practice" and app.profile.settings.show_hints:
                panel.hint = problem.hint

        # Only the human's answers should shape the saved profile.
        if side == self.player_side:
            app.profile.record_answer(problem.skill, self.grade, correct,
                                      int(elapsed * 1000))
        self.streams[side].report(problem, correct, elapsed)

    def _celebrate(self, side: str, res: MD.PullResult) -> None:
        app = self.app
        pal = side_palette(side)
        x, y = self._rope_x(), ROPE_Y
        app.vfx.shockwave(x, y, pal.glow)
        app.vfx.burst(x, y, pal.glow, count=26)
        app.vfx.shake(7 + 3 * min(res.combo, 5), 0.22)
        app.audio.play("correct" if res.combo < 2 else "correct_streak",
                       pitch=1.0 + 0.05 * min(res.combo, 6))
        app.audio.play("rope_pull", 0.7)
        app.audio.play("kid_effort", 0.5)
        if res.combo >= 2:
            app.audio.play(f"combo_{min(5, res.combo)}", 0.8)
            # Rises from below the rope so it never climbs into the HUD chip.
            app.vfx.floating_text(x, y + 34, f"{res.combo}x COMBO", T.GOLD)
        for event in res.events:
            if event == "powerup_spawn":
                app.audio.play("powerup_pickup")
                app.toasts.push("Power-up!", "success", icon="bolt")

    def _rope_x(self) -> float:
        return W / 2 + self.rope_visual * ROPE_TRAVEL

    # ---- loop ----

    def handle_event(self, event) -> None:
        if self.finished:
            return
        if self.pause_btn.handle_event(event):
            return
        for panel in self.panels.values():
            if panel.handle_event(event):
                self.app.audio.play("keypad_press", 0.5)
                return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self._quit_to_menu()
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.panels[self.player_side].keypad.key("\r")
            elif event.key == pygame.K_BACKSPACE:
                self.panels[self.player_side].keypad.key("\b")
            elif event.unicode and event.unicode.isdigit():
                self.panels[self.player_side].keypad.key(event.unicode)

    def update(self, dt: float) -> None:
        app = self.app
        self.t += dt
        app.vfx.update(dt)

        if self.countdown > 0:
            before = int(math.ceil(self.countdown))
            self.countdown -= dt
            after = int(math.ceil(self.countdown))
            if after != before and after >= 0:
                app.audio.play("countdown_go" if after == 0 else "countdown_tick")
            if self.countdown <= 0:
                self.asked = {"left": self.t, "right": self.t}
            app.backdrop.update(dt, 0.0)
            return

        for panel in self.panels.values():
            panel.update(dt)

        if self.finished:
            self.result_delay -= dt
            self._settle(dt)
            app.backdrop.update(dt, -self.rope_visual)
            if self.result_delay <= 0:
                app.goto("results", result=self.finished, mode=self.mode_key,
                         grade=self.grade)
            return

        self.rules.update(self.state, dt)

        if self.ai is not None:
            verdict = self.ai.update(dt)
            if verdict is not None:
                problem = self.panels["right"].problem
                if problem is not None:
                    # A deliberate miss: nudge off the real answer.
                    self.submit("right",
                                problem.answer if verdict else problem.answer + 1)

        if self.rules.timed and not self._tense and 0 < self.state.time_left <= 15:
            self._tense = True
            app.audio.play_music("tense")

        if self.rules.timed:
            sec = int(math.ceil(self.state.time_left))
            if sec != self._last_tick and 0 < sec <= 5:
                app.audio.play("countdown_tick", 0.6)
            self._last_tick = sec

        self._settle(dt)
        app.backdrop.update(dt, -self.rope_visual)

        result = self.rules.check_end(self.state)
        if result is not None:
            self._finish(result)

    def _settle(self, dt: float) -> None:
        """Ease drawn state toward simulated state so pulls read as a heave."""
        self.rope_visual += (self.state.rope - self.rope_visual) * min(1.0, dt * 7.0)
        strain = {
            "left": max(0.0, min(1.0, self.rope_visual)),
            "right": max(0.0, min(1.0, -self.rope_visual)),
        }
        for side, kids in self.teams.items():
            self.pull[side] = max(0.0, self.pull[side] - dt * 1.6)
            for kid in kids:
                CH.update_kid(kid, dt, min(1.0, self.pull[side]), strain[side])

    def _finish(self, result: MD.MatchResult) -> None:
        app = self.app
        self.finished = result
        self.result_delay = 2.0
        won = (result.winner == self.player_side
               or result.reason == "boss_defeated")

        app.audio.stop_music()
        app.audio.play("whistle_end")
        app.audio.play("victory_fanfare" if won else "defeat_sting")
        app.audio.play("crowd_cheer" if won else "crowd_gasp", 0.85)
        if won:
            app.vfx.confetti(W / 2, 110, 140)
        app.vfx.flash(T.GOLD if won else T.BAD, 0.32, 0.4)

        rewards = app.profile.record_match(
            won=won, stars=result.stars, mode=self.mode_key,
            grade=self.grade, score=result.score)
        if self.mode_key == "survival":
            try:
                app.profile.record_survival(self.state.elapsed)
            except Exception:
                pass
        app.profile.save()
        if result.stats is None:
            result.stats = {}
        result.stats["rewards"] = rewards
        result.stats["won"] = won

    # ---- draw ----

    def draw(self, surf) -> None:
        app = self.app
        app.backdrop.draw(surf)
        app.vfx.draw_below(surf)

        self._draw_lines(surf)
        shift = self.rope_visual * 44
        CH.draw_team(surf, self.teams["left"], self.t, min(1.0, self.pull["left"]),
                     max(0.0, min(1.0, self.rope_visual)), ROPE_Y, shift)
        CH.draw_team(surf, self.teams["right"], self.t, min(1.0, self.pull["right"]),
                     max(0.0, min(1.0, -self.rope_visual)), ROPE_Y, shift)
        self._draw_rope(surf, shift)

        app.vfx.draw_above(surf)
        app.vfx.draw_additive(surf)
        app.backdrop.draw_foreground(surf)

        self._draw_hud(surf)
        for panel in self.panels.values():
            panel.draw(surf, self.t)
        self.pause_btn.draw(surf)

        if self.countdown > 0:
            self._draw_countdown(surf)
        app.vfx.draw_fullscreen(surf)

    def _draw_lines(self, surf) -> None:
        for side in ("left", "right"):
            x = W / 2 + (-1 if side == "left" else 1) * ROPE_TRAVEL
            pygame.draw.line(surf, T.with_alpha(side_palette(side).core, 90),
                             (x, ARENA_TOP + 46), (x, GROUND_Y + 8), 3)
        pygame.draw.line(surf, T.with_alpha(T.INK_FAINT, 70),
                         (W / 2, ARENA_TOP + 64), (W / 2, GROUND_Y + 8), 2)

    def _draw_rope(self, surf, shift: float) -> None:
        knot_x = self._rope_x()
        x0, x1 = W * 0.18 + shift, W * 0.82 + shift
        tension = self.pull["left"] + self.pull["right"]
        pts = []
        for i in range(29):
            k = i / 28
            x = x0 + (x1 - x0) * k
            sag = math.sin(k * math.pi) * 7
            tremor = math.sin(self.t * 22 + k * 10) * 1.6 * tension
            pts.append((x, ROPE_Y + sag + tremor))
        R.draw_polyline_glow(surf, pts, T.GOLD_DEEP, width=6,
                             glow_width=16, glow_alpha=60)

        R.add_glow(surf, (int(knot_x), int(ROPE_Y)), 46, T.GOLD, 120)
        pygame.draw.circle(surf, T.GOLD_DEEP, (int(knot_x), int(ROPE_Y)), 11)
        pygame.draw.circle(surf, T.GOLD_LIGHT, (int(knot_x), int(ROPE_Y)), 11, 2)
        lean = -self.rope_visual * 16
        pygame.draw.polygon(surf, T.GOLD, [
            (knot_x, ROPE_Y - 10),
            (knot_x + lean + 22, ROPE_Y - 30),
            (knot_x, ROPE_Y - 44),
        ])

    def _draw_hud(self, surf) -> None:
        st = self.state
        chip = pygame.Rect(0, 0, 430, 76)
        chip.midtop = (W // 2, T.SAFE_T)
        R.glass_panel(surf, chip, radius=T.R_LG)
        for side, ax in (("left", chip.left + 60), ("right", chip.right - 60)):
            R.draw_text(surf, str(st.scores.get(side, 0)),
                        R.font_num(T.T_HEAD, bold=True), side_palette(side).light,
                        center=(ax, chip.centery - 7))
            streak = st.streaks.get(side, 0)
            if streak >= 2:
                R.draw_text(surf, f"{streak} streak", R.font_text(T.T_MICRO, True),
                            T.GOLD, center=(ax, chip.centery + 20))
        if self.rules.timed:
            secs = max(0, int(st.time_left))
            label = f"{secs // 60}:{secs % 60:02d}"
            colour = T.BAD if st.time_left <= 10 else T.INK
        else:
            label = f"{int(st.elapsed)}s"
            colour = T.INK
        R.draw_text(surf, label, R.font_num(T.T_TITLE, bold=True), colour,
                    center=chip.center)

        self._draw_tug_bar(surf)
        self._draw_extras(surf, self.rules.hud_extras(st))

    def _draw_tug_bar(self, surf) -> None:
        bar = pygame.Rect(0, 0, 520, 14)
        bar.midtop = (W // 2, T.SAFE_T + 84)
        R.draw_rrect(surf, bar, (10, 12, 30, 180), T.R_PILL)
        mid = bar.centerx
        half = self.rope_visual * (bar.width / 2)
        if abs(half) > 1:
            pal = T.EMBER if half < 0 else T.FROST
            fill = pygame.Rect(int(min(mid, mid + half)), bar.top + 2,
                               int(abs(half)), bar.height - 4)
            R.draw_rrect(surf, fill, pal.core, T.R_PILL)
        pygame.draw.line(surf, T.INK_DIM, (mid, bar.top - 4), (mid, bar.bottom + 4), 2)
        pygame.draw.circle(surf, T.GOLD, (int(mid + half), bar.centery), 8)

    def _draw_extras(self, surf, extras: dict) -> None:
        """Boss HP and phase, plus any active power-ups."""
        # Sits clear of the tug bar, which ends at SAFE_T + 98.
        y = T.SAFE_T + 108
        if extras.get("boss_max_hp"):
            frac = extras.get("boss_hp", 0) / max(1e-6, extras["boss_max_hp"])
            bar = pygame.Rect(0, 0, 360, 22)
            bar.midtop = (W // 2, y)
            R.draw_rrect(surf, bar, (10, 12, 30, 200), T.R_PILL)
            fill = bar.copy()
            fill.width = max(0, int(bar.width * max(0.0, frac)))
            R.draw_rrect(surf, fill, T.BAD if frac < 0.35 else T.EMBER.core, T.R_PILL)
            R.draw_text(surf, str(extras.get("phase_name", "Boss")),
                        R.font_text(T.T_MICRO, True), T.INK, center=bar.center)
            y += 30
            if extras.get("telegraph"):
                R.draw_text(surf, "INCOMING!", R.font_display(T.T_HEAD, True),
                            T.BAD, midtop=(W // 2, y))

        for side in ("left", "right"):
            active = (extras.get("powerups") or {}).get(side) or {}
            base_x = 150 if side == "left" else W - 150
            for i, key in enumerate(list(active)[:3]):
                pu = MD.POWERUPS.get(key)
                rc = pygame.Rect(base_x - 22 + i * 50, T.SAFE_T + 108, 44, 44)
                R.glass_panel(surf, rc, radius=T.R_MD, shadow=False)
                IC.draw_icon(surf, getattr(pu, "icon", "bolt"), rc.inflate(-14, -14),
                             side_palette(side).light)

    def _draw_countdown(self, surf) -> None:
        _veil(surf, 150)
        n = int(math.ceil(self.countdown))
        label = str(n) if n > 0 else "PULL!"
        frac = self.countdown - math.floor(self.countdown)
        scale = 1.0 + 0.5 * (1.0 - frac)
        R.draw_text(surf, label, R.font_display(int(T.T_HERO * scale), True),
                    T.GOLD, center=(W // 2, H // 2),
                    outline=(40, 20, 10), outline_w=4)


# --------------------------------------------------------------------------- #
# Menu and navigation
# --------------------------------------------------------------------------- #

class MenuScene(Scene):
    music = "menu"

    def on_enter(self, **kw) -> None:
        app = self.app
        try:
            app.profile.touch_daily()
        except Exception:
            pass
        self.t = 0.0
        cx = W // 2
        self.buttons = [
            U.Button(pygame.Rect(cx - 150, 344, 300, 78), "PLAY", icon="play",
                     variant="primary", palette=T.EMBER, glow=True,
                     font_size=T.T_HEAD, on_click=lambda: self._go("modes")),
            U.Button(pygame.Rect(cx - 150, 434, 143, 60), "Stats", icon="chart",
                     variant="secondary", on_click=lambda: self._go("profile")),
            U.Button(pygame.Rect(cx + 7, 434, 143, 60), "Settings", icon="gear",
                     variant="secondary", on_click=lambda: self._go("settings")),
        ]
        # A silent exhibition match idles behind the menu. The squads sit hard
        # against the outer edges so the whole centre column - title, buttons,
        # level chip - stays clear of them.
        self.demo = {
            "left": CH.make_team(T.EMBER, "left", 2, GROUND_Y + 96, (W * 0.05, W * 0.15)),
            "right": CH.make_team(T.FROST, "right", 2, GROUND_Y + 96, (W * 0.85, W * 0.95)),
        }

    def _go(self, scene: str) -> None:
        self.app.audio.play("ui_tap")
        self.app.goto(scene)

    def handle_event(self, event) -> None:
        for b in self.buttons:
            if b.handle_event(event):
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.quit()

    def update(self, dt: float) -> None:
        self.t += dt
        self.app.backdrop.update(dt, math.sin(self.t * 0.4) * 0.5)
        self.app.vfx.update(dt)
        for kids in self.demo.values():
            for kid in kids:
                CH.update_kid(kid, dt, 0.55, 0.15)
        for b in self.buttons:
            b.update(dt)

    def draw(self, surf) -> None:
        app = self.app
        app.backdrop.draw(surf)
        rope_y = GROUND_Y + 96 - 54
        CH.draw_team(surf, self.demo["left"], self.t, 0.55, 0.1, rope_y)
        CH.draw_team(surf, self.demo["right"], self.t, 0.55, 0.1, rope_y)
        pygame.draw.line(surf, T.GOLD_DEEP, (W * 0.03, rope_y), (W * 0.97, rope_y), 5)
        _veil(surf, 120)

        title = R.gradient_text("ROPE RUSH", R.font_display(96, True),
                                T.GOLD_LIGHT, T.GOLD_DEEP,
                                outline=(30, 16, 8), outline_w=4)
        surf.blit(title, title.get_rect(center=(W // 2, 178)))
        R.draw_text(surf, "MATH TUG-OF-WAR", R.font_display(T.T_HEAD, True),
                    T.INK_DIM, center=(W // 2, 252))

        self._draw_level_chip(surf)
        for b in self.buttons:
            b.draw(surf)
        app.backdrop.draw_foreground(surf)

    def _draw_level_chip(self, surf) -> None:
        p = self.app.profile
        level, progress = PR.level_for_xp(p.xp)
        chip = pygame.Rect(0, 0, 310, 54)
        chip.midtop = (W // 2, 536)
        R.glass_panel(surf, chip, radius=T.R_PILL)
        R.draw_text(surf, f"Level {level}", R.font_text(T.T_LABEL, True), T.INK,
                    midleft=(chip.left + 20, chip.centery))
        IC.draw_icon(surf, "star",
                     pygame.Rect(chip.right - 36, chip.centery - 11, 22, 22), T.GOLD)
        R.draw_text(surf, str(p.total_stars), R.font_text(T.T_LABEL, True), T.GOLD,
                    midright=(chip.right - 42, chip.centery))
        bar = pygame.Rect(chip.left + 106, chip.centery - 5, 118, 10)
        R.draw_rrect(surf, bar, (10, 12, 30, 190), T.R_PILL)
        fill = bar.copy()
        fill.width = max(2, int(bar.width * progress))
        R.draw_rrect(surf, fill, T.FROST.core, T.R_PILL)


class ModeSelectScene(Scene):
    def on_enter(self, **kw) -> None:
        app = self.app
        self.back = U.IconButton(pygame.Rect(T.SAFE_L, T.SAFE_T + 8, 56, 56),
                                 "back", variant="secondary",
                                 on_click=lambda: self._go("menu"))
        keys = list(getattr(CO, "MODE_ORDER", None) or CO.MODES)
        self.cards: List[Tuple[CO.ModeDef, pygame.Rect, bool]] = []
        cols, cw, ch, gap = 3, 344, 186, 22
        x0 = (W - (cols * cw + (cols - 1) * gap)) // 2
        for i, key in enumerate(keys):
            md = CO.MODES[key]
            c, r = i % cols, i // cols
            rect = pygame.Rect(x0 + c * (cw + gap), 178 + r * (ch + gap), cw, ch)
            unlocked = (key in app.profile.unlocked_modes
                        or app.profile.level >= md.unlock_level)
            self.cards.append((md, rect, unlocked))

    def _go(self, scene: str, **kw) -> None:
        self.app.audio.play("ui_tap")
        self.app.goto(scene, **kw)

    def handle_event(self, event) -> None:
        if self.back.handle_event(event):
            return
        if event.type == pygame.MOUSEBUTTONDOWN and getattr(event, "button", 1) == 1:
            for md, rect, unlocked in self.cards:
                if not rect.collidepoint(event.pos):
                    continue
                if unlocked:
                    self._go("grade", mode=md.key)
                else:
                    self.app.audio.play("ui_error")
                    self.app.toasts.push(f"Reach level {md.unlock_level} to unlock",
                                         "warn", icon="lock")
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._go("menu")

    def update(self, dt: float) -> None:
        self.app.backdrop.update(dt, 0.0)
        self.back.update(dt)

    def draw(self, surf) -> None:
        self.app.backdrop.draw(surf)
        _veil(surf, 155)
        R.draw_text(surf, "CHOOSE A MODE", R.font_display(T.T_DISPLAY, True),
                    T.INK, center=(W // 2, T.SAFE_T + 58))
        for i, (md, rect, unlocked) in enumerate(self.cards):
            pal = T.EMBER if i % 2 == 0 else T.FROST
            R.glass_panel(surf, rect, radius=T.R_XL,
                          tint=(255, 255, 255, 20 if unlocked else 9))
            IC.draw_icon(surf, md.icon,
                         pygame.Rect(rect.left + 22, rect.top + 22, 52, 52),
                         pal.light if unlocked else T.INK_FAINT)
            R.draw_text(surf, md.name, R.font_display(T.T_HEAD, True),
                        T.INK if unlocked else T.INK_FAINT,
                        topleft=(rect.left + 88, rect.top + 26))
            R.draw_text(surf, md.tagline, R.font_text(T.T_LABEL), T.INK_DIM,
                        topleft=(rect.left + 88, rect.top + 60))

            # Three lines fit; anything longer is elided rather than cut off
            # mid-word, which read as a rendering bug.
            body = R.font_text(T.T_MICRO)
            wrapped = U.wrap_text(md.description, body, rect.width - 44)
            for j, line in enumerate(wrapped[:3]):
                if j == 2 and len(wrapped) > 3:
                    line = line.rstrip(" ,.") + "..."
                R.draw_text(surf, line, body, T.INK_FAINT,
                            topleft=(rect.left + 22, rect.top + 106 + j * 18))

            if not unlocked:
                # Top-right, clear of the description block.
                badge = pygame.Rect(rect.right - 50, rect.top + 18, 32, 32)
                IC.draw_icon(surf, "lock", badge, T.INK_FAINT)
                R.draw_text(surf, f"Lv {md.unlock_level}", R.font_text(T.T_MICRO, True),
                            T.INK_FAINT, midtop=(badge.centerx, badge.bottom + 2))
        self.back.draw(surf)


class GradeSelectScene(Scene):
    def on_enter(self, mode: str = "classic", **kw) -> None:
        self.mode = mode
        last = getattr(self.app, "last_grade", "1st")
        self.index = ME.GRADES.index(last) if last in ME.GRADES else 2
        self.seg = U.SegmentedControl(
            pygame.Rect(W // 2 - 420, 296, 840, 78),
            [ME.grade_label(g) for g in ME.GRADES], self.index,
            on_change=self._pick, palette=T.FROST)
        self.start = U.Button(pygame.Rect(W // 2 - 150, 430, 300, 78), "START",
                              icon="play", variant="primary", palette=T.EMBER,
                              glow=True, font_size=T.T_HEAD, on_click=self._start)
        self.back = U.IconButton(pygame.Rect(T.SAFE_L, T.SAFE_T + 8, 56, 56),
                                 "back", variant="secondary",
                                 on_click=lambda: self.app.goto("modes"))

    def _pick(self, index: int) -> None:
        self.index = index
        self.app.audio.play("ui_toggle")

    def _start(self) -> None:
        grade = ME.GRADES[self.index]
        self.app.last_grade = grade
        self.app.audio.play("ui_tap")
        self.app.goto("match", mode=self.mode, grade=grade,
                      versus=self.mode in ("classic", "blitz"))

    def handle_event(self, event) -> None:
        for wgt in (self.back, self.seg, self.start):
            if wgt.handle_event(event):
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.goto("modes")

    def update(self, dt: float) -> None:
        self.app.backdrop.update(dt, 0.0)
        for wgt in (self.back, self.seg, self.start):
            wgt.update(dt)

    def draw(self, surf) -> None:
        self.app.backdrop.draw(surf)
        _veil(surf, 155)
        R.draw_text(surf, "PICK A GRADE", R.font_display(T.T_DISPLAY, True),
                    T.INK, center=(W // 2, 152))
        band = CO.band_for(ME.GRADES[self.index]) if hasattr(CO, "band_for") else None
        if band is not None and getattr(band, "blurb", None):
            R.draw_text(surf, band.blurb, R.font_text(T.T_BODY), T.INK_DIM,
                        center=(W // 2, 216))
        for wgt in (self.seg, self.start, self.back):
            wgt.draw(surf)


class ResultsScene(Scene):
    def on_enter(self, result=None, mode: str = "classic", grade: str = "1st",
                 **kw) -> None:
        self.result = result
        self.mode = mode
        self.grade = grade
        stats = (getattr(result, "stats", None) or {}) if result else {}
        self.won = bool(stats.get("won"))
        self.rewards = stats.get("rewards") or {}

        self.stars = U.StarRating(pygame.Rect(W // 2 - 110, 262, 220, 66), 3, 0,
                                  palette=T.EMBER)
        self.stars.reveal(getattr(result, "stars", 0) or 0, delay=0.35)
        self.again = U.Button(pygame.Rect(W // 2 - 226, 516, 210, 70), "Again",
                              icon="retry", variant="primary", palette=T.EMBER,
                              on_click=self._again)
        self.menu = U.Button(pygame.Rect(W // 2 + 16, 516, 210, 70), "Menu",
                             icon="home", variant="secondary",
                             on_click=lambda: self.app.goto("menu"))

        if self.rewards.get("leveled_up"):
            self.app.audio.play("level_up")
            self.app.toasts.push(f"Level {self.rewards.get('level_after')}!",
                                 "success", icon="crown")
        for key in (self.rewards.get("unlocks") or []):
            toast = self._unlock_toast(key)
            if toast is not None:
                text, icon = toast
                self.app.audio.play("unlock", 0.8)
                self.app.toasts.push(text, "success", icon=icon)

    @staticmethod
    def _unlock_toast(key: str) -> Optional[Tuple[str, str]]:
        """Turn a raw unlock event key into something a child can read.

        `progression.add_xp` emits machine keys like 'mode:blitz' or 'level:7'.
        Showing those verbatim ("Unlocked: level:2") looks like a bug, so they
        are resolved to real names here - and level events are dropped because
        the level-up toast above already covers them.
        """
        kind, _, value = str(key).partition(":")
        if kind == "level":
            return None
        if kind == "mode":
            md = CO.MODES.get(value)
            return (f"New mode: {md.name}" if md else f"New mode: {value}"), "trophy"
        if kind == "cosmetic":
            name = value.replace("_", " ").title()
            for cos in getattr(CO, "COSMETICS", []):
                if getattr(cos, "key", None) == value:
                    name = cos.name
                    break
            return f"New look: {name}", "sparkle"
        if kind == "gems":
            return f"+{value} gems", "gem"
        return f"Unlocked: {value or key}", "unlock"

    def _again(self) -> None:
        self.app.audio.play("ui_tap")
        self.app.goto("match", mode=self.mode, grade=self.grade,
                      versus=self.mode in ("classic", "blitz"))

    def handle_event(self, event) -> None:
        for b in (self.again, self.menu):
            if b.handle_event(event):
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.goto("menu")

    def update(self, dt: float) -> None:
        self.app.backdrop.update(dt, 0.0)
        self.app.vfx.update(dt)
        self.stars.update(dt)
        for b in (self.again, self.menu):
            b.update(dt)

    def draw(self, surf) -> None:
        self.app.backdrop.draw(surf)
        _veil(surf, 175)

        pal = T.EMBER if self.won else T.FROST
        head = "VICTORY!" if self.won else "SO CLOSE!"
        title = R.gradient_text(head, R.font_display(84, True), pal.light, pal.deep,
                                outline=(24, 12, 8), outline_w=4)
        surf.blit(title, title.get_rect(center=(W // 2, 172)))
        self.stars.draw(surf)

        rows: List[Tuple[str, str]] = []
        if self.result is not None:
            stats = self.result.stats or {}
            correct = stats.get("correct")
            if isinstance(correct, dict):
                rows.append(("Correct", str(correct.get("left", "-"))))
            rows.append(("Score", str(self.result.score)))
        if self.rewards:
            rows.append(("XP earned", f"+{self.rewards.get('xp', 0)}"))
            rows.append(("Coins", f"+{self.rewards.get('coins', 0)}"))

        if rows:
            box = pygame.Rect(W // 2 - 220, 352, 440, 22 + 30 * len(rows[:4]))
            R.glass_panel(surf, box, radius=T.R_LG)
            for i, (key, value) in enumerate(rows[:4]):
                y = box.top + 24 + i * 30
                R.draw_text(surf, key, R.font_text(T.T_LABEL), T.INK_DIM,
                            midleft=(box.left + 26, y))
                R.draw_text(surf, value, R.font_num(T.T_LABEL, True), T.INK,
                            midright=(box.right - 26, y))

        self.again.draw(surf)
        self.menu.draw(surf)
        self.app.vfx.draw_additive(surf)


class ProfileScene(Scene):
    def on_enter(self, **kw) -> None:
        self.back = U.IconButton(pygame.Rect(T.SAFE_L, T.SAFE_T + 8, 56, 56),
                                 "back", variant="secondary",
                                 on_click=lambda: self.app.goto("menu"))

    def handle_event(self, event) -> None:
        if self.back.handle_event(event):
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.app.goto("menu")

    def update(self, dt: float) -> None:
        self.app.backdrop.update(dt, 0.0)
        self.back.update(dt)

    def draw(self, surf) -> None:
        p = self.app.profile
        self.app.backdrop.draw(surf)
        _veil(surf, 168)
        R.draw_text(surf, "YOUR PROGRESS", R.font_display(T.T_DISPLAY, True),
                    T.INK, center=(W // 2, T.SAFE_T + 56))

        level, _ = PR.level_for_xp(p.xp)
        tiles = [
            ("Level", str(level), "crown"),
            ("Stars", str(p.total_stars), "star"),
            ("Wins", str(p.matches_won), "trophy"),
            ("Solved", str(p.problems_solved), "check"),
            ("Accuracy", f"{p.accuracy() * 100:.0f}%", "target"),
            ("Coins", str(p.coins), "coin"),
        ]
        for i, (label, value, icon) in enumerate(tiles):
            c, r = i % 3, i // 3
            rect = pygame.Rect(140 + c * 340, 148 + r * 130, 300, 108)
            R.glass_panel(surf, rect, radius=T.R_LG)
            IC.draw_icon(surf, icon,
                         pygame.Rect(rect.left + 20, rect.centery - 20, 40, 40), T.GOLD)
            R.draw_text(surf, value, R.font_num(T.T_TITLE, True), T.INK,
                        midleft=(rect.left + 78, rect.centery - 10))
            R.draw_text(surf, label, R.font_text(T.T_MICRO), T.INK_DIM,
                        midleft=(rect.left + 78, rect.centery + 22))

        weak = p.weakest_skills(3)
        if weak:
            names = ",  ".join(ME.skill_label(s) for s in weak)
            R.draw_text(surf, f"Practise next:  {names}", R.font_text(T.T_BODY),
                        T.INK_DIM, center=(W // 2, 452))
        self.back.draw(surf)


class SettingsScene(Scene):
    def on_enter(self, **kw) -> None:
        s = self.app.profile.settings
        self.back = U.IconButton(pygame.Rect(T.SAFE_L, T.SAFE_T + 8, 56, 56),
                                 "back", variant="secondary", on_click=self._exit)
        specs = [
            ("Sound effects", "sfx", s.sfx_volume > 0.01),
            ("Music", "music", s.music_volume > 0.01),
            ("Reduce motion", "reduce_motion", s.reduce_motion),
            ("Show hints", "show_hints", s.show_hints),
            ("Colourblind palette", "colorblind_mode", s.colorblind_mode),
        ]
        self.rows: List[Tuple[str, U.Toggle]] = []
        for i, (label, key, value) in enumerate(specs):
            rect = pygame.Rect(W // 2 + 132, 184 + i * 76, 92, 48)
            self.rows.append(
                (label, U.Toggle(rect, value,
                                 on_change=(lambda v, k=key: self._set(k, v)))))

    def _set(self, key: str, value: bool) -> None:
        app = self.app
        s = app.profile.settings
        app.audio.play("ui_toggle")
        if key == "sfx":
            s.sfx_volume = 0.9 if value else 0.0
            app.audio.set_sfx_volume(s.sfx_volume)
        elif key == "music":
            s.music_volume = 0.6 if value else 0.0
            app.audio.set_music_volume(s.music_volume)
        else:
            setattr(s, key, value)

    def _exit(self) -> None:
        try:
            self.app.profile.save()
        except Exception:
            pass
        self.app.goto("menu")

    def handle_event(self, event) -> None:
        if self.back.handle_event(event):
            return
        for _label, toggle in self.rows:
            if toggle.handle_event(event):
                return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._exit()

    def update(self, dt: float) -> None:
        self.app.backdrop.update(dt, 0.0)
        self.back.update(dt)
        for _label, toggle in self.rows:
            toggle.update(dt)

    def draw(self, surf) -> None:
        self.app.backdrop.draw(surf)
        _veil(surf, 172)
        R.draw_text(surf, "SETTINGS", R.font_display(T.T_DISPLAY, True), T.INK,
                    center=(W // 2, T.SAFE_T + 56))
        for label, toggle in self.rows:
            row = pygame.Rect(W // 2 - 320, toggle.rect.centery - 34, 640, 68)
            R.glass_panel(surf, row, radius=T.R_LG)
            R.draw_text(surf, label, R.font_text(T.T_BODY), T.INK,
                        midleft=(row.left + 26, row.centery))
            toggle.draw(surf)
        self.back.draw(surf)


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def build_app(fullscreen: bool = False) -> App:
    """Construct the app, wire shared services, and register every scene."""
    app = App(title="RopeRush - Math Tug-of-War", fullscreen=fullscreen)

    app.profile = PR.Profile.load()
    app.audio = AudioEngine()
    app.audio.set_sfx_volume(app.profile.settings.sfx_volume)
    app.audio.set_music_volume(app.profile.settings.music_volume)
    app.vfx = VFX.VFX()
    app.backdrop = BD.Backdrop()
    app.toasts = U.ToastStack()
    app.last_grade = "1st"

    # App has already mapped pointer events into logical space, so the widget
    # kit needs no further transform.
    U.set_pointer_transform(lambda p: p)

    app.register("menu", MenuScene(app))
    app.register("modes", ModeSelectScene(app))
    app.register("grade", GradeSelectScene(app))
    app.register("match", MatchScene(app))
    app.register("results", ResultsScene(app))
    app.register("profile", ProfileScene(app))
    app.register("settings", SettingsScene(app))

    app.push("menu")
    return app


def main() -> None:
    app = build_app("--fullscreen" in sys.argv)
    try:
        app.run()
    finally:
        try:
            app.profile.save()
        except Exception:
            pass


if __name__ == "__main__":
    main()

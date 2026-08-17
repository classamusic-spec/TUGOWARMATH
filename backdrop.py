"""The animated arena environment that sits behind everything in RopeRush.

Art direction
-------------
A night stadium at dusk. The eye should read, from the horizon outward: a deep
indigo sky, a distant city skyline, two tiers of crowd stands packed with
silhouettes and pinprick lights, and a polished floor whose perspective lines
converge on the centre of the rope. Two rival energy sources - EMBER on the
left, FROST on the right - light the arena from the wings and fight over it.

`tension` (-1 .. +1) is the single input that makes the environment feel
*alive*: the winning team's half brightens, its floor bloom swells, its crowd
lights get busier, and the centre line drifts toward the loser.

Cost model
----------
Everything static is composited once into a small number of surfaces at
construction time:

    _sky        opaque, full frame - sky gradient + haze + skyline
    _stars[]    3 parallax layers, tiled horizontally and scrolled
    _crowd[]    one surface per stand tier, blitted with a bob offset
    _shafts[]   one surface per light shaft, faded with set_alpha
    _floor      SRCALPHA - ground gradient, perspective grid, centre line

Per frame the class only blits those, plus a handful of cached
`render_utils.radial_glow` blobs whose radius and intensity are *quantised* so
the glow cache never grows. A full `draw()` + `draw_foreground()` is roughly 60
blits and no allocations.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

import render_utils as R
import theme as T

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Layout constants (fractions of the logical canvas)
# --------------------------------------------------------------------------- #

HORIZON_F = 0.575          # where sky meets floor
SKYLINE_TOP_F = 0.34       # tallest building
STAND_TOP_F = 0.30         # top of the highest crowd tier
VP_LIFT = 46               # vanishing point sits this far ABOVE the horizon

STAR_LAYERS = 3
DUST_COUNT = 46

# Glow blobs are cached by (radius, colour, intensity); quantising keeps the
# render_utils cache at a fixed handful of entries instead of one per frame.
_R_QUANT = 12
_I_QUANT = 10


def _q(v: float, step: int) -> int:
    return int(max(step, round(v / step) * step))


# --------------------------------------------------------------------------- #
# Backdrop
# --------------------------------------------------------------------------- #

class Backdrop:
    """The whole arena environment, back to front."""

    def __init__(self, width: int = T.LOGICAL_W, height: int = T.LOGICAL_H) -> None:
        self.w = int(width)
        self.h = int(height)
        self.horizon = int(self.h * HORIZON_F)
        self.cx = self.w // 2

        self.t = 0.0
        self.tension = 0.0          # smoothed toward the value passed in
        self._tension_target = 0.0

        rng = random.Random(0x50DA)

        # ---- static composites -------------------------------------- #
        self._sky = self._build_sky(rng)
        self._sky.blit(self._build_skyline(random.Random(0xC17E)),
                       (0, int(self.h * SKYLINE_TOP_F)))
        self._stars, self._star_speed = self._build_stars(rng)
        self._crowd = self._build_crowd(rng)
        self._shafts = self._build_shafts(rng)
        self._floor = self._build_floor()
        self._vignette = R.vignette((self.w, self.h), strength=132)

        # ---- animated bits ------------------------------------------ #
        self._bokeh = self._make_bokeh(rng)
        self._lights = self._make_crowd_lights(rng)
        self._motes = self._make_motes(rng)

        self._star_off = [0.0] * STAR_LAYERS

    # ================================================================== #
    # Layer 1 - sky gradient + atmospheric haze + skyline silhouette
    # ================================================================== #

    def _build_sky(self, rng: random.Random) -> pygame.Surface:
        """Sky gradient with the distant skyline burnt in. Opaque, full frame."""
        surf = pygame.Surface((self.w, self.h))
        grad = R.multi_gradient(
            (self.w, self.h),
            (
                (0.00, T.SKY_TOP),
                (0.30, T.SKY_MID),
                (0.50, T.SKY_LOW),
                (float(HORIZON_F), T.SKY_HORIZON),
                (1.00, T.GROUND_NEAR),
            ),
        )
        surf.blit(grad, (0, 0))

        # A wide, very soft warm bloom sitting on the horizon line: this is
        # what stops the gradient from looking like a flat CSS background.
        haze = R.radial_glow(int(self.w * 0.62), T.lerp_color(T.SKY_HORIZON, T.GOLD, 0.28),
                             intensity=46, falloff=2.6)
        surf.blit(haze, (self.cx - haze.get_width() // 2, self.horizon - haze.get_height() // 2),
                  special_flags=pygame.BLEND_RGB_ADD)
        return surf

    def _build_skyline(self, rng: random.Random) -> pygame.Surface:
        """Geometric city silhouette, two depths, darker than the sky behind."""
        top = int(self.h * SKYLINE_TOP_F)
        surf = pygame.Surface((self.w, self.horizon - top + 4), pygame.SRCALPHA)
        base = surf.get_height() - 2

        for depth, (tint, alpha, hmul) in enumerate((
            (T.lerp_color(T.SKY_LOW, T.SKY_TOP, 0.45), 210, 1.00),   # far
            (T.lerp_color(T.SKY_TOP, (0, 0, 0), 0.35), 240, 0.68),   # near
        )):
            x = -rng.randint(0, 60)
            while x < self.w + 40:
                bw = rng.randint(34, 96)
                bh = int(rng.randint(30, base - 8) * hmul)
                if depth == 1:
                    bh = int(bh * 0.75)
                y = base - bh
                pygame.draw.rect(surf, (*tint, alpha), (x, y, bw, bh + 4))

                # Roof furniture - masts and boxes break up the flat tops.
                if rng.random() < 0.32:
                    mx = x + bw // 2
                    pygame.draw.line(surf, (*tint, alpha), (mx, y),
                                     (mx, y - rng.randint(8, 26)), 3)
                if rng.random() < 0.25:
                    kw = rng.randint(8, bw // 2 + 4)
                    pygame.draw.rect(surf, (*tint, alpha),
                                     (x + rng.randint(0, max(1, bw - kw)), y - 8, kw, 10))

                # Sparse lit windows, warmer on the near row.
                if depth == 1 and bw > 40 and bh > 40:
                    wc = T.lerp_color(T.GOLD, T.SKY_HORIZON, 0.35)
                    for wy in range(y + 10, base - 8, 13):
                        for wx in range(x + 7, x + bw - 8, 12):
                            if rng.random() < 0.14:
                                pygame.draw.rect(surf, (*wc, 150), (wx, wy, 3, 5))
                x += bw + rng.randint(-6, 16)
        return surf

    # ================================================================== #
    # Layer 2 - parallax star / bokeh field
    # ================================================================== #

    def _build_stars(self, rng: random.Random) -> Tuple[List[pygame.Surface], List[float]]:
        """Three depths of drifting points of light, pre-baked and scrolled.

        Each layer is a full-width surface blitted twice (at `off` and
        `off - w`) so it wraps seamlessly - two blits per depth per frame.
        """
        layers: List[pygame.Surface] = []
        speeds: List[float] = []
        band = int(self.h * 0.62)
        specs = (
            # (count, max radius, alpha, drift px/s, colour mix toward gold)
            (150, 1.15, 120, 1.6, 0.10),
            (90, 1.8, 165, 3.4, 0.22),
            (38, 2.7, 210, 6.2, 0.40),
        )
        for count, rad, alpha, speed, warm in specs:
            s = pygame.Surface((self.w, band), pygame.SRCALPHA)
            for _ in range(count):
                x = rng.uniform(0, self.w)
                # Bias density toward the top - the sky thins near the horizon.
                y = band * (rng.random() ** 1.7)
                c = T.lerp_color(T.INK, T.GOLD, rng.random() * warm)
                a = int(alpha * rng.uniform(0.45, 1.0))
                r = max(1, int(rad * rng.uniform(0.6, 1.0) + 0.5))
                if r <= 1:
                    s.set_at((int(x), int(y)), (*c, a))
                else:
                    pygame.draw.circle(s, (*c, a), (int(x), int(y)), r)
            layers.append(s)
            speeds.append(speed)
        return layers, speeds

    def _make_bokeh(self, rng: random.Random) -> List[dict]:
        """A dozen big soft out-of-focus lights that breathe on their own."""
        out = []
        for _ in range(12):
            out.append({
                "x": rng.uniform(0, self.w),
                "y": rng.uniform(30, self.h * 0.46),
                "r": rng.uniform(8, 22),
                "c": T.lerp_color(T.INK, rng.choice((T.GOLD, T.FROST.light, T.EMBER.light)),
                                  rng.uniform(0.25, 0.8)),
                "ph": rng.uniform(0, math.tau),
                "sp": rng.uniform(0.35, 0.95),
                "drift": rng.uniform(2.0, 7.0),
            })
        return out

    # ================================================================== #
    # Layer 5 - crowd stands
    # ================================================================== #

    def _build_crowd(self, rng: random.Random) -> List[dict]:
        """Tiered stands flanking the arena, one baked surface per tier.

        The middle of the frame is left open so the skyline and the rope read
        cleanly; the stands wrap around the wings like a real venue.
        """
        tiers: List[dict] = []
        n_tiers = 5
        top = int(self.h * STAND_TOP_F)
        bottom = self.horizon + 6
        span = bottom - top

        for i in range(n_tiers):
            # Tier 0 is the highest / furthest back and therefore the darkest.
            f = i / max(1, n_tiers - 1)
            y = int(top + span * (f ** 0.85) * 0.92)
            row_h = int(28 + 16 * f)
            surf = pygame.Surface((self.w, row_h + 26), pygame.SRCALPHA)

            deck = T.lerp_color(T.SKY_TOP, (0, 0, 0), 0.30 + 0.32 * f)
            body = T.lerp_color(T.SKY_TOP, (0, 0, 0), 0.52 + 0.30 * f)
            alpha = int(190 + 55 * f)

            # How far in from each edge this tier reaches.
            reach = self.w * (0.30 + 0.10 * f)
            blocks = ((0, reach), (self.w - reach, self.w))

            for x0, x1 in blocks:
                pygame.draw.rect(surf, (*deck, alpha),
                                 (int(x0), row_h - 6, int(x1 - x0), 32))
                # Heads: overlapping circles of two sizes read as a dense crowd.
                x = x0 + rng.uniform(0, 9)
                while x < x1 - 4:
                    hr = rng.uniform(3.4, 5.6) * (0.72 + 0.42 * f)
                    hy = row_h - 6 - hr * rng.uniform(0.9, 1.5)
                    pygame.draw.circle(surf, (*body, alpha), (int(x), int(hy)),
                                       max(2, int(hr)))
                    # shoulders
                    pygame.draw.ellipse(surf, (*body, alpha),
                                        (int(x - hr * 1.35), int(hy + hr * 0.5),
                                         int(hr * 2.7), int(hr * 2.2)))
                    x += hr * rng.uniform(1.25, 2.0)

            # Support columns under the deck give the stands some structure.
            col = T.lerp_color(T.SKY_TOP, (0, 0, 0), 0.62)
            for x0, x1 in blocks:
                for cx in range(int(x0) + 18, int(x1) - 10, 74):
                    pygame.draw.rect(surf, (*col, alpha), (cx, row_h + 24, 7, 14))

            tiers.append({
                "surf": surf,
                "y": y - row_h,
                "phase": rng.uniform(0, math.tau),
                "bob": 1.6 + 1.5 * (1.0 - f),
                "sway": 3.0 * (1.0 - f) + 1.0,
                "side_f": f,
                "left": (0.0, reach),
                "right": (self.w - reach, float(self.w)),
            })
        return tiers

    def _make_crowd_lights(self, rng: random.Random) -> List[dict]:
        """Camera flashes / phone screens twinkling in the stands."""
        out = []
        for _ in range(46):
            side = -1 if rng.random() < 0.5 else 1
            if side < 0:
                x = rng.uniform(6, self.w * 0.36)
            else:
                x = rng.uniform(self.w * 0.64, self.w - 6)
            out.append({
                "x": x,
                "y": rng.uniform(self.h * STAND_TOP_F + 6, self.horizon - 4),
                "c": rng.choice((T.INK, T.GOLD_LIGHT, T.FROST.light, T.EMBER.light)),
                "ph": rng.uniform(0, math.tau),
                "sp": rng.uniform(0.7, 2.6),
                "side": side,
                "r": rng.uniform(5.0, 11.0),
            })
        return out

    # ================================================================== #
    # Layer 6 - volumetric light shafts
    # ================================================================== #

    def _build_shafts(self, rng: random.Random) -> List[dict]:
        """Angled translucent beams raking down from above the stands.

        Baked once each and faded with `set_alpha`; normal blend over a very
        dark sky is visually indistinguishable from additive here and costs a
        single cached surface instead of one per brightness step.
        """
        shafts: List[dict] = []
        h = int(self.h * 0.80)
        specs = (
            (0.10, 150, 300, T.EMBER.light, 0.55),
            (0.30, 110, 230, T.GOLD_LIGHT, 0.40),
            (0.62, 120, 250, T.INK, 0.35),
            (0.86, 160, 320, T.FROST.light, 0.55),
        )
        for fx, wtop, wbot, color, amp in specs:
            pad = 90
            sw = int(max(wtop, wbot)) + pad * 2
            s = pygame.Surface((sw, h), pygame.SRCALPHA)
            lean = rng.uniform(-0.22, 0.22)
            cx_top = pad + max(wtop, wbot) * 0.5 - lean * h * 0.5
            cx_bot = pad + max(wtop, wbot) * 0.5 + lean * h * 0.5
            # Draw the beam as horizontal slices so it can fade out downward.
            steps = 56
            for i in range(steps):
                t = i / (steps - 1)
                yy = int(t * h)
                hh = max(1, h // steps + 1)
                ww = T.lerp(wtop, wbot, t)
                a = int(30 * (1.0 - t) ** 1.5 + 4)
                cxx = T.lerp(cx_top, cx_bot, t)
                pygame.draw.rect(s, (*color, a),
                                 (int(cxx - ww * 0.5), yy, int(ww), hh))
            s = R.blur(s, amount=0.22, passes=2)
            shafts.append({
                "surf": s,
                "x": int(self.w * fx) - sw // 2,
                "ph": rng.uniform(0, math.tau),
                "sp": rng.uniform(0.25, 0.6),
                "amp": amp,
                "side": -1 if fx < 0.5 else 1,
            })
        return shafts

    # ================================================================== #
    # Layer 7 - arena floor
    # ================================================================== #

    def _build_floor(self) -> pygame.Surface:
        """Ground plane with a perspective grid converging on the arena centre."""
        top = self.horizon
        fh = self.h - top
        surf = pygame.Surface((self.w, fh), pygame.SRCALPHA)

        grad = R.vertical_gradient((self.w, fh), T.GROUND_FAR, T.GROUND_NEAR)
        surf.blit(grad, (0, 0))

        vp = (self.cx, -VP_LIFT)          # in floor-local coordinates
        line = T.GROUND_LINE

        # --- radial lines ------------------------------------------------ #
        # Spaced by angle rather than by x so the fan stays even at the edges.
        for i in range(-13, 14):
            if i == 0:
                continue
            # Project a point far off the bottom edge; extreme values are
            # clipped by the surface anyway.
            k = i / 13.0
            bx = self.cx + math.copysign(abs(k) ** 1.35, k) * self.w * 1.5
            a = int(58 * (1.0 - abs(k) * 0.45))
            pygame.draw.line(surf, (*line, a), vp, (int(bx), fh), 2)

        # --- depth lines -------------------------------------------------- #
        # Geometric spacing = constant spacing in world Z under perspective.
        y = 6.0
        step = 4.0
        while y < fh:
            a = int(16 + 62 * (y / fh) ** 1.2)
            pygame.draw.line(surf, (*line, a), (0, int(y)), (self.w, int(y)), 1)
            y += step
            step *= 1.42

        # --- centre line -------------------------------------------------- #
        pygame.draw.line(surf, (*T.GOLD, 120), (self.cx, 0), (self.cx, fh), 3)
        pygame.draw.line(surf, (*T.GOLD_LIGHT, 190), (self.cx, 0), (self.cx, fh), 1)

        # --- horizon lip -------------------------------------------------- #
        # A bright edge where the floor meets the stands anchors the two.
        lip = R.vertical_gradient((self.w, 12), T.lerp_color(T.SKY_HORIZON, T.GOLD, 0.35),
                                  T.GROUND_FAR, alpha=120)
        surf.blit(lip, (0, 0))
        pygame.draw.line(surf, (*T.lerp_color(T.SKY_HORIZON, T.GOLD_LIGHT, 0.5), 150),
                         (0, 0), (self.w, 0), 2)

        # Sheen: the floor is polished, so it should hold a soft reflection.
        sheen = R.vertical_gradient((self.w, int(fh * 0.55)), (255, 255, 255), (255, 255, 255))
        sheen = sheen.copy()
        for yy in range(sheen.get_height()):
            k = 1.0 - yy / max(1, sheen.get_height() - 1)
            pygame.draw.line(sheen, (255, 255, 255, int(15 * k * k)),
                             (0, yy), (self.w, yy))
        surf.blit(sheen, (0, 0))
        return surf

    # ================================================================== #
    # Layer 8 - foreground dust motes
    # ================================================================== #

    def _make_motes(self, rng: random.Random) -> List[dict]:
        out = []
        for _ in range(DUST_COUNT):
            out.append({
                "x": rng.uniform(0, self.w),
                "y": rng.uniform(self.h * 0.18, self.h),
                "r": rng.uniform(1.6, 5.0),
                "vx": rng.uniform(-9.0, 9.0),
                "vy": rng.uniform(-16.0, -3.0),
                "ph": rng.uniform(0, math.tau),
                "sp": rng.uniform(0.5, 1.6),
                "c": T.lerp_color(T.INK, T.GOLD_LIGHT, rng.random() * 0.6),
            })
        return out

    # ================================================================== #
    # Update
    # ================================================================== #

    def update(self, dt: float, tension: float = 0.0) -> None:
        """Advance the animation. `tension` in -1..+1 = who is currently winning.

        The stored value is critically damped toward the target so a sudden
        swing in the rope does not make the whole arena strobe.
        """
        dt = max(0.0, min(0.05, float(dt)))
        self.t += dt
        self._tension_target = max(-1.0, min(1.0, float(tension)))
        self.tension += (self._tension_target - self.tension) * min(1.0, dt * 3.2)

        for i in range(STAR_LAYERS):
            self._star_off[i] = (self._star_off[i] + self._star_speed[i] * dt) % self.w

        for b in self._bokeh:
            b["x"] += b["drift"] * dt
            if b["x"] > self.w + 40:
                b["x"] = -40.0

        for m in self._motes:
            m["x"] += m["vx"] * dt
            m["y"] += m["vy"] * dt
            if m["y"] < self.h * 0.16:
                m["y"] = self.h + 8.0
                m["x"] = random.uniform(0, self.w)
            if m["x"] < -10:
                m["x"] = self.w + 10.0
            elif m["x"] > self.w + 10:
                m["x"] = -10.0

    # ================================================================== #
    # Draw
    # ================================================================== #

    def draw(self, surf: pygame.Surface) -> None:
        """Everything from the sky down to the arena floor."""
        t = self.t
        tension = self.tension

        # -- 1. sky (skyline is baked into it) --------------------------- #
        surf.blit(self._sky, (0, 0))

        # -- 2. parallax stars ------------------------------------------- #
        for i, layer in enumerate(self._stars):
            off = int(self._star_off[i])
            surf.blit(layer, (off, 0))
            surf.blit(layer, (off - self.w, 0))

        # -- 2b. breathing bokeh ----------------------------------------- #
        for b in self._bokeh:
            k = 0.55 + 0.45 * math.sin(t * b["sp"] + b["ph"])
            r = _q(b["r"] * (0.8 + 0.4 * k), _R_QUANT)
            inten = _q(70 * k + 26, _I_QUANT)
            g = R.radial_glow(r, b["c"], intensity=inten, falloff=2.2)
            surf.blit(g, (int(b["x"]) - r, int(b["y"]) - r),
                       special_flags=pygame.BLEND_RGB_ADD)

        # -- 4. the two rival energy sources ----------------------------- #
        self._draw_energy(surf, t, tension)

        # -- 5. crowd stands --------------------------------------------- #
        for tier in self._crowd:
            ph = tier["phase"]
            bob = math.sin(t * 1.25 + ph) * tier["bob"]
            sway = math.sin(t * 0.62 + ph * 1.7) * tier["sway"]
            surf.blit(tier["surf"], (int(sway), int(tier["y"] + bob)))

        self._draw_crowd_lights(surf, t, tension)

        # -- 6. volumetric shafts ---------------------------------------- #
        for sh in self._shafts:
            pulse = 0.5 + 0.5 * math.sin(t * sh["sp"] + sh["ph"])
            lean = tension * sh["side"]           # winning side's lights flare
            a = sh["amp"] * (0.45 + 0.55 * pulse) * (1.0 + 0.55 * max(0.0, lean))
            s = sh["surf"]
            s.set_alpha(int(max(0, min(255, a * 255))))
            drift = math.sin(t * 0.21 + sh["ph"]) * 14.0
            surf.blit(s, (int(sh["x"] + drift), 0))

        # -- 7. arena floor ---------------------------------------------- #
        surf.blit(self._floor, (0, self.horizon))
        self._draw_floor_glow(surf, t, tension)

    def draw_foreground(self, surf: pygame.Surface) -> None:
        """Dust motes drifting in front of the action, then the vignette."""
        t = self.t
        for m in self._motes:
            tw = 0.45 + 0.55 * math.sin(t * m["sp"] + m["ph"])
            r = _q(m["r"] * (2.2 + 0.8 * tw), _R_QUANT)
            inten = _q(38 * tw + 14, _I_QUANT)
            g = R.radial_glow(r, m["c"], intensity=inten, falloff=2.4)
            surf.blit(g, (int(m["x"]) - r, int(m["y"]) - r),
                       special_flags=pygame.BLEND_RGB_ADD)
        surf.blit(self._vignette, (0, 0))

    # ------------------------------------------------------------------ #
    # Animated sub-layers
    # ------------------------------------------------------------------ #

    def _draw_energy(self, surf: pygame.Surface, t: float, tension: float) -> None:
        """EMBER on the left, FROST on the right: pulsing wing floodlights.

        Each side is three stacked blobs - a wide ambient wash, a tighter core
        and a hot pip - so the falloff has some structure instead of reading as
        one flat circle.
        """
        for side, team in ((-1, T.EMBER), (+1, T.FROST)):
            # `lean` is >0 when this side is winning.
            lean = tension * side
            breathe = 0.5 + 0.5 * math.sin(t * (1.05 + 0.2 * side) + (0.0 if side < 0 else 1.9))
            boost = 1.0 + 0.75 * max(0.0, lean) - 0.28 * max(0.0, -lean)

            x = 0 if side < 0 else self.w
            y = int(self.h * 0.46)
            for rad_f, inten_f, colr in (
                (0.46, 0.42, team.deep),
                (0.26, 0.72, team.core),
                (0.11, 1.00, team.glow),
            ):
                r = _q(self.w * rad_f * (0.92 + 0.08 * breathe) * (0.9 + 0.16 * boost), _R_QUANT)
                inten = _q(74 * inten_f * boost * (0.78 + 0.22 * breathe), _I_QUANT)
                if inten <= 0:
                    continue
                g = R.radial_glow(r, colr, intensity=min(200, inten), falloff=2.5)
                surf.blit(g, (x - r, y - r), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_crowd_lights(self, surf: pygame.Surface, t: float, tension: float) -> None:
        """Sparse twinkles in the stands; the winning side's crowd is busier."""
        for L in self._lights:
            lean = tension * L["side"]
            # A sharp power curve keeps most lights off most of the time, so
            # the ones that fire read as individual flashes.
            k = (0.5 + 0.5 * math.sin(t * L["sp"] + L["ph"])) ** 5.0
            k *= 0.55 + 0.75 * max(0.0, lean) + 0.25
            if k < 0.06:
                continue
            r = _q(L["r"] * (0.7 + 0.6 * k), _R_QUANT)
            inten = _q(150 * k, _I_QUANT)
            if inten <= 0:
                continue
            g = R.radial_glow(r, L["c"], intensity=min(210, inten), falloff=2.0)
            surf.blit(g, (int(L["x"]) - r, int(L["y"]) - r),
                       special_flags=pygame.BLEND_RGB_ADD)

    def _draw_floor_glow(self, surf: pygame.Surface, t: float, tension: float) -> None:
        """Team colour spilling onto the polished floor, plus the centre pip."""
        gy = self.horizon + int((self.h - self.horizon) * 0.42)
        for side, team in ((-1, T.EMBER), (+1, T.FROST)):
            lean = tension * side
            boost = 1.0 + 0.9 * max(0.0, lean) - 0.35 * max(0.0, -lean)
            breathe = 0.5 + 0.5 * math.sin(t * 0.9 + (0.0 if side < 0 else 2.4))
            x = int(self.w * (0.16 if side < 0 else 0.84))
            r = _q(self.w * 0.24 * (0.92 + 0.1 * breathe) * (0.9 + 0.2 * boost), _R_QUANT)
            inten = _q(52 * boost * (0.8 + 0.2 * breathe), _I_QUANT)
            if inten <= 0:
                continue
            g = R.radial_glow(r, team.core, intensity=min(190, inten), falloff=2.6)
            surf.blit(g, (x - r, gy - r), special_flags=pygame.BLEND_RGB_ADD)

        # The centre marker drifts with the tension - a tiny cue that reads
        # even out of the corner of your eye.
        cx = int(self.cx - tension * self.w * 0.045)
        pulse = 0.62 + 0.38 * math.sin(t * 2.1)
        r = _q(46 * pulse + 22, _R_QUANT)
        inten = _q(96 * pulse, _I_QUANT)
        g = R.radial_glow(r, T.GOLD, intensity=min(200, inten), falloff=2.1)
        surf.blit(g, (cx - r, self.horizon - r + 6), special_flags=pygame.BLEND_RGB_ADD)

"""Pure-vector icon set for RopeRush.

No font glyphs, no image files - every icon is built from pygame primitives so
it stays crisp at any size and can be tinted per-team at runtime.

How it works
------------
Each icon is a small function that draws into a *unit box*: coordinates run
0..1 across a square that is centred inside the caller's rect. That makes every
icon automatically proportional, centred and resolution independent.

Drawing happens on a surface supersampled by `_SS` and then smoothscaled down,
which is by far the cheapest way to get genuinely anti-aliased polygons and
thick round-capped strokes out of pygame. The transparent background is
pre-filled with the icon colour at alpha 0 so the downscale never produces the
dark fringe you get when averaging against black.

Everything is memoised on (name, size, colour, accent, width), so the cost is
paid once per unique appearance.

    icons.draw_icon(surf, "trophy", rect, T.GOLD, accent=T.GOLD_DEEP)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pygame

import theme as T

Color = Tuple[int, int, int]
Pt = Tuple[float, float]

# Supersample factor used when rasterising an icon.
_SS = 4

_cache: Dict[tuple, pygame.Surface] = {}


def clear_cache() -> None:
    """Drop every rasterised icon (call if the palette is swapped at runtime)."""
    _cache.clear()


# --------------------------------------------------------------------------- #
# Small colour helpers
# --------------------------------------------------------------------------- #

def _auto_accent(c: Color) -> Color:
    """A readable secondary colour when the caller supplies none.

    Bright icons get a darkened accent (so detail reads *inside* a filled
    shape); dark icons get a lightened one.
    """
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return T.shade(c, -86) if lum > 110 else T.shade(c, 96)


# --------------------------------------------------------------------------- #
# The pen - a tiny unit-space drawing context
# --------------------------------------------------------------------------- #

class _Pen:
    """Draws in unit coordinates (0..1) inside a square box on a surface."""

    def __init__(
        self,
        surf: pygame.Surface,
        box: pygame.Rect,
        color: Color,
        accent: Optional[Color],
        width: float,
    ) -> None:
        self.s = surf
        self.box = box
        self.n = float(box.width)
        self.color: Color = (int(color[0]), int(color[1]), int(color[2]))
        self.acc: Color = tuple(accent) if accent is not None else _auto_accent(self.color)  # type: ignore[assignment]
        # Base stroke weight. 0.112 of the box reads as a modern 2px-at-24px line.
        self.sw = max(2.0, 0.112 * self.n * width)
        # "Erase" colour: same RGB as the fill so downscaling stays fringe-free.
        self.clear = (self.color[0], self.color[1], self.color[2], 0)

    # ---- coordinates -----------------------------------------------------
    def px(self, x: float, y: float) -> Tuple[int, int]:
        return (int(round(self.box.x + x * self.n)), int(round(self.box.y + y * self.n)))

    def length(self, v: float) -> float:
        return v * self.n

    def w(self, k: float = 1.0) -> int:
        return max(1, int(round(self.sw * k)))

    def _col(self, color) -> tuple:
        return self.color if color is None else color

    # ---- strokes ---------------------------------------------------------
    def line(self, a: Pt, b: Pt, k: float = 1.0, color=None, cap: bool = True) -> None:
        c = self._col(color)
        t = self.w(k)
        pa, pb = self.px(*a), self.px(*b)
        pygame.draw.line(self.s, c, pa, pb, t)
        if cap and t > 2:
            r = t // 2
            pygame.draw.circle(self.s, c, pa, r)
            pygame.draw.circle(self.s, c, pb, r)

    def path(
        self,
        pts: Sequence[Pt],
        k: float = 1.0,
        color=None,
        closed: bool = False,
        cap: bool = True,
    ) -> None:
        """Polyline with round joins (and round caps unless `cap` is False)."""
        if len(pts) < 2:
            return
        c = self._col(color)
        t = self.w(k)
        px = [self.px(*q) for q in pts]
        if closed:
            px = px + [px[0]]
        pygame.draw.lines(self.s, c, False, px, t)
        if t > 2:
            r = t // 2
            joints = px if (cap or closed) else px[1:-1]
            for q in joints:
                pygame.draw.circle(self.s, c, q, r)

    # ---- fills -----------------------------------------------------------
    def poly(self, pts: Sequence[Pt], color=None, k: float = 0.0) -> None:
        """Filled polygon; `k` > 0 also strokes the outline to round corners."""
        c = self._col(color)
        pygame.draw.polygon(self.s, c, [self.px(*q) for q in pts])
        if k:
            self.path(pts, k, c, closed=True)

    def circle(self, c_: Pt, r: float, color=None, k: float = 0.0) -> None:
        col = self._col(color)
        pygame.draw.circle(
            self.s, col, self.px(*c_), max(1, int(round(self.length(r)))),
            0 if not k else self.w(k),
        )

    def ellipse(self, x0: float, y0: float, x1: float, y1: float, color=None, k: float = 0.0) -> None:
        col = self._col(color)
        a, b = self.px(x0, y0), self.px(x1, y1)
        rect = pygame.Rect(a[0], a[1], max(1, b[0] - a[0]), max(1, b[1] - a[1]))
        pygame.draw.ellipse(self.s, col, rect, 0 if not k else self.w(k))

    def rrect(
        self, x0: float, y0: float, x1: float, y1: float,
        r: float = 0.10, color=None, k: float = 0.0,
    ) -> None:
        col = self._col(color)
        a, b = self.px(x0, y0), self.px(x1, y1)
        rect = pygame.Rect(a[0], a[1], max(1, b[0] - a[0]), max(1, b[1] - a[1]))
        rad = max(0, min(int(self.length(r)), min(rect.w, rect.h) // 2))
        pygame.draw.rect(self.s, col, rect, 0 if not k else self.w(k), border_radius=rad)

    def capsule(self, a: Pt, b: Pt, thick: float, color=None) -> None:
        """A stroked line of an explicit unit thickness (round caps)."""
        c = self._col(color)
        t = max(1, int(round(self.length(thick))))
        pa, pb = self.px(*a), self.px(*b)
        pygame.draw.line(self.s, c, pa, pb, t)
        pygame.draw.circle(self.s, c, pa, t // 2)
        pygame.draw.circle(self.s, c, pb, t // 2)

    # ---- arcs ------------------------------------------------------------
    def arc_pts(self, c_: Pt, r: float, a0: float, a1: float, steps: int = 0) -> List[Pt]:
        """Sampled arc. Angles in degrees, 0 = east, growing clockwise on screen."""
        cx, cy = c_
        steps = steps or max(6, int(abs(a1 - a0) / 5.0))
        out: List[Pt] = []
        for i in range(steps + 1):
            a = math.radians(a0 + (a1 - a0) * i / steps)
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return out

    def arc(self, c_: Pt, r: float, a0: float, a1: float, k: float = 1.0,
            color=None, cap: bool = True) -> None:
        self.path(self.arc_pts(c_, r, a0, a1), k, color, cap=cap)

    # ---- decorations -----------------------------------------------------
    def arrow_head(self, tip: Pt, deg: float, size: float, color=None) -> None:
        """Filled triangle at `tip` pointing along `deg`."""
        a = math.radians(deg)
        dx, dy = math.cos(a), math.sin(a)
        nx, ny = -dy, dx
        back = (tip[0] - dx * size, tip[1] - dy * size)
        pts = [
            tip,
            (back[0] + nx * size * 0.62, back[1] + ny * size * 0.62),
            (back[0] - nx * size * 0.62, back[1] - ny * size * 0.62),
        ]
        self.poly(pts, color, k=0.35)

    # ---- erasing ---------------------------------------------------------
    def punch_circle(self, c_: Pt, r: float) -> None:
        self.circle(c_, r, self.clear)

    def punch_poly(self, pts: Sequence[Pt]) -> None:
        self.poly(pts, self.clear)

    def punch_rrect(self, x0: float, y0: float, x1: float, y1: float, r: float = 0.08) -> None:
        self.rrect(x0, y0, x1, y1, r, self.clear)


# --------------------------------------------------------------------------- #
# Shared geometry helpers
# --------------------------------------------------------------------------- #

def _star_pts(cx: float, cy: float, ro: float, ri: float,
              points: int = 5, rot: float = -90.0) -> List[Pt]:
    pts: List[Pt] = []
    for i in range(points * 2):
        r = ro if i % 2 == 0 else ri
        a = math.radians(rot + i * 180.0 / points)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _fit(pts: Sequence[Pt], x0: float, y0: float, x1: float, y1: float) -> List[Pt]:
    """Scale a point cloud into a box, preserving aspect and centring it."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w = max(1e-6, max(xs) - min(xs))
    h = max(1e-6, max(ys) - min(ys))
    k = min((x1 - x0) / w, (y1 - y0) / h)
    ox = x0 + ((x1 - x0) - w * k) * 0.5 - min(xs) * k
    oy = y0 + ((y1 - y0) - h * k) * 0.5 - min(ys) * k
    return [(p[0] * k + ox, p[1] * k + oy) for p in pts]


def _speaker(p: _Pen) -> None:
    """Shared speaker body for the sound_* icons."""
    p.poly([(0.06, 0.36), (0.22, 0.36), (0.44, 0.14),
            (0.44, 0.86), (0.22, 0.64), (0.06, 0.64)], k=0.30)


def _person(p: _Pen, cx: float, scale: float, color=None) -> None:
    """A head + shoulders bust centred on `cx`, sized by `scale`."""
    c = p._col(color)
    hy = 0.50 - 0.20 * scale
    p.circle((cx, hy), 0.205 * scale, c)
    dome = p.arc_pts((cx, 0.50 + 0.44 * scale), 0.355 * scale, 180, 360)
    dome.append((dome[-1][0], 0.50 + 0.44 * scale))
    dome.append((dome[0][0], 0.50 + 0.44 * scale))
    p.poly(dome, c, k=0.16)


# --------------------------------------------------------------------------- #
# Icon definitions
# --------------------------------------------------------------------------- #
# Every function draws one icon inside the 0..1 unit box.

def _home(p: _Pen) -> None:
    p.poly([(0.50, 0.06), (0.97, 0.48), (0.03, 0.48)], k=0.34)
    p.rrect(0.16, 0.40, 0.84, 0.94, 0.10)
    p.punch_rrect(0.40, 0.62, 0.60, 0.95, 0.09)
    p.rrect(0.40, 0.62, 0.60, 0.95, 0.09, p.acc)


def _back(p: _Pen) -> None:
    p.line((0.90, 0.50), (0.20, 0.50))
    p.path([(0.46, 0.24), (0.18, 0.50), (0.46, 0.76)])


def _next_arrow(p: _Pen) -> None:
    p.line((0.10, 0.50), (0.80, 0.50))
    p.path([(0.54, 0.24), (0.82, 0.50), (0.54, 0.76)])


def _arrow_up(p: _Pen) -> None:
    p.line((0.50, 0.90), (0.50, 0.20))
    p.path([(0.24, 0.46), (0.50, 0.18), (0.76, 0.46)])


def _arrow_down(p: _Pen) -> None:
    p.line((0.50, 0.10), (0.50, 0.80))
    p.path([(0.24, 0.54), (0.50, 0.82), (0.76, 0.54)])


def _close(p: _Pen) -> None:
    p.line((0.22, 0.22), (0.78, 0.78), 1.05)
    p.line((0.78, 0.22), (0.22, 0.78), 1.05)


def _play(p: _Pen) -> None:
    p.poly([(0.28, 0.14), (0.87, 0.50), (0.28, 0.86)], k=0.42)


def _pause(p: _Pen) -> None:
    p.rrect(0.24, 0.14, 0.42, 0.86, 0.07)
    p.rrect(0.58, 0.14, 0.76, 0.86, 0.07)


def _stop(p: _Pen) -> None:
    p.rrect(0.18, 0.18, 0.82, 0.82, 0.12)


def _skip_next(p: _Pen) -> None:
    p.poly([(0.12, 0.16), (0.64, 0.50), (0.12, 0.84)], k=0.36)
    p.rrect(0.70, 0.16, 0.86, 0.84, 0.06)


def _skip_prev(p: _Pen) -> None:
    p.poly([(0.88, 0.16), (0.36, 0.50), (0.88, 0.84)], k=0.36)
    p.rrect(0.14, 0.16, 0.30, 0.84, 0.06)


def _settings(p: _Pen) -> None:
    # Three slider rails with knobs - reads better than a second gear.
    for y, kx in ((0.22, 0.66), (0.50, 0.34), (0.78, 0.58)):
        p.line((0.10, y), (0.90, y), 0.62)
        p.circle((kx, y), 0.145)
        p.punch_circle((kx, y), 0.062)
        p.circle((kx, y), 0.062, p.acc)


def _gear(p: _Pen) -> None:
    teeth = 8
    ro, ri = 0.48, 0.355
    half = 12.0          # angular half-width of a tooth
    gap = 10.0           # bevel between tooth and root
    pts: List[Pt] = []
    for i in range(teeth):
        base = i * (360.0 / teeth)
        for ang, r in (
            (base - half, ro), (base + half, ro),
            (base + half + gap, ri), (base + 360.0 / teeth - half - gap, ri),
        ):
            a = math.radians(ang)
            pts.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
    p.poly(pts, k=0.22)
    p.punch_circle((0.5, 0.5), 0.175)
    p.circle((0.5, 0.5), 0.175, p.acc)


def _sound_on(p: _Pen) -> None:
    _speaker(p)
    p.arc((0.44, 0.50), 0.20, -52, 52, 0.62)
    p.arc((0.44, 0.50), 0.36, -52, 52, 0.62)
    p.arc((0.44, 0.50), 0.52, -52, 52, 0.62)


def _sound_off(p: _Pen) -> None:
    _speaker(p)
    p.line((0.60, 0.32), (0.94, 0.68), 0.80)
    p.line((0.94, 0.32), (0.60, 0.68), 0.80)


def _note(p: _Pen, hx: float, hy: float, top: float, color=None) -> None:
    c = p._col(color)
    p.circle((hx, hy), 0.135, c)
    p.line((hx + 0.125, hy), (hx + 0.125, top), 0.62, c)


def _music_on(p: _Pen) -> None:
    _note(p, 0.24, 0.78, 0.24)
    _note(p, 0.66, 0.68, 0.14)
    p.poly([(0.365, 0.24), (0.785, 0.14), (0.785, 0.30), (0.365, 0.40)], k=0.16)


def _music_off(p: _Pen) -> None:
    _note(p, 0.32, 0.74, 0.22)
    p.poly([(0.445, 0.22), (0.80, 0.13), (0.80, 0.28), (0.445, 0.37)], k=0.16)
    # Cut a clean gutter out of the note before laying the slash into it.
    p.line((0.10, 0.90), (0.90, 0.10), 1.85, p.clear)
    p.line((0.12, 0.88), (0.88, 0.12), 0.80)


def _star(p: _Pen) -> None:
    p.poly(_star_pts(0.50, 0.52, 0.47, 0.216), k=0.30)


def _star_outline(p: _Pen) -> None:
    p.path(_star_pts(0.50, 0.52, 0.44, 0.202), 0.80, closed=True)


def _trophy(p: _Pen) -> None:
    p.arc((0.255, 0.28), 0.155, 80, 280, 0.62)      # left handle
    p.arc((0.745, 0.28), 0.155, -100, 100, 0.62)    # right handle
    p.poly([(0.255, 0.10), (0.745, 0.10), (0.725, 0.40),
            (0.615, 0.58), (0.385, 0.58), (0.275, 0.40)], k=0.18)
    p.rrect(0.435, 0.55, 0.565, 0.74, 0.03)
    p.rrect(0.315, 0.72, 0.685, 0.82, 0.04)
    p.rrect(0.215, 0.83, 0.785, 0.95, 0.05)
    p.poly(_star_pts(0.50, 0.31, 0.145, 0.066), p.acc)


def _crown(p: _Pen) -> None:
    p.poly([(0.06, 0.72), (0.13, 0.24), (0.315, 0.50), (0.50, 0.16),
            (0.685, 0.50), (0.87, 0.24), (0.94, 0.72)], k=0.20)
    p.rrect(0.06, 0.68, 0.94, 0.86, 0.06)
    for x, y in ((0.13, 0.26), (0.50, 0.19), (0.87, 0.26)):
        p.circle((x, y), 0.072, p.acc)
    p.circle((0.50, 0.77), 0.062, p.acc)


def _medal(p: _Pen) -> None:
    # Crossed ribbons behind the disc.
    p.poly([(0.10, 0.03), (0.29, 0.03), (0.55, 0.47), (0.38, 0.53)], k=0.08)
    # The far ribbon takes the accent so the crossing reads as two straps.
    p.poly([(0.90, 0.03), (0.71, 0.03), (0.45, 0.47), (0.62, 0.53)], p.acc, k=0.08)
    p.circle((0.50, 0.68), 0.305)
    p.circle((0.50, 0.68), 0.225, p.acc, k=0.42)
    p.poly(_star_pts(0.50, 0.68, 0.155, 0.070), p.acc)


def _heart(p: _Pen) -> None:
    raw: List[Pt] = []
    for i in range(60):
        t = i / 60.0 * math.tau
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t)
              - 2 * math.cos(3 * t) - math.cos(4 * t))
        raw.append((x, y))
    p.poly(_fit(raw, 0.06, 0.08, 0.94, 0.92), k=0.12)


def _flame(p: _Pen) -> None:
    # Teardrop body with the classic curl notch on the upper left.
    p.poly([
        (0.50, 0.02), (0.60, 0.16), (0.71, 0.26), (0.81, 0.43),
        (0.845, 0.61), (0.78, 0.81), (0.61, 0.96), (0.39, 0.96),
        (0.22, 0.81), (0.155, 0.61), (0.20, 0.42), (0.335, 0.325),
        (0.385, 0.20), (0.355, 0.075), (0.445, 0.135),
    ], k=0.12)
    p.poly([(0.50, 0.575), (0.585, 0.695), (0.575, 0.825),
            (0.50, 0.905), (0.425, 0.825), (0.415, 0.695)], p.acc, k=0.12)


def _snowflake(p: _Pen) -> None:
    for i in range(6):
        a = math.radians(i * 60.0)
        ex = 0.5 + 0.46 * math.cos(a)
        ey = 0.5 + 0.46 * math.sin(a)
        p.line((0.5, 0.5), (ex, ey), 0.56)
        for frac, blen in ((0.52, 0.155), (0.80, 0.115)):
            bx = 0.5 + 0.46 * frac * math.cos(a)
            by = 0.5 + 0.46 * frac * math.sin(a)
            for side in (-1, 1):
                b = a + side * math.radians(42)
                p.line((bx, by), (bx + blen * math.cos(b), by + blen * math.sin(b)), 0.44)
    p.circle((0.5, 0.5), 0.085, p.acc)


def _bolt(p: _Pen) -> None:
    p.poly([(0.62, 0.03), (0.24, 0.56), (0.46, 0.56),
            (0.38, 0.97), (0.76, 0.44), (0.54, 0.44)], k=0.20)


def _clock(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.44, k=0.80)
    p.line((0.5, 0.5), (0.5, 0.24), 0.62)
    p.line((0.5, 0.5), (0.70, 0.60), 0.62)
    p.circle((0.5, 0.5), 0.055, p.acc)


def _timer(p: _Pen) -> None:
    # Hourglass - instantly distinct from `clock` at a glance.
    p.poly([(0.335, 0.155), (0.665, 0.155), (0.50, 0.42)], p.acc)   # sand left
    p.poly([(0.50, 0.585), (0.665, 0.845), (0.335, 0.845)], p.acc)  # sand fallen
    p.path([(0.22, 0.10), (0.78, 0.10), (0.50, 0.50),
            (0.78, 0.90), (0.22, 0.90), (0.50, 0.50)], 0.70, closed=True)
    p.line((0.17, 0.085), (0.83, 0.085), 0.85)
    p.line((0.17, 0.915), (0.83, 0.915), 0.85)


def _check(p: _Pen) -> None:
    p.path([(0.16, 0.53), (0.40, 0.76), (0.84, 0.26)], 1.10)


def _plus(p: _Pen) -> None:
    p.line((0.50, 0.16), (0.50, 0.84), 1.05)
    p.line((0.16, 0.50), (0.84, 0.50), 1.05)


def _minus(p: _Pen) -> None:
    p.line((0.14, 0.50), (0.86, 0.50), 1.05)


def _multiply(p: _Pen) -> None:
    p.line((0.24, 0.24), (0.76, 0.76), 1.05)
    p.line((0.76, 0.24), (0.24, 0.76), 1.05)


def _divide(p: _Pen) -> None:
    p.line((0.14, 0.50), (0.86, 0.50), 1.05)
    p.circle((0.50, 0.22), 0.095)
    p.circle((0.50, 0.78), 0.095)


def _equals(p: _Pen) -> None:
    p.line((0.14, 0.34), (0.86, 0.34), 1.05)
    p.line((0.14, 0.66), (0.86, 0.66), 1.05)


def _percent(p: _Pen) -> None:
    p.line((0.78, 0.16), (0.22, 0.84), 0.85)
    p.circle((0.30, 0.28), 0.155, k=0.70)
    p.circle((0.70, 0.72), 0.155, k=0.70)


def _backspace(p: _Pen) -> None:
    p.path([(0.36, 0.14), (0.95, 0.14), (0.95, 0.86), (0.36, 0.86), (0.05, 0.50)],
           0.72, closed=True)
    p.line((0.55, 0.36), (0.80, 0.64), 0.62)
    p.line((0.80, 0.36), (0.55, 0.64), 0.62)


def _calculator(p: _Pen) -> None:
    p.rrect(0.14, 0.04, 0.86, 0.96, 0.13, k=0.72)
    p.rrect(0.26, 0.16, 0.74, 0.35, 0.05, p.acc)
    for gy in (0.51, 0.67, 0.83):
        for gx in (0.30, 0.50, 0.70):
            p.circle((gx, gy), 0.058)


def _chart(p: _Pen) -> None:
    p.rrect(0.10, 0.56, 0.31, 0.94, 0.05)
    p.rrect(0.395, 0.32, 0.605, 0.94, 0.05)
    p.rrect(0.69, 0.10, 0.90, 0.94, 0.05, p.acc)


def _lock(p: _Pen) -> None:
    p.arc((0.50, 0.40), 0.21, 180, 360, 0.72, cap=False)
    p.line((0.29, 0.40), (0.29, 0.50), 0.72)
    p.line((0.71, 0.40), (0.71, 0.50), 0.72)
    p.rrect(0.16, 0.44, 0.84, 0.94, 0.11)
    p.punch_circle((0.50, 0.63), 0.082)
    p.punch_poly([(0.455, 0.66), (0.545, 0.66), (0.525, 0.84), (0.475, 0.84)])
    p.circle((0.50, 0.63), 0.082, p.acc)
    p.poly([(0.455, 0.66), (0.545, 0.66), (0.525, 0.84), (0.475, 0.84)], p.acc)


def _unlock(p: _Pen) -> None:
    p.arc((0.70, 0.38), 0.21, 180, 360, 0.72, cap=False)
    p.line((0.91, 0.38), (0.91, 0.46), 0.72)
    p.line((0.49, 0.38), (0.49, 0.44), 0.72)
    p.rrect(0.10, 0.44, 0.72, 0.94, 0.11)
    p.punch_circle((0.41, 0.63), 0.082)
    p.punch_poly([(0.365, 0.66), (0.455, 0.66), (0.435, 0.84), (0.385, 0.84)])
    p.circle((0.41, 0.63), 0.082, p.acc)
    p.poly([(0.365, 0.66), (0.455, 0.66), (0.435, 0.84), (0.385, 0.84)], p.acc)


def _info(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.45, k=0.78)
    p.circle((0.5, 0.27), 0.070)
    p.line((0.5, 0.42), (0.5, 0.74), 0.80)


def _help(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.45, k=0.78)
    hook = p.arc_pts((0.50, 0.35), 0.155, 170, 375)
    hook.append((0.50, 0.62))
    p.path(hook, 0.72)
    p.circle((0.50, 0.76), 0.066)


def _retry(p: _Pen) -> None:
    r, cx, cy = 0.37, 0.50, 0.54
    end = 230.0
    p.arc((cx, cy), r, -30, end, 0.80, cap=False)
    # Continue along the tangent at the sweep end and cap it with a head.
    a = math.radians(end)
    ex, ey = cx + r * math.cos(a), cy + r * math.sin(a)
    dx, dy = -math.sin(a), math.cos(a)          # direction of travel
    p.arrow_head((ex + dx * 0.17, ey + dy * 0.17),
                 math.degrees(math.atan2(dy, dx)), 0.23)


def _chevron_left(p: _Pen) -> None:
    p.path([(0.64, 0.16), (0.32, 0.50), (0.64, 0.84)], 1.05)


def _chevron_right(p: _Pen) -> None:
    p.path([(0.36, 0.16), (0.68, 0.50), (0.36, 0.84)], 1.05)


def _chevron_up(p: _Pen) -> None:
    p.path([(0.16, 0.64), (0.50, 0.32), (0.84, 0.64)], 1.05)


def _chevron_down(p: _Pen) -> None:
    p.path([(0.16, 0.36), (0.50, 0.68), (0.84, 0.36)], 1.05)


def _coin(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.47)
    p.circle((0.5, 0.5), 0.355, p.acc, k=0.42)
    p.poly(_star_pts(0.50, 0.51, 0.235, 0.106), p.acc)


def _gem(p: _Pen) -> None:
    p.poly([(0.29, 0.16), (0.71, 0.16), (0.94, 0.44), (0.50, 0.94), (0.06, 0.44)], k=0.16)
    p.line((0.06, 0.44), (0.94, 0.44), 0.34, p.acc)
    p.line((0.29, 0.16), (0.345, 0.44), 0.34, p.acc)
    p.line((0.71, 0.16), (0.655, 0.44), 0.34, p.acc)
    p.line((0.345, 0.44), (0.50, 0.94), 0.34, p.acc)
    p.line((0.655, 0.44), (0.50, 0.94), 0.34, p.acc)


def _target(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.45, k=0.72)
    p.circle((0.5, 0.5), 0.265, k=0.72)
    p.circle((0.5, 0.5), 0.105, p.acc)


def _rope(p: _Pen) -> None:
    """A slack, braided rope - the game's own motif."""
    def sample(t: float) -> Pt:
        # A single sagging span - the tug-of-war silhouette.
        return (0.07 + t * 0.86, 0.40 + 0.19 * math.sin(t * math.pi))

    body = [sample(i / 28.0) for i in range(29)]
    p.path(body, 0.30 / 0.112, cap=True)        # thick capsule along the curve
    # Diagonal strands, each perpendicular-ish to the local tangent.
    for i in range(1, 9):
        t = i / 9.0
        a, b = sample(t - 0.012), sample(t + 0.012)
        tx, ty = b[0] - a[0], b[1] - a[1]
        m = math.hypot(tx, ty) or 1.0
        tx, ty = tx / m, ty / m
        nx, ny = -ty, tx
        c = sample(t)
        p.line((c[0] - nx * 0.15 - tx * 0.055, c[1] - ny * 0.15 - ty * 0.055),
               (c[0] + nx * 0.15 + tx * 0.055, c[1] + ny * 0.15 + ty * 0.055),
               0.42, p.acc)


def _hand(p: _Pen) -> None:
    p.capsule((0.335, 0.52), (0.335, 0.30), 0.155)
    p.capsule((0.495, 0.52), (0.495, 0.16), 0.155)
    p.capsule((0.655, 0.52), (0.655, 0.21), 0.155)
    p.capsule((0.805, 0.54), (0.805, 0.33), 0.150)
    p.capsule((0.315, 0.66), (0.135, 0.52), 0.155)
    p.rrect(0.255, 0.46, 0.885, 0.95, 0.22)


def _user(p: _Pen) -> None:
    _person(p, 0.50, 1.0)


def _users(p: _Pen) -> None:
    _person(p, 0.28, 0.78, p.acc)
    # Carve a rim so the front figure separates cleanly from the back one.
    _person_clear = p.clear
    p.circle((0.645, 0.335), 0.215, _person_clear)
    dome = p.arc_pts((0.645, 0.905), 0.375, 180, 360)
    dome.append((dome[-1][0], 0.905))
    dome.append((dome[0][0], 0.905))
    p.poly(dome, _person_clear)
    _person(p, 0.645, 0.92)


def _grid(p: _Pen) -> None:
    for x0, y0 in ((0.08, 0.08), (0.54, 0.08), (0.08, 0.54), (0.54, 0.54)):
        p.rrect(x0, y0, x0 + 0.38, y0 + 0.38, 0.10)


def _book(p: _Pen) -> None:
    left = [(0.05, 0.21), (0.455, 0.13), (0.455, 0.89), (0.05, 0.83)]
    right = [(0.95, 0.21), (0.545, 0.13), (0.545, 0.89), (0.95, 0.83)]
    p.poly(left, k=0.14)
    p.poly(right, k=0.14)
    p.line((0.50, 0.155), (0.50, 0.925), 0.45)   # spine
    for i, y in enumerate((0.34, 0.50, 0.66)):
        p.line((0.13, y + 0.012 * i), (0.40, y - 0.05 + 0.012 * i), 0.30, p.acc)
        p.line((0.60, y - 0.05 + 0.012 * i), (0.87, y + 0.012 * i), 0.30, p.acc)


def _sparkle(p: _Pen) -> None:
    def spark(cx: float, cy: float, r: float, color=None) -> None:
        pts: List[Pt] = []
        for i in range(8):
            rr = r if i % 2 == 0 else r * 0.24
            a = math.radians(-90 + i * 45)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        p.poly(pts, color, k=0.10)
    spark(0.42, 0.44, 0.42)
    spark(0.82, 0.80, 0.18, p.acc)
    spark(0.84, 0.20, 0.13, p.acc)


def _menu(p: _Pen) -> None:
    for y in (0.24, 0.50, 0.76):
        p.line((0.12, y), (0.88, y), 0.95)


def _trash(p: _Pen) -> None:
    p.line((0.08, 0.24), (0.92, 0.24), 0.72)
    p.path([(0.36, 0.22), (0.38, 0.09), (0.62, 0.09), (0.64, 0.22)], 0.60)
    p.poly([(0.19, 0.30), (0.81, 0.30), (0.74, 0.94), (0.26, 0.94)], k=0.16)
    for x in (0.38, 0.50, 0.62):
        p.line((x, 0.44), (x, 0.80), 0.30, p.acc)


def _pencil(p: _Pen) -> None:
    p.poly([(0.66, 0.05), (0.95, 0.34), (0.34, 0.95), (0.05, 0.95), (0.05, 0.66)], k=0.14)
    p.line((0.55, 0.16), (0.84, 0.45), 0.42, p.acc)
    p.poly([(0.05, 0.72), (0.28, 0.95), (0.05, 0.95)], p.acc)


def _eye(p: _Pen) -> None:
    # Two mirrored circular arcs meeting at sharp corners = a proper almond.
    top = p.arc_pts((0.50, 0.762), 0.512, 210.8, 329.2)
    bot = p.arc_pts((0.50, 0.238), 0.512, 30.8, 149.2)
    p.path(top + bot, 0.62, closed=True, cap=False)
    p.circle((0.50, 0.50), 0.165)
    p.circle((0.565, 0.435), 0.055, p.clear)


def _bell(p: _Pen) -> None:
    p.circle((0.50, 0.145), 0.078)
    dome = p.arc_pts((0.50, 0.58), 0.29, 180, 360)
    p.poly(dome + [(0.86, 0.74), (0.14, 0.74)], k=0.10)
    p.rrect(0.09, 0.70, 0.91, 0.83, 0.055)
    p.circle((0.50, 0.915), 0.095)


def _shield(p: _Pen) -> None:
    p.poly([(0.50, 0.04), (0.92, 0.20), (0.86, 0.60),
            (0.50, 0.96), (0.14, 0.60), (0.08, 0.20)], k=0.16)
    p.path([(0.32, 0.48), (0.45, 0.62), (0.70, 0.34)], 0.72, p.acc)


def _flag(p: _Pen) -> None:
    p.line((0.20, 0.06), (0.20, 0.96), 0.72)
    wave_top = [(0.20, 0.12)] + [
        (0.20 + t * 0.68, 0.12 + 0.07 * math.sin(t * math.pi * 1.6))
        for t in [i / 12 for i in range(1, 13)]
    ]
    wave_bot = [
        (0.20 + t * 0.68, 0.52 + 0.07 * math.sin(t * math.pi * 1.6))
        for t in [i / 12 for i in range(12, -1, -1)]
    ]
    p.poly(wave_top + wave_bot, k=0.10)


def _globe(p: _Pen) -> None:
    p.circle((0.5, 0.5), 0.45, k=0.66)
    p.line((0.08, 0.50), (0.92, 0.50), 0.50)
    p.path(p.arc_pts((0.50, 0.50), 0.45, -90, 90), 0.50, cap=False)
    p.ellipse(0.29, 0.05, 0.71, 0.95, k=0.50)
    p.path(p.arc_pts((0.50, -0.22), 0.86, 60, 120), 0.42, cap=False)
    p.path(p.arc_pts((0.50, 1.22), 0.86, -120, -60), 0.42, cap=False)


def _search(p: _Pen) -> None:
    p.circle((0.42, 0.42), 0.32, k=0.78)
    p.line((0.66, 0.66), (0.92, 0.92), 0.90)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_DRAW: Dict[str, Callable[[_Pen], None]] = {
    # navigation
    "home": _home, "back": _back, "close": _close, "menu": _menu,
    "next": _skip_next, "prev": _skip_prev,
    "arrow_left": _back, "arrow_right": _next_arrow,
    "arrow_up": _arrow_up, "arrow_down": _arrow_down,
    "chevron_left": _chevron_left, "chevron_right": _chevron_right,
    "chevron_up": _chevron_up, "chevron_down": _chevron_down,
    # transport / settings
    "play": _play, "pause": _pause, "stop": _stop,
    "settings": _settings, "gear": _gear,
    "sound_on": _sound_on, "sound_off": _sound_off,
    "music_on": _music_on, "music_off": _music_off,
    # rewards
    "star": _star, "star_outline": _star_outline, "trophy": _trophy,
    "crown": _crown, "medal": _medal, "coin": _coin, "gem": _gem,
    "sparkle": _sparkle,
    # status / vibe
    "heart": _heart, "flame": _flame, "snowflake": _snowflake, "bolt": _bolt,
    "zap": _bolt, "clock": _clock, "timer": _timer, "target": _target,
    "shield": _shield, "flag": _flag, "bell": _bell, "eye": _eye,
    # math
    "check": _check, "cross": _close, "plus": _plus, "minus": _minus,
    "multiply": _multiply, "divide": _divide, "equals": _equals,
    "percent": _percent, "backspace": _backspace, "calculator": _calculator,
    "chart": _chart,
    # meta
    "lock": _lock, "unlock": _unlock, "info": _info, "help": _help,
    "retry": _retry, "refresh": _retry, "grid": _grid, "book": _book,
    "search": _search, "trash": _trash, "pencil": _pencil, "globe": _globe,
    # game specific
    "rope": _rope, "hand": _hand, "user": _user, "users": _users,
}

ICONS: frozenset = frozenset(_DRAW)


def _placeholder(p: _Pen) -> None:
    p.rrect(0.10, 0.10, 0.90, 0.90, 0.14, k=0.62)
    p.line((0.28, 0.28), (0.72, 0.72), 0.50)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def icon_surface(
    name: str,
    size: Tuple[int, int],
    color: Color,
    accent: Optional[Color] = None,
    width: float = 1.0,
) -> pygame.Surface:
    """Rasterise (and memoise) one icon at an exact pixel size."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    color = (int(color[0]), int(color[1]), int(color[2]))
    acc = None if accent is None else (int(accent[0]), int(accent[1]), int(accent[2]))
    key = (name, w, h, color, acc, round(float(width), 3))
    cached = _cache.get(key)
    if cached is not None:
        return cached

    big = pygame.Surface((w * _SS, h * _SS), pygame.SRCALPHA)
    # Pre-fill with the icon colour at zero alpha so the downscale never
    # averages against black and haloes the edges.
    big.fill((color[0], color[1], color[2], 0))

    side = min(w, h) * _SS
    box = pygame.Rect((w * _SS - side) // 2, (h * _SS - side) // 2, side, side)
    pen = _Pen(big, box, color, acc, max(0.15, float(width)))
    _DRAW.get(name, _placeholder)(pen)

    out = pygame.transform.smoothscale(big, (w, h))
    _cache[key] = out
    return out


def draw_icon(
    surf: pygame.Surface,
    name: str,
    rect: pygame.Rect,
    color: Color,
    accent: Optional[Color] = None,
    width: float = 1.0,
) -> None:
    """Draw `name` centred inside `rect`, scaled to fill it proportionally.

    `color` is the primary ink; `accent` colours secondary detail (gem facets,
    a trophy's star, the slash on a muted speaker). If omitted an accent is
    derived automatically from `color`.
    """
    rect = pygame.Rect(rect)
    if rect.width <= 0 or rect.height <= 0:
        return
    surf.blit(icon_surface(name, rect.size, color, accent, width), rect.topleft)


def draw_icon_alpha(
    surf: pygame.Surface,
    name: str,
    rect: pygame.Rect,
    color: Color,
    alpha: int,
    accent: Optional[Color] = None,
    width: float = 1.0,
) -> None:
    """Same as `draw_icon` but faded - used by widgets that cross-fade icons."""
    if alpha >= 255:
        draw_icon(surf, name, rect, color, accent, width)
        return
    if alpha <= 0:
        return
    src = icon_surface(name, pygame.Rect(rect).size, color, accent, width)
    tmp = src.copy()
    tmp.fill((255, 255, 255, max(0, min(255, alpha))), special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(tmp, pygame.Rect(rect).topleft)

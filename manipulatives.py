"""Visual math manipulatives - the pictures that make problems make sense.

The math engine describes *what* to show as a `VisualSpec` (a `kind` string
plus a `data` dict); this module owns *how* it looks. Keeping the split here
means the curriculum logic never has to know about pygame, and the art can be
retuned without touching the problem generators.

Every renderer takes a target rect and draws itself centered and scaled to
fit, so callers can drop a manipulative into any panel without measuring.
Renderers are defensive: a missing or malformed `data` key degrades to a
sensible default rather than raising, because a kid staring at a crashed
screen is worse than a slightly wrong picture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pygame

import render_utils as R
import theme as T

Color = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _get(data: Dict[str, Any], key: str, default: Any) -> Any:
    """Read a key with a fallback, tolerating None values."""
    val = data.get(key, default) if isinstance(data, dict) else default
    return default if val is None else val


def _fit_grid(count: int, rect: pygame.Rect, max_cols: int = 10) -> Tuple[int, int, float]:
    """Choose (cols, rows, cell) that packs `count` items into `rect`."""
    count = max(1, count)
    cols = min(max_cols, count)
    rows = math.ceil(count / cols)
    cell = min(rect.width / max(1, cols), rect.height / max(1, rows))
    return cols, rows, cell


def _accent(palette=None) -> Color:
    return palette.core if palette is not None else T.GOLD


# --------------------------------------------------------------------------- #
# Counting: dots / objects
# --------------------------------------------------------------------------- #

def draw_dots(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """N countable tokens. Used for Pre-K counting and subitizing."""
    n = int(_get(data, "count", 5))
    color = _get(data, "color", _accent(palette))
    shape = _get(data, "shape", "circle")

    cols, rows, cell = _fit_grid(n, rect.inflate(-T.S4, -T.S4), max_cols=5)
    radius = cell * 0.32
    grid_w = cols * cell
    grid_h = rows * cell
    ox = rect.centerx - grid_w / 2 + cell / 2
    oy = rect.centery - grid_h / 2 + cell / 2

    for i in range(n):
        c, r = i % cols, i // cols
        # A gentle stagger so a row of tokens feels hand-placed, not gridded.
        bob = math.sin(t * 2.2 + i * 0.6) * cell * 0.04
        cx = ox + c * cell
        cy = oy + r * cell + bob

        R.add_glow(surf, (int(cx), int(cy)), int(radius * 2.1), color, 60)
        if shape == "square":
            box = pygame.Rect(0, 0, int(radius * 1.8), int(radius * 1.8))
            box.center = (int(cx), int(cy))
            surf.blit(
                R.gradient_rounded_rect(box.size, int(radius * 0.5),
                                        T.shade(color, 46), T.shade(color, -40)),
                box.topleft,
            )
        else:
            pygame.draw.circle(surf, T.shade(color, -46), (int(cx), int(cy + radius * 0.14)), int(radius))
            pygame.draw.circle(surf, color, (int(cx), int(cy)), int(radius))
            # Specular dot sells the 3D read at a glance.
            pygame.draw.circle(surf, T.shade(color, 78),
                               (int(cx - radius * 0.3), int(cy - radius * 0.34)),
                               max(1, int(radius * 0.26)))


# --------------------------------------------------------------------------- #
# Ten-frame
# --------------------------------------------------------------------------- #

def draw_tenframe(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """The classic K/1st ten-frame: two rows of five cells, `filled` occupied."""
    filled = int(_get(data, "filled", 0))
    capacity = int(_get(data, "capacity", 10))
    color = _get(data, "color", _accent(palette))

    cols = 5
    rows = max(1, math.ceil(capacity / cols))
    cell = min((rect.width - T.S3) / cols, (rect.height - T.S3) / rows)
    grid_w, grid_h = cell * cols, cell * rows
    ox = rect.centerx - grid_w / 2
    oy = rect.centery - grid_h / 2

    frame = pygame.Rect(int(ox), int(oy), int(grid_w), int(grid_h))
    R.draw_rrect(surf, frame.inflate(8, 8), T.with_alpha(T.INK, 34), T.R_SM, width=0)

    for i in range(capacity):
        c, r = i % cols, i // cols
        cx = ox + c * cell + cell / 2
        cy = oy + r * cell + cell / 2
        box = pygame.Rect(int(ox + c * cell), int(oy + r * cell), int(cell), int(cell))
        pygame.draw.rect(surf, T.with_alpha(T.INK_FAINT, 120), box, 2, border_radius=4)
        if i < filled:
            rad = cell * 0.32
            R.add_glow(surf, (int(cx), int(cy)), int(rad * 2.2), color, 70)
            pygame.draw.circle(surf, color, (int(cx), int(cy)), int(rad))
            pygame.draw.circle(surf, T.shade(color, 70),
                               (int(cx - rad * 0.3), int(cy - rad * 0.32)),
                               max(1, int(rad * 0.26)))


# --------------------------------------------------------------------------- #
# Number line
# --------------------------------------------------------------------------- #

def draw_numberline(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """A labeled number line with optional markers and a hop arc.

    Used for ordering, fractions-on-a-line (3rd) and add/sub-as-movement.
    """
    lo = float(_get(data, "min", 0))
    hi = float(_get(data, "max", 10))
    step = float(_get(data, "step", 1))
    marks: Sequence[float] = _get(data, "marks", [])
    hop = _get(data, "hop", None)            # (from, to)
    labels = bool(_get(data, "labels", True))
    color = _get(data, "color", _accent(palette))

    span = max(1e-6, hi - lo)
    y = rect.centery + rect.height * 0.16
    x0 = rect.left + T.S5
    x1 = rect.right - T.S5

    def to_x(v: float) -> float:
        return x0 + (x1 - x0) * ((v - lo) / span)

    pygame.draw.line(surf, T.INK_DIM, (x0, y), (x1, y), 3)
    # Arrowheads signal the line continues in both directions.
    for end, direction in ((x0, -1), (x1, 1)):
        pygame.draw.polygon(surf, T.INK_DIM, [
            (end + direction * 10, y),
            (end - direction * 2, y - 6),
            (end - direction * 2, y + 6),
        ])

    n_ticks = int(span / step) if step > 0 else 0
    if 0 < n_ticks <= 40:
        font = R.font_num(T.T_MICRO, bold=True)
        for i in range(n_ticks + 1):
            v = lo + i * step
            x = to_x(v)
            major = abs(v - round(v)) < 1e-6
            h = 12 if major else 7
            pygame.draw.line(surf, T.INK_DIM if major else T.INK_FAINT,
                             (x, y - h), (x, y + h), 2)
            if labels and major:
                txt = f"{int(v)}" if abs(v - int(v)) < 1e-6 else f"{v:g}"
                R.draw_text(surf, txt, font, T.INK_DIM, midtop=(int(x), int(y + h + 3)))

    if hop and len(hop) == 2:
        # An arc showing the jump from one value to another.
        hx0, hx1 = to_x(float(hop[0])), to_x(float(hop[1]))
        peak = y - max(24, abs(hx1 - hx0) * 0.34)
        pts = []
        for i in range(25):
            k = i / 24
            bx = (1 - k) ** 2 * hx0 + 2 * (1 - k) * k * ((hx0 + hx1) / 2) + k ** 2 * hx1
            by = (1 - k) ** 2 * y + 2 * (1 - k) * k * peak + k ** 2 * y
            pts.append((bx, by))
        R.draw_polyline_glow(surf, pts, color, width=3, glow_width=10, glow_alpha=80)
        # Arrowhead at the landing point, pointing down onto the line.
        tip_dir = 1 if hx1 >= hx0 else -1
        pygame.draw.polygon(surf, color, [
            (hx1, y - 1),
            (hx1 - tip_dir * 11, y - 13),
            (hx1 - tip_dir * 2, y - 15),
        ])

    for m in marks:
        x = to_x(float(m))
        R.add_glow(surf, (int(x), int(y)), 22, color, 110)
        pygame.draw.circle(surf, color, (int(x), int(y)), 8)
        pygame.draw.circle(surf, T.INK, (int(x), int(y)), 8, 2)


# --------------------------------------------------------------------------- #
# Fractions
# --------------------------------------------------------------------------- #

def draw_fraction_bar(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """One or two bars split into equal parts, with some shaded.

    Two bars lets us show equivalence (1/2 vs 2/4) side by side, which is the
    core 3rd/4th grade fraction concept.
    """
    bars = _get(data, "bars", None)
    if not bars:
        bars = [{"num": int(_get(data, "num", 1)), "den": int(_get(data, "den", 2))}]
    color = _get(data, "color", _accent(palette))

    n = len(bars)
    bar_h = min(56, (rect.height - T.S3 * (n + 1)) / max(1, n))
    total_h = n * bar_h + (n - 1) * T.S3
    top = rect.centery - total_h / 2
    width = min(rect.width - T.S5 * 2, 420)
    left = rect.centerx - width / 2

    font = R.font_num(T.T_LABEL, bold=True)
    for bi, bar in enumerate(bars):
        den = max(1, int(bar.get("den", 2)))
        num = max(0, int(bar.get("num", 0)))
        y = top + bi * (bar_h + T.S3)
        seg_w = width / den

        for i in range(den):
            seg = pygame.Rect(int(left + i * seg_w) + 1, int(y),
                              int(seg_w) - 2, int(bar_h))
            if i < num:
                surf.blit(
                    R.gradient_rounded_rect(seg.size, 6,
                                            T.shade(color, 50), T.shade(color, -50)),
                    seg.topleft,
                )
            else:
                R.draw_rrect(surf, seg, T.with_alpha(T.INK, 24), 6)
            pygame.draw.rect(surf, T.with_alpha(T.INK, 90), seg, 2, border_radius=6)

        R.draw_text(surf, f"{num}/{den}", font, T.INK,
                    midleft=(int(left + width + T.S3), int(y + bar_h / 2)))


def draw_fraction_circle(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """A pie split into `den` wedges with `num` filled."""
    den = max(1, int(_get(data, "den", 4)))
    num = max(0, int(_get(data, "num", 1)))
    color = _get(data, "color", _accent(palette))

    radius = int(min(rect.width, rect.height) * 0.40)
    cx, cy = rect.centerx, rect.centery
    R.add_glow(surf, (cx, cy), int(radius * 1.5), color, 55)

    for i in range(den):
        a0 = -math.pi / 2 + (math.tau * i / den)
        a1 = -math.pi / 2 + (math.tau * (i + 1) / den)
        pts = [(cx, cy)]
        steps = max(3, int(24 / den) + 3)
        for s in range(steps + 1):
            a = a0 + (a1 - a0) * (s / steps)
            pts.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
        fill = color if i < num else T.with_alpha(T.INK, 26)
        pygame.draw.polygon(surf, fill, pts)
        pygame.draw.polygon(surf, T.with_alpha(T.INK, 110), pts, 2)

    pygame.draw.circle(surf, T.INK_DIM, (cx, cy), radius, 3)


# --------------------------------------------------------------------------- #
# Arrays / base-ten
# --------------------------------------------------------------------------- #

def draw_array(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """A rows x cols dot array - multiplication as area/repeated addition."""
    rows = max(1, int(_get(data, "rows", 3)))
    cols = max(1, int(_get(data, "cols", 4)))
    color = _get(data, "color", _accent(palette))

    cell = min((rect.width - T.S4) / cols, (rect.height - T.S4) / rows)
    grid_w, grid_h = cell * cols, cell * rows
    ox = rect.centerx - grid_w / 2 + cell / 2
    oy = rect.centery - grid_h / 2 + cell / 2
    rad = max(3, cell * 0.30)

    for r in range(rows):
        for c in range(cols):
            cx, cy = ox + c * cell, oy + r * cell
            pygame.draw.circle(surf, T.shade(color, -50), (int(cx), int(cy + rad * 0.15)), int(rad))
            pygame.draw.circle(surf, color, (int(cx), int(cy)), int(rad))

    font = R.font_num(T.T_LABEL, bold=True)
    R.draw_text(surf, f"{rows} x {cols}", font, T.INK_DIM,
                midtop=(rect.centerx, int(oy + grid_h - cell / 2 + T.S3)))


def draw_base_ten(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """Place-value blocks: hundreds flats, tens rods, ones cubes."""
    hundreds = int(_get(data, "hundreds", 0))
    tens = int(_get(data, "tens", 0))
    ones = int(_get(data, "ones", 0))
    color = _get(data, "color", _accent(palette))

    # Size the unit cube so the whole set fits the panel: a hundreds flat is
    # 10 units wide (+1 gap), a tens rod 1 (+1 gap), and ones stack 2 wide.
    cols_needed = hundreds * 11 + tens * 2 + 2
    unit = min(rect.height * 0.080, (rect.width - T.S5) / max(1, cols_needed))
    unit = max(2.0, unit)

    total_w = cols_needed * unit
    x = rect.centerx - total_w / 2
    base_y = rect.centery + (10 * unit) / 2

    def cube(px, py, size, shade_delta=0):
        box = pygame.Rect(int(px), int(py), int(size), int(size))
        pygame.draw.rect(surf, T.shade(color, shade_delta), box)
        pygame.draw.rect(surf, T.shade(color, -70), box, 1)

    for _ in range(hundreds):
        for r in range(10):
            for c in range(10):
                cube(x + c * unit, base_y - 10 * unit + r * unit, unit, 20)
        x += unit * 11
    for _ in range(tens):
        for r in range(10):
            cube(x, base_y - 10 * unit + r * unit, unit, 0)
        x += unit * 2
    for i in range(ones):
        col, row = i % 2, i // 2
        cube(x + col * unit, base_y - (row + 1) * unit, unit, -18)


# --------------------------------------------------------------------------- #
# Clock / money / shapes
# --------------------------------------------------------------------------- #

def draw_clock(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """An analog clock face for telling-time problems."""
    hour = int(_get(data, "hour", 3)) % 12
    minute = int(_get(data, "minute", 0)) % 60
    color = _get(data, "color", _accent(palette))

    radius = int(min(rect.width, rect.height) * 0.42)
    cx, cy = rect.centerx, rect.centery

    R.add_glow(surf, (cx, cy), int(radius * 1.4), color, 45)
    pygame.draw.circle(surf, (250, 250, 255), (cx, cy), radius)
    pygame.draw.circle(surf, T.shade(color, -30), (cx, cy), radius, 5)

    font = R.font_num(max(11, int(radius * 0.20)), bold=True)
    for i in range(12):
        a = -math.pi / 2 + math.tau * i / 12
        # Hour numerals sit inside the tick ring.
        nx = cx + math.cos(a) * radius * 0.78
        ny = cy + math.sin(a) * radius * 0.78
        label = str(12 if i == 0 else i)
        R.draw_text(surf, label, font, T.INK_ON_LIGHT, center=(int(nx), int(ny)),
                    shadow_color=None)
        tx0 = cx + math.cos(a) * radius * 0.90
        ty0 = cy + math.sin(a) * radius * 0.90
        tx1 = cx + math.cos(a) * radius * 0.97
        ty1 = cy + math.sin(a) * radius * 0.97
        pygame.draw.line(surf, T.INK_ON_LIGHT, (tx0, ty0), (tx1, ty1), 3)

    for i in range(60):
        if i % 5 == 0:
            continue
        a = -math.pi / 2 + math.tau * i / 60
        pygame.draw.line(
            surf, (150, 155, 175),
            (cx + math.cos(a) * radius * 0.93, cy + math.sin(a) * radius * 0.93),
            (cx + math.cos(a) * radius * 0.97, cy + math.sin(a) * radius * 0.97), 1,
        )

    minute_angle = -math.pi / 2 + math.tau * (minute / 60)
    hour_angle = -math.pi / 2 + math.tau * ((hour + minute / 60) / 12)
    pygame.draw.line(surf, T.INK_ON_LIGHT, (cx, cy),
                     (cx + math.cos(hour_angle) * radius * 0.50,
                      cy + math.sin(hour_angle) * radius * 0.50), 7)
    pygame.draw.line(surf, T.shade(color, -20), (cx, cy),
                     (cx + math.cos(minute_angle) * radius * 0.74,
                      cy + math.sin(minute_angle) * radius * 0.74), 5)
    pygame.draw.circle(surf, T.shade(color, -40), (cx, cy), 6)


_COIN_STYLE = {
    # name: (value cents, radius scale, face color, label)
    "penny":   (1,  0.62, (196, 122, 76),  "1c"),
    "nickel":  (5,  0.74, (176, 180, 190), "5c"),
    "dime":    (10, 0.55, (198, 202, 212), "10c"),
    "quarter": (25, 0.86, (188, 192, 204), "25c"),
}


def draw_coins(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """A handful of US coins for 2nd grade money problems."""
    coins: List[str] = list(_get(data, "coins", ["quarter", "dime", "penny"]))
    coins = [c for c in coins if c in _COIN_STYLE][:12]
    if not coins:
        return

    # Cap columns at 4 so coins stay large enough for a kid to identify.
    cols, rows, cell = _fit_grid(len(coins), rect.inflate(-T.S3, -T.S3), max_cols=4)
    ox = rect.centerx - (cols * cell) / 2 + cell / 2
    oy = rect.centery - (rows * cell) / 2 + cell / 2
    font = R.font_num(max(11, int(cell * 0.26)), bold=True)

    for i, name in enumerate(coins):
        _, rscale, face, label = _COIN_STYLE[name]
        c, r = i % cols, i // cols
        cx, cy = ox + c * cell, oy + r * cell
        rad = cell * 0.50 * rscale
        pygame.draw.circle(surf, T.shade(face, -60), (int(cx), int(cy + rad * 0.14)), int(rad))
        pygame.draw.circle(surf, face, (int(cx), int(cy)), int(rad))
        pygame.draw.circle(surf, T.shade(face, 55), (int(cx), int(cy)), int(rad), 2)
        R.draw_text(surf, label, font, T.shade(face, -110), center=(int(cx), int(cy)),
                    shadow_color=None)


def draw_shapes(surf, rect: pygame.Rect, data: Dict[str, Any], palette=None, t: float = 0.0) -> None:
    """2D shape recognition (Pre-K/K) and angle/geometry hints (4th)."""
    names: List[str] = list(_get(data, "shapes", ["circle"]))[:6]
    color = _get(data, "color", _accent(palette))

    cols, rows, cell = _fit_grid(len(names), rect.inflate(-T.S4, -T.S4), max_cols=3)
    ox = rect.centerx - (cols * cell) / 2 + cell / 2
    oy = rect.centery - (rows * cell) / 2 + cell / 2

    for i, name in enumerate(names):
        c, r = i % cols, i // cols
        cx, cy = ox + c * cell, oy + r * cell
        rad = cell * 0.34
        pts: List[Tuple[float, float]] = []
        if name in ("triangle", "tri"):
            sides = 3
        elif name in ("square", "rect", "rectangle"):
            sides = 4
        elif name == "pentagon":
            sides = 5
        elif name == "hexagon":
            sides = 6
        else:
            sides = 0

        if sides == 0:
            pygame.draw.circle(surf, color, (int(cx), int(cy)), int(rad))
            pygame.draw.circle(surf, T.shade(color, -60), (int(cx), int(cy)), int(rad), 3)
        else:
            rot = -math.pi / 2 if sides % 2 else -math.pi / 4
            for s in range(sides):
                a = rot + math.tau * s / sides
                pts.append((cx + math.cos(a) * rad, cy + math.sin(a) * rad))
            pygame.draw.polygon(surf, color, pts)
            pygame.draw.polygon(surf, T.shade(color, -60), pts, 3)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

_RENDERERS = {
    "dots": draw_dots,
    "tenframe": draw_tenframe,
    "ten_frame": draw_tenframe,
    "numberline": draw_numberline,
    "number_line": draw_numberline,
    "fraction_bar": draw_fraction_bar,
    "fraction_circle": draw_fraction_circle,
    "array": draw_array,
    "base_ten": draw_base_ten,
    "clock": draw_clock,
    "coins": draw_coins,
    "shapes": draw_shapes,
}

SUPPORTED = frozenset(_RENDERERS)


def draw_visual(surf, rect: pygame.Rect, visual, palette=None, t: float = 0.0) -> bool:
    """Render any VisualSpec. Returns False if the kind is unknown.

    Accepts either a VisualSpec-like object (``.kind`` / ``.data``) or a plain
    dict, so this works regardless of how the math engine hands it over.
    """
    if visual is None:
        return False
    kind = getattr(visual, "kind", None)
    data = getattr(visual, "data", None)
    if kind is None and isinstance(visual, dict):
        kind = visual.get("kind")
        data = visual.get("data")
    if not kind:
        return False

    fn = _RENDERERS.get(str(kind).lower())
    if fn is None:
        return False
    try:
        fn(surf, rect, data or {}, palette, t)
    except Exception:
        # A malformed spec must never take down a match in progress.
        return False
    return True

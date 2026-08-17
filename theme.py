"""Design system for RopeRush - the single source of truth for look & feel.

Everything visual in the game reads its colors, spacing, type scale, timing
curves and layout metrics from here. Nothing else should hardcode a color.

Art direction
-------------
"Night arena at dusk." A deep indigo stadium lit by two rival energy sources:
EMBER (warm coral/amber, left) and FROST (electric cyan/blue, right). Gold is
the neutral accent used for rewards, stars and the rope itself. Surfaces are
dark translucent glass panels with soft inner light, floating above a gradient
sky with volumetric haze.

This is deliberately NOT a light/pastel UI - it reads clean and premium at
small sizes on a phone held horizontally, and the two team colors stay
unambiguous even at a glance.

Layout
------
The game renders to a fixed logical canvas (LOGICAL_W x LOGICAL_H, 16:9) and
is letterbox-scaled to whatever window/screen it lands on. All coordinates in
game code are in logical pixels. SAFE_* insets keep critical UI clear of
rounded corners and notches on real handsets.
"""

from __future__ import annotations

import math
from typing import Tuple

Color = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Canvas & layout
# --------------------------------------------------------------------------- #

LOGICAL_W = 1280
LOGICAL_H = 720

# Insets that keep critical UI away from notches / rounded display corners.
SAFE_L = 44
SAFE_R = 44
SAFE_T = 16
SAFE_B = 16

# Minimum comfortable touch target on a phone (logical px).
TOUCH_MIN = 64

# 8pt spacing scale.
S1, S2, S3, S4, S5, S6, S7, S8 = 4, 8, 12, 16, 24, 32, 48, 64

# Corner radii
R_SM, R_MD, R_LG, R_XL, R_PILL = 8, 14, 22, 30, 999


# --------------------------------------------------------------------------- #
# Core palette
# --------------------------------------------------------------------------- #

# Backdrop / sky gradient stops (top -> bottom)
SKY_TOP: Color = (11, 14, 38)
SKY_MID: Color = (28, 24, 66)
SKY_LOW: Color = (58, 38, 84)
SKY_HORIZON: Color = (104, 58, 88)

# Ground / arena floor
GROUND_FAR: Color = (36, 28, 62)
GROUND_NEAR: Color = (22, 18, 42)
GROUND_LINE: Color = (86, 72, 132)

# Neutral ink (text on dark)
INK: Color = (238, 242, 255)
INK_DIM: Color = (166, 176, 214)
INK_FAINT: Color = (108, 118, 158)
INK_ON_LIGHT: Color = (22, 24, 48)

# Glass surfaces (drawn as RGBA over the backdrop)
GLASS: RGBA = (255, 255, 255, 20)
GLASS_STRONG: RGBA = (255, 255, 255, 34)
GLASS_EDGE: RGBA = (255, 255, 255, 56)
SCRIM: RGBA = (7, 9, 24, 200)

# Accent (rewards, stars, rope, celebration)
GOLD: Color = (255, 200, 87)
GOLD_DEEP: Color = (226, 150, 40)
GOLD_LIGHT: Color = (255, 232, 168)

# Semantic
OK: Color = (74, 222, 128)
WARN: Color = (251, 191, 36)
BAD: Color = (248, 113, 113)


# --------------------------------------------------------------------------- #
# Teams
# --------------------------------------------------------------------------- #

class TeamPalette:
    """All the colors one team needs, derived from a single identity hue."""

    def __init__(
        self,
        key: str,
        name: str,
        core: Color,
        light: Color,
        deep: Color,
        glow: Color,
    ) -> None:
        self.key = key
        self.name = name
        self.core = core      # primary fill
        self.light = light    # highlight / gradient top
        self.deep = deep      # shadow / gradient bottom
        self.glow = glow      # additive glow + particle color

    @property
    def gradient(self) -> Tuple[Color, Color]:
        return (self.light, self.deep)


EMBER = TeamPalette(
    key="ember",
    name="Ember",
    core=(255, 107, 74),
    light=(255, 168, 94),
    deep=(198, 54, 62),
    glow=(255, 138, 76),
)

FROST = TeamPalette(
    key="frost",
    name="Frost",
    core=(56, 189, 248),
    light=(125, 226, 255),
    deep=(37, 99, 205),
    glow=(94, 214, 255),
)

TEAMS = (EMBER, FROST)


# --------------------------------------------------------------------------- #
# Typography
# --------------------------------------------------------------------------- #

# Font families are resolved at runtime with a fallback chain, since we ship no
# font files. render_utils.get_font() does the actual lookup + caching.
FONT_STACK_DISPLAY = ["Poppins", "Montserrat", "Verdana", "DejaVu Sans", "Arial"]
FONT_STACK_TEXT = ["Inter", "Segoe UI", "Verdana", "DejaVu Sans", "Arial"]
FONT_STACK_NUM = ["Poppins", "Montserrat", "Verdana", "DejaVu Sans", "Arial"]

# Type scale (logical px). Names describe role, not size.
T_HERO = 82        # win banner
T_DISPLAY = 54     # screen titles
T_TITLE = 38       # section headers, the live math problem
T_HEAD = 28        # panel headings, big numbers
T_BODY = 21        # standard copy, buttons
T_LABEL = 17       # small labels
T_MICRO = 14       # captions, hints


# --------------------------------------------------------------------------- #
# Motion
# --------------------------------------------------------------------------- #

# Durations in seconds.
D_INSTANT = 0.08
D_FAST = 0.16
D_BASE = 0.26
D_SLOW = 0.45
D_SCENE = 0.55


def ease_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - pow(1.0 - t, 3)


def ease_in_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t


def ease_in_out_cubic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def ease_out_back(t: float, overshoot: float = 1.70158) -> float:
    """Overshoots past 1 then settles - use for things popping into view."""
    t = max(0.0, min(1.0, t))
    c1 = overshoot
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def ease_out_elastic(t: float) -> float:
    t = max(0.0, min(1.0, t))
    if t == 0 or t == 1:
        return t
    c4 = (2 * math.pi) / 3
    return pow(2, -10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_quint(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - pow(1 - t, 5)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(a: Color, b: Color, t: float) -> Color:
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def with_alpha(c: Color, a: int) -> RGBA:
    return (c[0], c[1], c[2], max(0, min(255, a)))


def shade(c: Color, delta: int) -> Color:
    """Lighten (positive) or darken (negative) a color."""
    return (
        max(0, min(255, c[0] + delta)),
        max(0, min(255, c[1] + delta)),
        max(0, min(255, c[2] + delta)),
    )


def saturate(c: Color, factor: float) -> Color:
    """Push a color toward (factor>1) or away from (factor<1) its own hue."""
    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    return (
        max(0, min(255, int(lum + (c[0] - lum) * factor))),
        max(0, min(255, int(lum + (c[1] - lum) * factor))),
        max(0, min(255, int(lum + (c[2] - lum) * factor))),
    )

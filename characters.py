"""Character rendering for RopeRush - the chibi-athletic kids on the rope.

Art direction
-------------
"Night arena at dusk." Two rival squads face off across the rope: EMBER
(warm coral/amber, stands on the LEFT, faces right) and FROST (electric
cyan/blue, stands on the RIGHT, faces left). Each kid is a stylised
chibi-athlete - big head (~1/3 of body height), compact confident body,
chunky simplified limbs - tuned to read as a strong silhouette at phone size
while still carrying enough shading to feel hand-crafted rather than
programmer-drawn.

The look is built from six layered techniques, all done with pygame
primitives (no image assets):

1. **Rim light.** Every kid is composited on a private scratch surface, and
   the silhouette of that surface is differenced against a copy of itself
   nudged toward the arena centre. What survives is a 2-4px band hugging the
   edge that faces the team's own energy source, painted in `palette.glow`
   and blitted additively. This is the single strongest "AAA" tell: it
   separates the character from the dark backdrop and colour-codes the team
   even in a thumbnail.
2. **Gradient shading.** Nothing is a flat fill. Torso, shorts, shoes and
   head are pre-baked gradient surfaces (cached, then rotated per frame);
   limbs are drawn as tapered capsules with a lit upper edge and a dark
   lower edge, which fakes a cylinder.
3. **Ambient occlusion.** Soft blurred dark blobs sit under the chin, in the
   shoulder and hip sockets, and where the shoe meets the sole - the cheap
   trick that makes stacked shapes feel like one connected body.
4. **Ground shadow.** A blurred ellipse that squashes wide when the kid
   braces and pinches small and faint when they leave the ground.
5. **Cloth lag.** Hair tufts, ponytails, headband tails and the shirt hem
   ride a damped spring driven by the torso's angular velocity, so they
   whip after the body instead of moving with it.
6. **Squash, stretch, anticipation and follow-through.** The heave is not a
   sinusoid: it dips forward and compresses first (anticipation), snaps back
   with an ease-out-quint while the body stretches (the beat), then settles
   through a damped oscillation (follow-through).

Geometry contract
-----------------
`facing = +1` means the kid faces RIGHT and their hands reach toward +x
(this is EMBER, the LEFT team). `facing = -1` faces LEFT, hands reach toward
-x (FROST, the RIGHT team). `draw_kid` always returns a grip point that sits
exactly on `rope_y`.

Cache discipline
----------------
Every expensive surface (gradients, blurred blobs, baked body parts, rotated
variants) lives in the module-level `_CACHE`, keyed by quantised size plus
the kid's immutable `style_key`. Nothing blurred is built per frame. The
per-frame scratch surfaces come from `_SCRATCH`, a small size-keyed pool that
is cleared and reused rather than reallocated. Call `clear_caches()` if the
palette is ever swapped at runtime.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pygame

import render_utils as RU
import theme as T
from theme import TeamPalette

Color = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

# How long one heave beat lasts, end to end (seconds).
HEAVE_DUR = 0.62

# Physics is integrated with a clamped dt so a hitching frame can't explode
# the springs.
_MAX_DT = 1.0 / 20.0

# Body proportions, expressed as fractions of `Kid.height`. Heights are
# measured up from the ground line. These are the chibi tuning knobs - the
# head is deliberately huge and the legs deliberately short.
P_HEAD_R = 0.158        # head radius
P_HEAD_Y = 0.840        # head centre height
P_SHOULDER_Y = 0.650    # shoulder pivot height
P_HIP_Y = 0.415         # hip pivot height
P_TORSO_W = 0.330
P_TORSO_H = 0.310
P_SHORTS_W = 0.310
P_SHORTS_H = 0.145
P_UPPER_ARM = 0.190
P_FOREARM = 0.182
P_THIGH = 0.235
P_SHIN = 0.220
P_SHOE_W = 0.205
P_SHOE_H = 0.095
P_ARM_W = 0.078
P_LEG_W = 0.105

# Ink used for line work. Never pure black - it reads as a hole on a dark
# backdrop. This is the deep end of the sky gradient instead.
INK_LINE: Color = (16, 14, 34)


# --------------------------------------------------------------------------- #
# Caches
# --------------------------------------------------------------------------- #

_CACHE: Dict[tuple, pygame.Surface] = {}
_SCRATCH: Dict[Tuple[int, int], pygame.Surface] = {}


def clear_caches() -> None:
    """Drop every cached surface. Call after a runtime theme swap."""
    _CACHE.clear()
    _SCRATCH.clear()


def _q(v: float, step: int = 2) -> int:
    """Quantise a dimension so cache keys stay bounded."""
    return int(round(v / step) * step)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _scratch(w: int, h: int) -> pygame.Surface:
    """A cleared, pooled SRCALPHA surface at least (w, h) big."""
    key = (max(8, (w + 31) // 32 * 32), max(8, (h + 31) // 32 * 32))
    s = _SCRATCH.get(key)
    if s is None:
        s = pygame.Surface(key, pygame.SRCALPHA)
        _SCRATCH[key] = s
    s.fill((0, 0, 0, 0))
    return s


# --------------------------------------------------------------------------- #
# Small drawing primitives
# --------------------------------------------------------------------------- #

def _soft_blob(w: int, h: int, color: Color, alpha: int, amount: float = 0.30) -> pygame.Surface:
    """A blurred ellipse - the workhorse for AO and ground shadows.

    Returns a padded surface; blit it centred on the point you want occluded.
    """
    w, h = max(2, _q(w, 2)), max(2, _q(h, 2))
    key = ("blob", w, h, color, alpha, amount)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    pad = max(4, int(min(w, h) * 0.6))
    s = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (color[0], color[1], color[2], alpha), (pad, pad, w, h))
    s = RU.blur(s, amount=amount, passes=2)
    _CACHE[key] = s
    return s


def _blob(dst: pygame.Surface, cx: float, cy: float, w: float, h: float,
          color: Color, alpha: int, amount: float = 0.30) -> None:
    s = _soft_blob(int(w), int(h), color, alpha, amount)
    dst.blit(s, s.get_rect(center=(int(cx), int(cy))))


def _grad_ellipse(w: int, h: int, top: Color, bottom: Color) -> pygame.Surface:
    """Vertically shaded ellipse, cached."""
    w, h = max(2, int(w)), max(2, int(h))
    key = ("gell", w, h, top, bottom)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    grad = RU.vertical_gradient((w, h), top, bottom)
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(mask, (255, 255, 255, 255), (0, 0, w, h))
    out = grad.copy()
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    _CACHE[key] = out
    return out


def _alpha_grad(w: int, h: int, color: Color, a_top: int, a_bot: int) -> pygame.Surface:
    """A single-colour band whose alpha ramps top to bottom. Used for AO
    bands baked into body parts."""
    w, h = max(1, int(w)), max(1, int(h))
    key = ("agrad", w, h, color, a_top, a_bot)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    strip = pygame.Surface((1, h), pygame.SRCALPHA)
    for y in range(h):
        k = y / max(1, h - 1)
        strip.set_at((0, y), (color[0], color[1], color[2], int(a_top + (a_bot - a_top) * k)))
    out = pygame.transform.smoothscale(strip, (w, h))
    _CACHE[key] = out
    return out


def _rot(key: tuple, src: pygame.Surface, deg: float, step: int = 4) -> pygame.Surface:
    """Rotate a baked body part, cached at `step`-degree granularity.

    Rotating a cached gradient is far cheaper than rebuilding it, and
    quantising the angle keeps the cache to a couple dozen entries per part
    without any visible stepping at 60fps.
    """
    a = int(round(deg / step) * step) % 360
    k = ("rot", key, a)
    cached = _CACHE.get(k)
    if cached is not None:
        return cached
    out = src if a == 0 else pygame.transform.rotozoom(src, a, 1.0)
    _CACHE[k] = out
    return out


def _capsule(dst: pygame.Surface, p0, p1, w0: float, w1: float, color: Color) -> None:
    """A tapered round-ended bar from p0 (width w0) to p1 (width w1)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    h0, h1 = w0 * 0.5, w1 * 0.5
    pygame.draw.polygon(dst, color, [
        (p0[0] + nx * h0, p0[1] + ny * h0),
        (p1[0] + nx * h1, p1[1] + ny * h1),
        (p1[0] - nx * h1, p1[1] - ny * h1),
        (p0[0] - nx * h0, p0[1] - ny * h0),
    ])
    pygame.draw.circle(dst, color, (int(p0[0]), int(p0[1])), max(1, int(h0)))
    pygame.draw.circle(dst, color, (int(p1[0]), int(p1[1])), max(1, int(h1)))


def _shaded_limb(dst: pygame.Surface, p0, p1, w0: float, w1: float, base: Color) -> None:
    """A cylinder-shaded limb: dark contact edge, body, lit upper edge.

    The key light is up-and-forward, so the highlight always rides whichever
    normal points more upward. The team rim light is applied later, globally.
    """
    dark = T.lerp_color(base, INK_LINE, 0.55)
    lit = T.lerp_color(base, (255, 246, 232), 0.34)

    # 1. Dark under-edge: a slightly fatter capsule showing only at the rim.
    _capsule(dst, p0, p1, w0 + 2.2, w1 + 2.2, dark)
    # 2. Body.
    _capsule(dst, p0, p1, w0, w1, base)
    # 3. Lit edge, offset along the upward normal.
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L
    if ny > 0:
        nx, ny = -nx, -ny
    off = w0 * 0.21
    _capsule(dst,
             (p0[0] + nx * off, p0[1] + ny * off),
             (p1[0] + nx * off * 0.85, p1[1] + ny * off * 0.85),
             w0 * 0.42, w1 * 0.40, lit)


def _ik2(a, b, l1: float, l2: float, sign: int) -> Tuple[float, float]:
    """Two-bone IK: elbow/knee position for a chain from `a` reaching `b`.

    `sign` picks which of the two mirror solutions to use. If the target is
    out of reach the joint is placed straight along the line, so the limb
    stays visually connected (the second bone just reads as fully extended).
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy) or 1e-4
    base = math.atan2(dy, dx)
    if d >= l1 + l2:
        return (a[0] + math.cos(base) * l1, a[1] + math.sin(base) * l1)
    cos_a = (d * d + l1 * l1 - l2 * l2) / (2.0 * d * l1)
    ang = math.acos(_clamp(cos_a, -1.0, 1.0))
    th = base + sign * ang
    return (a[0] + math.cos(th) * l1, a[1] + math.sin(th) * l1)


def _rotp(px: float, py: float, cx: float, cy: float, a: float) -> Tuple[float, float]:
    """Rotate a point about a centre. Positive `a` is clockwise on screen."""
    s, c = math.sin(a), math.cos(a)
    dx, dy = px - cx, py - cy
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


# --------------------------------------------------------------------------- #
# The Kid
# --------------------------------------------------------------------------- #

@dataclass
class Kid:
    """One character. Pose is derived; only the spring state is persistent."""

    # --- placement -------------------------------------------------------
    x: float                    # foot anchor x, logical px
    y: float                    # ground line y, logical px
    facing: int                 # +1 faces right (EMBER), -1 faces left (FROST)
    phase: float                # per-kid animation offset, radians
    height: float               # ground-to-crown height, logical px
    palette: TeamPalette
    skin: Color
    hair: Color

    # --- style (immutable per kid; feeds the static-layer cache key) ------
    hair_style: int = 0         # 0 crop, 1 spikes, 2 ponytail, 3 curls
    accessory: int = 0          # 0 headband, 1 cap, 2 sweatband only
    shorts: Color = (46, 44, 78)
    shoe: Color = (238, 240, 250)
    band: Color = (250, 250, 255)
    jersey_num: int = 7
    freckles: bool = False
    depth: float = 0.0          # 0 = front row, 1 = back row
    style_key: tuple = ()

    # --- animation state (advanced by update_kid) ------------------------
    anim_t: float = 0.0
    lean: float = 0.0           # smoothed 0..1 pull-lean amount
    lean_v: float = 0.0
    squash: float = 1.0         # 1 = neutral, <1 compressed, >1 stretched
    squash_v: float = 0.0
    heave_t: float = -1.0       # <0 = no heave running
    heave_power: float = 0.0
    hair_lag: float = 0.0       # cloth springs, in fractions of height
    hair_lag_v: float = 0.0
    hem_lag: float = 0.0
    hem_lag_v: float = 0.0
    warmed: bool = False        # True once update_kid has run at least once


# --------------------------------------------------------------------------- #
# Team construction
# --------------------------------------------------------------------------- #

_SKINS: Tuple[Color, ...] = (
    (255, 219, 186), (245, 200, 160), (226, 172, 130),
    (196, 138, 100), (152, 100, 70), (110, 72, 52),
)
_HAIRS: Tuple[Color, ...] = (
    (38, 26, 30), (58, 38, 30), (96, 60, 34),
    (168, 112, 56), (218, 176, 104), (74, 52, 66), (28, 24, 34),
)
_SHORTS: Tuple[Color, ...] = (
    (44, 42, 76), (34, 32, 60), (58, 48, 74), (30, 38, 66),
)


def make_team(palette: TeamPalette, side: str, count: int,
              ground_y: float, x_range: Tuple[float, float]) -> List[Kid]:
    """Build one squad.

    `side` is 'left' or 'right'. The LEFT squad faces right (+1) and the
    RIGHT squad faces left (-1), so the two teams pull toward each other.

    The RNG is seeded from the side string, so a given team looks identical
    every match while still reading as a crowd of individuals: skin tone,
    hair colour and style, height, shorts, shoes and accessories all vary,
    and kids are staggered in depth so the row has some parallax.
    """
    facing = +1 if side == "left" else -1
    # Explicit seed - str.hash() is salted per process, which would make the
    # squads look different every launch.
    rng = random.Random(0x50FA if side == "left" else 0x1CE9)

    kids: List[Kid] = []
    span = x_range[1] - x_range[0]
    step = span / max(1, count - 1) if count > 1 else 0.0

    for i in range(count):
        # Depth alternates so neighbours never overlap identically. Back-row
        # kids stand a little further up the screen and a little shorter.
        depth = (i % 2) * 0.62 + rng.uniform(0.0, 0.30)
        x = x_range[0] + step * i + rng.uniform(-6.0, 6.0)
        h = rng.uniform(128.0, 152.0) * (1.0 - 0.13 * depth)

        shoe_pool = (
            (240, 242, 250),
            T.lerp_color(palette.light, (255, 255, 255), 0.35),
            (38, 40, 62),
            T.GOLD,
        )
        band_pool = (
            (250, 250, 255),
            T.lerp_color(palette.light, (255, 255, 255), 0.5),
            T.GOLD_LIGHT,
        )

        kid = Kid(
            x=x,
            y=ground_y - depth * 13.0,
            facing=facing,
            phase=rng.uniform(0.0, math.tau),
            height=h,
            palette=palette,
            skin=rng.choice(_SKINS),
            hair=rng.choice(_HAIRS),
            hair_style=rng.randrange(4),
            accessory=rng.choice((0, 0, 0, 1, 2)),
            shorts=rng.choice(_SHORTS),
            shoe=rng.choice(shoe_pool),
            band=rng.choice(band_pool),
            jersey_num=rng.randrange(1, 10),
            freckles=rng.random() < 0.35,
            depth=depth,
        )
        kid.style_key = (
            palette.key, kid.skin, kid.hair, kid.hair_style, kid.accessory,
            kid.shorts, kid.shoe, kid.band, kid.jersey_num, kid.freckles,
            _q(kid.height, 2), facing,
        )
        kids.append(kid)
    return kids


# --------------------------------------------------------------------------- #
# Animation state
# --------------------------------------------------------------------------- #

def _heave_env(kid: Kid) -> float:
    """The heave beat as a signed curve, in [-0.42, 1.15].

    Deliberately not a sinusoid. Three acts:
      0.00-0.28  anticipation - dips NEGATIVE, i.e. the kid rocks forward
                 toward the rope and coils before pulling.
      0.28-0.50  the beat - snaps back with ease-out-quint, overshooting to
                 1.15 so the body visibly stretches.
      0.50-1.00  follow-through - a damped oscillation settling to zero.
    """
    if kid.heave_t < 0.0:
        return 0.0
    u = kid.heave_t / HEAVE_DUR
    p = kid.heave_power
    if u < 0.28:
        return -0.42 * p * math.sin((u / 0.28) * math.pi)
    if u < 0.50:
        return 1.15 * p * T.ease_out_quint((u - 0.28) / 0.22)
    k = (u - 0.50) / 0.50
    return 1.15 * p * math.exp(-4.2 * k) * math.cos(k * 4.6)


def _lean_target(pull_intensity: float, strain: float, hv: float) -> float:
    """Where the torso wants to be, before the spring smooths it."""
    return 0.18 + 0.42 * pull_intensity + 0.34 * strain + 0.58 * hv


def update_kid(kid: Kid, dt: float, pull_intensity: float, strain: float) -> None:
    """Advance the springs, cloth lag and squash/stretch for one kid.

    Safe to call at any frame rate; dt is clamped so a stall can't blow up
    the integrators. `draw_kid` reads the state this leaves behind.
    """
    dt = _clamp(dt, 0.0, _MAX_DT)
    if dt <= 0.0:
        return
    kid.warmed = True
    kid.anim_t += dt
    pull_intensity = _clamp(pull_intensity, 0.0, 1.0)
    strain = _clamp(strain, 0.0, 1.0)

    # --- heave clock -----------------------------------------------------
    if kid.heave_t >= 0.0:
        kid.heave_t += dt
        if kid.heave_t >= HEAVE_DUR:
            kid.heave_t = -1.0
            kid.heave_power = 0.0
    hv = _heave_env(kid)

    # --- torso lean: critically-damped spring ----------------------------
    target = _lean_target(pull_intensity, strain, hv)
    kid.lean_v += (210.0 * (target - kid.lean) - 28.0 * kid.lean_v) * dt
    kid.lean += kid.lean_v * dt
    kid.lean = _clamp(kid.lean, -0.55, 1.9)

    # --- squash & stretch ------------------------------------------------
    # Anticipation (hv < 0) compresses; the pull beat (hv > 0) stretches.
    # Heavy strain leaves the kid permanently a bit compressed and dug in.
    s_target = 1.0 + 0.115 * hv - 0.045 * strain
    kid.squash_v += (240.0 * (s_target - kid.squash) - 22.0 * kid.squash_v) * dt
    kid.squash += kid.squash_v * dt
    kid.squash = _clamp(kid.squash, 0.80, 1.22)

    # --- cloth lag -------------------------------------------------------
    # Hair and hem chase a target proportional to (negative) torso velocity,
    # so they trail behind the motion and whip forward when it reverses.
    h_target = _clamp(-kid.lean_v * 0.019, -0.34, 0.34)
    kid.hair_lag_v += (150.0 * (h_target - kid.hair_lag) - 15.0 * kid.hair_lag_v) * dt
    kid.hair_lag += kid.hair_lag_v * dt
    kid.hair_lag = _clamp(kid.hair_lag, -0.42, 0.42)

    m_target = _clamp(-kid.lean_v * 0.013, -0.26, 0.26)
    kid.hem_lag_v += (190.0 * (m_target - kid.hem_lag) - 19.0 * kid.hem_lag_v) * dt
    kid.hem_lag += kid.hem_lag_v * dt
    kid.hem_lag = _clamp(kid.hem_lag, -0.30, 0.30)


def trigger_heave(kid: Kid, power: float = 1.0) -> None:
    """Fire the punchy pull-impulse beat on this kid.

    Restarts the beat from the top and kicks the squash spring downward so
    the anticipation crouch snaps in on the very first frame rather than
    easing in over several.
    """
    kid.heave_t = 0.0
    kid.heave_power = _clamp(power, 0.15, 1.6)
    kid.squash_v -= 2.4 * kid.heave_power


# --------------------------------------------------------------------------- #
# Baked (cached) body parts
# --------------------------------------------------------------------------- #

def _mask_to(surface: pygame.Surface, radius: int) -> pygame.Surface:
    """Clip a surface to a rounded rect, so decals can't spill past the edge."""
    w, h = surface.get_size()
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h),
                     border_radius=max(0, min(radius, min(w, h) // 2)))
    surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return surface


def _torso_surface(kid: Kid, w: int, h: int) -> pygame.Surface:
    """The jersey: gradient body, collar, chest sash, number, hem AO.

    Baked once per (style, size) and then just rotated - this is the single
    biggest static layer, so it is worth caching aggressively.
    """
    key = ("torso", kid.style_key, w, h)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    pal = kid.palette
    radius = int(min(w, h) * 0.36)
    top = T.lerp_color(pal.light, (255, 255, 255), 0.20)
    bot = T.lerp_color(pal.deep, INK_LINE, 0.28)
    s = RU.gradient_rounded_rect((w, h), radius, top, bot).copy()

    deco = pygame.Surface((w, h), pygame.SRCALPHA)

    # Diagonal chest sash in the bright team tint - gives the silhouette an
    # internal read so the torso isn't one undifferentiated blob.
    sash = [
        (w * 0.02, h * 0.30), (w * 0.98, h * 0.10),
        (w * 0.98, h * 0.30), (w * 0.02, h * 0.50),
    ]
    if kid.facing < 0:
        sash = [(w - px, py) for px, py in sash]
    pygame.draw.polygon(deco, T.with_alpha(T.lerp_color(pal.light, (255, 255, 255), 0.55), 130), sash)

    # Jersey number, low contrast so it reads as fabric print not UI text.
    try:
        font = RU.font_display(max(9, int(h * 0.40)), True)
        num = font.render(str(kid.jersey_num), True, (255, 255, 255))
        num.set_alpha(96)
        deco.blit(num, num.get_rect(center=(int(w * 0.5), int(h * 0.62))))
    except Exception:  # pragma: no cover - font backends can be absent
        pass

    # Collar shadow under the chin.
    collar = pygame.Rect(0, 0, int(w * 0.46), int(h * 0.17))
    collar.midtop = (w // 2, -int(h * 0.05))
    pygame.draw.ellipse(deco, (12, 10, 26, 130), collar)

    # Hem ambient occlusion where the shirt tucks over the shorts.
    band_h = max(3, int(h * 0.30))
    deco.blit(_alpha_grad(w, band_h, INK_LINE, 0, 120), (0, h - band_h))

    # Top light wrap.
    deco.blit(_alpha_grad(w, max(2, int(h * 0.22)), (255, 255, 255), 60, 0), (0, 0))

    s.blit(deco, (0, 0))
    _mask_to(s, radius)
    _CACHE[key] = s
    return s


def _shorts_surface(kid: Kid, w: int, h: int) -> pygame.Surface:
    key = ("shorts", kid.style_key, w, h)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    base = kid.shorts
    s = RU.gradient_rounded_rect(
        (w, h), int(min(w, h) * 0.40),
        T.lerp_color(base, (255, 255, 255), 0.22),
        T.lerp_color(base, INK_LINE, 0.35),
    ).copy()
    deco = pygame.Surface((w, h), pygame.SRCALPHA)
    # Waistband stripe in team colour.
    pygame.draw.rect(deco, T.with_alpha(kid.palette.core, 190), (0, 0, w, max(2, int(h * 0.22))))
    # Leg-split shadow.
    pygame.draw.rect(deco, (10, 8, 22, 150),
                     (int(w * 0.46), int(h * 0.45), max(2, int(w * 0.08)), int(h * 0.6)))
    s.blit(deco, (0, 0))
    _mask_to(s, int(min(w, h) * 0.40))
    _CACHE[key] = s
    return s


def _shoe_surface(kid: Kid, w: int, h: int) -> pygame.Surface:
    """A chunky sneaker drawn pointing toward +x, then flipped if needed."""
    key = ("shoe", kid.style_key, w, h)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    w, h = max(6, w), max(4, h)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    col = kid.shoe

    # Upper - gradient, toe box swollen toward +x.
    up_h = int(h * 0.72)
    upper = RU.gradient_rounded_rect(
        (w, up_h), int(up_h * 0.48),
        T.lerp_color(col, (255, 255, 255), 0.30),
        T.lerp_color(col, INK_LINE, 0.30),
    )
    s.blit(upper, (0, 0))
    # Ankle cuff (back, i.e. -x side) sits higher.
    cuff = RU.gradient_rounded_rect(
        (int(w * 0.42), int(h * 0.55)), int(h * 0.22),
        T.lerp_color(col, (255, 255, 255), 0.42),
        T.lerp_color(col, INK_LINE, 0.10),
    )
    s.blit(cuff, (0, -int(h * 0.10)))

    # Outsole - bright, slightly wider, with a dark welt line above it.
    sole_h = max(3, int(h * 0.34))
    sole = RU.gradient_rounded_rect(
        (w, sole_h), int(sole_h * 0.5), (250, 250, 255), (176, 180, 200)
    )
    s.blit(sole, (0, h - sole_h))
    pygame.draw.line(s, (14, 12, 30, 170), (1, h - sole_h), (w - 1, h - sole_h), 1)

    # Swoosh stripe in the team glow colour.
    pygame.draw.line(s, kid.palette.glow,
                     (int(w * 0.20), int(h * 0.52)), (int(w * 0.78), int(h * 0.24)),
                     max(2, int(h * 0.13)))

    if kid.facing < 0:
        s = pygame.transform.flip(s, True, False)
    _CACHE[key] = s
    return s


def _head_surface(kid: Kid, r: int) -> pygame.Surface:
    """Head base: ear, skull gradient, hair cap, accessory, blush, chin AO.

    Facial features are NOT baked - they animate every frame and are drawn
    live on top of this. Everything here is style-constant.
    """
    key = ("head", kid.style_key, r)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    f = 1  # baked facing right; flipped at the end for left-facers
    pad = int(r * 0.80)
    size = int(r * 2 + pad * 2)
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = cy = size // 2

    skin = kid.skin
    skin_lit = T.lerp_color(skin, (255, 250, 238), 0.30)
    skin_dark = T.lerp_color(skin, (68, 34, 52), 0.42)
    hair = kid.hair
    hair_lit = T.lerp_color(hair, (255, 240, 220), 0.30)
    hair_dark = T.lerp_color(hair, INK_LINE, 0.45)

    # --- ear on the far (back) side --------------------------------------
    ear = _grad_ellipse(int(r * 0.40), int(r * 0.50), skin, skin_dark)
    s.blit(ear, ear.get_rect(center=(int(cx - f * r * 0.80), int(cy + r * 0.08))))

    # --- skull ------------------------------------------------------------
    hw, hh = int(r * 2.02), int(r * 2.0)
    head = _grad_ellipse(hw, hh, skin_lit, T.lerp_color(skin, INK_LINE, 0.22))
    s.blit(head, head.get_rect(center=(cx, cy)))

    # Chin / jaw ambient occlusion.
    _blob(s, cx, cy + r * 0.80, r * 1.25, r * 0.55, INK_LINE, 90, amount=0.35)
    # Cheek blush on the forward cheek.
    _blob(s, cx + f * r * 0.42, cy + r * 0.36, r * 0.62, r * 0.36,
          T.lerp_color(skin, kid.palette.core, 0.55), 120, amount=0.4)

    # --- hair -------------------------------------------------------------
    cap = pygame.Surface((size, size), pygame.SRCALPHA)
    if kid.hair_style == 0:            # tight crop
        cap.blit(_grad_ellipse(int(r * 2.06), int(r * 1.52), hair_lit, hair_dark),
                 (cx - int(r * 1.03), cy - int(r * 1.06)))
    elif kid.hair_style == 1:          # spikes
        cap.blit(_grad_ellipse(int(r * 2.04), int(r * 1.40), hair_lit, hair_dark),
                 (cx - int(r * 1.02), cy - int(r * 1.02)))
        for i in range(5):
            k = i / 4.0
            bx = cx + (k - 0.5) * r * 1.7
            pygame.draw.polygon(cap, hair_lit, [
                (bx - r * 0.20, cy - r * 0.62),
                (bx + r * 0.20, cy - r * 0.62),
                (bx + f * r * 0.26, cy - r * (1.18 + 0.20 * math.sin(i * 2.1))),
            ])
    elif kid.hair_style == 2:          # slicked back (ponytail drawn live)
        cap.blit(_grad_ellipse(int(r * 2.00), int(r * 1.36), hair_lit, hair_dark),
                 (cx - int(r * 1.00), cy - int(r * 1.00)))
    else:                              # curls
        for i in range(7):
            a = math.pi * (0.06 + 0.88 * i / 6.0)
            px = cx - math.cos(a) * r * 0.94
            py = cy - math.sin(a) * r * 0.86
            pygame.draw.circle(cap, hair_dark, (int(px), int(py)), int(r * 0.42))
        for i in range(6):
            a = math.pi * (0.12 + 0.78 * i / 5.0)
            px = cx - math.cos(a) * r * 0.86
            py = cy - math.sin(a) * r * 0.84
            pygame.draw.circle(cap, hair_lit, (int(px - r * 0.06), int(py - r * 0.08)), int(r * 0.30))

    # Forelock sweeping toward the face, so the hairline isn't a flat arc.
    pygame.draw.polygon(cap, hair_lit, [
        (cx - f * r * 0.10, cy - r * 0.92),
        (cx + f * r * 0.96, cy - r * 0.52),
        (cx + f * r * 0.86, cy - r * 0.14),
        (cx + f * r * 0.30, cy - r * 0.60),
    ])
    # Keep hair inside the skull silhouette (plus a little for spikes/curls).
    clip = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.ellipse(clip, (255, 255, 255, 255),
                        (cx - int(r * 1.04), cy - int(r * 1.02), int(r * 2.08), int(r * 2.04)))
    if kid.hair_style in (1, 3):
        pygame.draw.rect(clip, (255, 255, 255, 255),
                         (cx - int(r * 1.2), cy - int(r * 1.6), int(r * 2.4), int(r * 1.0)))
    cap.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    s.blit(cap, (0, 0))

    # --- accessory --------------------------------------------------------
    if kid.accessory == 1:             # backwards cap
        crown = _grad_ellipse(int(r * 2.06), int(r * 1.32),
                              T.lerp_color(kid.palette.light, (255, 255, 255), 0.25),
                              T.lerp_color(kid.palette.deep, INK_LINE, 0.20))
        s.blit(crown, (cx - int(r * 1.03), cy - int(r * 1.16)))
        brim = pygame.Rect(0, 0, int(r * 0.95), int(r * 0.34))
        brim.center = (int(cx - f * r * 1.05), int(cy - r * 0.52))
        pygame.draw.ellipse(s, T.lerp_color(kid.palette.deep, INK_LINE, 0.35), brim)
        pygame.draw.circle(s, kid.band, (cx, int(cy - r * 1.06)), max(2, int(r * 0.13)))
    else:                              # headband (tails drawn live)
        bh = max(4, int(r * 0.34))
        band = pygame.Rect(0, 0, int(r * 2.02), bh)
        band.center = (cx, int(cy - r * 0.50))
        pygame.draw.rect(s, kid.band, band, border_radius=bh // 2)
        pygame.draw.rect(s, T.lerp_color(kid.band, INK_LINE, 0.30), band, 1, border_radius=bh // 2)
        stripe = pygame.Rect(band.left + 2, band.centery - max(1, bh // 6),
                             band.width - 4, max(1, bh // 3))
        pygame.draw.rect(s, kid.palette.core, stripe, border_radius=2)

    if kid.freckles:
        fr = T.lerp_color(skin, (120, 60, 40), 0.45)
        for i in range(6):
            fx = cx + f * r * (0.18 + 0.16 * (i % 3)) * (1 if i < 3 else -0.3)
            fy = cy + r * (0.30 + 0.09 * (i % 2))
            pygame.draw.circle(s, fr, (int(fx), int(fy)), max(1, int(r * 0.045)))

    if kid.facing < 0:
        s = pygame.transform.flip(s, True, False)
    _CACHE[key] = s
    return s


# --------------------------------------------------------------------------- #
# Expression
# --------------------------------------------------------------------------- #

@dataclass
class _Expr:
    """Everything the face needs, resolved from mode + intensity + strain."""
    lid: float = 0.0        # 0 wide open .. 1 fully closed
    happy: float = 0.0      # 0 normal .. 1 upturned-arc eyes
    wide: float = 0.0       # startled scale on the sclera
    brow: float = 0.0       # 0 relaxed .. 1 hard angry-V
    brow_up: float = 0.0    # raised worried/surprised brows
    mouth: str = "set"      # 'set' | 'shout' | 'grit' | 'grin' | 'o'
    open_amt: float = 0.0
    sweat: float = 0.0


def _expression(mode: str, pull: float, strain: float, hv: float, t: float, phase: float) -> _Expr:
    e = _Expr()
    if mode == "celebrate":
        e.happy = 1.0
        e.mouth = "grin"
        e.open_amt = 0.55 + 0.25 * math.sin(t * 7.0 + phase)
        e.brow_up = 0.7
        return e
    if mode == "stagger":
        e.wide = 1.0
        e.brow_up = 1.0
        e.mouth = "o"
        e.open_amt = 0.8
        return e

    effort = _clamp(max(pull, hv * 0.9), 0.0, 1.0)
    e.brow = _clamp(0.30 + effort * 0.65 + strain * 0.45, 0.0, 1.0)
    e.lid = _clamp(0.12 + strain * 0.52 + effort * 0.20, 0.0, 0.85)
    e.sweat = _clamp((strain - 0.42) / 0.58, 0.0, 1.0)
    if strain > 0.55:
        e.mouth = "grit"
        e.open_amt = 0.35 + 0.25 * strain
    elif effort > 0.45:
        e.mouth = "shout"
        e.open_amt = 0.35 + 0.6 * effort
    else:
        e.mouth = "set"
        e.open_amt = 0.16 + 0.2 * effort
    return e


def _draw_face(dst: pygame.Surface, cx: float, cy: float, r: float, f: int,
               ang: float, kid: Kid, e: _Expr) -> None:
    """Live facial features, rotated with the head tilt.

    Three-quarter view: both eyes visible but crowded toward the facing side,
    the near one slightly larger. Everything is placed in head-local space
    and then rotated by `ang` (clockwise-positive radians).
    """
    def P(ux: float, uy: float) -> Tuple[float, float]:
        """Head-local (in units of r, +x = forward) -> rotated screen point."""
        return _rotp(cx + f * ux * r, cy + uy * r, cx, cy, ang)

    ink = (26, 22, 44)
    eye_r = r * (0.215 + 0.06 * e.wide)
    near = P(0.46, 0.02)
    far = P(-0.08, 0.02)

    if e.happy > 0.5:
        # ^ ^ arcs - no sclera, reads instantly as joy at thumbnail size.
        for c, sc in ((near, 1.0), (far, 0.9)):
            rr = pygame.Rect(0, 0, int(eye_r * 2.1 * sc), int(eye_r * 1.9 * sc))
            rr.center = (int(c[0]), int(c[1] + eye_r * 0.3))
            pygame.draw.arc(dst, ink, rr, 0.35, math.pi - 0.35, max(2, int(r * 0.09)))
    else:
        for c, sc in ((near, 1.0), (far, 0.88)):
            rx, ry = eye_r * 0.92 * sc, eye_r * 1.06 * sc
            rect = pygame.Rect(0, 0, int(rx * 2), int(ry * 2))
            rect.center = (int(c[0]), int(c[1]))
            pygame.draw.ellipse(dst, (252, 250, 255), rect)
            # Iris darts toward the rope (forward).
            ix = c[0] + f * rx * 0.30
            iy = c[1] + ry * 0.10
            pygame.draw.circle(dst, (30, 34, 66), (int(ix), int(iy)), max(2, int(ry * 0.62)))
            pygame.draw.circle(dst, kid.palette.glow, (int(ix), int(iy)), max(1, int(ry * 0.34)))
            # Specular.
            pygame.draw.circle(dst, (255, 255, 255),
                               (int(ix - f * ry * 0.22), int(iy - ry * 0.34)),
                               max(1, int(ry * 0.24)))
            # Upper lid.
            if e.lid > 0.04:
                lid_h = int(ry * 2 * e.lid)
                if lid_h > 0:
                    lid = pygame.Rect(0, 0, int(rx * 2.4), lid_h)
                    lid.midtop = (int(c[0]), int(c[1] - ry))
                    lid_col = T.lerp_color(kid.skin, INK_LINE, 0.18)
                    pygame.draw.ellipse(dst, lid_col, lid.inflate(0, int(ry)))

    # --- brows ------------------------------------------------------------
    brow_w = max(2, int(r * 0.11))
    brow_col = T.lerp_color(kid.hair, INK_LINE, 0.25)
    inner_dy = -0.30 - e.brow_up * 0.22 + e.brow * 0.30
    outer_dy = -0.30 - e.brow_up * 0.30 - e.brow * 0.16
    # near brow: inner edge is the one toward the back of the head
    n0 = P(0.46 - 0.26, outer_dy - 0.14)
    n1 = P(0.46 + 0.24, inner_dy - 0.14)
    pygame.draw.line(dst, brow_col, n0, n1, brow_w)
    fa0 = P(-0.08 - 0.22, outer_dy - 0.10)
    fa1 = P(-0.08 + 0.22, inner_dy - 0.12)
    pygame.draw.line(dst, brow_col, fa0, fa1, max(2, brow_w - 1))

    # --- nose -------------------------------------------------------------
    nz0 = P(0.30, 0.26)
    nz1 = P(0.50, 0.40)
    pygame.draw.line(dst, T.lerp_color(kid.skin, INK_LINE, 0.40), nz0, nz1, max(2, int(r * 0.07)))

    # --- mouth ------------------------------------------------------------
    mc = P(0.24, 0.60)
    mw = r * 0.52
    mh = r * (0.10 + 0.42 * e.open_amt)
    if e.mouth == "set":
        a = pygame.Rect(0, 0, int(mw * 1.5), int(r * 0.42))
        a.center = (int(mc[0]), int(mc[1] - r * 0.10))
        pygame.draw.arc(dst, (58, 30, 46), a, math.pi * 1.10, math.pi * 1.90, max(2, int(r * 0.075)))
    elif e.mouth == "grin":
        a = pygame.Rect(0, 0, int(mw * 1.9), int(r * 0.80 * (0.5 + e.open_amt)))
        a.center = (int(mc[0]), int(mc[1] - r * 0.10))
        pygame.draw.ellipse(dst, (52, 22, 38), a)
        teeth = pygame.Rect(0, 0, a.width - max(2, int(r * 0.10)), max(2, int(a.height * 0.34)))
        teeth.midtop = (a.centerx, a.top + max(1, int(r * 0.045)))
        pygame.draw.rect(dst, (255, 252, 246), teeth, border_radius=2)
    elif e.mouth == "o":
        a = pygame.Rect(0, 0, int(mw * 0.9), int(mh * 1.5))
        a.center = (int(mc[0]), int(mc[1]))
        pygame.draw.ellipse(dst, (52, 22, 38), a)
    elif e.mouth == "grit":
        a = pygame.Rect(0, 0, int(mw * 1.5), int(mh * 1.1))
        a.center = (int(mc[0]), int(mc[1]))
        pygame.draw.rect(dst, (44, 18, 32), a, border_radius=max(2, int(r * 0.06)))
        pygame.draw.rect(dst, (255, 250, 244), a.inflate(-max(2, int(r * 0.08)), -max(2, int(r * 0.10))),
                         border_radius=2)
        # Clenched teeth lines.
        for i in range(1, 4):
            gx = a.left + a.width * i / 4.0
            pygame.draw.line(dst, (150, 130, 140), (gx, a.top + 2), (gx, a.bottom - 2), 1)
    else:  # shout
        a = pygame.Rect(0, 0, int(mw * 1.25), int(mh * 1.7))
        a.center = (int(mc[0]), int(mc[1]))
        pygame.draw.ellipse(dst, (46, 18, 34), a)
        tongue = a.inflate(-int(a.width * 0.36), -int(a.height * 0.52))
        tongue.bottom = a.bottom - max(1, int(r * 0.04))
        pygame.draw.ellipse(dst, (222, 96, 116), tongue)
        teeth = pygame.Rect(0, 0, a.width - max(2, int(r * 0.10)), max(2, int(a.height * 0.22)))
        teeth.midtop = (a.centerx, a.top + 1)
        pygame.draw.rect(dst, (255, 252, 246), teeth, border_radius=2)


# --------------------------------------------------------------------------- #
# Rim light
# --------------------------------------------------------------------------- #

#: Transparent margin added around the rim before blurring, so the bloom's
#: falloff never touches the surface edge (which would read as a rectangle).
RIM_BLOOM_PAD = 14

_RIM_SCRATCH: Dict[Tuple[int, int], pygame.Surface] = {}
_CARVE_SCRATCH: Dict[Tuple[int, int], pygame.Surface] = {}


def _exact_scratch(pool: Dict[Tuple[int, int], pygame.Surface],
                   size: Tuple[int, int]) -> pygame.Surface:
    """A cleared SRCALPHA surface of exactly `size`, pooled per call site.

    The rim pass needs two buffers alive at once and both must match the
    body surface exactly, so they can't share the general rounded-up pool.
    """
    s = pool.get(size)
    if s is None:
        s = pygame.Surface(size, pygame.SRCALPHA)
        pool[size] = s
    s.fill((0, 0, 0, 0))
    return s


def _rim_scratch(size: Tuple[int, int]) -> pygame.Surface:
    return _exact_scratch(_RIM_SCRATCH, size)


def _carve_scratch(size: Tuple[int, int]) -> pygame.Surface:
    return _exact_scratch(_CARVE_SCRATCH, size)


def _apply_rim(body: pygame.Surface, glow: Color, edge_x: int, thickness: int,
               strength: int, bloom: bool) -> Optional[pygame.Surface]:
    """Carve a rim band out of `body`'s silhouette and add it back as light.

    The band is `silhouette - silhouette_shifted_toward_the_light`, which is
    exactly the set of pixels on the lit edge, for any shape, with no shape
    knowledge required. Returns the (un-blitted) bloom layer if requested, so
    the caller can lay it into the scene behind the character.
    """
    size = body.get_size()

    # The band is computed on the alpha channel directly, in numpy.
    #
    # Doing this with surface blits is a trap: blitting a per-pixel-alpha
    # source onto a fully transparent surface copies the source RGB even where
    # its alpha is 0, leaving (glow_rgb, 0) outside the silhouette. Since the
    # result is composited with BLEND_RGB_ADD - which ignores alpha entirely -
    # that invisible RGB gets added across the whole scratch buffer and shows
    # up in the scene as a glowing rectangle.
    #
    # Working on the alpha mask and writing premultiplied RGB keeps everything
    # outside the body at a true zero, so the additive pass is well-behaved.
    alpha = pygame.surfarray.array_alpha(body).astype(np.float32) / 255.0

    dx = -edge_x * thickness
    dy = int(thickness * 0.55)
    shifted = np.zeros_like(alpha)
    w, h = size
    # shifted[x, y] = alpha[x - dx, y - dy], zero-filled outside.
    sx0, sx1 = max(0, dx), min(w, w + dx)
    sy0, sy1 = max(0, dy), min(h, h + dy)
    if sx1 > sx0 and sy1 > sy0:
        shifted[sx0:sx1, sy0:sy1] = alpha[sx0 - dx:sx1 - dx, sy0 - dy:sy1 - dy]

    band = np.clip(alpha - shifted, 0.0, 1.0) * (strength / 255.0)

    rim = _rim_scratch(size)
    rgb = pygame.surfarray.pixels3d(rim)
    rgb[:, :, 0] = (band * glow[0]).astype(np.uint8)
    rgb[:, :, 1] = (band * glow[1]).astype(np.uint8)
    rgb[:, :, 2] = (band * glow[2]).astype(np.uint8)
    del rgb
    a_view = pygame.surfarray.pixels_alpha(rim)
    a_view[:, :] = (band * 255.0).astype(np.uint8)
    del a_view

    bloom_layer = None
    if bloom:
        # Blur into a transparent margin. Blurring `rim` directly would smear
        # its bright edge pixels along the surface border, and since the bloom
        # is composited additively that border reads as a glowing rectangle
        # around the character. The padding gives the falloff somewhere to go.
        w, h = size
        padded = pygame.Surface((w + RIM_BLOOM_PAD * 2, h + RIM_BLOOM_PAD * 2),
                                pygame.SRCALPHA)
        padded.blit(rim, (RIM_BLOOM_PAD, RIM_BLOOM_PAD))
        bloom_layer = RU.blur(padded, amount=0.34, passes=1)

    body.blit(rim, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    return bloom_layer


# --------------------------------------------------------------------------- #
# Main draw
# --------------------------------------------------------------------------- #

def draw_kid(surf, kid: Kid, t: float, pull_intensity: float, strain: float,
             rope_y: float, mode: str = "play") -> Tuple[int, int]:
    """Draw one kid and return the (x, y) where their front hand grips the rope.

    `mode` is 'play' | 'celebrate' | 'stagger'. The returned point always sits
    exactly on `rope_y` so callers can anchor the rope to it; in 'celebrate'
    the hands are raised, so the returned point is the body's rope-line
    position rather than a literal hand.
    """
    pull_intensity = _clamp(pull_intensity, 0.0, 1.0)
    strain = _clamp(strain, 0.0, 1.0)

    H = kid.height
    f = 1 if kid.facing >= 0 else -1
    pal = kid.palette
    at = kid.anim_t if kid.warmed else t

    # ---- resolve animation scalars --------------------------------------
    hv = _heave_env(kid)
    lean = kid.lean if kid.warmed else _lean_target(pull_intensity, strain, hv)
    # Idle breathing / weight-shift so nobody is ever perfectly still.
    breathe = math.sin(at * 2.3 + kid.phase)
    lean += 0.055 * breathe * (1.0 - 0.5 * pull_intensity)

    sy = kid.squash * (1.0 + 0.012 * math.sin(at * 2.3 + kid.phase + 0.6))
    hop = 0.0

    if mode == "celebrate":
        # Two-beat hop with a compressed landing.
        u = (at * 2.05 + kid.phase * 0.31) % 1.0
        hop = math.sin(math.pi * min(1.0, u / 0.82)) ** 0.72 if u < 0.82 else 0.0
        land = 0.0 if u < 0.82 else math.sin((u - 0.82) / 0.18 * math.pi)
        sy = sy * (1.0 + 0.10 * hop - 0.16 * land)
        lean = -0.12 + 0.10 * hop
    elif mode == "stagger":
        # Yanked off balance: the torso pitches forward, past vertical.
        lean = -0.95 - 0.25 * math.sin(at * 9.0 + kid.phase)
        sy = sy * 0.97

    sy = _clamp(sy, 0.78, 1.24)
    sx = 1.0 / math.sqrt(max(0.55, sy))       # rough volume preservation

    theta = _clamp(0.14 + lean * 0.40, -0.62, 0.72)   # + = leaning away from rope
    deg = f * math.degrees(theta)                      # rotozoom is CCW-positive
    ang = -f * theta                                   # screen-clockwise radians

    # ---- scratch surface layout -----------------------------------------
    # Local space so the whole character can be composited (and rim-lit) as
    # one image before it touches the scene.
    pad_l = int(H * 0.78)
    pad_r = int(H * 0.90)
    pad_t = int(max(H * 1.42, (kid.y - rope_y) + H * 0.55))
    pad_b = int(H * 0.24)
    if f < 0:
        pad_l, pad_r = pad_r, pad_l
    body = _scratch(pad_l + pad_r, pad_t + pad_b)

    bx = int(math.floor(kid.x)) - pad_l
    by = int(math.floor(kid.y)) - pad_t
    gx = kid.x - bx                     # foot anchor in local space (float)
    gy = kid.y - by                     # ground line in local space
    rope_local = rope_y - by

    # ---- skeleton --------------------------------------------------------
    lean01 = _clamp(lean, -1.2, 1.6)
    hip_x = gx + f * H * (0.085 * lean01 + 0.02)
    hip_y = gy - H * P_HIP_Y * sy - hop * H * 0.22
    up = (-f * math.sin(theta), -math.cos(theta))

    spine = (P_SHOULDER_Y - P_HIP_Y) * H * sy
    sh_x = hip_x + up[0] * spine
    sh_y = hip_y + up[1] * spine
    head_r = H * P_HEAD_R
    head_x = hip_x + up[0] * ((P_HEAD_Y - P_HIP_Y) * H * sy)
    head_y = hip_y + up[1] * ((P_HEAD_Y - P_HIP_Y) * H * sy)

    # Feet: stance widens as the kid digs in.
    spread = 1.0 + 0.30 * _clamp(lean01, 0.0, 1.6) + 0.20 * strain
    back_ax = gx - f * H * 0.205 * spread
    front_ax = gx + f * H * 0.130 * spread
    ankle_y = gy - H * P_SHOE_H * 0.55
    back_ay = ankle_y - hop * H * 0.24
    front_ay = ankle_y - hop * H * 0.24
    if mode == "stagger":
        # Back foot rips off the ground as they're dragged forward.
        drag = 0.5 + 0.5 * math.sin(at * 9.0 + kid.phase)
        back_ax -= f * H * 0.02
        back_ay -= H * 0.11 * drag
    if mode == "celebrate":
        back_ay -= hop * H * 0.10
        front_ay -= hop * H * 0.06

    # Hands.
    if mode == "celebrate":
        fh = (gx + f * H * 0.20, head_y - H * 0.34 - hop * H * 0.05)
        bh = (gx - f * H * 0.07, head_y - H * 0.28 + math.sin(at * 8 + kid.phase) * H * 0.02)
    else:
        reach = 0.285 + 0.055 * _clamp(lean01, -0.6, 1.4)
        if mode == "stagger":
            reach = 0.40
        fh = (gx + f * H * reach, rope_local)
        bh = (gx + f * H * (reach - 0.115), rope_local + 1.0)

    # ---- ground shadow (scene-space, under everything) -------------------
    lift = hop + (0.11 if mode == "stagger" else 0.0)
    sh_w = H * (0.62 + 0.16 * _clamp(lean01, 0.0, 1.5)) * (1.0 - 0.42 * lift) * sx
    sh_h = sh_w * 0.29
    sh_a = int(150 * (1.0 - 0.55 * lift) * (1.0 - 0.25 * kid.depth))
    _blob(surf, kid.x + f * H * 0.02, kid.y + 2, sh_w, sh_h, (4, 3, 14), sh_a, amount=0.28)

    # Team energy pooling at their feet - cheap, cached, and it plants the
    # character in the arena lighting.
    RU.add_glow(surf, (int(kid.x), int(kid.y - H * 0.06)), int(H * 0.42), pal.glow,
                int(30 + 34 * pull_intensity))

    # ---- limbs (back layer) ---------------------------------------------
    arm_w = H * P_ARM_W * sx
    leg_w = H * P_LEG_W * sx
    sleeve = T.lerp_color(pal.core, INK_LINE, 0.10)
    sleeve_back = T.lerp_color(sleeve, INK_LINE, 0.32)
    skin_back = T.lerp_color(kid.skin, INK_LINE, 0.30)
    pants_back = T.lerp_color(kid.shorts, INK_LINE, 0.34)

    # Back arm.
    bsh = (sh_x - f * H * 0.055, sh_y + H * 0.018)
    b_elbow = _ik2(bsh, bh, H * P_UPPER_ARM * sy, H * P_FOREARM * sy, f)
    _shaded_limb(body, bsh, b_elbow, arm_w * 1.02, arm_w * 0.86, sleeve_back)
    _shaded_limb(body, b_elbow, bh, arm_w * 0.84, arm_w * 0.70, T.lerp_color(skin_back, INK_LINE, 0.05))

    # Back leg.
    bhip = (hip_x - f * H * 0.045, hip_y + H * 0.01)
    b_knee = _ik2(bhip, (back_ax, back_ay), H * P_THIGH * sy, H * P_SHIN * sy, -f)
    _shaded_limb(body, bhip, b_knee, leg_w * 1.02, leg_w * 0.88, pants_back)
    _shaded_limb(body, b_knee, (back_ax, back_ay), leg_w * 0.86, leg_w * 0.66, skin_back)
    _blit_shoe(body, kid, back_ax, back_ay, H, sx, dim=True)

    # ---- front leg -------------------------------------------------------
    fhip = (hip_x + f * H * 0.045, hip_y + H * 0.01)
    f_knee = _ik2(fhip, (front_ax, front_ay), H * P_THIGH * sy, H * P_SHIN * sy, -f)
    _shaded_limb(body, fhip, f_knee, leg_w * 1.08, leg_w * 0.92, kid.shorts)
    _shaded_limb(body, f_knee, (front_ax, front_ay), leg_w * 0.90, leg_w * 0.70, kid.skin)
    _blit_shoe(body, kid, front_ax, front_ay, H, sx, dim=False)

    # ---- shorts ----------------------------------------------------------
    sw, shh = max(4, int(H * P_SHORTS_W * sx)), max(4, int(H * P_SHORTS_H * sy))
    shorts_img = _rot(("shorts", kid.style_key, sw, shh), _shorts_surface(kid, sw, shh), deg * 0.55)
    body.blit(shorts_img, shorts_img.get_rect(center=(
        int(hip_x + up[0] * H * 0.012), int(hip_y + up[1] * H * 0.012 + H * 0.015))))

    # Hip-socket AO.
    _blob(body, hip_x, hip_y + H * 0.055, H * 0.30 * sx, H * 0.10, INK_LINE, 110, amount=0.35)

    # ---- torso -----------------------------------------------------------
    tw, th = max(6, int(H * P_TORSO_W * sx)), max(6, int(H * P_TORSO_H * sy))
    torso_img = _rot(("torso", kid.style_key, tw, th), _torso_surface(kid, tw, th), deg)
    tcx = hip_x + up[0] * (H * P_TORSO_H * sy * 0.46)
    tcy = hip_y + up[1] * (H * P_TORSO_H * sy * 0.46)
    body.blit(torso_img, torso_img.get_rect(center=(int(tcx), int(tcy))))

    # Shirt hem flapping with the cloth spring.
    hem_dx = f * kid.hem_lag * H * 0.55
    hem_pts = [
        _rotp(tcx - tw * 0.48, tcy + th * 0.34, tcx, tcy, ang),
        _rotp(tcx + tw * 0.48, tcy + th * 0.34, tcx, tcy, ang),
        (tcx + tw * 0.44 + hem_dx, tcy + th * 0.56 + abs(hem_dx) * 0.20),
        (tcx - tw * 0.44 + hem_dx * 0.7, tcy + th * 0.58 + abs(hem_dx) * 0.20),
    ]
    pygame.draw.polygon(body, T.lerp_color(pal.deep, INK_LINE, 0.32), hem_pts)

    # Shoulder-socket AO, drawn after the torso so it darkens the join.
    _blob(body, sh_x - f * H * 0.05, sh_y + H * 0.02, H * 0.16 * sx, H * 0.11, INK_LINE, 120, amount=0.4)

    # ---- neck ------------------------------------------------------------
    # Runs from the shoulder pivot up into the base of the skull, so the head
    # never floats free of the torso at any lean angle.
    neck_top = (head_x - up[0] * head_r * 0.45, head_y - up[1] * head_r * 0.45)
    _shaded_limb(body, (sh_x, sh_y - H * 0.005), neck_top,
                 H * 0.108 * sx, H * 0.096 * sx,
                 T.lerp_color(kid.skin, INK_LINE, 0.22))

    # ---- hair / band tails behind the head (cloth lag) -------------------
    _draw_tails(body, kid, head_x, head_y, head_r, f, ang, at)

    # ---- head ------------------------------------------------------------
    hr = max(4, int(head_r))
    head_img = _rot(("head", kid.style_key, hr), _head_surface(kid, hr), deg * 0.92)
    body.blit(head_img, head_img.get_rect(center=(int(head_x), int(head_y))))

    # Chin contact shadow onto the chest.
    _blob(body, head_x + f * head_r * 0.10, head_y + head_r * 0.95,
          head_r * 1.5, head_r * 0.62, INK_LINE, 95, amount=0.4)

    expr = _expression(mode, pull_intensity, strain, hv, at, kid.phase)
    _draw_face(body, head_x, head_y, head_r, f, ang, kid, expr)

    # ---- front arm + fists ----------------------------------------------
    fsh = (sh_x + f * H * 0.030, sh_y - H * 0.012)
    f_elbow = _ik2(fsh, fh, H * P_UPPER_ARM * sy, H * P_FOREARM * sy, f)
    _shaded_limb(body, fsh, f_elbow, arm_w * 1.10, arm_w * 0.92, sleeve)
    _shaded_limb(body, f_elbow, fh, arm_w * 0.90, arm_w * 0.76, kid.skin)

    for hand, rad, dim in ((bh, arm_w * 0.62, True), (fh, arm_w * 0.68, False)):
        c = T.lerp_color(kid.skin, INK_LINE, 0.30 if dim else 0.0)
        pygame.draw.circle(body, T.lerp_color(c, INK_LINE, 0.45),
                           (int(hand[0]), int(hand[1])), int(rad + 1.6))
        pygame.draw.circle(body, c, (int(hand[0]), int(hand[1])), int(rad))
        pygame.draw.circle(body, T.lerp_color(c, (255, 248, 236), 0.32),
                           (int(hand[0] - f * rad * 0.18), int(hand[1] - rad * 0.30)),
                           max(1, int(rad * 0.48)))

    # ---- rim light + bloom ----------------------------------------------
    # The energy source is behind the kid, on their own team's side of the
    # arena, so the lit edge is the -facing side.
    edge = -f
    thickness = max(2, int(H * 0.023))
    strength = int(200 + 40 * pull_intensity - 60 * kid.depth)
    bloom = _apply_rim(body, pal.glow, edge, thickness, strength,
                       bloom=(pull_intensity > 0.12 or mode == "celebrate"))

    # ---- composite -------------------------------------------------------
    # Strain makes them tremble; a sub-pixel-ish jitter sells the effort.
    tr = strain * strain * 1.7
    jx = int(round(math.sin(at * 41.0 + kid.phase * 7.1) * tr))
    jy = int(round(math.sin(at * 53.0 + kid.phase * 3.3) * tr * 0.6))

    if bloom is not None:
        surf.blit(bloom, (bx + jx - RIM_BLOOM_PAD, by + jy - RIM_BLOOM_PAD),
                  special_flags=pygame.BLEND_RGB_ADD)
    if kid.depth > 0.05:
        # Back-row kids sink into the arena haze. Safe to mutate `body` here -
        # it is a pooled scratch surface and we are done reading it.
        body.fill((255, 255, 255, int(255 * (1.0 - 0.26 * kid.depth))),
                  special_flags=pygame.BLEND_RGBA_MULT)
    surf.blit(body, (bx + jx, by + jy))

    # ---- sweat (scene space, so it can leave the silhouette) -------------
    if expr.sweat > 0.05:
        _draw_sweat(surf, kid, bx + head_x, by + head_y, head_r, f, at, expr.sweat)

    # ---- grip point ------------------------------------------------------
    return int(round(bx + fh[0])), int(round(rope_y))


def _blit_shoe(dst: pygame.Surface, kid: Kid, ax: float, ay: float,
               H: float, sx: float, dim: bool) -> None:
    """Sneaker + its contact AO, planted at an ankle position."""
    sw = max(6, int(H * P_SHOE_W * sx))
    sh = max(4, int(H * P_SHOE_H))
    img = _shoe_surface(kid, sw, sh)
    if dim:
        key = ("shoedim", kid.style_key, sw, sh)
        cached = _CACHE.get(key)
        if cached is None:
            cached = img.copy()
            cached.fill((168, 168, 190, 255), special_flags=pygame.BLEND_RGBA_MULT)
            _CACHE[key] = cached
        img = cached
    # Shoe points toward +facing; the anchor is the ankle, so the shoe sits
    # forward of it.
    r = img.get_rect(center=(int(ax + kid.facing * sw * 0.16), int(ay + sh * 0.40)))
    dst.blit(img, r)
    _blob(dst, r.centerx, r.bottom - 1, sw * 1.05, sh * 0.55, INK_LINE, 110, amount=0.35)


def _draw_tails(dst: pygame.Surface, kid: Kid, hx: float, hy: float, r: float,
                f: int, ang: float, at: float) -> None:
    """Ponytail and headband tails - the visible payoff of the cloth spring.

    Both hang off the BACK of the head (the -facing side) and are swung by
    `kid.hair_lag`, which trails the torso's angular velocity.
    """
    lag = kid.hair_lag + 0.035 * math.sin(at * 3.1 + kid.phase)
    swing = -f * lag * r * 3.4
    droop = abs(lag) * r * 0.9

    hair_lit = T.lerp_color(kid.hair, (255, 240, 220), 0.22)
    hair_dark = T.lerp_color(kid.hair, INK_LINE, 0.40)

    root = _rotp(hx - f * r * 0.82, hy - r * 0.30, hx, hy, ang)

    if kid.hair_style == 2:
        # Ponytail: two tapered segments so it can actually bend.
        mid = (root[0] - f * r * 0.62 + swing * 0.55, root[1] + r * 0.30 + droop * 0.5)
        tip = (root[0] - f * r * 1.20 + swing, root[1] + r * 0.86 + droop)
        _capsule(dst, root, mid, r * 0.52, r * 0.40, hair_dark)
        _capsule(dst, mid, tip, r * 0.40, r * 0.14, hair_dark)
        _capsule(dst, (root[0], root[1] - r * 0.06), (mid[0], mid[1] - r * 0.05),
                 r * 0.26, r * 0.20, hair_lit)
        # Scrunchie in the team colour.
        pygame.draw.circle(dst, kid.palette.core, (int(root[0]), int(root[1])), max(2, int(r * 0.20)))
    elif kid.hair_style in (1, 3):
        # A back tuft that whips with the same spring.
        tip = (root[0] - f * r * 0.85 + swing * 0.8, root[1] + r * 0.28 + droop * 0.7)
        _capsule(dst, root, tip, r * 0.55, r * 0.18, hair_dark)

    if kid.accessory != 1:
        # Headband tails: two thin ribbons, offset in phase so they don't
        # move as a single rigid unit.
        band_root = _rotp(hx - f * r * 0.92, hy - r * 0.48, hx, hy, ang)
        for i, (l_mul, w_mul) in enumerate(((1.0, 1.0), (0.78, 0.78))):
            s2 = swing * (1.0 + 0.22 * i)
            tip = (band_root[0] - f * r * 0.95 * l_mul + s2,
                   band_root[1] + r * (0.55 + 0.22 * i) * l_mul + droop * 0.8)
            mid = ((band_root[0] + tip[0]) * 0.5 - f * r * 0.08,
                   (band_root[1] + tip[1]) * 0.5 - r * 0.10)
            pygame.draw.lines(dst, kid.band, False, [band_root, mid, tip],
                              max(2, int(r * 0.16 * w_mul)))


def _draw_sweat(dst: pygame.Surface, kid: Kid, hx: float, hy: float, r: float,
                f: int, at: float, amount: float) -> None:
    """Beads flung off the brow while straining."""
    for i in range(3):
        u = (at * (1.1 + 0.28 * i) + kid.phase * 0.4 + i * 0.37) % 1.0
        if u > amount + 0.25:
            continue
        k = u
        px = hx + f * r * (0.55 + 0.75 * k) + i * r * 0.10
        py = hy - r * (0.35 - 1.15 * k) + k * k * r * 1.1
        rad = max(1, int(r * 0.14 * (1.0 - k * 0.4)))
        a = int(230 * (1.0 - k))
        drop = pygame.Surface((rad * 4, rad * 5), pygame.SRCALPHA)
        pygame.draw.ellipse(drop, (196, 236, 255, a), (rad, rad, rad * 2, rad * 3))
        pygame.draw.ellipse(drop, (255, 255, 255, a), (int(rad * 1.4), int(rad * 1.5), rad, rad))
        dst.blit(drop, drop.get_rect(center=(int(px), int(py))))


# --------------------------------------------------------------------------- #
# Team draw
# --------------------------------------------------------------------------- #

def draw_team(surf, kids: List[Kid], t: float, pull_intensity: float,
              strain: float, rope_y: float, shift: float = 0.0,
              mode: str = "play") -> List[Tuple[int, int]]:
    """Draw a whole squad back-to-front and return every grip point.

    `shift` slides the entire team along x as the rope moves. Kids further
    up the screen (smaller y) are further from camera, so they paint first
    and get overlapped by the front row.
    """
    grips: List[Tuple[int, int]] = []
    for kid in sorted(kids, key=lambda k: k.y):
        if shift:
            kid.x += shift
            try:
                grips.append(draw_kid(surf, kid, t, pull_intensity, strain, rope_y, mode))
            finally:
                kid.x -= shift
        else:
            grips.append(draw_kid(surf, kid, t, pull_intensity, strain, rope_y, mode))
    return grips

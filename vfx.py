"""Transient visual effects, camera juice and screen-space impact for RopeRush.

This is the "game feel" layer. Nothing here is gameplay - it is the spray of
sparks when an answer lands, the punch of the camera when the rope snaps, the
slow-motion beat before a knockout. It exists to make a correct answer feel
*expensive*.

Architecture
------------
One `VFX` instance owns everything. Effects are drawn in three passes so that
additive glow composites on top of the scene instead of being occluded by it:

    backdrop.draw()
    vfx.draw_below(surf)      # dust, ground scuffs - behind the characters
    ... characters, rope ...
    vfx.draw_above(surf)      # confetti, floating text - in front
    vfx.draw_additive(surf)   # every glowing thing, BLEND_RGB_ADD
    ... HUD ...
    vfx.draw_fullscreen(surf) # flashes, chromatic aberration, edge bloom

Performance discipline
----------------------
* Particles live in a **fixed pool** (`MAX_PARTICLES`). Spawning never
  allocates; when the pool is full the oldest slot is recycled round-robin.
* **No surface is created per particle per frame.** Every particle sprite is
  baked once into `_SpriteBank`, keyed by (kind, size bucket, colour,
  brightness step, rotation step). Additive fades are done by *quantising the
  baked colour* rather than by `set_alpha`, because `BLEND_RGB_ADD` ignores a
  surface's alpha entirely.
* The additive pass renders into a single persistent scratch surface and is
  composited with a **dirty rectangle**, so a quiet frame costs almost nothing.

Time
----
`update()` wants the *real* frame delta. Particles internally advance by
`dt * get_time_scale()` so they slow down with the game during a `slowmo`,
while camera shake, flashes and the slowmo envelope itself always run on
wall-clock time.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

import render_utils as R
import theme as T

Color = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #

MAX_PARTICLES = 1200

# Size buckets every particle snaps to. Fine enough that shrinking reads as
# smooth, coarse enough that the sprite bank stays small.
_SIZES: Tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 26, 32, 40, 52)

# Additive particles fade by stepping down their baked colour.
_BRIGHT_STEPS = 8

# Rotation tables. Streaks and confetti are 180-symmetric, stars are 90-.
_STREAK_ANGLES = 24
_SQUARE_ANGLES = 12
_SQUARE_SQUASH = 5
_STAR_ANGLES = 8

# Aspect ratios for the three velocity-stretch tiers.
_STREAK_ASPECT = {"streak_s": 1.9, "streak_m": 3.2, "streak_l": 5.4}
_STREAK_KINDS = ("streak_s", "streak_m", "streak_l")

# Speed thresholds (px/s) at which a dot promotes to each streak tier.
_STRETCH_MIN = 110.0
_STRETCH_T1 = 300.0
_STRETCH_T2 = 620.0

# Layers for non-additive particles.
LAYER_BELOW = 0
LAYER_ABOVE = 1

# Fade curves.
_FADE_OUT = 0        # full, then falls away
_FADE_INOUT = 1      # blooms in and out - sparkles, dust
_FADE_LATE = 2       # holds bright, drops hard at the end - confetti

CONFETTI_COLORS: Tuple[Color, ...] = (
    T.GOLD, T.GOLD_LIGHT, T.EMBER.core, T.EMBER.light,
    T.FROST.core, T.FROST.light, T.OK, T.INK,
)


# --------------------------------------------------------------------------- #
# Smooth value noise - camera shake reads as an impact, not as TV static
# --------------------------------------------------------------------------- #

def _hash01(i: int, seed: int) -> float:
    """Deterministic 0..1 hash of an integer lattice point."""
    n = (i * 374761393 + seed * 668265263) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def _vnoise(t: float, seed: int) -> float:
    """1-D value noise in -1..1 with smoothstep interpolation."""
    i = math.floor(t)
    f = t - i
    a = _hash01(int(i), seed)
    b = _hash01(int(i) + 1, seed)
    u = f * f * (3.0 - 2.0 * f)
    return (a + (b - a) * u) * 2.0 - 1.0


def _shake_noise(t: float, seed: int) -> float:
    """Two octaves - a heavy low wobble with a fine rattle on top."""
    return _vnoise(t, seed) * 0.72 + _vnoise(t * 2.7 + 31.5, seed + 91) * 0.28


def _bucket(size: float) -> int:
    """Snap a size to the nearest baked bucket."""
    if size <= _SIZES[0]:
        return _SIZES[0]
    if size >= _SIZES[-1]:
        return _SIZES[-1]
    lo = _SIZES[0]
    for s in _SIZES:
        if s >= size:
            return s if (s - size) < (size - lo) else lo
        lo = s
    return _SIZES[-1]


# --------------------------------------------------------------------------- #
# Sprite bank - every particle image the game will ever need, baked once
# --------------------------------------------------------------------------- #

class _SpriteBank:
    """Lazily bakes and caches particle sprites.

    Keys are fully quantised, so after a few seconds of play the cache is warm
    and `get()` is a plain dict lookup. Never returns a surface the caller may
    mutate destructively - `set_alpha` on the way out is fine (it is a cheap
    per-surface flag) but nothing may draw onto these.
    """

    _MAX_ENTRIES = 6000

    def __init__(self) -> None:
        self._cache: Dict[tuple, pygame.Surface] = {}

    # -- public -------------------------------------------------------- #

    def get(
        self,
        kind: str,
        size: int,
        color: Color,
        additive: bool,
        bright: int = _BRIGHT_STEPS - 1,
        angle_idx: int = 0,
        squash_idx: int = 0,
    ) -> pygame.Surface:
        key = (kind, size, color, additive, bright if additive else -1,
               angle_idx, squash_idx)
        got = self._cache.get(key)
        if got is not None:
            return got

        if len(self._cache) > self._MAX_ENTRIES:      # pathological colour churn
            self._cache.clear()

        col = color
        if additive:
            f = (bright + 1) / _BRIGHT_STEPS
            col = (int(color[0] * f), int(color[1] * f), int(color[2] * f))

        if kind == "dot":
            surf = _bake_dot(size, col, additive)
        elif kind in _STREAK_ASPECT:
            surf = _bake_streak(size, col, additive, _STREAK_ASPECT[kind])
            surf = _rotate(surf, angle_idx, _STREAK_ANGLES, 180.0)
        elif kind == "square":
            surf = _bake_square(size, col, squash_idx)
            surf = _rotate(surf, angle_idx, _SQUARE_ANGLES, 180.0)
        elif kind == "star":
            surf = _bake_star(size, col, additive)
            surf = _rotate(surf, angle_idx, _STAR_ANGLES, 90.0)
        else:                                          # pragma: no cover
            surf = _bake_dot(size, col, additive)

        self._cache[key] = surf
        return surf

    def warm(self, colors: Sequence[Color]) -> None:
        """Pre-bake the hot path so the first burst of the game never hitches."""
        for c in colors:
            for s in (3, 4, 5, 6, 8, 10, 13, 16):
                for b in range(_BRIGHT_STEPS):
                    self.get("dot", s, c, True, b)
            for s in (4, 5, 6, 8, 10):
                for b in (7, 5, 3, 1):
                    for a in range(0, _STREAK_ANGLES, 2):
                        self.get("streak_m", s, c, True, b, a)

    def stats(self) -> int:
        return len(self._cache)


def _rotate(surf: pygame.Surface, idx: int, count: int, span: float) -> pygame.Surface:
    """Rotate a baked sprite to table slot `idx` (screen-space, y-down)."""
    if idx == 0:
        return surf
    deg = (idx / count) * span
    # pygame rotates counter-clockwise in a y-up frame; screen y grows down.
    return pygame.transform.rotate(surf, -deg)


# -- bakers ------------------------------------------------------------- #
#
# Additive sprites bake their falloff into RGB (BLEND_RGB_ADD ignores alpha).
# Normal-blend sprites bake it into alpha and keep RGB flat, so the caller can
# fade them with a free `set_alpha`.

def _bake_dot(size: int, color: Color, additive: bool) -> pygame.Surface:
    """A soft round blob with a hot core."""
    src = 36
    s = pygame.Surface((src, src), pygame.SRCALPHA)
    r = src // 2
    for i in range(r, 0, -1):
        t = i / r
        f = (1.0 - t) ** 1.55
        if additive:
            c = (int(color[0] * f), int(color[1] * f), int(color[2] * f), 255)
        else:
            c = (color[0], color[1], color[2], int(255 * min(1.0, f * 1.25)))
        pygame.draw.circle(s, c, (r, r), i)
    # Blown-out centre sells "this is emitting light".
    if additive:
        hot = T.lerp_color(color, (255, 255, 255), 0.45)
        pygame.draw.circle(s, (hot[0], hot[1], hot[2], 255), (r, r), max(1, r // 4))
    d = max(2, int(size))
    return pygame.transform.smoothscale(s, (d, d))


def _bake_streak(size: int, color: Color, additive: bool, aspect: float) -> pygame.Surface:
    """An elongated blob - a particle smeared along its own velocity."""
    sh = 32
    sw = int(sh * aspect)
    s = pygame.Surface((sw, sh), pygame.SRCALPHA)
    steps = 16
    for i in range(steps, 0, -1):
        t = i / steps
        f = (1.0 - t) ** 1.35
        w = max(1, int(sw * t))
        h = max(1, int(sh * t))
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (sw // 2, sh // 2)
        if additive:
            c = (int(color[0] * f), int(color[1] * f), int(color[2] * f), 255)
        else:
            c = (color[0], color[1], color[2], int(255 * min(1.0, f * 1.3)))
        pygame.draw.ellipse(s, c, rect)
    h = max(2, int(size))
    w = max(3, int(size * aspect))
    return pygame.transform.smoothscale(s, (w, h))


def _bake_square(size: int, color: Color, squash_idx: int) -> pygame.Surface:
    """A confetti chip: a flat rectangle, vertically squashed to fake tumbling.

    The squash index walks the chip through an out-of-plane rotation - at index
    4 it is nearly edge-on, and the shaded lower half keeps it reading as a
    solid object rather than a flickering line.
    """
    sq = (1.0, 0.80, 0.56, 0.30, 0.11)[max(0, min(4, squash_idx))]
    w = max(2, int(size))
    h = max(1, int(size * 0.62 * sq))
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    # Edge-on chips catch the light; face-on chips show their own colour.
    lit = T.lerp_color(color, (255, 255, 255), 0.30 + 0.35 * (1.0 - sq))
    dim = T.lerp_color(color, (0, 0, 0), 0.34)
    top_h = max(1, h // 2)
    pygame.draw.rect(s, (*lit, 255), (0, 0, w, top_h))
    pygame.draw.rect(s, (*dim, 255), (0, top_h, w, h - top_h))
    return s


def _bake_star(size: int, color: Color, additive: bool) -> pygame.Surface:
    """A four-point sparkle - two tapered spikes crossed over a hot core."""
    src = 48
    s = pygame.Surface((src, src), pygame.SRCALPHA)
    c = src // 2
    arm = c - 1
    for i in range(6, 0, -1):
        t = i / 6.0
        f = (1.0 - t) ** 1.2
        if additive:
            col = (int(color[0] * f), int(color[1] * f), int(color[2] * f), 255)
        else:
            col = (color[0], color[1], color[2], int(255 * f))
        thick = max(1, int(arm * 0.30 * t))
        a = int(arm * (0.35 + 0.65 * (1.0 - t) ** 0.4))
        pygame.draw.polygon(s, col, [(c - a, c), (c, c - thick), (c + a, c), (c, c + thick)])
        pygame.draw.polygon(s, col, [(c, c - a), (c - thick, c), (c, c + a), (c + thick, c)])
    core = T.lerp_color(color, (255, 255, 255), 0.6)
    pygame.draw.circle(s, (*core, 255), (c, c), max(1, src // 12))
    d = max(3, int(size * 2.2))          # stars read bigger than their nominal size
    return pygame.transform.smoothscale(s, (d, d))


# --------------------------------------------------------------------------- #
# Particle
# --------------------------------------------------------------------------- #

class _P:
    """One pooled particle. Plain slots - dataclasses are too slow here."""

    __slots__ = (
        "alive", "kind", "x", "y", "vx", "vy", "grav", "drag",
        "t", "life", "s0", "s1", "color", "additive", "layer",
        "rot", "spin", "fade", "stretch", "sway", "sway_f", "seed", "born",
    )

    def __init__(self) -> None:
        self.alive = False
        self.kind = "dot"
        self.x = self.y = 0.0
        self.vx = self.vy = 0.0
        self.grav = 0.0
        self.drag = 0.0
        self.t = 0.0
        self.life = 1.0
        self.s0 = self.s1 = 4.0
        self.color: Color = (255, 255, 255)
        self.additive = True
        self.layer = LAYER_ABOVE
        self.rot = 0.0
        self.spin = 0.0
        self.fade = _FADE_OUT
        self.stretch = True
        self.sway = 0.0
        self.sway_f = 0.0
        self.seed = 0
        self.born = 0.0


# --------------------------------------------------------------------------- #
# Non-particle effects
# --------------------------------------------------------------------------- #

class _Shockwave:
    __slots__ = ("x", "y", "color", "r_max", "t", "life", "width")

    def __init__(self, x, y, color, r_max, life, width):
        self.x, self.y, self.color = x, y, color
        self.r_max, self.life, self.width = r_max, life, width
        self.t = 0.0


class _ImpactLines:
    __slots__ = ("x", "y", "color", "count", "length", "t", "life", "seed")

    def __init__(self, x, y, color, count, length, seed):
        self.x, self.y, self.color = x, y, color
        self.count, self.length = count, length
        self.t, self.life, self.seed = 0.0, 0.34, seed


class _Beam:
    __slots__ = ("x0", "y0", "x1", "y1", "color", "t", "life", "width")

    def __init__(self, x0, y0, x1, y1, color, life, width):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.color, self.life, self.width = color, life, width
        self.t = 0.0


class _FloatText:
    __slots__ = ("x", "y", "surf", "rise", "t", "life")

    def __init__(self, x, y, surf, rise, life):
        self.x, self.y, self.surf = x, y, surf
        self.rise, self.life = rise, life
        self.t = 0.0


class _Shake:
    __slots__ = ("mag", "life", "t", "seed")

    def __init__(self, mag, life, seed):
        self.mag, self.life, self.seed = mag, life, seed
        self.t = 0.0


class _Kick:
    __slots__ = ("dx", "dy", "life", "t")

    def __init__(self, dx, dy, life):
        self.dx, self.dy, self.life = dx, dy, life
        self.t = 0.0


class _Zoom:
    __slots__ = ("amount", "life", "t")

    def __init__(self, amount, life):
        self.amount, self.life = amount, life
        self.t = 0.0


class _Flash:
    __slots__ = ("color", "strength", "life", "t")

    def __init__(self, color, strength, life):
        self.color, self.strength, self.life = color, strength, life
        self.t = 0.0


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #

class VFX:
    """Owns every transient effect on screen.

    Create one, feed it `update(real_dt)` once per frame, and call the three
    draw passes at the right points in the render order.
    """

    def __init__(self, width: int = T.LOGICAL_W, height: int = T.LOGICAL_H) -> None:
        self.w = int(width)
        self.h = int(height)
        self._rng = random.Random(20240517)
        self._time = 0.0

        # --- particle pool ---
        self._pool: List[_P] = [_P() for _ in range(MAX_PARTICLES)]
        self._free: List[int] = list(range(MAX_PARTICLES - 1, -1, -1))
        self._active: List[int] = []
        self._steal = 0

        # --- effect lists ---
        self._waves: List[_Shockwave] = []
        self._ilines: List[_ImpactLines] = []
        self._beams: List[_Beam] = []
        self._texts: List[_FloatText] = []

        # --- camera ---
        self._shakes: List[_Shake] = []
        self._kicks: List[_Kick] = []
        self._zooms: List[_Zoom] = []

        # --- full screen ---
        self._flashes: List[_Flash] = []
        self._chroma_t = 0.0
        self._chroma_life = 0.0
        self._chroma_strength = 0.0
        self._slow_t = 0.0
        self._slow_life = 0.0
        self._slow_scale = 1.0
        self._time_scale = 1.0

        # --- surfaces ---
        self.bank = _SpriteBank()
        self._add_surf = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        self._add_dirty: Optional[pygame.Rect] = None
        self._chroma_a: Optional[pygame.Surface] = None
        self._chroma_b: Optional[pygame.Surface] = None
        self._edge_cache: Dict[tuple, pygame.Surface] = {}

        # Smear continuity for `trail()`.
        self._last_trail: Dict[tuple, Tuple[float, float, float]] = {}

        self.bank.warm((T.EMBER.glow, T.FROST.glow, T.GOLD, T.GOLD_LIGHT, T.INK))

    # ------------------------------------------------------------------ #
    # Pool
    # ------------------------------------------------------------------ #

    def _alloc(self) -> _P:
        if self._free:
            idx = self._free.pop()
        else:
            # Pool exhausted: recycle round-robin so we never scan 1200 slots.
            idx = self._active[self._steal % len(self._active)]
            self._steal += 1
            self._active.remove(idx)
        p = self._pool[idx]
        p.alive = True
        p.t = 0.0
        p.born = self._time
        self._active.append(idx)
        return p

    @property
    def particle_count(self) -> int:
        return len(self._active)

    def clear(self) -> None:
        """Drop every live effect. Camera and time scale reset to neutral."""
        for idx in self._active:
            self._pool[idx].alive = False
        self._active.clear()
        self._free = list(range(MAX_PARTICLES - 1, -1, -1))
        self._waves.clear()
        self._ilines.clear()
        self._beams.clear()
        self._texts.clear()
        self._shakes.clear()
        self._kicks.clear()
        self._zooms.clear()
        self._flashes.clear()
        self._chroma_life = 0.0
        self._slow_life = 0.0
        self._time_scale = 1.0
        self._last_trail.clear()
        if self._add_dirty:
            self._add_surf.fill((0, 0, 0, 0), self._add_dirty)
            self._add_dirty = None

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #

    def update(self, dt: float) -> None:
        """Advance everything. `dt` is the REAL frame delta, unscaled."""
        dt = max(0.0, min(0.05, float(dt)))
        self._time += dt

        # Slow-motion envelope runs on wall clock, then scales particle time.
        self._update_slowmo(dt)
        sdt = dt * self._time_scale

        self._update_particles(sdt)

        for lst in (self._waves, self._ilines, self._beams, self._texts):
            i = 0
            while i < len(lst):
                e = lst[i]
                e.t += sdt
                if e.t >= e.life:
                    lst[i] = lst[-1]
                    lst.pop()
                else:
                    i += 1

        for lst in (self._shakes, self._kicks, self._zooms, self._flashes):
            i = 0
            while i < len(lst):
                e = lst[i]
                e.t += dt
                if e.t >= e.life:
                    lst[i] = lst[-1]
                    lst.pop()
                else:
                    i += 1

        if self._chroma_life > 0.0:
            self._chroma_t += dt
            if self._chroma_t >= self._chroma_life:
                self._chroma_life = 0.0

    def _update_particles(self, dt: float) -> None:
        if dt <= 0.0:
            return
        pool = self._pool
        active = self._active
        i = 0
        while i < len(active):
            p = pool[active[i]]
            p.t += dt
            if p.t >= p.life:
                p.alive = False
                self._free.append(active[i])
                active[i] = active[-1]
                active.pop()
                continue
            # Semi-implicit Euler with exponential-ish drag.
            if p.grav:
                p.vy += p.grav * dt
            if p.drag:
                d = 1.0 - p.drag * dt
                if d < 0.0:
                    d = 0.0
                p.vx *= d
                p.vy *= d
            if p.sway:
                # Confetti and dust slide sideways as they fall.
                p.x += p.sway * math.sin((p.t + p.seed) * p.sway_f) * dt
            p.x += p.vx * dt
            p.y += p.vy * dt
            if p.spin:
                p.rot += p.spin * dt
            i += 1

    def _update_slowmo(self, dt: float) -> None:
        if self._slow_life <= 0.0:
            self._time_scale = 1.0
            return
        self._slow_t += dt
        if self._slow_t >= self._slow_life:
            self._slow_life = 0.0
            self._time_scale = 1.0
            return
        u = self._slow_t / self._slow_life
        # Ease in over the first 18%, hold, ease back out over the last 40%.
        if u < 0.18:
            env = T.ease_in_out_cubic(u / 0.18)
        elif u > 0.60:
            env = 1.0 - T.ease_in_out_cubic((u - 0.60) / 0.40)
        else:
            env = 1.0
        self._time_scale = T.lerp(1.0, self._slow_scale, env)

    # ------------------------------------------------------------------ #
    # Emitters
    # ------------------------------------------------------------------ #

    def burst(
        self,
        x: float,
        y: float,
        color: Color,
        count: int = 24,
        speed: float = 260.0,
        spread: float = math.tau,
        angle: float = 0.0,
        size: Tuple[float, float] = (3, 7),
        life: Tuple[float, float] = (0.4, 0.9),
        gravity: float = 520.0,
        additive: bool = True,
    ) -> None:
        """A cone (or full circle) of sparks flung from a point.

        Fast particles automatically render as velocity-aligned streaks, so a
        high `speed` reads as a whip-crack rather than a cloud of dots.
        """
        rng = self._rng
        half = spread * 0.5
        for _ in range(int(count)):
            a = angle + rng.uniform(-half, half)
            # Bias toward the fast end so the silhouette has long leaders.
            sp = speed * (0.35 + 0.65 * rng.random() ** 0.55)
            p = self._alloc()
            p.kind = "dot"
            p.x, p.y = x, y
            p.vx = math.cos(a) * sp
            p.vy = math.sin(a) * sp
            p.grav = gravity
            p.drag = 1.9
            p.life = rng.uniform(life[0], life[1])
            p.s0 = rng.uniform(size[0], size[1])
            p.s1 = p.s0 * 0.18
            p.color = color
            p.additive = additive
            p.layer = LAYER_ABOVE
            p.rot = 0.0
            p.spin = 0.0
            p.fade = _FADE_OUT
            p.stretch = True
            p.sway = 0.0
            p.seed = rng.randrange(1000)

    def sparkle(self, x: float, y: float, color: Color, count: int = 12) -> None:
        """Slow, twinkling four-point stars - the 'magic' garnish on a reward."""
        rng = self._rng
        for _ in range(int(count)):
            a = rng.uniform(0.0, math.tau)
            sp = rng.uniform(18.0, 130.0)
            p = self._alloc()
            p.kind = "star"
            p.x = x + math.cos(a) * rng.uniform(0.0, 26.0)
            p.y = y + math.sin(a) * rng.uniform(0.0, 26.0)
            p.vx = math.cos(a) * sp
            p.vy = math.sin(a) * sp - rng.uniform(10.0, 60.0)
            p.grav = -20.0                      # drift upward like embers
            p.drag = 2.6
            p.life = rng.uniform(0.45, 1.05)
            p.s0 = rng.uniform(2.5, 6.0)
            p.s1 = p.s0 * 0.35
            p.color = color
            p.additive = True
            p.layer = LAYER_ABOVE
            p.rot = rng.uniform(0.0, math.pi)
            p.spin = rng.uniform(-4.0, 4.0)
            p.fade = _FADE_INOUT
            p.stretch = False
            p.sway = 0.0
            p.seed = rng.randrange(1000)

    def confetti(
        self,
        x: float,
        y: float,
        count: int = 90,
        colors: Optional[Sequence[Color]] = None,
    ) -> None:
        """A victory shower of tumbling paper chips.

        Each chip carries its own out-of-plane spin, so it flashes between
        face-on and edge-on as it falls - that flicker is what makes flat
        rectangles read as 3-D.
        """
        rng = self._rng
        pal = tuple(colors) if colors else CONFETTI_COLORS
        for _ in range(int(count)):
            a = rng.uniform(-math.pi * 0.92, -math.pi * 0.08)
            sp = rng.uniform(180.0, 620.0)
            p = self._alloc()
            p.kind = "square"
            p.x = x + rng.uniform(-18.0, 18.0)
            p.y = y + rng.uniform(-12.0, 12.0)
            p.vx = math.cos(a) * sp
            p.vy = math.sin(a) * sp
            p.grav = rng.uniform(420.0, 640.0)
            p.drag = rng.uniform(1.1, 2.0)
            p.life = rng.uniform(1.6, 3.1)
            p.s0 = rng.uniform(9.0, 17.0)
            p.s1 = p.s0
            p.color = rng.choice(pal)
            p.additive = False
            p.layer = LAYER_ABOVE
            p.rot = rng.uniform(0.0, math.tau)
            p.spin = rng.uniform(-11.0, 11.0)
            p.fade = _FADE_LATE
            p.stretch = False
            p.sway = rng.uniform(40.0, 130.0) * rng.choice((-1.0, 1.0))
            p.sway_f = rng.uniform(3.0, 7.0)
            p.seed = rng.randrange(1000)

    def dust(self, x: float, y: float, direction: int = 1, count: int = 14) -> None:
        """Ground scuff kicked sideways - drawn *behind* the characters."""
        rng = self._rng
        d = 1 if direction >= 0 else -1
        for _ in range(int(count)):
            a = rng.uniform(-0.95, -0.12) * 1.0
            sp = rng.uniform(50.0, 210.0)
            p = self._alloc()
            p.kind = "dot"
            p.x = x + rng.uniform(-10.0, 10.0)
            p.y = y + rng.uniform(-5.0, 4.0)
            p.vx = math.cos(a) * sp * d
            p.vy = math.sin(a) * sp * 0.55
            p.grav = 110.0
            p.drag = 2.9
            p.life = rng.uniform(0.45, 1.0)
            p.s0 = rng.uniform(7.0, 18.0)
            p.s1 = p.s0 * 2.4                 # dust puffs *expand* as they die
            p.color = T.lerp_color(T.GROUND_LINE, T.INK_DIM, rng.random() * 0.5)
            p.additive = False
            p.layer = LAYER_BELOW
            p.rot = 0.0
            p.spin = 0.0
            p.fade = _FADE_INOUT
            p.stretch = False
            p.sway = 0.0
            p.seed = rng.randrange(1000)

    def trail(self, x: float, y: float, color: Color, size: float = 6.0,
              life: float = 0.35) -> None:
        """A short additive smear.

        Consecutive calls with the same colour are joined: the particle is
        stretched along the gap since the last call, so a moving emitter draws
        a continuous ribbon instead of a dotted line.
        """
        key = (color, int(size))
        prev = self._last_trail.get(key)
        vx = vy = 0.0
        if prev is not None:
            px, py, pt = prev
            gap = self._time - pt
            if 0.0 < gap < 0.12:
                vx = (x - px) / gap
                vy = (y - py) / gap
        self._last_trail[key] = (x, y, self._time)

        p = self._alloc()
        p.kind = "dot"
        p.x, p.y = x, y
        # Keep a fraction of the emitter velocity: enough to orient the streak,
        # little enough that the smear stays put.
        p.vx, p.vy = vx * 0.22, vy * 0.22
        p.grav = 0.0
        p.drag = 3.2
        p.life = max(0.05, life)
        p.s0 = size
        p.s1 = size * 0.15
        p.color = color
        p.additive = True
        p.layer = LAYER_ABOVE
        p.rot = 0.0
        p.spin = 0.0
        p.fade = _FADE_OUT
        p.stretch = True
        p.sway = 0.0
        p.seed = 0

    def shockwave(self, x: float, y: float, color: Color, max_radius: float = 180.0,
                  life: float = 0.45, width: float = 6.0) -> None:
        """An expanding ring that thins and fades. Fire one on every hit."""
        self._waves.append(_Shockwave(x, y, color, max_radius, max(0.05, life), width))

    def impact_lines(self, x: float, y: float, color: Color, count: int = 8,
                     length: float = 70.0) -> None:
        """Radial speed lines punched outward from a point of contact."""
        self._ilines.append(_ImpactLines(x, y, color, int(count), length,
                                         self._rng.randrange(10000)))

    def beam(self, x0: float, y0: float, x1: float, y1: float, color: Color,
             life: float = 0.25, width: float = 10.0) -> None:
        """A bright additive bolt between two points that thins as it fades."""
        self._beams.append(_Beam(x0, y0, x1, y1, color, max(0.05, life), width))

    def floating_text(
        self,
        x: float,
        y: float,
        text: str,
        color: Color,
        size: int = T.T_HEAD,
        rise: float = 70.0,
        life: float = 0.9,
        outline: Optional[Color] = None,
    ) -> None:
        """Damage-number style text that pops, rises and fades."""
        font = R.font_display(int(size), bold=True)
        # `.copy()` because we fade these with set_alpha and render_utils hands
        # back a shared cached surface that other UI code may also be blitting.
        surf = R.text_surface(
            str(text), font, color,
            shadow_color=(0, 0, 0, 170), shadow_offset=(0, 3),
            outline=outline if outline is not None else (10, 12, 30),
            outline_w=3 if size >= T.T_BODY else 2,
        ).copy()
        self._texts.append(_FloatText(x, y, surf, rise, max(0.1, life)))

    # ------------------------------------------------------------------ #
    # Camera
    # ------------------------------------------------------------------ #

    def shake(self, magnitude: float, duration: float = 0.25) -> None:
        """Decaying smooth-noise camera shake. Stacks with other shakes."""
        if magnitude <= 0.0 or duration <= 0.0:
            return
        self._shakes.append(_Shake(float(magnitude), float(duration),
                                   self._rng.randrange(10000)))

    def kick(self, dx: float, dy: float, duration: float = 0.2) -> None:
        """A directional punch that springs back - use it for one-sided hits."""
        if duration <= 0.0:
            return
        self._kicks.append(_Kick(float(dx), float(dy), float(duration)))

    def zoom_punch(self, amount: float = 0.03, duration: float = 0.2) -> None:
        """A quick push-in that settles back to 1.0."""
        if duration <= 0.0:
            return
        self._zooms.append(_Zoom(float(amount), float(duration)))

    def get_camera(self) -> Tuple[float, float, float]:
        """Current (offset_x, offset_y, zoom). Apply before drawing the scene."""
        ox = oy = 0.0
        for s in self._shakes:
            u = s.t / s.life
            amp = s.mag * (1.0 - u) ** 2.0
            f = s.t * 34.0
            ox += _shake_noise(f, s.seed) * amp
            oy += _shake_noise(f, s.seed + 777) * amp * 0.78
        for k in self._kicks:
            u = k.t / k.life
            # Damped oscillation: full offset at t=0, overshoots back once.
            d = math.exp(-5.0 * u) * math.cos(u * math.pi * 2.1)
            ox += k.dx * d
            oy += k.dy * d
        zoom = 1.0
        for z in self._zooms:
            u = z.t / z.life
            zoom += z.amount * math.exp(-5.0 * u) * math.cos(u * math.pi * 1.9)
        return ox, oy, zoom

    # ------------------------------------------------------------------ #
    # Full-screen
    # ------------------------------------------------------------------ #

    def flash(self, color: Color, strength: float = 0.5, duration: float = 0.25) -> None:
        """Blow the whole frame out toward `color`, then fall off."""
        if duration <= 0.0 or strength <= 0.0:
            return
        self._flashes.append(_Flash(color, float(strength), float(duration)))

    def chromatic_pulse(self, strength: float = 1.0, duration: float = 0.3) -> None:
        """Split the red and blue channels apart briefly. Use sparingly."""
        self._chroma_strength = max(self._chroma_strength if self._chroma_life > 0 else 0.0,
                                    float(strength))
        self._chroma_t = 0.0
        self._chroma_life = max(0.05, float(duration))

    def slowmo(self, scale: float = 0.35, duration: float = 0.6) -> None:
        """Ease the world down to `scale` speed and back up again."""
        self._slow_scale = max(0.02, min(1.0, float(scale)))
        self._slow_t = 0.0
        self._slow_life = max(0.1, float(duration))

    def get_time_scale(self) -> float:
        """Multiply gameplay dt by this. 1.0 when no slowmo is running."""
        return self._time_scale

    # ------------------------------------------------------------------ #
    # Draw passes
    # ------------------------------------------------------------------ #

    def draw_below(self, surf: pygame.Surface) -> None:
        """Non-additive particles that belong behind the characters."""
        self._draw_particles(surf, LAYER_BELOW)

    def draw_above(self, surf: pygame.Surface) -> None:
        """Non-additive particles and floating text, in front of everything."""
        self._draw_particles(surf, LAYER_ABOVE)
        self._draw_texts(surf)

    def _draw_particles(self, surf: pygame.Surface, layer: int) -> None:
        pool = self._pool
        bank = self.bank
        blit = surf.blit
        for idx in self._active:
            p = pool[idx]
            if p.additive or p.layer != layer:
                continue
            a = _fade(p)
            if a <= 0.01:
                continue
            size = p.s0 + (p.s1 - p.s0) * (p.t / p.life)
            if size < 1.0:
                continue
            if p.kind == "square":
                # Tumble: |cos(rot)| drives the out-of-plane squash, the sign
                # flip is what makes a chip look like it flipped over.
                sq = abs(math.cos(p.rot))
                si = min(4, int((1.0 - sq) * 5.0))
                ai = int(p.rot / math.pi * _SQUARE_ANGLES) % _SQUARE_ANGLES
                spr = bank.get("square", _bucket(size), p.color, False,
                               angle_idx=ai, squash_idx=si)
            else:
                spr = bank.get("dot", _bucket(size), p.color, False)
            spr.set_alpha(int(a * 255))
            w, h = spr.get_size()
            blit(spr, (int(p.x - w * 0.5), int(p.y - h * 0.5)))

    def _draw_texts(self, surf: pygame.Surface) -> None:
        for ft in self._texts:
            u = ft.t / ft.life
            y = ft.y - ft.rise * T.ease_out_quint(min(1.0, u * 1.35))
            a = 1.0 if u < 0.55 else 1.0 - (u - 0.55) / 0.45
            s = ft.surf
            # Pop-in overshoot for the first beat only - keeps the scaling
            # allocation off the steady-state path.
            if u < 0.22:
                k = 0.55 + 0.45 * T.ease_out_back(u / 0.22, 2.6)
                w, h = s.get_size()
                s = pygame.transform.smoothscale(s, (max(1, int(w * k)), max(1, int(h * k))))
            s.set_alpha(int(max(0.0, min(1.0, a)) * 255))
            r = s.get_rect(center=(int(ft.x), int(y)))
            surf.blit(s, r)

    def draw_additive(self, surf: pygame.Surface) -> None:
        """Every glowing effect, composited with BLEND_RGB_ADD in one blit."""
        scratch = self._add_surf
        if self._add_dirty is not None:
            scratch.fill((0, 0, 0, 0), self._add_dirty)
            self._add_dirty = None

        dirty: Optional[pygame.Rect] = None

        def mark(r: Optional[pygame.Rect]) -> None:
            nonlocal dirty
            if r is None:
                return
            dirty = r.copy() if dirty is None else dirty.union(r)

        # Rings / lines / beams first: pygame.draw overwrites rather than adds,
        # so they must not sit on top of accumulated particle glow.
        for wv in self._waves:
            mark(self._draw_shockwave(scratch, wv))
        for il in self._ilines:
            mark(self._draw_impact_lines(scratch, il))
        for bm in self._beams:
            mark(self._draw_beam(scratch, bm))

        pool = self._pool
        bank = self.bank
        blit = scratch.blit
        add = pygame.BLEND_RGB_ADD
        for idx in self._active:
            p = pool[idx]
            if not p.additive:
                continue
            a = _fade(p)
            if a <= 0.02:
                continue
            bi = int(a * (_BRIGHT_STEPS - 1) + 0.5)
            if bi <= 0:
                continue
            size = p.s0 + (p.s1 - p.s0) * (p.t / p.life)
            if size < 1.0:
                continue

            kind = "dot"
            ai = 0
            if p.stretch:
                sp2 = p.vx * p.vx + p.vy * p.vy
                if sp2 > _STRETCH_MIN * _STRETCH_MIN:
                    sp = math.sqrt(sp2)
                    tier = 0 if sp < _STRETCH_T1 else (1 if sp < _STRETCH_T2 else 2)
                    kind = _STREAK_KINDS[tier]
                    size *= (0.92, 0.80, 0.68)[tier]   # keep visual mass even
                    deg = math.degrees(math.atan2(p.vy, p.vx)) % 180.0
                    ai = int(deg / (180.0 / _STREAK_ANGLES) + 0.5) % _STREAK_ANGLES
            elif p.kind == "star":
                ai = int(math.degrees(p.rot) / (90.0 / _STAR_ANGLES) + 0.5) % _STAR_ANGLES
                kind = "star"

            if size < 1.0:
                continue
            spr = bank.get(kind, _bucket(size), p.color, True, bi, ai)
            w, h = spr.get_size()
            mark(blit(spr, (int(p.x - w * 0.5), int(p.y - h * 0.5)), special_flags=add))

        if dirty is not None:
            dirty = dirty.clip(scratch.get_rect())
            if dirty.width > 0 and dirty.height > 0:
                self._add_dirty = dirty
                surf.blit(scratch, dirty.topleft, dirty, special_flags=add)

    # -- individual additive effects ------------------------------------ #

    def _draw_shockwave(self, dst: pygame.Surface, w: _Shockwave) -> Optional[pygame.Rect]:
        u = w.t / w.life
        r = w.r_max * T.ease_out_quint(u)
        if r < 2.0:
            return None
        fade = (1.0 - u) ** 1.7
        thick = max(1, int(w.width * (1.0 - u) ** 0.6))
        c = w.color
        # Three concentric passes fake an anti-aliased, soft-shouldered ring.
        rect = None
        for k, (ro, tw, mul) in enumerate((
            (0, thick + 4, 0.22), (0, thick + 1, 0.55), (0, max(1, thick - 1), 1.0),
        )):
            f = fade * mul
            if f <= 0.02:
                continue
            col = (int(c[0] * f), int(c[1] * f), int(c[2] * f))
            rr = pygame.draw.circle(dst, col, (int(w.x), int(w.y)),
                                    int(r + ro), min(int(r), tw))
            rect = rr if rect is None else rect.union(rr)
        return rect

    def _draw_impact_lines(self, dst: pygame.Surface, il: _ImpactLines) -> Optional[pygame.Rect]:
        u = il.t / il.life
        fade = (1.0 - u) ** 1.6
        if fade <= 0.02:
            return None
        c = il.color
        col = (int(c[0] * fade), int(c[1] * fade), int(c[2] * fade))
        inner = il.length * (0.18 + 0.62 * T.ease_out_cubic(u))
        outer = inner + il.length * 0.55 * (1.0 - u * 0.6)
        rect = None
        for i in range(il.count):
            a = (i / il.count) * math.tau + _hash01(i, il.seed) * 0.5
            ca, sa = math.cos(a), math.sin(a)
            x0, y0 = il.x + ca * inner, il.y + sa * inner
            x1, y1 = il.x + ca * outer, il.y + sa * outer
            # Tapered spike drawn as a triangle - wide at the base, sharp tip.
            tw = max(1.0, 5.0 * (1.0 - u))
            nx, ny = -sa * tw, ca * tw
            rr = pygame.draw.polygon(dst, col, (
                (x0 + nx, y0 + ny), (x0 - nx, y0 - ny), (x1, y1)))
            rect = rr if rect is None else rect.union(rr)
        return rect

    def _draw_beam(self, dst: pygame.Surface, b: _Beam) -> Optional[pygame.Rect]:
        u = b.t / b.life
        fade = (1.0 - u) ** 1.4
        if fade <= 0.02:
            return None
        c = b.color
        p0 = (int(b.x0), int(b.y0))
        p1 = (int(b.x1), int(b.y1))
        rect = None
        for mul, wmul in ((0.18, 2.4), (0.45, 1.25), (1.0, 0.45)):
            f = fade * mul
            col = (int(c[0] * f), int(c[1] * f), int(c[2] * f))
            wdt = max(1, int(b.width * wmul * (1.0 - u * 0.75)))
            rr = pygame.draw.line(dst, col, p0, p1, wdt)
            rect = rr if rect is None else rect.union(rr)
        # Hot caps so the bolt looks like it is anchored to something.
        cap = int(b.width * 1.1 * (1.0 - u))
        if cap > 1:
            hot = T.lerp_color(c, (255, 255, 255), 0.4)
            hc = (int(hot[0] * fade), int(hot[1] * fade), int(hot[2] * fade))
            for p in (p0, p1):
                rr = pygame.draw.circle(dst, hc, p, cap)
                rect = rr if rect is None else rect.union(rr)
        return rect

    # ------------------------------------------------------------------ #
    # Full-screen pass
    # ------------------------------------------------------------------ #

    def draw_fullscreen(self, surf: pygame.Surface) -> None:
        """Flashes, edge bloom and chromatic aberration. Draw this last."""
        if self._chroma_life > 0.0:
            self._apply_chroma(surf)

        for fl in self._flashes:
            u = fl.t / fl.life
            a = fl.strength * (1.0 - u) ** 2.2
            if a <= 0.004:
                continue
            c = fl.color
            surf.fill((int(c[0] * a), int(c[1] * a), int(c[2] * a)),
                      special_flags=pygame.BLEND_RGB_ADD)
            # A hotter rim on top of the flat wash reads as light spilling in
            # from off-screen rather than as a plain colour cast. Normal blend
            # (not additive) so a single cached surface can be faded for free.
            rim = int(min(1.0, a * 1.6) * 190.0)
            if rim > 3:
                glow = self._edge_glow(c)
                glow.set_alpha(rim)
                surf.blit(glow, (0, 0))

    def _edge_glow(self, color: Color) -> pygame.Surface:
        """Inverse vignette: bright at the frame edges, clear in the middle."""
        got = self._edge_cache.get(color)
        if got is not None:
            return got
        if len(self._edge_cache) >= 4:
            self._edge_cache.clear()
        small = pygame.Surface((64, 36), pygame.SRCALPHA)
        cx, cy = 32.0, 18.0
        maxd = math.hypot(cx, cy)
        for y in range(36):
            for x in range(64):
                d = math.hypot(x - cx, y - cy) / maxd
                k = max(0.0, (d - 0.42) / 0.58) ** 1.5
                small.set_at((x, y), (color[0], color[1], color[2], int(255 * k)))
        out = pygame.transform.smoothscale(small, (self.w, self.h))
        self._edge_cache[color] = out
        return out

    def _apply_chroma(self, surf: pygame.Surface) -> None:
        """Displace the red and blue channels apart, then let them snap back.

        Costs two full-screen copies while active, so it is deliberately a
        short, rare effect. Scratch surfaces are allocated once and reused.
        """
        u = self._chroma_t / self._chroma_life
        amp = self._chroma_strength * (1.0 - u) ** 1.8
        off = int(round(amp * 7.0))
        if off < 1:
            return
        if self._chroma_a is None:
            try:
                self._chroma_a = pygame.Surface((self.w, self.h)).convert()
                self._chroma_b = pygame.Surface((self.w, self.h)).convert()
            except pygame.error:            # no display mode set yet
                self._chroma_a = pygame.Surface((self.w, self.h))
                self._chroma_b = pygame.Surface((self.w, self.h))

        red, blue = self._chroma_a, self._chroma_b
        red.blit(surf, (0, 0))
        red.fill((255, 0, 0), special_flags=pygame.BLEND_RGB_MULT)
        blue.blit(surf, (0, 0))
        blue.fill((0, 0, 255), special_flags=pygame.BLEND_RGB_MULT)

        # Pull each channel out of place, then add it back displaced.
        surf.blit(red, (0, 0), special_flags=pygame.BLEND_RGB_SUB)
        surf.blit(blue, (0, 0), special_flags=pygame.BLEND_RGB_SUB)
        surf.blit(red, (off, 0), special_flags=pygame.BLEND_RGB_ADD)
        surf.blit(blue, (-off, 0), special_flags=pygame.BLEND_RGB_ADD)


# --------------------------------------------------------------------------- #
# Fade curves
# --------------------------------------------------------------------------- #

def _fade(p: _P) -> float:
    u = p.t / p.life
    if p.fade == _FADE_OUT:
        return (1.0 - u) ** 1.45
    if p.fade == _FADE_INOUT:
        # Quick bloom in, long tail out.
        return math.sin(min(1.0, u ** 0.55) * math.pi) ** 0.85
    # _FADE_LATE - hold full, then cut.
    return 1.0 if u < 0.72 else 1.0 - (u - 0.72) / 0.28

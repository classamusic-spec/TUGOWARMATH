"""Low-level rendering primitives shared by every visual module.

These wrap pygame's fairly bare drawing API with the things a polished game
actually needs: gradients, soft shadows, additive glow, glass panels, blurs,
outlined text and squircles - all aggressively cached, because most of them
are far too slow to rebuild every frame.

Cache discipline
----------------
Anything keyed on (size, color, radius, ...) is memoized in a module-level
dict. Surfaces are immutable once cached, so callers must never mutate a
returned surface - blit it, don't draw on it. Call `clear_caches()` if the
theme is ever swapped at runtime.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import pygame

import theme as T

Color = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

_surface_cache: Dict[tuple, pygame.Surface] = {}
_font_cache: Dict[tuple, pygame.font.Font] = {}


def clear_caches() -> None:
    _surface_cache.clear()
    _font_cache.clear()


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

def get_font(size: int, bold: bool = False, stack: Optional[Sequence[str]] = None) -> pygame.font.Font:
    """Resolve the first available family in `stack`, cached.

    Falls back to pygame's default font if nothing in the stack is installed,
    so this never raises on a bare machine.
    """
    stack = tuple(stack or T.FONT_STACK_TEXT)
    key = (stack, size, bold)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached

    if not pygame.font.get_init():
        pygame.font.init()

    available = {n.lower().replace(" ", "") for n in pygame.font.get_fonts()}
    chosen: Optional[pygame.font.Font] = None
    for family in stack:
        norm = family.lower().replace(" ", "")
        if norm in available:
            chosen = pygame.font.SysFont(family, size, bold=bold)
            break
    if chosen is None:
        chosen = pygame.font.Font(None, int(size * 1.12))
        chosen.set_bold(bold)

    _font_cache[key] = chosen
    return chosen


def font_display(size: int, bold: bool = True) -> pygame.font.Font:
    return get_font(size, bold, T.FONT_STACK_DISPLAY)


def font_text(size: int, bold: bool = False) -> pygame.font.Font:
    return get_font(size, bold, T.FONT_STACK_TEXT)


def font_num(size: int, bold: bool = True) -> pygame.font.Font:
    return get_font(size, bold, T.FONT_STACK_NUM)


# --------------------------------------------------------------------------- #
# Gradients
# --------------------------------------------------------------------------- #

def vertical_gradient(
    size: Tuple[int, int],
    top: Color,
    bottom: Color,
    alpha: int = 255,
) -> pygame.Surface:
    """A cached top-to-bottom linear gradient."""
    key = ("vgrad", size, top, bottom, alpha)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    w, h = max(1, size[0]), max(1, size[1])
    # Build 1px wide then scale - dramatically faster than per-pixel fills.
    strip = pygame.Surface((1, h), pygame.SRCALPHA)
    for y in range(h):
        t = y / max(1, h - 1)
        strip.set_at((0, y), (
            int(top[0] + (bottom[0] - top[0]) * t),
            int(top[1] + (bottom[1] - top[1]) * t),
            int(top[2] + (bottom[2] - top[2]) * t),
            alpha,
        ))
    surf = pygame.transform.smoothscale(strip, (w, h))
    _surface_cache[key] = surf
    return surf


def multi_gradient(
    size: Tuple[int, int],
    stops: Sequence[Tuple[float, Color]],
    alpha: int = 255,
) -> pygame.Surface:
    """Vertical gradient through N stops given as (position 0..1, color)."""
    key = ("mgrad", size, tuple(stops), alpha)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    w, h = max(1, size[0]), max(1, size[1])
    ordered = sorted(stops, key=lambda s: s[0])
    strip = pygame.Surface((1, h), pygame.SRCALPHA)
    for y in range(h):
        t = y / max(1, h - 1)
        # find bracketing stops
        lo = ordered[0]
        hi = ordered[-1]
        for i in range(len(ordered) - 1):
            if ordered[i][0] <= t <= ordered[i + 1][0]:
                lo, hi = ordered[i], ordered[i + 1]
                break
        span = max(1e-6, hi[0] - lo[0])
        k = (t - lo[0]) / span
        c = T.lerp_color(lo[1], hi[1], k)
        strip.set_at((0, y), (c[0], c[1], c[2], alpha))
    surf = pygame.transform.smoothscale(strip, (w, h))
    _surface_cache[key] = surf
    return surf


def radial_glow(radius: int, color: Color, intensity: int = 180, falloff: float = 2.0) -> pygame.Surface:
    """A soft additive glow blob, cached. Blit with BLEND_RGB_ADD."""
    key = ("glow", radius, color, intensity, falloff)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    r = max(2, radius)
    size = r * 2
    # This surface is blitted with BLEND_RGB_ADD, which ignores alpha entirely
    # and adds RGB straight into the destination. So the falloff has to be
    # baked into the RGB values (premultiplied), not into the alpha channel -
    # otherwise the glow reads as a hard-edged disc.
    src_r = 32
    src = pygame.Surface((src_r * 2, src_r * 2))
    src.fill((0, 0, 0))  # black adds nothing outside the falloff
    peak = intensity / 255.0
    for y in range(src_r * 2):
        dy = (y - src_r) / src_r
        for x in range(src_r * 2):
            dx = (x - src_r) / src_r
            d = math.hypot(dx, dy)
            if d >= 1.0:
                continue
            k = peak * pow(1.0 - d, falloff)
            src.set_at((x, y), (
                int(color[0] * k),
                int(color[1] * k),
                int(color[2] * k),
            ))
    surf = pygame.transform.smoothscale(src, (size, size))
    _surface_cache[key] = surf
    return surf


# --------------------------------------------------------------------------- #
# Blur
# --------------------------------------------------------------------------- #

def blur(surface: pygame.Surface, amount: float = 0.25, passes: int = 2) -> pygame.Surface:
    """Cheap gaussian-ish blur via repeated downscale/upscale.

    `amount` is the fraction of original size to shrink to (smaller = blurrier).
    Not cached - callers that need this every frame should cache themselves.
    """
    w, h = surface.get_size()
    small_w = max(1, int(w * amount))
    small_h = max(1, int(h * amount))
    out = surface
    for _ in range(max(1, passes)):
        small = pygame.transform.smoothscale(out, (small_w, small_h))
        out = pygame.transform.smoothscale(small, (w, h))
    return out


# --------------------------------------------------------------------------- #
# Shapes
# --------------------------------------------------------------------------- #

def rounded_rect_surface(
    size: Tuple[int, int],
    radius: int,
    color: RGBA,
) -> pygame.Surface:
    """A cached solid rounded rect with alpha."""
    key = ("rrect", size, radius, color)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached
    w, h = max(1, size[0]), max(1, size[1])
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(surf, color, (0, 0, w, h), border_radius=_clamp_radius(radius, w, h))
    _surface_cache[key] = surf
    return surf


def gradient_rounded_rect(
    size: Tuple[int, int],
    radius: int,
    top: Color,
    bottom: Color,
    alpha: int = 255,
) -> pygame.Surface:
    """Rounded rect filled with a vertical gradient, cached."""
    key = ("grrect", size, radius, top, bottom, alpha)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    w, h = max(1, size[0]), max(1, size[1])
    grad = vertical_gradient((w, h), top, bottom, alpha)
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h),
                     border_radius=_clamp_radius(radius, w, h))
    out = grad.copy()
    out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    _surface_cache[key] = out
    return out


def _clamp_radius(radius: int, w: int, h: int) -> int:
    return max(0, min(int(radius), min(w, h) // 2))


def draw_rrect(
    surf: pygame.Surface,
    rect: pygame.Rect,
    color,
    radius: int = T.R_MD,
    width: int = 0,
) -> None:
    """Direct rounded-rect draw (no cache) - fine for outlines."""
    pygame.draw.rect(surf, color, rect, width, border_radius=_clamp_radius(radius, rect.width, rect.height))


def drop_shadow(
    surf: pygame.Surface,
    rect: pygame.Rect,
    radius: int = T.R_LG,
    spread: int = 18,
    opacity: int = 110,
    offset: Tuple[int, int] = (0, 10),
) -> None:
    """Soft shadow beneath a rounded rect. Cached by shape."""
    key = ("shadow", rect.size, radius, spread, opacity)
    shadow = _surface_cache.get(key)
    if shadow is None:
        pad = spread * 2
        w, h = rect.width + pad * 2, rect.height + pad * 2
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(
            s, (0, 0, 0, opacity), (pad, pad, rect.width, rect.height),
            border_radius=_clamp_radius(radius, rect.width, rect.height),
        )
        shadow = blur(s, amount=0.30, passes=2)
        _surface_cache[key] = shadow
    pad = spread * 2
    surf.blit(shadow, (rect.left - pad + offset[0], rect.top - pad + offset[1]))


def glass_panel(
    surf: pygame.Surface,
    rect: pygame.Rect,
    radius: int = T.R_LG,
    tint: RGBA = T.GLASS,
    edge: RGBA = T.GLASS_EDGE,
    shadow: bool = True,
    inner_light: bool = True,
) -> None:
    """The standard frosted surface: soft shadow, translucent fill, lit edge.

    This is the single building block for every panel, card and HUD chip, so
    the whole UI stays visually consistent.
    """
    if shadow:
        drop_shadow(surf, rect, radius=radius, spread=16, opacity=120, offset=(0, 8))

    body = rounded_rect_surface(rect.size, radius, tint)
    surf.blit(body, rect.topleft)

    if inner_light:
        # A brighter band across the top third fakes a light source above.
        key = ("glasslight", rect.size, radius)
        lit = _surface_cache.get(key)
        if lit is None:
            w, h = rect.size
            lit = pygame.Surface((w, h), pygame.SRCALPHA)
            band_h = max(2, int(h * 0.42))
            grad = vertical_gradient((w, band_h), (255, 255, 255), (255, 255, 255))
            grad = grad.copy()
            # fade the band out downward
            for y in range(band_h):
                a = int(30 * (1.0 - y / max(1, band_h - 1)))
                pygame.draw.line(grad, (255, 255, 255, a), (0, y), (w, y))
            lit.blit(grad, (0, 0))
            mask = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(mask, (255, 255, 255, 255), (0, 0, w, h),
                             border_radius=_clamp_radius(radius, w, h))
            lit.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            _surface_cache[key] = lit
        surf.blit(lit, rect.topleft)

    if edge:
        draw_rrect(surf, rect, edge, radius, width=1)


def add_glow(
    surf: pygame.Surface,
    center: Tuple[int, int],
    radius: int,
    color: Color,
    intensity: int = 150,
) -> None:
    """Additive glow centered on a point."""
    g = radial_glow(radius, color, intensity)
    surf.blit(g, (center[0] - radius, center[1] - radius), special_flags=pygame.BLEND_RGB_ADD)


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #

def text_surface(
    txt: str,
    font: pygame.font.Font,
    color: Color,
    shadow_color: Optional[RGBA] = (0, 0, 0, 150),
    shadow_offset: Tuple[int, int] = (0, 2),
    outline: Optional[Color] = None,
    outline_w: int = 2,
) -> pygame.Surface:
    """Render text with an optional outline and drop shadow, cached."""
    key = ("text", txt, id(font), color, shadow_color, shadow_offset, outline, outline_w)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    base = font.render(txt, True, color)
    pad_x = outline_w * 2 + abs(shadow_offset[0]) + 2
    pad_y = outline_w * 2 + abs(shadow_offset[1]) + 2
    w = base.get_width() + pad_x * 2
    h = base.get_height() + pad_y * 2
    out = pygame.Surface((w, h), pygame.SRCALPHA)
    ox, oy = pad_x, pad_y

    if shadow_color:
        sh = font.render(txt, True, shadow_color[:3])
        sh.set_alpha(shadow_color[3])
        out.blit(sh, (ox + shadow_offset[0], oy + shadow_offset[1]))

    if outline and outline_w > 0:
        ol = font.render(txt, True, outline)
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx * dx + dy * dy > outline_w * outline_w:
                    continue
                if dx == 0 and dy == 0:
                    continue
                out.blit(ol, (ox + dx, oy + dy))

    out.blit(base, (ox, oy))
    _surface_cache[key] = out
    return out


def draw_text(
    surf: pygame.Surface,
    txt: str,
    font: pygame.font.Font,
    color: Color,
    center: Optional[Tuple[int, int]] = None,
    midleft: Optional[Tuple[int, int]] = None,
    midright: Optional[Tuple[int, int]] = None,
    midtop: Optional[Tuple[int, int]] = None,
    midbottom: Optional[Tuple[int, int]] = None,
    topleft: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> pygame.Rect:
    """Blit text anchored by whichever anchor kwarg is supplied."""
    s = text_surface(txt, font, color, **kwargs)
    anchors = {
        "center": center, "midleft": midleft, "midright": midright,
        "midtop": midtop, "midbottom": midbottom, "topleft": topleft,
    }
    for name, val in anchors.items():
        if val is not None:
            r = s.get_rect(**{name: val})
            surf.blit(s, r)
            return r
    r = s.get_rect(topleft=(0, 0))
    surf.blit(s, r)
    return r


def gradient_text(
    txt: str,
    font: pygame.font.Font,
    top: Color,
    bottom: Color,
    outline: Optional[Color] = None,
    outline_w: int = 3,
) -> pygame.Surface:
    """Big display text filled with a vertical gradient - for titles/banners."""
    key = ("gtext", txt, id(font), top, bottom, outline, outline_w)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached

    base = font.render(txt, True, (255, 255, 255))
    w, h = base.get_size()
    pad = outline_w * 2 + 4
    out = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)

    if outline and outline_w > 0:
        ol = font.render(txt, True, outline)
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx * dx + dy * dy > outline_w * outline_w:
                    continue
                out.blit(ol, (pad + dx, pad + dy))

    grad = vertical_gradient((w, h), top, bottom)
    filled = grad.copy()
    filled.blit(base, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    out.blit(filled, (pad, pad))
    _surface_cache[key] = out
    return out


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #

def draw_polyline_glow(
    surf: pygame.Surface,
    points: Sequence[Tuple[float, float]],
    color: Color,
    width: int = 4,
    glow_width: int = 14,
    glow_alpha: int = 70,
) -> None:
    """A line with a soft additive halo - used for the rope and energy trails."""
    if len(points) < 2:
        return
    halo = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
    pygame.draw.lines(halo, (color[0], color[1], color[2], glow_alpha), False,
                      [(int(p[0]), int(p[1])) for p in points], glow_width)
    halo = blur(halo, amount=0.35, passes=1)
    surf.blit(halo, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    pygame.draw.lines(surf, color, False, [(int(p[0]), int(p[1])) for p in points], width)


def scanline_overlay(size: Tuple[int, int], alpha: int = 10, spacing: int = 3) -> pygame.Surface:
    """Subtle horizontal texture that keeps large flat areas from banding."""
    key = ("scan", size, alpha, spacing)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached
    w, h = size
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(0, h, spacing):
        pygame.draw.line(s, (255, 255, 255, alpha), (0, y), (w, y))
    _surface_cache[key] = s
    return s


def vignette(size: Tuple[int, int], strength: int = 120) -> pygame.Surface:
    """Darkened corners to focus the eye on the center of the arena."""
    key = ("vig", size, strength)
    cached = _surface_cache.get(key)
    if cached is not None:
        return cached
    w, h = size
    small = pygame.Surface((64, 36), pygame.SRCALPHA)
    cx, cy = 32, 18
    maxd = math.hypot(cx, cy)
    for y in range(36):
        for x in range(64):
            d = math.hypot(x - cx, y - cy) / maxd
            a = int(strength * max(0.0, (d - 0.45) / 0.55) ** 1.6)
            small.set_at((x, y), (0, 0, 0, a))
    out = pygame.transform.smoothscale(small, (w, h))
    _surface_cache[key] = out
    return out

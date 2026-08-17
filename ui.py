"""Touch-first UI widget kit for RopeRush.

Every screen in the game is assembled out of the widgets in here. They all
share one tiny lifecycle - `handle_event` / `update` / `draw` - so a screen is
just a list of widgets plus layout code.

House rules
-----------
* Nothing appears, disappears or moves instantly. Every state change runs
  through one of `theme`'s easing curves over one of its durations.
* All colour, spacing, radii and type sizes come from `theme`; all drawing
  goes through `render_utils`. Neither is ever bypassed.
* Everything interactive is at least `theme.TOUCH_MIN` on its short axis, and
  press handling uses press-inside / release-inside semantics with a cancel
  when the pointer slides off - the behaviour a phone player expects.
* Depth is non-negotiable: soft shadow beneath, gradient fill, lit top rim.

Pointer input
-------------
Mouse and touch are unified by `pointer_pos()`. The game renders to a fixed
logical canvas and letterboxes it, so screen->logical conversion is injected
once via `set_pointer_transform()` rather than being repeated in every widget.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pygame

import icons
import render_utils as R
import theme as T

Color = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]

__all__ = [
    "Widget", "Group", "Button", "IconButton", "Panel", "ProgressBar",
    "SegmentedControl", "Toast", "ToastStack", "Modal", "StarRating",
    "Toggle", "Marquee", "CountUpNumber", "ScrollList",
    "set_pointer_transform", "pointer_pos", "wrap_text", "approach",
]


# --------------------------------------------------------------------------- #
# Motion helpers
# --------------------------------------------------------------------------- #

def approach(current: float, target: float, dt: float, tau: float) -> float:
    """Frame-rate independent exponential move toward `target`.

    `tau` is roughly "time to cover 63% of the remaining distance", so the
    theme's duration constants can be used directly.
    """
    if tau <= 1e-6:
        return target
    k = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * k


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _lum(c: Color) -> float:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


# --------------------------------------------------------------------------- #
# Pointer plumbing (mouse + touch unified)
# --------------------------------------------------------------------------- #

_POINTER_XFORM: Optional[Callable[[Tuple[int, int]], Tuple[int, int]]] = None

_DOWN = (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN)
_UP = (pygame.MOUSEBUTTONUP, pygame.FINGERUP)
_MOVE = (pygame.MOUSEMOTION, pygame.FINGERMOTION)


def set_pointer_transform(fn: Optional[Callable[[Tuple[int, int]], Tuple[int, int]]]) -> None:
    """Install the window->logical-canvas mapping used by every widget.

    The engine owns the letterbox maths; widgets just ask for logical coords.
    """
    global _POINTER_XFORM
    _POINTER_XFORM = fn


def pointer_pos(event: pygame.event.Event) -> Optional[Tuple[int, int]]:
    """Logical-canvas position for a mouse or touch event, or None."""
    if hasattr(event, "pos"):
        p = (int(event.pos[0]), int(event.pos[1]))
        return _POINTER_XFORM(p) if _POINTER_XFORM else p
    if hasattr(event, "x") and hasattr(event, "y") and event.type in _DOWN + _UP + _MOVE:
        # SDL touch events carry normalised 0..1 coordinates.
        return (int(event.x * T.LOGICAL_W), int(event.y * T.LOGICAL_H))
    return None


def _is_primary(event: pygame.event.Event) -> bool:
    return getattr(event, "button", 1) == 1


# --------------------------------------------------------------------------- #
# Drawing helpers shared by the widgets
# --------------------------------------------------------------------------- #

class _Clip:
    """`with _Clip(surf, rect):` - intersects with any clip already in force."""

    def __init__(self, surf: pygame.Surface, rect: pygame.Rect) -> None:
        self.surf = surf
        self.rect = pygame.Rect(rect)
        self.prev: Optional[pygame.Rect] = None

    def __enter__(self) -> "_Clip":
        self.prev = self.surf.get_clip()
        self.surf.set_clip(self.rect.clip(self.prev) if self.prev else self.rect)
        return self

    def __exit__(self, *exc) -> None:
        self.surf.set_clip(self.prev)


def _rim(
    surf: pygame.Surface,
    rect: pygame.Rect,
    radius: int,
    light: int = 105,
    dark: int = 78,
) -> None:
    """A 1px lit top edge and shaded bottom edge - the 'real material' cue."""
    half = pygame.Rect(rect.left, rect.top, rect.width, max(1, rect.height // 2))
    with _Clip(surf, pygame.Rect(rect.left, rect.centery, rect.width, rect.height)):
        R.draw_rrect(surf, rect, (0, 0, 0, dark), radius, 2)
    with _Clip(surf, half):
        R.draw_rrect(surf, rect, (255, 255, 255, light), radius, 2)


def _scaled(rect: pygame.Rect, scale: float, lift: float = 0.0) -> pygame.Rect:
    """Rect scaled about its centre, quantised to 2px so surface caches hit."""
    w = max(2, int(round(rect.width * scale / 2.0)) * 2)
    h = max(2, int(round(rect.height * scale / 2.0)) * 2)
    out = pygame.Rect(0, 0, w, h)
    out.center = (rect.centerx, int(round(rect.centery - lift)))
    return out


def _dim(c: Color, amount: float = 0.55) -> Color:
    """Desaturate + sink toward the backdrop: the universal 'disabled' look."""
    flat = T.saturate(c, 0.22)
    return T.lerp_color(flat, T.SKY_MID, amount)


def fit_font(
    txt: str,
    size: int,
    max_w: int,
    bold: bool = True,
    display: bool = True,
    min_size: int = 11,
) -> pygame.font.Font:
    """Largest font <= `size` whose rendering of `txt` fits `max_w`."""
    maker = R.font_display if display else R.font_text
    s = int(size)
    while s > min_size:
        f = maker(s, bold)
        if f.size(txt)[0] <= max_w:
            return f
        s -= 1
    return maker(min_size, bold)


def wrap_text(txt: str, font: pygame.font.Font, max_w: int) -> List[str]:
    """Greedy word wrap; honours explicit newlines."""
    lines: List[str] = []
    for para in txt.split("\n"):
        words = para.split()
        if not words:
            lines.append("")
            continue
        cur = words[0]
        for w in words[1:]:
            trial = cur + " " + w
            if font.size(trial)[0] <= max_w:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------- #
# Base widget
# --------------------------------------------------------------------------- #

class Widget:
    """Common lifecycle for everything in the kit."""

    def __init__(self, rect: Union[pygame.Rect, Tuple[int, int, int, int]]) -> None:
        self.rect = pygame.Rect(rect)
        self.visible: bool = True
        self.enabled: bool = True

    # ---- lifecycle -------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Return True if the event was consumed and must not propagate."""
        return False

    def update(self, dt: float) -> None:
        pass

    def draw(self, surf: pygame.Surface) -> None:
        pass

    # ---- helpers ---------------------------------------------------------
    def contains(self, pos: Optional[Tuple[int, int]]) -> bool:
        return bool(pos) and self.rect.collidepoint(pos)

    def cancel_press(self) -> None:
        """Called by scroll containers when a drag steals the gesture."""


class Group:
    """A flat list of widgets sharing one lifecycle. Topmost gets events first."""

    def __init__(self, *widgets: Widget) -> None:
        self.widgets: List[Widget] = list(widgets)

    def add(self, *w: Widget) -> None:
        self.widgets.extend(w)

    def remove(self, w: Widget) -> None:
        if w in self.widgets:
            self.widgets.remove(w)

    def clear(self) -> None:
        self.widgets.clear()

    def handle_event(self, event: pygame.event.Event) -> bool:
        for w in reversed(self.widgets):
            if w.visible and w.handle_event(event):
                return True
        return False

    def update(self, dt: float) -> None:
        for w in self.widgets:
            w.update(dt)

    def draw(self, surf: pygame.Surface) -> None:
        for w in self.widgets:
            if w.visible:
                w.draw(surf)


# --------------------------------------------------------------------------- #
# Press animation - shared by Button, IconButton, Toggle, SegmentedControl
# --------------------------------------------------------------------------- #

class _PressAnim:
    """Squash on press, spring back on release with `ease_out_back`."""

    def __init__(self, min_scale: float = 0.94) -> None:
        self.min_scale = min_scale
        self.held = False
        self.hover = 0.0          # 0..1 desktop hover/focus lift
        self.scale = 1.0
        self._spring_t = -1.0
        self._spring_from = 1.0

    def press(self) -> None:
        self.held = True
        self._spring_t = -1.0

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        self._spring_from = self.scale
        self._spring_t = 0.0

    def update(self, dt: float, hovered: bool) -> None:
        self.hover = approach(self.hover, 1.0 if hovered else 0.0, dt, T.D_FAST)
        if self.held:
            self.scale = approach(self.scale, self.min_scale, dt, T.D_INSTANT)
        elif self._spring_t >= 0.0:
            self._spring_t += dt
            k = _clamp(self._spring_t / T.D_BASE)
            self.scale = T.lerp(self._spring_from, 1.0, T.ease_out_back(k))
            if k >= 1.0:
                self._spring_t = -1.0
                self.scale = 1.0


# --------------------------------------------------------------------------- #
# Button
# --------------------------------------------------------------------------- #

VARIANTS = ("primary", "secondary", "ghost", "danger")


class Button(Widget):
    """The workhorse. Gradient/glass/ghost/danger, optional icon, springy press.

        Button(rect, "PLAY", icon="play", variant="primary", on_click=start)
    """

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        label: str = "",
        icon: Optional[str] = None,
        variant: str = "primary",
        on_click: Optional[Callable[[], None]] = None,
        palette: Optional[T.TeamPalette] = None,
        font_size: int = T.T_BODY,
        *,
        radius: int = T.R_LG,
        icon_side: str = "left",        # "left" | "right" | "top" | "only"
        icon_scale: float = 1.0,
        text_color: Optional[Color] = None,
        glow: bool = False,
        uppercase: bool = False,
    ) -> None:
        super().__init__(rect)
        self.label = label
        self.icon = icon
        self.variant = variant if variant in VARIANTS else "primary"
        self.on_click = on_click
        self.palette = palette
        self.font_size = font_size
        self.radius = radius
        self.icon_side = icon_side
        self.icon_scale = icon_scale
        self.text_color = text_color
        self.glow = glow
        self.uppercase = uppercase

        self.press = _PressAnim(0.94)
        self._hovered = False
        self._armed = False
        self._pulse = 0.0

    # ---- palette ---------------------------------------------------------
    def _fill_colors(self) -> Tuple[Color, Color]:
        if self.variant == "danger":
            return (T.shade(T.BAD, 26), T.shade(T.BAD, -84))
        if self.palette is not None:
            return (self.palette.light, self.palette.deep)
        return (T.GOLD, T.GOLD_DEEP)

    def _ink(self, top: Color, bottom: Color) -> Color:
        if self.text_color is not None:
            return self.text_color
        if self.variant in ("secondary", "ghost"):
            return T.INK
        mid = T.lerp_color(top, bottom, 0.5)
        return T.INK_ON_LIGHT if _lum(mid) > 186 else (255, 255, 255)

    # ---- events ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pos = pointer_pos(event)
        if event.type in _DOWN and _is_primary(event):
            if self.enabled and self.contains(pos):
                self._armed = True
                self._hovered = True
                self.press.press()
                return True
            return False
        if event.type in _MOVE:
            inside = self.contains(pos)
            self._hovered = inside and self.enabled
            if self._armed and not inside:
                # Pointer slid off while held: cancel, don't fire.
                self._armed = False
                self.press.release()
            return False
        if event.type in _UP:
            if self._armed:
                self._armed = False
                self.press.release()
                if self.contains(pos):
                    self._fire()
                    return True
            return False
        return False

    def cancel_press(self) -> None:
        if self._armed:
            self._armed = False
            self.press.release()

    def _fire(self) -> None:
        if self.enabled and self.on_click is not None:
            self.on_click()

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self.press.update(dt, self._hovered and self.enabled)
        self._pulse = (self._pulse + dt) % 100.0

    # ---- drawing ---------------------------------------------------------
    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        off = not self.enabled
        hover = 0.0 if off else self.press.hover
        r = _scaled(self.rect, self.press.scale, lift=hover * 3.0)
        rad = min(self.radius, min(r.width, r.height) // 2)

        top, bottom = self._fill_colors()
        ink = self._ink(top, bottom)
        if off:
            top, bottom = _dim(top), _dim(bottom)
            ink = T.INK_FAINT

        if self.variant in ("primary", "danger"):
            self._draw_solid(surf, r, rad, top, bottom, hover, off)
        elif self.variant == "secondary":
            self._draw_glass(surf, r, rad, hover, off)
        else:
            self._draw_ghost(surf, r, rad, hover)

        self._draw_content(surf, r, ink, off)

    def _draw_solid(self, surf, r, rad, top, bottom, hover, off) -> None:
        if not off:
            R.drop_shadow(surf, r, radius=rad, spread=18,
                          opacity=int(120 + 40 * hover), offset=(0, int(9 + 3 * hover)))
            if self.glow or hover > 0.02:
                base = T.lerp_color(top, bottom, 0.4)
                amt = 46 + 60 * hover + (26 * (0.5 + 0.5 * math.sin(self._pulse * 3.0)) if self.glow else 0)
                R.add_glow(surf, r.center, int(r.width * 0.62), base, int(amt))
        else:
            R.drop_shadow(surf, r, radius=rad, spread=12, opacity=60, offset=(0, 5))

        lift = 0 if off else int(22 * hover)
        body = R.gradient_rounded_rect(r.size, rad, T.shade(top, lift), T.shade(bottom, lift // 2))
        surf.blit(body, r.topleft)
        _rim(surf, r, rad, light=118 if not off else 40, dark=86)

    def _draw_glass(self, surf, r, rad, hover, off) -> None:
        tint = T.GLASS if off else (255, 255, 255, int(26 + 22 * hover))
        R.glass_panel(surf, r, radius=rad, tint=tint,
                      edge=(255, 255, 255, int(56 + 46 * hover)) if not off else (255, 255, 255, 26),
                      shadow=not off)
        if self.palette is not None and not off:
            # Team-tinted wash so a glass button still reads as "ours".
            wash = R.rounded_rect_surface(r.size, rad, T.with_alpha(self.palette.core, 34))
            surf.blit(wash, r.topleft)

    def _draw_ghost(self, surf, r, rad, hover) -> None:
        a = int(40 * hover + (34 if self.press.held else 0))
        if a > 0:
            surf.blit(R.rounded_rect_surface(r.size, rad, (255, 255, 255, a)), r.topleft)

    def _draw_content(self, surf, r: pygame.Rect, ink: Color, off: bool) -> None:
        label = self.label.upper() if (self.uppercase and self.label) else self.label
        pad = max(T.S3, int(r.height * 0.22))
        avail = max(8, r.width - pad * 2)
        shadow: Optional[RGBA] = (0, 0, 0, 0 if self.variant == "ghost" else 120)

        has_icon = bool(self.icon)
        if not label or self.icon_side == "only":
            if has_icon:
                s = int(min(r.height, r.width) * 0.52 * self.icon_scale)
                box = pygame.Rect(0, 0, s, s)
                box.center = r.center
                icons.draw_icon(surf, self.icon, box, ink)
            elif label:
                f = fit_font(label, self.font_size, avail)
                R.draw_text(surf, label, f, ink, center=r.center,
                            shadow_color=shadow, shadow_offset=(0, 2))
            return

        if not has_icon:
            f = fit_font(label, self.font_size, avail)
            R.draw_text(surf, label, f, ink, center=r.center,
                        shadow_color=shadow, shadow_offset=(0, 2))
            return

        if self.icon_side == "top":
            s = int(r.height * 0.40 * self.icon_scale)
            f = fit_font(label, min(self.font_size, int(r.height * 0.28)), avail)
            th = f.get_height()
            block = s + T.S1 + th
            top_y = r.centery - block // 2
            box = pygame.Rect(0, 0, s, s)
            box.midtop = (r.centerx, top_y)
            icons.draw_icon(surf, self.icon, box, ink)
            R.draw_text(surf, label, f, ink, midtop=(r.centerx, top_y + s + T.S1),
                        shadow_color=shadow, shadow_offset=(0, 2))
            return

        # Side-by-side: icon and label centred as one block.
        s = int(r.height * 0.46 * self.icon_scale)
        gap = T.S2
        f = fit_font(label, self.font_size, max(8, avail - s - gap))
        tw = f.size(label)[0]
        block = s + gap + tw
        left = r.centerx - block // 2
        box = pygame.Rect(0, 0, s, s)
        if self.icon_side == "right":
            R.draw_text(surf, label, f, ink, midleft=(left, r.centery),
                        shadow_color=shadow, shadow_offset=(0, 2))
            box.midleft = (left + tw + gap, r.centery)
        else:
            box.midleft = (left, r.centery)
            R.draw_text(surf, label, f, ink, midleft=(left + s + gap, r.centery),
                        shadow_color=shadow, shadow_offset=(0, 2))
        icons.draw_icon(surf, self.icon, box, ink)


# --------------------------------------------------------------------------- #
# IconButton
# --------------------------------------------------------------------------- #

class IconButton(Button):
    """Icon-only circular / squircle button, never smaller than TOUCH_MIN."""

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        icon: str,
        variant: str = "secondary",
        on_click: Optional[Callable[[], None]] = None,
        palette: Optional[T.TeamPalette] = None,
        *,
        shape: str = "circle",          # "circle" | "squircle"
        icon_scale: float = 1.0,
        glow: bool = False,
        badge: Optional[str] = None,
    ) -> None:
        r = pygame.Rect(rect)
        # Enforce the minimum comfortable touch target, growing about centre.
        c = r.center
        r.size = (max(r.width, T.TOUCH_MIN), max(r.height, T.TOUCH_MIN))
        r.center = c
        radius = T.R_PILL if shape == "circle" else max(T.R_MD, int(min(r.size) * 0.30))
        super().__init__(r, "", icon, variant, on_click, palette,
                         radius=radius, icon_side="only",
                         icon_scale=icon_scale * 1.02, glow=glow)
        self.shape = shape
        self.badge = badge

    def draw(self, surf: pygame.Surface) -> None:
        super().draw(surf)
        if not self.visible or not self.badge:
            return
        # Small count/status pill hanging off the top-right corner.
        r = _scaled(self.rect, self.press.scale, lift=self.press.hover * 3.0)
        f = R.font_display(T.T_MICRO, True)
        w = max(22, f.size(self.badge)[0] + T.S2)
        chip = pygame.Rect(0, 0, w, 22)
        chip.center = (r.right - 6, r.top + 6)
        R.drop_shadow(surf, chip, radius=11, spread=8, opacity=110, offset=(0, 3))
        surf.blit(R.gradient_rounded_rect(chip.size, 11, T.shade(T.BAD, 30), T.BAD), chip.topleft)
        R.draw_text(surf, self.badge, f, (255, 255, 255), center=chip.center,
                    shadow_color=(0, 0, 0, 120))


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #

class Panel(Widget):
    """Glass card with an optional title bar and team-coloured accent edge."""

    TITLE_H = 52

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        title: Optional[str] = None,
        palette: Optional[T.TeamPalette] = None,
        *,
        radius: int = T.R_XL,
        tint: RGBA = T.GLASS,
        accent_edge: bool = True,
        accent_side: str = "left",      # "left" | "top"
        title_icon: Optional[str] = None,
        padding: int = T.S5,
        shadow: bool = True,
    ) -> None:
        super().__init__(rect)
        self.title = title
        self.palette = palette
        self.radius = radius
        self.tint = tint
        self.accent_edge = accent_edge and palette is not None
        self.accent_side = accent_side
        self.title_icon = title_icon
        self.padding = padding
        self.shadow = shadow
        self.children: List[Widget] = []

    # ---- layout ----------------------------------------------------------
    @property
    def content_rect(self) -> pygame.Rect:
        top = self.rect.top + self.padding
        if self.title:
            top = self.rect.top + self.TITLE_H + T.S2
        return pygame.Rect(
            self.rect.left + self.padding, top,
            max(0, self.rect.width - self.padding * 2),
            max(0, self.rect.bottom - self.padding - top),
        )

    def add(self, *w: Widget) -> None:
        self.children.extend(w)

    # ---- lifecycle -------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.enabled:
            return False
        for c in reversed(self.children):
            if c.visible and c.handle_event(event):
                return True
        return False

    def update(self, dt: float) -> None:
        for c in self.children:
            c.update(dt)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        R.glass_panel(surf, self.rect, radius=self.radius, tint=self.tint,
                      shadow=self.shadow)

        if self.accent_edge and self.palette is not None:
            self._draw_accent(surf)

        if self.title:
            self._draw_title(surf)

        for c in self.children:
            if c.visible:
                c.draw(surf)

    def _draw_accent(self, surf: pygame.Surface) -> None:
        p = self.palette
        assert p is not None
        if self.accent_side == "top":
            bar = pygame.Rect(self.rect.left + self.radius, self.rect.top + 3,
                              self.rect.width - self.radius * 2, 5)
            grad = R.gradient_rounded_rect(bar.size, 3, p.light, p.core)
        else:
            bar = pygame.Rect(self.rect.left + 4, self.rect.top + self.radius // 2,
                              5, self.rect.height - self.radius)
            grad = R.gradient_rounded_rect(bar.size, 3, p.light, p.deep)
        R.add_glow(surf, bar.center, max(bar.width, bar.height), p.glow, 42)
        surf.blit(grad, bar.topleft)

    def _draw_title(self, surf: pygame.Surface) -> None:
        assert self.title is not None
        bar = pygame.Rect(self.rect.left, self.rect.top, self.rect.width, self.TITLE_H)
        x = bar.left + self.padding
        if self.accent_edge and self.accent_side == "left":
            x += T.S2
        ink = T.INK
        if self.title_icon:
            s = 26
            box = pygame.Rect(0, 0, s, s)
            box.midleft = (x, bar.centery)
            tint = self.palette.light if self.palette else T.GOLD
            icons.draw_icon(surf, self.title_icon, box, tint)
            x += s + T.S2
        f = fit_font(self.title, T.T_HEAD, max(20, bar.right - self.padding - x))
        R.draw_text(surf, self.title, f, ink, midleft=(x, bar.centery),
                    shadow_color=(0, 0, 0, 130))
        # Hairline divider under the title bar.
        y = bar.bottom - 1
        pygame.draw.line(surf, (255, 255, 255, 0), (0, 0), (0, 0))
        line = pygame.Surface((bar.width - self.padding * 2, 1), pygame.SRCALPHA)
        line.fill((255, 255, 255, 40))
        surf.blit(line, (bar.left + self.padding, y))


# --------------------------------------------------------------------------- #
# ProgressBar
# --------------------------------------------------------------------------- #

class ProgressBar(Widget):
    """Rounded-cap meter that eases toward its target and glows when near full."""

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        value: float = 0.0,
        palette: Optional[T.TeamPalette] = None,
        show_pct: bool = False,
        *,
        label: Optional[str] = None,
        glow_from: float = 0.82,
        font_size: int = T.T_LABEL,
    ) -> None:
        super().__init__(rect)
        self._target = _clamp(value)
        self.display = self._target
        self.palette = palette
        self.show_pct = show_pct
        self.label = label
        self.glow_from = glow_from
        self.font_size = font_size
        self._pulse = 0.0

    # ---- value -----------------------------------------------------------
    @property
    def value(self) -> float:
        return self._target

    @value.setter
    def value(self, v: float) -> None:
        self.set_value(v)

    def set_value(self, v: float, animate: bool = True) -> None:
        self._target = _clamp(float(v))
        if not animate:
            self.display = self._target

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self.display = approach(self.display, self._target, dt, T.D_BASE)
        if abs(self.display - self._target) < 0.0015:
            self.display = self._target
        self._pulse += dt

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        r = self.rect
        rad = r.height // 2
        top, core = self._colors()

        # Track: a dark inset trough with a lit lower rim.
        R.drop_shadow(surf, r, radius=rad, spread=10, opacity=90, offset=(0, 4))
        surf.blit(R.gradient_rounded_rect(r.size, rad, (10, 12, 30), (26, 28, 56), 220), r.topleft)
        R.draw_rrect(surf, r, (255, 255, 255, 34), rad, 1)

        frac = self.display
        if frac > 0.001:
            fw = max(r.height, int(round((r.width - 4) * frac)))
            fill = pygame.Rect(r.left + 2, r.top + 2, fw, r.height - 4)
            frad = fill.height // 2
            near = frac >= self.glow_from
            if near:
                pulse = 0.5 + 0.5 * math.sin(self._pulse * 5.0)
                R.add_glow(surf, (fill.right - fill.height // 2, fill.centery),
                           int(fill.height * 1.7), top, int(70 + 90 * pulse))
            surf.blit(R.gradient_rounded_rect(fill.size, frad, top, core), fill.topleft)
            # Sheen along the top of the fill sells the gloss.
            sheen = pygame.Rect(fill.left + 3, fill.top + 2, max(2, fill.width - 6),
                                max(2, fill.height // 3))
            surf.blit(R.rounded_rect_surface(sheen.size, sheen.height // 2,
                                             (255, 255, 255, 62)), sheen.topleft)
            # Bright cap at the leading edge.
            pygame.draw.circle(surf, T.with_alpha(T.shade(top, 40), 200),
                               (fill.right - frad, fill.centery), max(2, frad - 2), 2)

        self._draw_text(surf, r)

    def _colors(self) -> Tuple[Color, Color]:
        if self.palette is not None:
            return (self.palette.light, self.palette.core)
        return (T.GOLD_LIGHT, T.GOLD_DEEP)

    def _draw_text(self, surf: pygame.Surface, r: pygame.Rect) -> None:
        if self.label:
            f = R.font_text(self.font_size, True)
            R.draw_text(surf, self.label, f, T.INK,
                        midleft=(r.left + T.S3, r.centery), shadow_color=(0, 0, 0, 160))
        if self.show_pct:
            f = R.font_num(self.font_size, True)
            txt = f"{int(round(self.display * 100))}%"
            R.draw_text(surf, txt, f, T.INK,
                        midright=(r.right - T.S3, r.centery), shadow_color=(0, 0, 0, 170))


# --------------------------------------------------------------------------- #
# SegmentedControl
# --------------------------------------------------------------------------- #

class SegmentedControl(Widget):
    """Pill selector with a thumb that slides behind the chosen segment."""

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        options: Sequence[str],
        index: int = 0,
        on_change: Optional[Callable[[int], None]] = None,
        palette: Optional[T.TeamPalette] = None,
        *,
        font_size: int = T.T_LABEL,
        option_icons: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        r = pygame.Rect(rect)
        if r.height < T.TOUCH_MIN:
            c = r.center
            r.height = T.TOUCH_MIN
            r.center = c
        super().__init__(r)
        self.options = list(options) or [""]
        self.index = max(0, min(int(index), len(self.options) - 1))
        self.on_change = on_change
        self.palette = palette
        self.font_size = font_size
        self.option_icons = list(option_icons) if option_icons else [None] * len(self.options)

        self._thumb_x = float(self.seg_rect(self.index).left)
        self._press_i: Optional[int] = None
        self._press = _PressAnim(0.965)

    # ---- layout ----------------------------------------------------------
    @property
    def _inset(self) -> int:
        return 4

    def seg_rect(self, i: int) -> pygame.Rect:
        n = len(self.options)
        inner = self.rect.inflate(-self._inset * 2, -self._inset * 2)
        w = inner.width / n
        return pygame.Rect(int(inner.left + w * i), inner.top,
                           int(w * (i + 1)) - int(w * i), inner.height)

    def _hit(self, pos: Optional[Tuple[int, int]]) -> Optional[int]:
        if not self.contains(pos):
            return None
        assert pos is not None
        n = len(self.options)
        inner = self.rect.inflate(-self._inset * 2, -self._inset * 2)
        i = int((pos[0] - inner.left) / max(1.0, inner.width / n))
        return max(0, min(i, n - 1))

    # ---- events ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.enabled:
            return False
        pos = pointer_pos(event)
        if event.type in _DOWN and _is_primary(event):
            i = self._hit(pos)
            if i is not None:
                self._press_i = i
                self._press.press()
                return True
        elif event.type in _MOVE:
            if self._press_i is not None and self._hit(pos) is None:
                self._press_i = None
                self._press.release()
        elif event.type in _UP:
            if self._press_i is not None:
                i = self._hit(pos)
                target = self._press_i
                self._press_i = None
                self._press.release()
                if i is not None and i == target:
                    self.select(i)
                    return True
        return False

    def select(self, i: int, notify: bool = True) -> None:
        i = max(0, min(int(i), len(self.options) - 1))
        changed = i != self.index
        self.index = i
        if changed and notify and self.on_change:
            self.on_change(i)

    def cancel_press(self) -> None:
        self._press_i = None
        self._press.release()

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self._press.update(dt, False)
        self._thumb_x = approach(self._thumb_x, float(self.seg_rect(self.index).left),
                                 dt, T.D_FAST)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        r = self.rect
        rad = r.height // 2
        off = not self.enabled

        R.drop_shadow(surf, r, radius=rad, spread=12, opacity=100, offset=(0, 5))
        surf.blit(R.gradient_rounded_rect(r.size, rad, (14, 16, 38), (26, 28, 58), 225), r.topleft)
        R.draw_rrect(surf, r, (255, 255, 255, 40), rad, 1)

        seg = self.seg_rect(self.index)
        thumb = pygame.Rect(int(round(self._thumb_x)), seg.top, seg.width, seg.height)
        thumb = _scaled(thumb, self._press.scale)
        trad = thumb.height // 2

        top, bot = ((self.palette.light, self.palette.deep) if self.palette
                    else (T.GOLD, T.GOLD_DEEP))
        if off:
            top, bot = _dim(top), _dim(bot)
        R.drop_shadow(surf, thumb, radius=trad, spread=12, opacity=120, offset=(0, 5))
        if not off:
            R.add_glow(surf, thumb.center, int(thumb.width * 0.55),
                       T.lerp_color(top, bot, 0.4), 44)
        surf.blit(R.gradient_rounded_rect(thumb.size, trad, top, bot), thumb.topleft)
        _rim(surf, thumb, trad, light=110, dark=70)

        sel_ink = T.INK_ON_LIGHT if _lum(T.lerp_color(top, bot, 0.5)) > 186 else (255, 255, 255)
        for i, label in enumerate(self.options):
            box = self.seg_rect(i)
            selected = i == self.index
            ink = (sel_ink if selected else T.INK_DIM) if not off else T.INK_FAINT
            ic = self.option_icons[i] if i < len(self.option_icons) else None
            avail = box.width - T.S3
            if ic:
                s = int(box.height * 0.42)
                f = fit_font(label, self.font_size, max(8, avail - s - T.S2), bold=selected)
                tw = f.size(label)[0] if label else 0
                block = s + (T.S2 + tw if label else 0)
                left = box.centerx - block // 2
                ib = pygame.Rect(0, 0, s, s)
                ib.midleft = (left, box.centery)
                icons.draw_icon(surf, ic, ib, ink)
                if label:
                    R.draw_text(surf, label, f, ink, midleft=(left + s + T.S2, box.centery),
                                shadow_color=(0, 0, 0, 110) if not selected else None)
            else:
                f = fit_font(label, self.font_size, max(8, avail), bold=True)
                R.draw_text(surf, label, f, ink, center=box.center,
                            shadow_color=(0, 0, 0, 110) if not selected else (0, 0, 0, 60))


# --------------------------------------------------------------------------- #
# Toasts
# --------------------------------------------------------------------------- #

TOAST_KINDS: Dict[str, Tuple[Color, str]] = {
    "info": (T.FROST.core, "info"),
    "success": (T.OK, "check"),
    "warn": (T.WARN, "warning"),
    "error": (T.BAD, "close"),
}


class Toast(Widget):
    """One transient message. Slides down, holds, then fades back up."""

    H = 58

    def __init__(
        self,
        text: str,
        kind: str = "info",
        icon: Optional[str] = None,
        duration: float = 2.2,
        width: Optional[int] = None,
    ) -> None:
        self.text = text
        self.kind = kind if kind in TOAST_KINDS else "info"
        color, default_icon = TOAST_KINDS[self.kind]
        self.color = color
        self.icon = icon or default_icon
        self.duration = duration

        f = R.font_text(T.T_LABEL, True)
        w = width or min(560, max(240, f.size(text)[0] + 40 + T.S5 * 2))
        super().__init__(pygame.Rect(0, 0, int(w), self.H))

        self.slot_y = float(T.SAFE_T + T.S3)
        self.y = self.slot_y - 46.0
        self._t = 0.0
        self._out = 0.0
        self.dying = False

    # ---- state -----------------------------------------------------------
    @property
    def alive(self) -> bool:
        return not (self.dying and self._out >= 1.0)

    def dismiss(self) -> None:
        if not self.dying:
            self.dying = True
            self._out = 0.0

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self._t += dt
        if not self.dying and self._t >= self.duration + T.D_BASE:
            self.dismiss()
        if self.dying:
            self._out = min(1.0, self._out + dt / T.D_BASE)
        # Ease toward the slot the stack assigned us.
        self.y = approach(self.y, self.slot_y, dt, T.D_FAST)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        enter = T.ease_out_back(_clamp(self._t / T.D_BASE))
        exit_k = T.ease_in_cubic(self._out)
        alpha = int(255 * min(enter if self._t < T.D_BASE else 1.0, 1.0 - exit_k))
        if alpha <= 2:
            return

        rise = (1.0 - enter) * 44.0 + exit_k * 28.0
        r = pygame.Rect(0, 0, self.rect.width, self.H)
        r.centerx = T.LOGICAL_W // 2
        r.top = int(self.y - rise)
        self.rect = r

        # Compose off-screen so the whole toast can fade as one object.
        card = pygame.Surface((r.width + 48, r.height + 48), pygame.SRCALPHA)
        local = pygame.Rect(24, 24, r.width, r.height)
        rad = T.R_LG
        R.drop_shadow(card, local, radius=rad, spread=16, opacity=140, offset=(0, 8))
        card.blit(R.gradient_rounded_rect(local.size, rad, (44, 46, 84), (24, 25, 52), 246),
                  local.topleft)
        R.draw_rrect(card, local, (255, 255, 255, 46), rad, 1)
        # Kind stripe down the leading edge.
        stripe = pygame.Rect(local.left + 5, local.top + 9, 5, local.height - 18)
        card.blit(R.gradient_rounded_rect(stripe.size, 3, T.shade(self.color, 40), self.color),
                  stripe.topleft)

        s = 26
        ibox = pygame.Rect(0, 0, s, s)
        ibox.midleft = (local.left + 22, local.centery)
        icons.draw_icon(card, self.icon, ibox, self.color)

        tx = ibox.right + T.S3
        f = fit_font(self.text, T.T_LABEL, local.right - T.S4 - tx, bold=True, display=False)
        R.draw_text(card, self.text, f, T.INK, midleft=(tx, local.centery),
                    shadow_color=(0, 0, 0, 140))

        if alpha < 255:
            card.fill((255, 255, 255, alpha), special_flags=pygame.BLEND_RGBA_MULT)
        surf.blit(card, (r.left - 24, r.top - 24))


class ToastStack:
    """Queues toasts, lays them out from the top and reflows as they expire."""

    MAX_VISIBLE = 3

    def __init__(self, top: int = T.SAFE_T + T.S3, spacing: int = T.S2) -> None:
        self.top = top
        self.spacing = spacing
        self.active: List[Toast] = []
        self.queue: List[Toast] = []

    # ---- api -------------------------------------------------------------
    def push(
        self,
        text: str,
        kind: str = "info",
        icon: Optional[str] = None,
        duration: float = 2.2,
    ) -> Toast:
        t = Toast(text, kind, icon, duration)
        if len(self.active) < self.MAX_VISIBLE:
            self._admit(t)
        else:
            self.queue.append(t)
        return t

    def clear(self) -> None:
        for t in self.active:
            t.dismiss()
        self.queue.clear()

    def _admit(self, t: Toast) -> None:
        t.slot_y = float(self.top + len(self.active) * (Toast.H + self.spacing))
        t.y = t.slot_y - 46.0
        self.active.append(t)

    # ---- lifecycle -------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        return False        # toasts are non-interactive by design

    def update(self, dt: float) -> None:
        for t in self.active:
            t.update(dt)
        self.active = [t for t in self.active if t.alive]
        while self.queue and len(self.active) < self.MAX_VISIBLE:
            self._admit(self.queue.pop(0))
        for i, t in enumerate(self.active):
            t.slot_y = float(self.top + i * (Toast.H + self.spacing))

    def draw(self, surf: pygame.Surface) -> None:
        for t in self.active:
            t.draw(surf)


# --------------------------------------------------------------------------- #
# Modal
# --------------------------------------------------------------------------- #

ButtonSpec = Union[
    Tuple[str, Callable[[], None]],
    Tuple[str, str, Callable[[], None]],
]


class Modal(Widget):
    """Dimmed scrim + a glass dialog that pops in with `ease_out_back`.

    While open it consumes every event, so the screen behind it is inert.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (620, 340),
        title: str = "",
        body: str = "",
        buttons: Optional[Sequence[ButtonSpec]] = None,
        *,
        icon: Optional[str] = None,
        palette: Optional[T.TeamPalette] = None,
        dismissible: bool = True,
        on_close: Optional[Callable[[], None]] = None,
    ) -> None:
        r = pygame.Rect(0, 0, size[0], size[1])
        r.center = (T.LOGICAL_W // 2, T.LOGICAL_H // 2)
        super().__init__(r)
        self.title = title
        self.body = body
        self.icon = icon
        self.palette = palette
        self.dismissible = dismissible
        self.on_close = on_close

        self.is_open = False
        self._closing = False
        self._t = 0.0                    # 0 = hidden, 1 = fully open
        self.buttons: List[Button] = []
        self.set_buttons(buttons or [])

    # ---- construction ----------------------------------------------------
    def set_buttons(self, specs: Sequence[ButtonSpec]) -> None:
        self.buttons = []
        if not specs:
            return
        n = len(specs)
        gap = T.S3
        pad = T.S5
        total_w = self.rect.width - pad * 2
        bw = int((total_w - gap * (n - 1)) / n)
        bh = T.TOUCH_MIN
        y = self.rect.bottom - pad - bh
        for i, spec in enumerate(specs):
            if len(spec) == 3:
                label, variant, cb = spec           # type: ignore[misc]
            else:
                label, cb = spec                    # type: ignore[misc]
                variant = "primary" if i == n - 1 else "secondary"
            br = pygame.Rect(self.rect.left + pad + i * (bw + gap), y, bw, bh)
            self.buttons.append(Button(
                br, label, variant=variant, on_click=cb,
                palette=self.palette if variant == "primary" else None,
                uppercase=True,
            ))

    # ---- open/close ------------------------------------------------------
    def open(self) -> None:
        self.is_open = True
        self._closing = False
        self.visible = True

    def close(self) -> None:
        if self.is_open and not self._closing:
            self._closing = True
            if self.on_close:
                self.on_close()

    # ---- events ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.is_open:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.dismissible:
            self.close()
            return True
        for b in self.buttons:
            b.handle_event(event)
        if event.type in _DOWN and self.dismissible and _is_primary(event):
            pos = pointer_pos(event)
            if pos and not self.rect.collidepoint(pos):
                self.close()
        return True      # swallow everything underneath

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        if self._closing:
            self._t = max(0.0, self._t - dt / T.D_FAST)
            if self._t <= 0.0:
                self.is_open = False
                self._closing = False
        elif self.is_open:
            self._t = min(1.0, self._t + dt / T.D_BASE)
        if not self.is_open:
            return
        for b in self.buttons:
            b.update(dt)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.is_open or self._t <= 0.0:
            return
        # Scrim
        scrim = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        scrim.fill(T.with_alpha(T.SCRIM[:3], int(T.SCRIM[3] * T.ease_out_cubic(self._t))))
        surf.blit(scrim, (0, 0))

        k = T.ease_out_back(self._t) if not self._closing else T.ease_in_cubic(self._t)
        scale = 0.84 + 0.16 * k
        r = _scaled(self.rect, scale)
        dy = self.rect.centery - r.centery

        R.drop_shadow(surf, r, radius=T.R_XL, spread=30, opacity=170, offset=(0, 16))
        R.glass_panel(surf, r, radius=T.R_XL, tint=(30, 33, 68, 246), shadow=False)
        if self.palette:
            R.add_glow(surf, (r.centerx, r.top + 8), int(r.width * 0.45), self.palette.glow, 46)
        _rim(surf, r, T.R_XL, light=70, dark=40)

        pad = T.S5
        y = r.top + pad + int(T.S2 * (1 - k))

        if self.icon:
            s = 54
            box = pygame.Rect(0, 0, s, s)
            box.midtop = (r.centerx, y)
            tint = self.palette.light if self.palette else T.GOLD
            R.add_glow(surf, box.center, s, tint, 60)
            icons.draw_icon(surf, self.icon, box, tint)
            y += s + T.S3

        if self.title:
            f = fit_font(self.title, T.T_DISPLAY, r.width - pad * 2)
            surf.blit(
                R.gradient_text(self.title, f, T.INK, T.lerp_color(T.INK, T.INK_DIM, 0.7)),
                R.gradient_text(self.title, f, T.INK, T.INK_DIM).get_rect(midtop=(r.centerx, y)),
            )
            y += f.get_height() + T.S3

        if self.body:
            f = R.font_text(T.T_BODY)
            for line in wrap_text(self.body, f, r.width - pad * 2 - T.S4):
                R.draw_text(surf, line, f, T.INK_DIM, midtop=(r.centerx, y),
                            shadow_color=(0, 0, 0, 120))
                y += f.get_height() + 2

        # Buttons ride the panel's pop-in.
        for b in self.buttons:
            saved = b.rect.copy()
            b.rect = b.rect.move(0, -dy)
            b.rect.width = saved.width
            try:
                b.draw(surf)
            finally:
                b.rect = saved


# --------------------------------------------------------------------------- #
# StarRating
# --------------------------------------------------------------------------- #

class StarRating(Widget):
    """1-3 stars with a staggered pop-in reveal and a glow burst per star."""

    POP = 0.52          # seconds one star takes to settle
    BURST = 0.55        # seconds the glow burst lasts

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        count: int = 3,
        value: int = 0,
        palette: Optional[T.TeamPalette] = None,
        *,
        stagger: float = 0.22,
    ) -> None:
        super().__init__(rect)
        self.count = max(1, int(count))
        self.value = max(0, min(int(value), self.count))
        self.palette = palette
        self.stagger = stagger
        # Per-star elapsed time; negative = still waiting for its cue.
        self._t: List[float] = [1e9 if i < self.value else -1e9 for i in range(self.count)]

    # ---- api -------------------------------------------------------------
    def set(self, n: int) -> None:
        """Show `n` stars immediately, no animation."""
        self.value = max(0, min(int(n), self.count))
        self._t = [1e9 if i < self.value else -1e9 for i in range(self.count)]

    def reveal(self, n: int, delay: float = 0.0) -> None:
        """Pop `n` stars in sequence, starting after `delay` seconds."""
        self.value = max(0, min(int(n), self.count))
        self._t = [
            -(delay + i * self.stagger) if i < self.value else -1e9
            for i in range(self.count)
        ]

    @property
    def done(self) -> bool:
        return all(t >= self.POP + self.BURST or t <= -1e8 for t in self._t)

    # ---- layout ----------------------------------------------------------
    def _star_boxes(self) -> List[Tuple[pygame.Rect, float]]:
        n = self.count
        gap = T.S2
        s = min(int(self.rect.height * 0.86),
                int((self.rect.width - gap * (n - 1)) / n))
        total = s * n + gap * (n - 1)
        left = self.rect.centerx - total // 2
        out: List[Tuple[pygame.Rect, float]] = []
        mid = (n - 1) / 2.0
        for i in range(n):
            # Centre star sits a touch larger and higher - the arcade look.
            hero = 1.0 - abs(i - mid) / max(1.0, mid)
            k = 1.0 + 0.14 * hero
            box = pygame.Rect(0, 0, int(s * k), int(s * k))
            box.center = (left + s * i + gap * i + s // 2,
                          self.rect.centery - int(s * 0.11 * hero))
            out.append((box, k))
        return out

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        for i, t in enumerate(self._t):
            if t <= -1e8:
                continue
            self._t[i] = min(t + dt, 1e9)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        gold, deep = T.GOLD, T.GOLD_DEEP
        if self.palette is not None:
            gold, deep = self.palette.light, self.palette.deep

        for i, (box, _k) in enumerate(self._star_boxes()):
            t = self._t[i]
            if t <= 0.0:
                # Not revealed yet: empty socket.
                icons.draw_icon(surf, "star_outline", box, T.INK_FAINT, width=0.85)
                continue

            pop = _clamp(t / self.POP)
            scale = T.ease_out_back(pop, overshoot=2.6)
            b = _scaled(box, max(0.05, scale))

            # Expanding burst ring + additive glow on arrival.
            bt = _clamp(t / self.BURST)
            if bt < 1.0:
                fade = 1.0 - bt
                rr = int(box.width * (0.55 + 1.15 * T.ease_out_cubic(bt)))
                R.add_glow(surf, box.center, rr, gold, int(150 * fade))
                ring = int(box.width * (0.5 + 0.9 * T.ease_out_quint(bt)))
                if ring > 2:
                    pygame.draw.circle(surf, T.with_alpha(T.GOLD_LIGHT, int(150 * fade)),
                                       box.center, ring, max(1, int(4 * fade) + 1))
            else:
                R.add_glow(surf, box.center, int(box.width * 0.75), gold, 62)

            icons.draw_icon(surf, "star", b, gold, accent=deep)


# --------------------------------------------------------------------------- #
# Toggle
# --------------------------------------------------------------------------- #

class Toggle(Widget):
    """iOS-style switch: animated knob, colour-lerping track, squish on press."""

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        value: bool = False,
        on_change: Optional[Callable[[bool], None]] = None,
        palette: Optional[T.TeamPalette] = None,
        *,
        label: Optional[str] = None,
    ) -> None:
        r = pygame.Rect(rect)
        r.height = max(r.height, 40)
        r.width = max(r.width, int(r.height * 1.78))
        super().__init__(r)
        self.value = bool(value)
        self.on_change = on_change
        self.palette = palette
        self.label = label
        self._k = 1.0 if self.value else 0.0
        self._press = _PressAnim(0.96)
        self._armed = False

    # ---- hit area is padded out to a comfortable touch target ------------
    @property
    def hit_rect(self) -> pygame.Rect:
        r = self.rect.copy()
        if r.height < T.TOUCH_MIN:
            r = r.inflate(0, T.TOUCH_MIN - r.height)
        return r

    def contains(self, pos: Optional[Tuple[int, int]]) -> bool:
        return bool(pos) and self.hit_rect.collidepoint(pos)

    # ---- events ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.enabled:
            return False
        pos = pointer_pos(event)
        if event.type in _DOWN and _is_primary(event) and self.contains(pos):
            self._armed = True
            self._press.press()
            return True
        if event.type in _MOVE and self._armed and not self.contains(pos):
            self._armed = False
            self._press.release()
        if event.type in _UP and self._armed:
            self._armed = False
            self._press.release()
            if self.contains(pos):
                self.set(not self.value)
                return True
        return False

    def set(self, v: bool, notify: bool = True) -> None:
        v = bool(v)
        changed = v != self.value
        self.value = v
        if changed and notify and self.on_change:
            self.on_change(v)

    def cancel_press(self) -> None:
        self._armed = False
        self._press.release()

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self._press.update(dt, False)
        self._k = approach(self._k, 1.0 if self.value else 0.0, dt, T.D_FAST * 0.75)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        r = self.rect
        rad = r.height // 2
        off = not self.enabled
        on_top, on_bot = ((self.palette.light, self.palette.deep) if self.palette
                          else (T.OK, T.shade(T.OK, -70)))
        k = T.ease_out_cubic(self._k)

        R.drop_shadow(surf, r, radius=rad, spread=10, opacity=95, offset=(0, 4))
        track_top = T.lerp_color((18, 20, 44), on_top, k)
        track_bot = T.lerp_color((34, 36, 70), on_bot, k)
        if off:
            track_top, track_bot = _dim(track_top), _dim(track_bot)
        surf.blit(R.gradient_rounded_rect(r.size, rad, track_top, track_bot), r.topleft)
        R.draw_rrect(surf, r, (255, 255, 255, int(40 + 40 * k)), rad, 1)
        if k > 0.5 and not off:
            R.add_glow(surf, r.center, int(r.width * 0.5), on_top, int(46 * (k - 0.5) * 2))

        # Knob: slides, and stretches slightly while held.
        pad = 4
        kd = r.height - pad * 2
        stretch = int(kd * 0.16) if self._press.held else 0
        x0 = r.left + pad
        x1 = r.right - pad - kd - stretch
        kx = int(T.lerp(x0, x1, k))
        knob = pygame.Rect(kx, r.top + pad, kd + stretch, kd)
        R.drop_shadow(surf, knob, radius=kd // 2, spread=8, opacity=120, offset=(0, 3))
        surf.blit(R.gradient_rounded_rect(knob.size, knob.height // 2,
                                          (255, 255, 255), (216, 222, 240)), knob.topleft)
        R.draw_rrect(surf, knob, (255, 255, 255, 200), knob.height // 2, 1)

        if self.label:
            f = R.font_text(T.T_LABEL, True)
            R.draw_text(surf, self.label, f, T.INK if not off else T.INK_FAINT,
                        midright=(r.left - T.S3, r.centery), shadow_color=(0, 0, 0, 140))


# --------------------------------------------------------------------------- #
# Marquee
# --------------------------------------------------------------------------- #

class Marquee(Widget):
    """Clipped, edge-faded text that scrolls only when it doesn't fit."""

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        text: str = "",
        font_size: int = T.T_BODY,
        color: Color = T.INK,
        *,
        speed: float = 46.0,
        gap: int = 64,
        pause: float = 1.4,
        display_font: bool = False,
    ) -> None:
        super().__init__(rect)
        self.color = color
        self.speed = speed
        self.gap = gap
        self.pause = pause
        self.font = (R.font_display if display_font else R.font_text)(font_size, True)
        self._text = ""
        self._x = 0.0
        self._wait = 0.0
        self.text = text

    # ---- content ---------------------------------------------------------
    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, v: str) -> None:
        if v == self._text:
            return
        self._text = v
        self._x = 0.0
        self._wait = self.pause
        self._tw = self.font.size(v)[0]

    @property
    def scrolling(self) -> bool:
        return self._tw > self.rect.width

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        if not self.scrolling:
            self._x = 0.0
            return
        if self._wait > 0.0:
            self._wait -= dt
            return
        self._x += self.speed * dt
        if self._x >= self._tw + self.gap:
            self._x = 0.0
            self._wait = self.pause

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible or not self._text:
            return
        r = self.rect
        band = pygame.Surface(r.size, pygame.SRCALPHA)
        ts = R.text_surface(self._text, self.font, self.color, shadow_color=(0, 0, 0, 150))
        y = (r.height - ts.get_height()) // 2
        if self.scrolling:
            band.blit(ts, (-int(self._x), y))
            band.blit(ts, (-int(self._x) + self._tw + self.gap, y))
            band.blit(_edge_mask(r.size, min(48, r.width // 4)), (0, 0),
                      special_flags=pygame.BLEND_RGBA_MULT)
        else:
            band.blit(ts, ((r.width - ts.get_width()) // 2, y))
        surf.blit(band, r.topleft)


def _edge_mask(size: Tuple[int, int], fade: int) -> pygame.Surface:
    """Cached alpha ramp used to soften both ends of a scrolling band."""
    key = ("ui_edge", size, fade)
    cached = R._surface_cache.get(key)
    if cached is not None:
        return cached
    w, h = size
    m = pygame.Surface((w, h), pygame.SRCALPHA)
    m.fill((255, 255, 255, 255))
    for i in range(max(1, fade)):
        a = int(255 * (i / max(1, fade)))
        pygame.draw.line(m, (255, 255, 255, a), (i, 0), (i, h))
        pygame.draw.line(m, (255, 255, 255, a), (w - 1 - i, 0), (w - 1 - i, h))
    R._surface_cache[key] = m
    return m


# --------------------------------------------------------------------------- #
# CountUpNumber
# --------------------------------------------------------------------------- #

class CountUpNumber(Widget):
    """Score/XP readout that eases to a new value and rolls the digits.

    The value itself is interpolated with `ease_out_quint` so it decelerates
    into place, each digit that flips slides up into position, and the whole
    block gets a short scale punch - together that reads as *earned*, which a
    plain text swap never does.
    """

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        value: float = 0,
        font_size: int = T.T_HEAD,
        color: Color = T.INK,
        *,
        palette: Optional[T.TeamPalette] = None,
        prefix: str = "",
        suffix: str = "",
        commas: bool = True,
        decimals: int = 0,
        align: str = "center",          # "left" | "center" | "right"
        duration: float = T.D_SLOW * 1.6,
        roll: bool = True,
        gradient: bool = False,
    ) -> None:
        super().__init__(rect)
        self.font_size = font_size
        self.color = color
        self.palette = palette
        self.prefix = prefix
        self.suffix = suffix
        self.commas = commas
        self.decimals = decimals
        self.align = align
        self.duration = max(0.05, duration)
        self.roll = roll
        self.gradient = gradient

        self._target = float(value)
        self._from = float(value)
        self.display = float(value)
        self._t = self.duration
        self._prev_str = self._format(self.display)
        self._digit_age: Dict[int, float] = {}

    # ---- api -------------------------------------------------------------
    @property
    def value(self) -> float:
        return self._target

    def set(self, value: float, animate: bool = True) -> None:
        self._target = float(value)
        if not animate:
            self._from = self.display = self._target
            self._t = self.duration
            self._prev_str = self._format(self.display)
            return
        self._from = self.display
        self._t = 0.0

    def add(self, delta: float) -> None:
        self.set(self._target + delta)

    @property
    def animating(self) -> bool:
        return self._t < self.duration

    # ---- formatting ------------------------------------------------------
    def _format(self, v: float) -> str:
        if self.decimals > 0:
            body = f"{v:,.{self.decimals}f}" if self.commas else f"{v:.{self.decimals}f}"
        else:
            body = f"{int(round(v)):,}" if self.commas else str(int(round(v)))
        return f"{self.prefix}{body}{self.suffix}"

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        if self._t < self.duration:
            self._t = min(self.duration, self._t + dt)
            k = T.ease_out_quint(self._t / self.duration)
            self.display = T.lerp(self._from, self._target, k)
        txt = self._format(self.display)
        if self.roll and txt != self._prev_str:
            # Age-stamp only the glyphs that actually changed.
            old = self._prev_str
            pad = max(len(txt), len(old))
            a, b = old.rjust(pad), txt.rjust(pad)
            aged = {}
            for i in range(pad):
                if a[i] != b[i]:
                    aged[i] = 0.0
                elif i in self._digit_age:
                    aged[i] = self._digit_age[i]
            self._digit_age = aged
            self._prev_str = txt
        for i in list(self._digit_age):
            self._digit_age[i] += dt
            if self._digit_age[i] > T.D_FAST:
                del self._digit_age[i]

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        txt = self._format(self.display)
        f = fit_font(txt, self.font_size, self.rect.width, display=False) \
            if R.font_num(self.font_size, True).size(txt)[0] > self.rect.width \
            else R.font_num(self.font_size, True)

        top = self.palette.light if self.palette else T.GOLD_LIGHT
        bot = self.palette.deep if self.palette else T.GOLD_DEEP

        # Scale punch at the start of a change.
        punch = 1.0
        if self._t < T.D_BASE:
            punch = 1.0 + 0.18 * (1.0 - T.ease_out_cubic(self._t / T.D_BASE))

        block = self._render_block(txt, f, top, bot)
        if punch > 1.001:
            block = pygame.transform.smoothscale(
                block, (max(1, int(block.get_width() * punch)),
                        max(1, int(block.get_height() * punch))))

        if self.align == "left":
            pos = block.get_rect(midleft=(self.rect.left, self.rect.centery))
        elif self.align == "right":
            pos = block.get_rect(midright=(self.rect.right, self.rect.centery))
        else:
            pos = block.get_rect(center=self.rect.center)
        surf.blit(block, pos)

    def _render_block(self, txt: str, f: pygame.font.Font,
                      top: Color, bot: Color) -> pygame.Surface:
        """Compose the glyphs, offsetting any that just rolled over."""
        pad = 6
        w = f.size(txt)[0] + pad * 2
        h = f.get_height() + pad * 2
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        x = pad
        n = len(txt)
        for i, ch in enumerate(txt):
            if self.gradient or self.palette is not None:
                g = R.gradient_text(ch, f, top, bot, outline=(0, 0, 0), outline_w=2)
                gw = f.size(ch)[0]
                dy = 0.0
                idx = i + (max(0, len(self._prev_str) - n))
                age = self._digit_age.get(idx if idx in self._digit_age else i)
                if age is not None and self.roll:
                    dy = (1.0 - T.ease_out_cubic(age / T.D_FAST)) * f.get_height() * 0.30
                out.blit(g, g.get_rect(midleft=(x - 5, h // 2 + dy)))
                x += gw
            else:
                s = R.text_surface(ch, f, self.color, shadow_color=(0, 0, 0, 160),
                                   shadow_offset=(0, 2))
                gw = f.size(ch)[0]
                dy = 0.0
                age = self._digit_age.get(i)
                if age is not None and self.roll:
                    dy = (1.0 - T.ease_out_cubic(age / T.D_FAST)) * f.get_height() * 0.30
                out.blit(s, s.get_rect(midleft=(x - (s.get_width() - gw) // 2,
                                                int(h // 2 + dy))))
                x += gw
        return out


# --------------------------------------------------------------------------- #
# ScrollList
# --------------------------------------------------------------------------- #

class ScrollList(Widget):
    """Touch-first vertical scroller: drag, momentum, rubber-band overscroll.

    Children live in *content* coordinates - (0, 0) is the top-left of the
    scrollable content, not of the screen - and are drawn clipped to the list
    rect with their positions translated on the fly.
    """

    DRAG_SLOP = 8           # px before a press becomes a drag
    FRICTION = 4.2          # momentum decay (1/s)
    RUBBER = 0.42           # drag resistance past the ends
    SPRING = 0.075          # tau of the snap-back

    def __init__(
        self,
        rect: Union[pygame.Rect, Tuple[int, int, int, int]],
        *,
        padding: int = T.S3,
        spacing: int = T.S2,
        fade_edges: bool = True,
        scrollbar: bool = True,
    ) -> None:
        super().__init__(rect)
        self.padding = padding
        self.spacing = spacing
        self.fade_edges = fade_edges
        self.scrollbar = scrollbar

        self.children: List[Widget] = []
        self.scroll = 0.0
        self._vel = 0.0
        self._dragging = False
        self._armed = False
        self._down_y = 0
        self._last_y = 0
        self._last_dt = 1 / 60
        self._bar_alpha = 0.0
        self._stack_y = padding

    # ---- content ---------------------------------------------------------
    def add(self, *widgets: Widget, stack: bool = True) -> None:
        """Append children; by default they auto-stack down the content area."""
        for w in widgets:
            if stack:
                w.rect.top = self._stack_y
                w.rect.left = self.padding + w.rect.left if w.rect.left < 0 else w.rect.left
                self._stack_y = w.rect.bottom + self.spacing
            self.children.append(w)

    def clear(self) -> None:
        self.children.clear()
        self._stack_y = self.padding
        self.scroll = 0.0
        self._vel = 0.0

    @property
    def content_height(self) -> int:
        if not self.children:
            return 0
        return max(c.rect.bottom for c in self.children) + self.padding

    @property
    def max_scroll(self) -> float:
        return max(0.0, float(self.content_height - self.rect.height))

    def scroll_to(self, y: float, animate: bool = True) -> None:
        y = max(0.0, min(float(y), self.max_scroll))
        self._vel = 0.0
        if animate:
            self._target_scroll = y
        else:
            self.scroll = y

    # ---- coordinate mapping ----------------------------------------------
    def _to_content(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        return (pos[0] - self.rect.left, int(pos[1] - self.rect.top + self.scroll))

    def _shifted(self, event: pygame.event.Event, pos: Tuple[int, int]) -> pygame.event.Event:
        d = dict(event.__dict__)
        d["pos"] = self._to_content(pos)
        return pygame.event.Event(event.type, d)

    # ---- events ----------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible or not self.enabled:
            return False
        pos = pointer_pos(event)

        if event.type == pygame.MOUSEWHEEL:
            mouse = pygame.mouse.get_pos()
            if _POINTER_XFORM:
                mouse = _POINTER_XFORM(mouse)
            if self.rect.collidepoint(mouse):
                self._vel = 0.0
                self.scroll = max(0.0, min(self.scroll - event.y * 64, self.max_scroll))
                self._bar_alpha = 1.0
                return True
            return False

        if event.type in _DOWN and _is_primary(event) and self.contains(pos):
            assert pos is not None
            self._armed = True
            self._dragging = False
            self._down_y = pos[1]
            self._last_y = pos[1]
            self._vel = 0.0
            # Offer the press to a child; a drag will cancel it later.
            for c in reversed(self.children):
                if c.visible and c.handle_event(self._shifted(event, pos)):
                    break
            return True

        if event.type in _MOVE and self._armed and pos is not None:
            dy = pos[1] - self._last_y
            self._last_y = pos[1]
            if not self._dragging and abs(pos[1] - self._down_y) > self.DRAG_SLOP:
                self._dragging = True
                for c in self.children:
                    c.cancel_press()
            if self._dragging:
                # Resist past the ends so the list feels physically bounded.
                nxt = self.scroll - dy
                if nxt < 0 or nxt > self.max_scroll:
                    nxt = self.scroll - dy * self.RUBBER
                self.scroll = nxt
                self._vel = -dy / max(1e-4, self._last_dt)
                self._bar_alpha = 1.0
            return True

        if event.type in _UP and self._armed:
            self._armed = False
            if self._dragging:
                self._dragging = False
                return True
            for c in reversed(self.children):
                if c.visible and pos is not None and c.handle_event(self._shifted(event, pos)):
                    return True
            return True

        # Non-pointer events (keys, etc.) still reach the children.
        if event.type not in _DOWN + _UP + _MOVE:
            for c in reversed(self.children):
                if c.visible and c.handle_event(event):
                    return True
        return False

    def cancel_press(self) -> None:
        self._armed = False
        self._dragging = False
        for c in self.children:
            c.cancel_press()

    # ---- lifecycle -------------------------------------------------------
    def update(self, dt: float) -> None:
        self._last_dt = max(1e-4, dt)
        for c in self.children:
            c.update(dt)

        if not self._dragging:
            if abs(self._vel) > 1.0:
                self.scroll += self._vel * dt
                self._vel *= math.exp(-self.FRICTION * dt)
            else:
                self._vel = 0.0
            # Spring back from overscroll.
            clamped = max(0.0, min(self.scroll, self.max_scroll))
            if abs(clamped - self.scroll) > 0.4:
                self.scroll = approach(self.scroll, clamped, dt, self.SPRING)
                self._vel *= 0.55
            else:
                self.scroll = clamped

        target_alpha = 1.0 if (self._dragging or abs(self._vel) > 6.0) else 0.0
        self._bar_alpha = approach(self._bar_alpha, target_alpha, dt, T.D_BASE)

    def draw(self, surf: pygame.Surface) -> None:
        if not self.visible:
            return
        dy = int(round(self.scroll))
        with _Clip(surf, self.rect):
            for c in self.children:
                if not c.visible:
                    continue
                top = c.rect.top - dy + self.rect.top
                if top > self.rect.bottom or top + c.rect.height < self.rect.top:
                    continue      # cull off-screen rows
                saved = c.rect.topleft
                c.rect.topleft = (saved[0] + self.rect.left, saved[1] - dy + self.rect.top)
                try:
                    c.draw(surf)
                finally:
                    c.rect.topleft = saved

            if self.fade_edges:
                self._draw_fades(surf)
        if self.scrollbar:
            self._draw_bar(surf)

    def _draw_fades(self, surf: pygame.Surface) -> None:
        h = 26
        if self.scroll > 2:
            surf.blit(_fade_strip((self.rect.width, h), True), self.rect.topleft)
        if self.scroll < self.max_scroll - 2:
            surf.blit(_fade_strip((self.rect.width, h), False),
                      (self.rect.left, self.rect.bottom - h))

    def _draw_bar(self, surf: pygame.Surface) -> None:
        if self.max_scroll <= 0 or self._bar_alpha <= 0.02:
            return
        track_h = self.rect.height - T.S3 * 2
        frac = self.rect.height / max(1.0, float(self.content_height))
        bh = max(28, int(track_h * frac))
        t = _clamp(self.scroll / self.max_scroll)
        bar = pygame.Rect(self.rect.right - 8, int(self.rect.top + T.S3 + (track_h - bh) * t),
                          4, bh)
        surf.blit(R.rounded_rect_surface(bar.size, 2,
                                         (255, 255, 255, int(120 * self._bar_alpha))),
                  bar.topleft)


def _fade_strip(size: Tuple[int, int], top: bool) -> pygame.Surface:
    """Cached darkening ramp that tucks list content under its own edge."""
    key = ("ui_fade", size, top)
    cached = R._surface_cache.get(key)
    if cached is not None:
        return cached
    w, h = size
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    for y in range(h):
        k = (1.0 - y / max(1, h - 1)) if top else (y / max(1, h - 1))
        pygame.draw.line(s, (6, 8, 22, int(150 * k * k)), (0, y), (w, y))
    R._surface_cache[key] = s
    return s

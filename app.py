"""Application shell: window, logical canvas, scene stack and transitions.

Everything in the game draws into a fixed logical canvas (theme.LOGICAL_W x
theme.LOGICAL_H, 16:9). The shell letterbox-scales that canvas to whatever the
actual window or device screen is, and translates pointer coordinates back
into logical space. Game code therefore never has to think about the device's
real resolution or aspect ratio - it just draws at 1280x720.

On a phone held horizontally, aspect ratios run from 16:9 up to about 20:9.
Wider-than-16:9 screens get pillarboxed bars that are filled with an extended
backdrop tint rather than hard black, so it reads as an intentional frame.

Scene model
-----------
Scenes are pushed/replaced on a stack. Only the top scene receives input, but
scenes beneath it can still be drawn (used for modal overlays like pause).
Transitions run as a full-screen wipe/fade between two rendered canvases.
"""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional, Tuple

import pygame

import render_utils as R
import theme as T


# --------------------------------------------------------------------------- #
# Scene base
# --------------------------------------------------------------------------- #

class Scene:
    """One screen of the game.

    `app` is the App instance and is the only channel a scene has to shared
    services (audio, profile, vfx, navigation).
    """

    #: If True, the scene below this one keeps drawing (overlay scenes).
    transparent: bool = False
    #: Music track to request on enter; None leaves the current track alone.
    music: Optional[str] = None

    def __init__(self, app: "App") -> None:
        self.app = app

    def on_enter(self, **kwargs) -> None:
        """Called every time this scene becomes the active scene."""

    def on_exit(self) -> None:
        """Called when this scene is popped or replaced."""

    def handle_event(self, event: pygame.event.Event) -> None:
        """Receive one input event, already translated to logical coords."""

    def update(self, dt: float) -> None:
        """Advance simulation. `dt` is already time-scaled."""

    def draw(self, surf: pygame.Surface) -> None:
        """Render into the logical canvas."""


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #

class _Transition:
    """Cross-fade + slide between two canvas snapshots."""

    def __init__(self, outgoing: pygame.Surface, duration: float = T.D_SCENE,
                 kind: str = "fade") -> None:
        self.outgoing = outgoing
        self.duration = duration
        self.kind = kind
        self.elapsed = 0.0

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration

    def update(self, dt: float) -> None:
        self.elapsed += dt

    def draw(self, surf: pygame.Surface) -> None:
        """Composite the outgoing snapshot over the freshly drawn scene."""
        t = min(1.0, self.elapsed / max(1e-6, self.duration))
        if self.kind == "slide":
            e = T.ease_in_out_cubic(t)
            offset = int(-T.LOGICAL_W * e)
            surf.blit(self.outgoing, (offset, 0))
        else:
            e = T.ease_out_cubic(t)
            snap = self.outgoing.copy()
            snap.set_alpha(int(255 * (1.0 - e)))
            surf.blit(snap, (0, 0))


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

class App:
    """Owns the window, the main loop, shared services and the scene stack."""

    def __init__(self, title: str = "RopeRush", windowed_size: Tuple[int, int] = (1280, 720),
                 fullscreen: bool = False) -> None:
        pygame.init()

        flags = pygame.SCALED | pygame.RESIZABLE
        if fullscreen:
            flags |= pygame.FULLSCREEN
        self.window = pygame.display.set_mode(windowed_size, flags)
        pygame.display.set_caption(title)

        # The fixed logical canvas everything draws into.
        self.canvas = pygame.Surface((T.LOGICAL_W, T.LOGICAL_H)).convert()
        self.clock = pygame.time.Clock()
        self.running = True

        # Letterbox mapping, recomputed on resize.
        self._scale = 1.0
        self._origin = (0, 0)
        self._recompute_viewport()

        # Scene stack + registry
        self.scenes: Dict[str, Scene] = {}
        self.stack: List[Scene] = []
        self._transition: Optional[_Transition] = None
        self._pending: Optional[Tuple[str, dict, str]] = None

        # Shared services - assigned by main.py after construction.
        self.audio = None
        self.profile = None
        self.vfx = None
        self.backdrop = None
        self.toasts = None

        # Debug/perf
        self.show_fps = False
        self._frame_ms = 0.0

    # ---- viewport ------------------------------------------------------

    def _recompute_viewport(self) -> None:
        """Fit the logical canvas into the window, preserving aspect ratio."""
        win_w, win_h = self.window.get_size()
        scale = min(win_w / T.LOGICAL_W, win_h / T.LOGICAL_H)
        self._scale = scale
        draw_w = int(T.LOGICAL_W * scale)
        draw_h = int(T.LOGICAL_H * scale)
        self._origin = ((win_w - draw_w) // 2, (win_h - draw_h) // 2)

    def to_logical(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Translate a window-space pointer position into logical canvas space."""
        x = (pos[0] - self._origin[0]) / max(1e-6, self._scale)
        y = (pos[1] - self._origin[1]) / max(1e-6, self._scale)
        return int(x), int(y)

    # ---- scene management ----------------------------------------------

    def register(self, name: str, scene: Scene) -> None:
        self.scenes[name] = scene

    @property
    def current(self) -> Optional[Scene]:
        return self.stack[-1] if self.stack else None

    def goto(self, name: str, transition: str = "fade", **kwargs) -> None:
        """Replace the whole stack with `name`, running a transition."""
        self._pending = (name, kwargs, transition)

    def push(self, name: str, **kwargs) -> None:
        """Layer a scene on top (e.g. pause menu) without a transition."""
        scene = self.scenes[name]
        self.stack.append(scene)
        scene.on_enter(**kwargs)
        self._apply_music(scene)

    def pop(self) -> None:
        if len(self.stack) <= 1:
            return
        scene = self.stack.pop()
        scene.on_exit()
        if self.current:
            self._apply_music(self.current)

    def _apply_music(self, scene: Scene) -> None:
        if scene.music and self.audio is not None:
            try:
                self.audio.play_music(scene.music)
            except Exception:
                pass

    def _commit_pending(self) -> None:
        if not self._pending:
            return
        name, kwargs, kind = self._pending
        self._pending = None

        snapshot = self.canvas.copy()
        for s in self.stack:
            s.on_exit()
        self.stack.clear()

        # Toasts belong to the screen that raised them. Carrying them across a
        # transition drops last match's "unlocked" popups on top of the next
        # match's HUD. ToastStack.clear() only starts a dismiss animation, so
        # drop the lists outright - a hard scene change should take them with
        # it rather than fading them over the incoming screen.
        if self.toasts is not None:
            try:
                self.toasts.clear()
                self.toasts.active.clear()
                self.toasts.queue.clear()
            except Exception:
                pass

        scene = self.scenes[name]
        self.stack.append(scene)
        scene.on_enter(**kwargs)
        self._apply_music(scene)
        self._transition = _Transition(snapshot, kind=kind)

    # ---- main loop -----------------------------------------------------

    def _pump_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.VIDEORESIZE:
                self._recompute_viewport()
                continue
            if event.type == pygame.KEYDOWN and event.key == pygame.K_F3:
                self.show_fps = not self.show_fps
                continue

            # Translate pointer coordinates into logical space before the
            # scene ever sees them, so scenes are resolution-agnostic.
            if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                              pygame.MOUSEMOTION):
                event = self._translate_pointer(event)
            elif event.type in (pygame.FINGERDOWN, pygame.FINGERUP, pygame.FINGERMOTION):
                event = self._finger_to_mouse(event)
                if event is None:
                    continue

            if self.current and self._transition is None:
                self.current.handle_event(event)

    def _translate_pointer(self, event: pygame.event.Event) -> pygame.event.Event:
        attrs = {"pos": self.to_logical(event.pos)}
        if hasattr(event, "button"):
            attrs["button"] = event.button
        if hasattr(event, "rel"):
            attrs["rel"] = (int(event.rel[0] / max(1e-6, self._scale)),
                            int(event.rel[1] / max(1e-6, self._scale)))
        if hasattr(event, "buttons"):
            attrs["buttons"] = event.buttons
        if hasattr(event, "touch"):
            attrs["touch"] = event.touch
        return pygame.event.Event(event.type, attrs)

    def _finger_to_mouse(self, event: pygame.event.Event) -> Optional[pygame.event.Event]:
        """Normalize touch events into mouse events so widgets handle one path."""
        mapping = {
            pygame.FINGERDOWN: pygame.MOUSEBUTTONDOWN,
            pygame.FINGERUP: pygame.MOUSEBUTTONUP,
            pygame.FINGERMOTION: pygame.MOUSEMOTION,
        }
        mtype = mapping.get(event.type)
        if mtype is None:
            return None
        # Finger coords are normalized 0..1 across the window.
        win_w, win_h = self.window.get_size()
        pos = self.to_logical((int(event.x * win_w), int(event.y * win_h)))
        return pygame.event.Event(mtype, {"pos": pos, "button": 1, "touch": True})

    def _present(self) -> None:
        """Scale the logical canvas into the window with letterbox bars."""
        win_w, win_h = self.window.get_size()
        draw_w = int(T.LOGICAL_W * self._scale)
        draw_h = int(T.LOGICAL_H * self._scale)

        if (draw_w, draw_h) != (win_w, win_h):
            # Bars are tinted with the sky color so the frame looks deliberate.
            self.window.fill(T.SKY_TOP)

        if (draw_w, draw_h) == (T.LOGICAL_W, T.LOGICAL_H):
            self.window.blit(self.canvas, self._origin)
        else:
            scaled = pygame.transform.smoothscale(self.canvas, (draw_w, draw_h))
            self.window.blit(scaled, self._origin)

        pygame.display.flip()

    def run(self, target_fps: int = 60) -> None:
        while self.running:
            raw_dt = self.clock.tick(target_fps) / 1000.0
            # Clamp so a hitch (or a debugger pause) can't teleport the sim.
            raw_dt = min(raw_dt, 1.0 / 20.0)

            self._pump_events()
            if not self.running:
                break

            time_scale = 1.0
            if self.vfx is not None:
                try:
                    time_scale = self.vfx.get_time_scale()
                except Exception:
                    time_scale = 1.0
            dt = raw_dt * time_scale

            if self.audio is not None:
                try:
                    self.audio.update(raw_dt)
                except Exception:
                    pass

            if self._transition is not None:
                self._transition.update(raw_dt)
                if self._transition.done:
                    self._transition = None

            if self.current:
                self.current.update(dt)

            # Draw: walk down to the last opaque scene, then paint upward.
            start = len(self.stack) - 1
            while start > 0 and self.stack[start].transparent:
                start -= 1
            self.canvas.fill(T.SKY_TOP)
            for scene in self.stack[start:]:
                scene.draw(self.canvas)

            if self.toasts is not None:
                try:
                    self.toasts.update(raw_dt)
                    self.toasts.draw(self.canvas)
                except Exception:
                    pass

            if self._transition is not None:
                self._transition.draw(self.canvas)

            if self.show_fps:
                self._draw_debug()

            self._present()
            self._commit_pending()

            self._frame_ms = self.clock.get_rawtime()

        pygame.quit()

    def _draw_debug(self) -> None:
        font = R.font_text(T.T_MICRO, bold=True)
        txt = f"{self.clock.get_fps():5.1f} fps   {self._frame_ms:2d} ms"
        R.draw_text(self.canvas, txt, font, T.OK,
                    topleft=(T.SAFE_L, T.LOGICAL_H - 28))

    def quit(self) -> None:
        self.running = False

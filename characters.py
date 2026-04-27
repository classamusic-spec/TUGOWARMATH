"""Cartoon kid characters for Tug-of-War Math.

Each kid is a forward-facing 3/4-view cartoon: round head with a headband,
patterned shirt, dark pants, bright sneakers, and both arms extended toward
the rope. The animation cycle leans the upper body backward (using the
rope's resistance for weight) and bobs the arms in time with their phase.

Drawn entirely with pygame primitives - no image assets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pygame

Color = Tuple[int, int, int]


def _shade(c: Color, delta: int) -> Color:
    return (
        max(0, min(255, c[0] + delta)),
        max(0, min(255, c[1] + delta)),
        max(0, min(255, c[2] + delta)),
    )


@dataclass
class Kid:
    x: float            # foot anchor x (between the feet)
    y: float            # foot anchor y (ground line)
    facing: int         # +1 right-team (faces left), -1 left-team (faces right)
    phase: float        # animation phase offset
    skin: Color
    shirt: Color
    pants: Color
    height: float = 130.0
    shoe: Color = (220, 60, 60)
    headband: Color = (250, 240, 240)
    hair: Color = (40, 25, 15)
    pattern_seed: int = 0


def make_team(side: str, count: int, ground_y: int, x_range: Tuple[int, int]) -> List[Kid]:
    facing = +1 if side == "right" else -1
    rng = random.Random(hash(("team", side)) & 0xFFFF)

    skins = [(255, 220, 178), (240, 196, 145), (210, 160, 120), (160, 110, 80), (110, 75, 55)]
    if side == "left":
        # warm reds / oranges
        shirts = [(220, 70, 70), (240, 110, 90), (200, 50, 100), (240, 140, 80)]
        bands = [(250, 240, 230), (255, 220, 80), (255, 210, 200)]
        shoes = [(230, 50, 60), (250, 200, 60), (40, 40, 50), (250, 250, 250)]
    else:
        # cool blues / teals
        shirts = [(60, 130, 230), (100, 180, 240), (50, 90, 200), (90, 200, 220)]
        bands = [(255, 255, 255), (60, 200, 255), (255, 220, 80)]
        shoes = [(40, 80, 200), (250, 250, 250), (40, 40, 50), (250, 200, 60)]
    pants_choices = [(45, 55, 85), (35, 35, 55), (60, 45, 35), (40, 40, 60)]
    hair_choices = [(40, 25, 15), (90, 60, 30), (180, 120, 60), (30, 20, 10), (50, 30, 20)]

    kids: List[Kid] = []
    spacing = (x_range[1] - x_range[0]) / max(1, count - 1) if count > 1 else 0
    for i in range(count):
        x = x_range[0] + spacing * i
        kids.append(
            Kid(
                x=x,
                y=ground_y,
                facing=facing,
                phase=rng.uniform(0, math.tau),
                skin=rng.choice(skins),
                shirt=rng.choice(shirts),
                pants=rng.choice(pants_choices),
                height=rng.uniform(125, 150),
                shoe=rng.choice(shoes),
                headband=rng.choice(bands),
                hair=rng.choice(hair_choices),
                pattern_seed=rng.randint(0, 9999),
            )
        )
    return kids


# --------------------------------------------------------------------------- #
# Drawing primitives
# --------------------------------------------------------------------------- #

def _filled_rounded_rect(
    surf: pygame.Surface,
    color: Color,
    rect: pygame.Rect,
    radius: int,
    outline: Optional[Color] = (40, 30, 30),
    outline_w: int = 2,
) -> None:
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if outline:
        pygame.draw.rect(surf, outline, rect, outline_w, border_radius=radius)


def _filled_circle(
    surf: pygame.Surface,
    color: Color,
    pos: Tuple[float, float],
    radius: float,
    outline: Optional[Color] = (40, 30, 30),
    outline_w: int = 2,
) -> None:
    pygame.draw.circle(surf, color, (int(pos[0]), int(pos[1])), int(radius))
    if outline and outline_w > 0:
        pygame.draw.circle(surf, outline, (int(pos[0]), int(pos[1])), int(radius), outline_w)


def _draw_thick_segment(
    surf: pygame.Surface,
    color: Color,
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    width: int,
    outline: Optional[Color] = (40, 30, 30),
) -> None:
    """A thick line with rounded ends (drawn by capping with circles)."""
    pygame.draw.line(surf, color, p0, p1, width)
    pygame.draw.circle(surf, color, (int(p0[0]), int(p0[1])), width // 2)
    pygame.draw.circle(surf, color, (int(p1[0]), int(p1[1])), width // 2)
    if outline:
        # Outline the segment by drawing a slightly thicker dark line first
        # would require redraw order; cheaper: just outline the end caps lightly
        pygame.draw.circle(surf, outline, (int(p0[0]), int(p0[1])), width // 2, 1)
        pygame.draw.circle(surf, outline, (int(p1[0]), int(p1[1])), width // 2, 1)


# --------------------------------------------------------------------------- #
# Body parts
# --------------------------------------------------------------------------- #

def _draw_shadow(surf: pygame.Surface, x: float, y: float, w: float) -> None:
    s = pygame.Surface((int(w), int(w * 0.3)), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0, 0, 0, 90), s.get_rect())
    surf.blit(s, s.get_rect(center=(int(x), int(y + 4))))


def _draw_sneaker(
    surf: pygame.Surface, cx: float, cy: float, w: float, h: float, color: Color, facing: int
) -> None:
    # Sole
    sole = pygame.Rect(0, 0, int(w), int(h * 0.45))
    sole.midbottom = (int(cx), int(cy))
    pygame.draw.rect(surf, (245, 245, 245), sole, border_radius=int(h * 0.2))
    pygame.draw.rect(surf, (40, 30, 30), sole, 1, border_radius=int(h * 0.2))
    # Upper
    upper = pygame.Rect(0, 0, int(w * 0.92), int(h * 0.7))
    upper.midbottom = (int(cx + facing * 1), int(cy - h * 0.35))
    pygame.draw.rect(surf, color, upper, border_radius=int(h * 0.3))
    pygame.draw.rect(surf, (40, 30, 30), upper, 1, border_radius=int(h * 0.3))
    # Tongue / stripe
    stripe = pygame.Rect(0, 0, int(w * 0.6), int(h * 0.18))
    stripe.center = (int(cx + facing * 1), int(cy - h * 0.55))
    pygame.draw.rect(surf, _shade(color, 60), stripe, border_radius=4)


def _draw_leg(
    surf: pygame.Surface,
    hip: Tuple[float, float],
    foot: Tuple[float, float],
    width: float,
    color: Color,
) -> None:
    # Trapezoidal leg from hip to ankle
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    length = max(1.0, math.hypot(dx, dy))
    nx, ny = -dy / length, dx / length  # perpendicular
    top_w = width * 0.6
    bot_w = width * 0.5
    pts = [
        (hip[0] + nx * top_w, hip[1] + ny * top_w),
        (hip[0] - nx * top_w, hip[1] - ny * top_w),
        (foot[0] - nx * bot_w, foot[1] - ny * bot_w),
        (foot[0] + nx * bot_w, foot[1] + ny * bot_w),
    ]
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, (30, 25, 25), pts, 2)


def _draw_torso(
    surf: pygame.Surface,
    cx: float,
    cy: float,
    w: float,
    h: float,
    shirt: Color,
    pattern_seed: int,
) -> pygame.Rect:
    rect = pygame.Rect(0, 0, int(w), int(h))
    rect.center = (int(cx), int(cy))
    radius = int(min(w, h) * 0.28)
    _filled_rounded_rect(surf, shirt, rect, radius)

    # Pattern: soft tie-dye-ish blobs (deterministic per kid)
    rng = random.Random(pattern_seed)
    for _ in range(6):
        bx = rect.left + rng.randint(int(w * 0.1), int(w * 0.9))
        by = rect.top + rng.randint(int(h * 0.15), int(h * 0.85))
        br = rng.randint(int(min(w, h) * 0.06), int(min(w, h) * 0.15))
        delta = rng.choice([-45, -30, 30, 45])
        col = _shade(shirt, delta)
        # Clip blob to torso rect
        clip = surf.get_clip()
        surf.set_clip(rect)
        pygame.draw.circle(surf, col, (bx, by), br)
        surf.set_clip(clip)

    # Collar V
    collar_pts = [
        (rect.centerx - w * 0.15, rect.top + 2),
        (rect.centerx + w * 0.15, rect.top + 2),
        (rect.centerx, rect.top + h * 0.18),
    ]
    pygame.draw.polygon(surf, _shade(shirt, -50), collar_pts)
    return rect


def _draw_face(
    surf: pygame.Surface,
    cx: float,
    cy: float,
    radius: float,
    facing: int,
    skin: Color,
    hair: Color,
    headband: Color,
    pull_intensity: float,
    losing: float,
    t: float,
    phase: float,
) -> None:
    # Hair: a thin crescent peeking above the headband, plus side tufts.
    hair_rect = pygame.Rect(0, 0, int(radius * 2.05), int(radius * 1.55))
    hair_rect.center = (int(cx), int(cy - radius * 0.05))
    pygame.draw.ellipse(surf, hair, hair_rect)
    # Side tufts near the temples
    pygame.draw.circle(surf, hair, (int(cx - radius * 0.85), int(cy - radius * 0.30)), max(2, int(radius * 0.18)))
    pygame.draw.circle(surf, hair, (int(cx + radius * 0.85), int(cy - radius * 0.30)), max(2, int(radius * 0.18)))

    # Head (skin)
    _filled_circle(surf, skin, (cx, cy), radius)
    # Cheek blush
    blush = _shade(skin, -30)
    cheek_r = max(2, int(radius * 0.18))
    for sgn in (-1, +1):
        pygame.draw.circle(
            surf, blush,
            (int(cx + sgn * radius * 0.55), int(cy + radius * 0.25)),
            cheek_r,
        )

    # Headband across the forehead
    band_h = max(5, int(radius * 0.32))
    band_rect = pygame.Rect(0, 0, int(radius * 2.15), band_h)
    band_rect.center = (int(cx), int(cy - radius * 0.55))
    pygame.draw.rect(surf, headband, band_rect, border_radius=int(band_h / 2))
    pygame.draw.rect(surf, (40, 30, 30), band_rect, 1, border_radius=int(band_h / 2))
    # Headband stripe
    stripe = pygame.Rect(band_rect.left + 4, band_rect.centery - 2, band_rect.width - 8, 3)
    pygame.draw.rect(surf, _shade(headband, -60), stripe)
    # Knot on the back-of-head side (opposite of facing)
    knot_x = band_rect.left + band_h if facing > 0 else band_rect.right - band_h
    pygame.draw.circle(surf, headband, (int(knot_x), band_rect.centery), band_h)
    pygame.draw.circle(surf, (40, 30, 30), (int(knot_x), band_rect.centery), band_h, 1)
    # Tassel
    tx = knot_x - facing * band_h * 0.7
    ty = band_rect.centery + band_h * 0.3
    pygame.draw.line(surf, headband, (knot_x, band_rect.centery), (tx, ty + 8), 3)

    # Eyes - both visible, slight 3/4 turn means one a bit smaller
    eye_y = cy + radius * 0.05
    eye_dx = radius * 0.32
    near_x = cx + facing * eye_dx       # eye on the side toward the rope
    far_x = cx - facing * eye_dx * 0.85
    near_r = radius * 0.18
    far_r = radius * 0.16
    # Whites
    _filled_circle(surf, (255, 255, 255), (near_x, eye_y), near_r, outline=(40, 30, 30), outline_w=1)
    _filled_circle(surf, (255, 255, 255), (far_x, eye_y), far_r, outline=(40, 30, 30), outline_w=1)
    # Pupils - dart slightly toward the rope (forward)
    look = facing * radius * 0.05
    pygame.draw.circle(surf, (30, 30, 60), (int(near_x + look), int(eye_y + 1)), max(2, int(near_r * 0.55)))
    pygame.draw.circle(surf, (30, 30, 60), (int(far_x + look), int(eye_y + 1)), max(2, int(far_r * 0.55)))
    # Highlights
    pygame.draw.circle(surf, (255, 255, 255), (int(near_x + look + 1), int(eye_y - 2)), max(1, int(near_r * 0.22)))
    pygame.draw.circle(surf, (255, 255, 255), (int(far_x + look + 1), int(eye_y - 2)), max(1, int(far_r * 0.22)))

    # Eyebrows - tense more when losing or pulling hard
    tense = max(pull_intensity, losing)
    brow_y = eye_y - radius * 0.34
    brow_color = (40, 25, 15)
    brow_w = max(2, int(radius * 0.12))
    # Inner brow drops when tense
    inner_drop = tense * radius * 0.18
    pygame.draw.line(
        surf, brow_color,
        (near_x - radius * 0.18, brow_y + (inner_drop if facing > 0 else 0)),
        (near_x + radius * 0.18, brow_y + (0 if facing > 0 else inner_drop)),
        brow_w,
    )
    pygame.draw.line(
        surf, brow_color,
        (far_x - radius * 0.16, brow_y + (inner_drop if facing < 0 else 0)),
        (far_x + radius * 0.16, brow_y + (0 if facing < 0 else inner_drop)),
        brow_w,
    )

    # Nose - a tiny shaded curve on the forward side
    nose_x = cx + facing * radius * 0.05
    nose_y = cy + radius * 0.18
    pygame.draw.line(
        surf, _shade(skin, -50),
        (nose_x, nose_y - 2), (nose_x + facing * 4, nose_y + 4), 2,
    )

    # Mouth - opens wider with effort/losing; smile when calm
    mouth_open = max(pull_intensity * 0.75, losing) + 0.12 * (math.sin(t * 9 + phase) * 0.5 + 0.5)
    mouth_open = min(1.0, mouth_open)
    mouth_y = cy + radius * 0.5
    if mouth_open < 0.25:
        # Tiny smile arc
        arc = pygame.Rect(int(cx - radius * 0.35), int(mouth_y - radius * 0.25), int(radius * 0.7), int(radius * 0.45))
        pygame.draw.arc(surf, (60, 30, 30), arc, math.pi * 1.05, math.pi * 1.95, 3)
    else:
        mw = max(6, int(radius * 0.55))
        mh = max(4, int(radius * 0.18 + mouth_open * radius * 0.55))
        m = pygame.Rect(0, 0, mw, mh)
        m.center = (int(cx), int(mouth_y))
        pygame.draw.ellipse(surf, (60, 25, 25), m)
        pygame.draw.ellipse(surf, (30, 15, 15), m, 2)
        # Tongue
        if mouth_open > 0.4:
            tongue = m.inflate(-mw // 3, -mh // 2)
            tongue.bottom = m.bottom - 2
            pygame.draw.ellipse(surf, (220, 90, 100), tongue)
        # Top teeth strip
        if mouth_open > 0.3:
            teeth = pygame.Rect(0, 0, m.width - 6, 4)
            teeth.midtop = (m.centerx, m.top + 2)
            pygame.draw.rect(surf, (255, 250, 240), teeth)

    # Sweat drop when really losing
    if losing > 0.55:
        sweat_x = cx + facing * radius * 0.85
        sweat_y = cy - radius * 0.15 + math.sin(t * 4 + phase) * 2
        pygame.draw.circle(surf, (160, 220, 255), (int(sweat_x), int(sweat_y)), 4)
        pygame.draw.circle(surf, (90, 160, 220), (int(sweat_x), int(sweat_y)), 4, 1)


# --------------------------------------------------------------------------- #
# Main draw
# --------------------------------------------------------------------------- #

def draw_kid(
    surface: pygame.Surface,
    kid: Kid,
    t: float,
    pull_intensity: float,
    losing: float,
    rope_y: float,
) -> Tuple[int, int]:
    """Draw a cartoon kid pulling a rope; return (x, y) where their hands grip."""
    H = kid.height
    facing = kid.facing
    foot_y = kid.y

    # Animation: lean back away from the rope. Backward direction = -facing.
    pull_phase = math.sin(t * 5 + kid.phase)
    base_lean = 0.45 + pull_intensity * 0.30
    lean = base_lean + 0.10 * pull_phase
    # lean_dx > 0 means upper body shifts toward +x.
    # Right team (facing=+1, rope to the left) leans to the right: lean_dx = +lean*H*0.18
    # Left team (facing=-1, rope to the right) leans to the left: lean_dx = -lean*H*0.18
    lean_dx = facing * lean * H * 0.18

    # Anchors
    feet_spread = H * 0.18
    foot_back_x = kid.x - facing * feet_spread * 0.55
    foot_front_x = kid.x + facing * feet_spread * 0.55

    hip_cx = kid.x + lean_dx * 0.35
    hip_cy = foot_y - H * 0.42
    torso_cx = kid.x + lean_dx * 0.85
    torso_cy = foot_y - H * 0.58
    shoulder_y = foot_y - H * 0.70
    head_cx = kid.x + lean_dx * 1.20
    head_cy = foot_y - H * 0.86
    head_r = H * 0.16

    # Hands grip the rope on the side toward the center of the screen
    hand_x = kid.x + facing * H * 0.42
    hand_y = rope_y + math.sin(t * 8 + kid.phase) * 2

    # ---- shadow ----
    _draw_shadow(surface, kid.x, foot_y, H * 0.55)

    # ---- legs + sneakers ----
    sneak_w = H * 0.22
    sneak_h = H * 0.10
    # Back leg first (further from rope) so front leg overlaps it
    hip_back = (hip_cx - feet_spread * 0.20, hip_cy + 2)
    hip_front = (hip_cx + feet_spread * 0.20, hip_cy + 2)
    foot_back = (foot_back_x, foot_y - sneak_h * 0.45)
    foot_front = (foot_front_x, foot_y - sneak_h * 0.45)
    _draw_leg(surface, hip_back, foot_back, H * 0.16, kid.pants)
    _draw_leg(surface, hip_front, foot_front, H * 0.18, kid.pants)
    _draw_sneaker(surface, foot_back_x, foot_y, sneak_w, sneak_h, kid.shoe, facing)
    _draw_sneaker(surface, foot_front_x, foot_y, sneak_w, sneak_h, kid.shoe, facing)

    # ---- torso (shirt) ----
    torso_w = H * 0.42
    torso_h = H * 0.36
    torso_rect = _draw_torso(
        surface, torso_cx, torso_cy, torso_w, torso_h, kid.shirt, kid.pattern_seed
    )

    # ---- back arm (drawn first so front arm overlaps it) ----
    back_shoulder = (torso_cx - facing * torso_w * 0.30, shoulder_y + H * 0.04)
    back_hand = (hand_x - facing * 8, hand_y + 4)
    arm_w = max(6, int(H * 0.10))
    _draw_thick_segment(surface, kid.shirt, back_shoulder, back_hand, arm_w)

    # ---- front arm ----
    front_shoulder = (torso_cx + facing * torso_w * 0.10, shoulder_y - H * 0.02)
    front_hand = (hand_x, hand_y)
    _draw_thick_segment(surface, kid.shirt, front_shoulder, front_hand, arm_w + 1)

    # ---- forearms transition to skin near the wrist ----
    # Draw a small skin sleeve cuff stub then the fist
    for shoulder, hand in ((back_shoulder, back_hand), (front_shoulder, front_hand)):
        # Unit vec
        dx, dy = hand[0] - shoulder[0], hand[1] - shoulder[1]
        L = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / L, dy / L
        wrist = (hand[0] - ux * arm_w * 0.4, hand[1] - uy * arm_w * 0.4)
        pygame.draw.circle(surface, kid.skin, (int(wrist[0]), int(wrist[1])), int(arm_w * 0.55))

    # ---- fists gripping rope ----
    fist_r = max(6, int(H * 0.075))
    _filled_circle(surface, kid.skin, back_hand, fist_r, outline=(40, 30, 25), outline_w=2)
    _filled_circle(surface, kid.skin, front_hand, fist_r + 1, outline=(40, 30, 25), outline_w=2)
    # Knuckle lines on the front fist for definition
    for i in range(3):
        kx = front_hand[0] - fist_r * 0.5 + i * fist_r * 0.5
        pygame.draw.line(
            surface, _shade(kid.skin, -50),
            (kx, front_hand[1] - fist_r * 0.3),
            (kx, front_hand[1] + fist_r * 0.1),
            1,
        )

    # ---- neck ----
    neck = pygame.Rect(0, 0, int(H * 0.10), int(H * 0.08))
    neck.midbottom = (int(head_cx + lean_dx * 0.05), int(torso_rect.top + 4))
    pygame.draw.rect(surface, kid.skin, neck)
    pygame.draw.rect(surface, _shade(kid.skin, -60), neck, 1)

    # ---- head + face ----
    _draw_face(
        surface,
        head_cx,
        head_cy,
        head_r,
        facing,
        kid.skin,
        kid.hair,
        kid.headband,
        pull_intensity,
        losing,
        t,
        kid.phase,
    )

    return int(front_hand[0]), int(front_hand[1])

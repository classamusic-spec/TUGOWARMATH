"""Procedural stick-figure kids that lean back and pull a rope.

Drawn entirely with pygame primitives so the game has zero asset dependencies.
Each kid has a slightly different palette and an animation phase offset so the
team looks like a small, frantic crowd rather than a single body.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import pygame

Color = Tuple[int, int, int]


@dataclass
class Kid:
    x: float            # foot anchor x
    y: float            # foot anchor y (ground line)
    facing: int         # +1 for right team (faces left), -1 for left team (faces right)
    phase: float        # animation phase offset
    skin: Color
    shirt: Color
    pants: Color
    height: float = 110.0


def make_team(side: str, count: int, ground_y: int, x_range: Tuple[int, int]) -> List[Kid]:
    """Build a small team of kids on one side of the screen.

    side: 'left' or 'right'.
    """
    facing = +1 if side == "right" else -1  # right team faces left (toward rope)
    skins = [(255, 220, 178), (240, 196, 145), (200, 150, 110), (140, 95, 65)]
    if side == "left":
        shirts = [(220, 60, 60), (240, 100, 80), (200, 40, 90), (255, 140, 100)]
    else:
        shirts = [(60, 130, 230), (90, 180, 240), (50, 90, 200), (110, 200, 240)]
    pants = [(60, 60, 90), (40, 40, 60), (90, 70, 50), (70, 50, 90)]

    kids: List[Kid] = []
    rng = random.Random(hash(side) & 0xFFFF)
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
                pants=rng.choice(pants),
                height=rng.uniform(95, 125),
            )
        )
    return kids


def _draw_mouth(surface: pygame.Surface, cx: int, cy: int, openness: float, color: Color) -> None:
    """Open mouth = screaming. openness 0..1."""
    w = max(4, int(8 + openness * 6))
    h = max(2, int(2 + openness * 14))
    rect = pygame.Rect(0, 0, w, h)
    rect.center = (cx, cy)
    pygame.draw.ellipse(surface, color, rect)


def draw_kid(
    surface: pygame.Surface,
    kid: Kid,
    t: float,
    pull_intensity: float,
    losing: float,
    rope_y: float,
) -> Tuple[int, int]:
    """Draw a kid and return the (x, y) screen point where their hands grip the rope."""
    # Animation: lean back with a sinusoid; bigger amplitude when team is pulling hard
    base_lean = 0.35 + pull_intensity * 0.25
    lean = base_lean + 0.15 * math.sin(t * 6 + kid.phase) * (0.4 + pull_intensity)
    # Lean direction: left team leans left (away from rope on right), right team mirrors.
    lean_dir = -kid.facing  # facing +1 (right team) leans to the right (positive x), so dir = -facing? we'll fix sign
    # We want kids to lean BACKWARD (away from rope center).
    # left team (facing = -1, rope to the right) -> lean to the left -> dx negative
    # right team (facing = +1, rope to the left) -> lean to the right -> dx positive
    # so lean_dx = -facing * lean * height -> for facing=-1 gives +lean*h ... wrong.
    # We want left team dx negative: with facing=-1 we need dx = facing*lean*h = -1*lean*h. Good.
    # Right team dx positive: facing=+1 -> dx = +lean*h. Good.
    head_dx = kid.facing * lean * kid.height * 0.3
    torso_top = (kid.x + head_dx * 0.6, kid.y - kid.height * 0.55)
    head_pos = (kid.x + head_dx, kid.y - kid.height * 0.85)
    head_radius = max(8, int(kid.height * 0.13))

    # Legs braced
    leg_spread = kid.height * 0.18
    foot_back = (kid.x - kid.facing * leg_spread, kid.y)
    foot_front = (kid.x + kid.facing * leg_spread * 0.6, kid.y)

    # Body
    pygame.draw.line(surface, kid.pants, foot_back, (kid.x, kid.y - kid.height * 0.45), 6)
    pygame.draw.line(surface, kid.pants, foot_front, (kid.x, kid.y - kid.height * 0.45), 6)
    pygame.draw.line(
        surface,
        kid.shirt,
        (kid.x, kid.y - kid.height * 0.45),
        torso_top,
        9,
    )

    # Arms reach out toward rope - hands meet at the rope line in front of the kid
    hand_x = kid.x + kid.facing * (kid.height * 0.35) - head_dx * 0.4
    # Hands hover near rope_y with a small bob
    hand_y = rope_y + math.sin(t * 8 + kid.phase) * 3
    pygame.draw.line(surface, kid.shirt, torso_top, (hand_x, hand_y), 6)
    # Second arm slightly offset for depth
    pygame.draw.line(
        surface,
        kid.shirt,
        torso_top,
        (hand_x - kid.facing * 6, hand_y + 4),
        5,
    )

    # Head
    pygame.draw.circle(surface, kid.skin, (int(head_pos[0]), int(head_pos[1])), head_radius)
    pygame.draw.circle(surface, (40, 30, 20), (int(head_pos[0]), int(head_pos[1])), head_radius, 2)

    # Hair (a couple of dark strokes)
    pygame.draw.arc(
        surface,
        (40, 25, 15),
        pygame.Rect(
            int(head_pos[0] - head_radius),
            int(head_pos[1] - head_radius - 2),
            head_radius * 2,
            head_radius * 2,
        ),
        math.pi * 0.1,
        math.pi * 0.9,
        3,
    )

    # Eyes
    eye_offset = head_radius * 0.4
    eye_y = int(head_pos[1] - 2)
    pygame.draw.circle(surface, (0, 0, 0), (int(head_pos[0] - eye_offset), eye_y), 2)
    pygame.draw.circle(surface, (0, 0, 0), (int(head_pos[0] + eye_offset), eye_y), 2)

    # Mouth - open more when losing
    mouth_open = max(pull_intensity * 0.5, losing) + 0.1 * (math.sin(t * 9 + kid.phase) * 0.5 + 0.5)
    _draw_mouth(surface, int(head_pos[0]), int(head_pos[1] + head_radius * 0.45), min(1.0, mouth_open), (60, 20, 20))

    return int(hand_x), int(hand_y)

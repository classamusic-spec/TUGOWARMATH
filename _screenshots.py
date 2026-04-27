"""Headless screenshot harness - drives the game under SDL dummy and saves PNGs.

Not part of the shipped game; used only to verify rendering in environments
without a display.
"""

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import pygame
import main as m


def step(g, frames=1):
    for _ in range(frames):
        g.update(1 / 60)
        g.draw()


def save(g, name):
    pygame.image.save(g.screen, name)
    print(f"  wrote {name}  ({os.path.getsize(name)} bytes)")


def main():
    random.seed(42)
    g = m.Game()

    # ---- menu ----
    step(g, 30)
    save(g, "shot_menu.png")

    # ---- mid-game (some pulls applied) ----
    g.start_round()
    step(g, 60)
    # Have RED solve a couple in a row
    for _ in range(2):
        g._submit("left", str(g.left.problem.answer))
        step(g, 12)
    # BLUE gets one wrong
    g._submit("right", "0")
    step(g, 24)
    # Type something into BLUE's calc display so we can see the buffer
    for ch in "12+":
        g.right.calc.press(ch)
    step(g, 6)
    save(g, "shot_play.png")

    # ---- close to a win for RED ----
    for _ in range(8):
        if g.state != g.STATE_PLAY:
            break
        g._submit("left", str(g.left.problem.answer))
        step(g, 6)
    save(g, "shot_late_play.png")

    # ---- game over ----
    while g.state == g.STATE_PLAY:
        g._submit("left", str(g.left.problem.answer))
        step(g, 4)
    step(g, 30)
    save(g, "shot_over.png")


if __name__ == "__main__":
    main()

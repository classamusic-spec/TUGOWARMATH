"""Procedurally synthesized sound effects for Tug-of-War Math.

We don't ship any audio files - everything is generated with numpy at startup.
If audio init fails (no sound device, etc.) the engine still runs silently.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

import numpy as np
import pygame

SAMPLE_RATE = 44100


def _to_stereo_int16(mono: np.ndarray) -> np.ndarray:
    mono = np.clip(mono, -1.0, 1.0)
    samples = (mono * 32767).astype(np.int16)
    return np.column_stack((samples, samples))


def _envelope(length: int, attack: float = 0.01, release: float = 0.2) -> np.ndarray:
    env = np.ones(length)
    a = max(1, int(attack * SAMPLE_RATE))
    r = max(1, int(release * SAMPLE_RATE))
    a = min(a, length // 2)
    r = min(r, length - a)
    env[:a] = np.linspace(0.0, 1.0, a)
    env[length - r:] = np.linspace(1.0, 0.0, r)
    return env


def _kid_scream(duration: float = 0.55, base_freq: float = 600.0) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    # Pitch wobbles up then down - a kid yelping
    pitch = base_freq * (1.0 + 0.25 * np.sin(2 * np.pi * 6 * t))
    pitch *= np.linspace(1.0, 1.4, n)
    phase = 2 * np.pi * np.cumsum(pitch) / SAMPLE_RATE
    saw = 2 * (phase / (2 * np.pi) - np.floor(0.5 + phase / (2 * np.pi)))
    sine = np.sin(phase * 2)
    noise = np.random.uniform(-1, 1, n) * 0.15
    sig = 0.55 * saw + 0.25 * sine + noise
    sig *= _envelope(n, 0.02, 0.18)
    return sig * 0.7


def _crowd_scream(duration: float = 1.0, lower: bool = False) -> np.ndarray:
    """Layer several kid screams at slightly different pitches/offsets."""
    n = int(duration * SAMPLE_RATE)
    out = np.zeros(n)
    voices = 6
    for _ in range(voices):
        f = random.uniform(420, 780) * (0.85 if lower else 1.0)
        d = random.uniform(0.4, 0.7)
        offset = random.uniform(0.0, max(0.001, duration - d - 0.01))
        clip = _kid_scream(d, f)
        start = int(offset * SAMPLE_RATE)
        end = start + len(clip)
        end = min(end, n)
        out[start:end] += clip[: end - start] * (0.5 / voices ** 0.5)
    out = np.clip(out, -1, 1)
    return out


def _whistle(duration: float = 0.45) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = np.linspace(1100, 1500, n)
    sig = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE)
    sig += 0.05 * np.random.uniform(-1, 1, n)
    sig *= _envelope(n, 0.02, 0.12)
    return sig * 0.5


def _whoosh(duration: float = 0.25) -> np.ndarray:
    n = int(duration * SAMPLE_RATE)
    noise = np.random.uniform(-1, 1, n)
    # Cheap low-pass: cumulative average for a swoosh
    kernel_size = 50
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(noise, kernel, mode="same")
    sig = smooth * np.linspace(0.2, 1.0, n)
    sig *= _envelope(n, 0.01, 0.15)
    return sig * 0.6


def _ding(correct: bool = True) -> np.ndarray:
    duration = 0.28
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    if correct:
        f1, f2 = 660, 990
    else:
        f1, f2 = 330, 220
    half = n // 2
    sig = np.zeros(n)
    sig[:half] = np.sin(2 * np.pi * f1 * t[:half])
    sig[half:] = np.sin(2 * np.pi * f2 * t[half:])
    sig *= _envelope(n, 0.005, 0.18)
    return sig * 0.4


def _tick() -> np.ndarray:
    duration = 0.05
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    sig = np.sin(2 * np.pi * 1800 * t) * np.exp(-40 * t)
    return sig * 0.3


def _click() -> np.ndarray:
    duration = 0.04
    n = int(duration * SAMPLE_RATE)
    sig = np.random.uniform(-1, 1, n) * np.linspace(1, 0, n)
    return sig * 0.3


class SoundBank:
    """Lazy, fault-tolerant sound bank. play(name) is a no-op if audio failed."""

    def __init__(self) -> None:
        self._enabled = False
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        try:
            pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 512)
            pygame.mixer.init()
            self._enabled = True
        except pygame.error:
            self._enabled = False
            return

        self._build()

    def _add(self, name: str, mono: np.ndarray) -> None:
        try:
            arr = _to_stereo_int16(mono)
            self._sounds[name] = pygame.sndarray.make_sound(arr)
        except Exception:
            pass

    def _build(self) -> None:
        random.seed(7)
        np.random.seed(7)
        self._add("scream_left", _crowd_scream(1.0, lower=False))
        self._add("scream_right", _crowd_scream(1.0, lower=True))
        self._add("whistle", _whistle())
        self._add("whoosh", _whoosh())
        self._add("right", _ding(True))
        self._add("wrong", _ding(False))
        self._add("tick", _tick())
        self._add("click", _click())

    def play(self, name: str, volume: float = 1.0) -> None:
        if not self._enabled:
            return
        snd = self._sounds.get(name)
        if snd is None:
            return
        snd.set_volume(max(0.0, min(1.0, volume)))
        snd.play()

"""Procedural audio engine for RopeRush - synthesis, mixing and music.

We ship ZERO audio files. Every sound effect and every music bed in the game
is synthesized with numpy the moment the engine boots, then handed to pygame's
mixer as an in-memory buffer.

Why bother
----------
A tug-of-war game lives or dies on feel: the rope has to creak, the crowd has
to swell, and a five-in-a-row streak has to *climb*. Canned beeps read as
cheap. So this module implements a small synthesis toolkit - ADSR envelopes,
FM and additive voices, detuned oscillator stacks, formant vowels, IIR and
FFT filters, block-based filter sweeps and a Schroeder reverb - and then uses
it to design ~35 distinct effects plus three loopable music beds.

Architecture
------------
    numpy float buffers  ->  pygame.mixer.Sound  ->  channels
                             (SFX)                   (auto-allocated)
                             (music)                 (2 reserved, crossfaded)

* SFX are built synchronously at init (fast, well under a second).
* Music beds are much longer, so they are rendered on a background thread and
  converted to Sounds on the next `update()` call. Startup never blocks on
  them; a `play_music()` issued before the bed is ready is queued.
* Everything is fault tolerant. If there is no audio device - or pygame's
  mixer refuses to initialize, or numpy chokes - the engine flips to a silent
  mode where every public method is a well-behaved no-op.

Conventions
-----------
* Internal buffers are float64/float32 in the range [-1, 1]. Mono buffers are
  shape (n,), stereo buffers are shape (n, 2).
* `play(name, volume, pitch)` resamples on demand, so `correct_streak` can
  literally walk up a pentatonic scale as the streak grows.
* Legacy names used by older call sites (`click`, `right`, `whistle`, ...)
  are aliased onto the new bank, and `SoundBank` is an alias of `AudioEngine`.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pygame is a hard dependency of the game, but never of this module
    import pygame
except Exception:  # pragma: no cover - only hit on a broken install
    pygame = None  # type: ignore


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 44100
BIT_DEPTH = -16          # signed 16-bit, what pygame.sndarray expects
NUM_CHANNELS = 2         # stereo
MIXER_BUFFER = 512       # low latency; a tap must feel instant on a phone
POLYPHONY = 28           # total mixer channels
MUSIC_CHANNELS = 2       # reserved pair, used for crossfading beds

# Equal temperament helper: multiply a frequency by this to move one semitone.
SEMITONE = 2.0 ** (1.0 / 12.0)

# A major pentatonic scale in semitone offsets - used by combo / streak sounds
# because it is impossible to make it sound wrong, no matter how far it climbs.
PENTATONIC = (0, 2, 4, 7, 9, 12, 14, 16, 19, 21, 24)

# Deterministic noise so the bank sounds identical on every launch.
_SEED = 0x50FA


# --------------------------------------------------------------------------- #
# DSP toolkit - envelopes
# --------------------------------------------------------------------------- #

def _nsamp(dur: float) -> int:
    """Sample count for a duration in seconds (always at least 1)."""
    return max(1, int(round(dur * SAMPLE_RATE)))


def _time(n: int) -> np.ndarray:
    """Time axis in seconds for `n` samples."""
    return np.arange(n, dtype=np.float64) / SAMPLE_RATE


def adsr(
    n: int,
    attack: float = 0.005,
    decay: float = 0.08,
    sustain: float = 0.6,
    release: float = 0.2,
    curve: float = 1.8,
) -> np.ndarray:
    """Classic four-stage envelope.

    `curve` shapes the decay/release segments: 1.0 is linear, larger values
    bend toward an exponential fall, which is what real acoustic sources do.
    Stage lengths are clamped so the envelope always fits in `n` samples.
    """
    na = min(n, max(1, int(attack * SAMPLE_RATE)))
    nd = max(0, int(decay * SAMPLE_RATE))
    nr = max(1, int(release * SAMPLE_RATE))
    # Squeeze decay/release into whatever room the attack left.
    room = n - na
    if nd + nr > room:
        scale = room / max(1, nd + nr)
        nd = int(nd * scale)
        nr = room - nd
    ns = n - na - nd - nr

    env = np.empty(n, dtype=np.float64)
    env[:na] = np.linspace(0.0, 1.0, na, endpoint=False) ** (1.0 / curve)
    if nd:
        fall = np.linspace(0.0, 1.0, nd, endpoint=False) ** curve
        env[na:na + nd] = 1.0 + (sustain - 1.0) * fall
    if ns > 0:
        env[na + nd:na + nd + ns] = sustain
    if nr:
        fall = np.linspace(0.0, 1.0, nr) ** curve
        env[n - nr:] = sustain * (1.0 - fall)
    return env


def perc_env(n: int, decay: float, attack: float = 0.0015) -> np.ndarray:
    """Percussive envelope: near-instant attack, exponential tail."""
    t = _time(n)
    na = min(n, max(1, int(attack * SAMPLE_RATE)))
    env = np.exp(-t / max(1e-4, decay))
    env[:na] *= np.linspace(0.0, 1.0, na)
    return env


def swell_env(n: int, peak: float = 0.35, tail: float = 2.2) -> np.ndarray:
    """Crowd-style swell: slow rise to `peak` (0..1 of the buffer) then decay."""
    x = np.linspace(0.0, 1.0, n)
    peak = min(0.95, max(0.02, peak))
    rise = np.clip(x / peak, 0.0, 1.0) ** 1.6
    fall = np.exp(-tail * np.clip((x - peak) / (1.0 - peak), 0.0, 1.0))
    return rise * fall


def fade_edges(sig: np.ndarray, ms: float = 4.0) -> np.ndarray:
    """Taper both ends so a buffer never clicks when it starts or stops."""
    k = min(len(sig) // 2, max(1, int(ms * 0.001 * SAMPLE_RATE)))
    ramp = np.linspace(0.0, 1.0, k)
    out = sig.copy()
    if out.ndim == 1:
        out[:k] *= ramp
        out[-k:] *= ramp[::-1]
    else:
        out[:k] *= ramp[:, None]
        out[-k:] *= ramp[::-1, None]
    return out


# --------------------------------------------------------------------------- #
# DSP toolkit - oscillators
# --------------------------------------------------------------------------- #

def _phase(freq, n: int) -> np.ndarray:
    """Running phase (radians) for a constant or per-sample frequency."""
    if np.isscalar(freq):
        f = np.full(n, float(freq))
    else:
        f = np.asarray(freq, dtype=np.float64)
        if len(f) != n:  # allow envelopes authored at a different length
            f = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(f)), f)
    return 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE


def sine(freq, n: int, phase0: float = 0.0) -> np.ndarray:
    return np.sin(_phase(freq, n) + phase0)


def saw(freq, n: int) -> np.ndarray:
    """Naive ramp saw. Aliasing is masked by the lowpasses we run it through."""
    p = _phase(freq, n) / (2.0 * np.pi)
    return 2.0 * (p - np.floor(p + 0.5))


def square(freq, n: int, duty: float = 0.5) -> np.ndarray:
    p = _phase(freq, n) / (2.0 * np.pi)
    return np.where((p - np.floor(p)) < duty, 1.0, -1.0)


def triangle(freq, n: int) -> np.ndarray:
    p = _phase(freq, n) / (2.0 * np.pi)
    return 4.0 * np.abs(p - np.floor(p + 0.5)) - 1.0


def supersaw(freq, n: int, voices: int = 5, detune: float = 0.012) -> np.ndarray:
    """Stack of detuned saws - the cheapest route to a "big" sound."""
    out = np.zeros(n)
    for i in range(voices):
        k = (i - (voices - 1) / 2.0) / max(1.0, (voices - 1) / 2.0)
        f = np.asarray(freq, dtype=np.float64) * (1.0 + detune * k)
        out += saw(f if f.ndim else float(f), n) * (1.0 - 0.25 * abs(k))
    return out / voices


def fm(
    carrier,
    ratio: float,
    index,
    n: int,
    feedback: float = 0.0,
) -> np.ndarray:
    """Two-operator FM voice.

    `index` may be a scalar or a per-sample envelope; sweeping it from bright
    to dark in a few tens of milliseconds is what makes bells and mallets read
    as struck rather than blown.
    """
    c = np.full(n, float(carrier)) if np.isscalar(carrier) else np.asarray(carrier, float)
    idx = np.full(n, float(index)) if np.isscalar(index) else np.asarray(index, float)
    mod = np.sin(_phase(c * ratio, n))
    if feedback:
        mod = np.sin(_phase(c * ratio, n) + feedback * mod)
    return np.sin(_phase(c, n) + idx * mod)


def additive(base: float, n: int, partials: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Sum of (harmonic multiplier, amplitude) partials - for bells and chimes."""
    out = np.zeros(n)
    for mult, amp in partials:
        out += amp * sine(base * mult, n)
    return out


def noise(n: int, rng: np.random.Generator, color: str = "white") -> np.ndarray:
    """White or pink noise. Pink is shaped in the frequency domain (1/sqrt(f))."""
    w = rng.uniform(-1.0, 1.0, n)
    if color == "white":
        return w
    spec = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    f[0] = f[1] if len(f) > 1 else 1.0
    if color == "pink":
        spec /= np.sqrt(f)
    elif color == "brown":
        spec /= f
    spec[0] = 0.0
    out = np.fft.irfft(spec, n)
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def glide(n: int, f0: float, f1: float, curve: float = 1.0) -> np.ndarray:
    """Pitch envelope from f0 to f1. curve>1 lingers low, curve<1 lingers high."""
    x = np.linspace(0.0, 1.0, n) ** curve
    return f0 * (f1 / f0) ** x  # exponential = musically linear


def vibrato(n: int, rate: float, depth: float) -> np.ndarray:
    """Multiplicative pitch wobble, e.g. `freq * vibrato(n, 5.5, 0.02)`."""
    return 1.0 + depth * np.sin(2 * np.pi * rate * _time(n))


# --------------------------------------------------------------------------- #
# DSP toolkit - filters
# --------------------------------------------------------------------------- #

def one_pole_lp(sig: np.ndarray, cutoff: float) -> np.ndarray:
    """True one-pole IIR lowpass: y[n] = y[n-1] + a*(x[n] - y[n-1]).

    Evaluated block-wise in closed form so it stays vectorized:
        y[i] = b^(i+1) * y_prev + a * b^i * cumsum(x[i] / b^i)
    The block length is chosen from the pole so b^len never underflows.
    """
    a = 1.0 - math.exp(-2.0 * math.pi * max(1.0, cutoff) / SAMPLE_RATE)
    a = min(1.0, max(1e-6, a))
    b = 1.0 - a
    if b < 1e-6:
        return sig * a
    block = int(min(4096, max(8, -30.0 / math.log10(b))))
    x = np.asarray(sig, dtype=np.float64)
    out = np.empty_like(x)
    prev = 0.0
    for start in range(0, len(x), block):
        chunk = x[start:start + block]
        m = len(chunk)
        w = b ** np.arange(m)                       # b^i
        acc = np.cumsum(chunk / w)                  # sum x[k] / b^k
        y = b * prev * w + a * w * acc              # closed-form recursion
        out[start:start + m] = y
        prev = y[-1]
    return out


def one_pole_hp(sig: np.ndarray, cutoff: float) -> np.ndarray:
    """Complementary one-pole highpass (signal minus its lowpassed self)."""
    return np.asarray(sig, dtype=np.float64) - one_pole_lp(sig, cutoff)


def _gain_curve(n: int, kind: str, fc: float, q: float, order: int) -> np.ndarray:
    """Magnitude response sampled on the rfft grid for `n` samples."""
    f = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    f = np.maximum(f, 1e-6)
    if kind == "lowpass":
        return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * order))
    if kind == "highpass":
        return 1.0 / np.sqrt(1.0 + (fc / f) ** (2 * order))
    if kind == "bandpass":
        # Lorentzian resonance - smooth, so no pre-ringing artifacts.
        bw = max(1.0, fc / max(0.3, q))
        return 1.0 / (1.0 + ((f - fc) / bw) ** 2)
    if kind == "notch":
        bw = max(1.0, fc / max(0.3, q))
        return 1.0 - 1.0 / (1.0 + ((f - fc) / bw) ** 2)
    raise ValueError(kind)


def spectral_filter(
    sig: np.ndarray,
    kind: str,
    fc: float,
    q: float = 1.0,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase filter applied in the frequency domain.

    Cheaper and cleaner than running a biquad sample-by-sample in Python, and
    for one-shot buffers the lack of phase distortion is a bonus.
    """
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(x)
    spec *= _gain_curve(n, kind, fc, q, order)
    return np.fft.irfft(spec, n)


def sweep_filter(
    sig: np.ndarray,
    kind: str,
    cutoff_env: np.ndarray,
    q: float = 1.2,
    order: int = 2,
    block: int = 1024,
) -> np.ndarray:
    """Time-varying filter via overlap-add with a per-block cutoff.

    This is what turns flat noise into a whoosh: the cutoff traces an arc and
    the ear reads it as something moving past.
    """
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    env = np.interp(np.linspace(0, 1, max(2, n // (block // 2) + 2)),
                    np.linspace(0, 1, len(cutoff_env)), cutoff_env)
    hop = block // 2
    win = np.hanning(block)
    out = np.zeros(n + block)
    padded = np.concatenate([x, np.zeros(block)])
    for bi, start in enumerate(range(0, n, hop)):
        seg = padded[start:start + block] * win
        fc = float(env[min(bi, len(env) - 1)])
        spec = np.fft.rfft(seg)
        spec *= _gain_curve(block, kind, max(20.0, fc), q, order)
        out[start:start + block] += np.fft.irfft(spec, block)
    return out[:n]


def formant(sig: np.ndarray, freqs: Sequence[float], q: float = 6.0,
            gains: Optional[Sequence[float]] = None) -> np.ndarray:
    """Push a raw waveform through a set of resonant peaks -> a vowel."""
    gains = gains or [1.0, 0.55, 0.3][:len(freqs)]
    out = np.zeros(len(sig))
    for f, g in zip(freqs, list(gains) + [0.25] * len(freqs)):
        out += g * spectral_filter(sig, "bandpass", f, q=q)
    return out


# --------------------------------------------------------------------------- #
# DSP toolkit - time domain effects
# --------------------------------------------------------------------------- #

def _feedback_comb(sig: np.ndarray, delay_s: float, feedback: float) -> np.ndarray:
    """y[n] = x[n] + g*y[n-D], computed a delay-line at a time (vectorized)."""
    d = max(1, int(delay_s * SAMPLE_RATE))
    y = np.asarray(sig, dtype=np.float64).copy()
    for i in range(d, len(y), d):
        j = min(i + d, len(y))
        y[i:j] += feedback * y[i - d:i - d + (j - i)]
    return y


def allpass(sig: np.ndarray, delay_s: float, g: float = 0.5) -> np.ndarray:
    """Schroeder allpass: smears transients without coloring the spectrum."""
    d = max(1, int(delay_s * SAMPLE_RATE))
    v = _feedback_comb(sig, delay_s, g)
    out = -g * v
    out[d:] += v[:-d]
    return out


def reverb(
    sig: np.ndarray,
    room: float = 0.7,
    mix: float = 0.28,
    tail: float = 0.9,
    damp: float = 5200.0,
) -> np.ndarray:
    """Small Schroeder reverb: 4 parallel combs -> 2 series allpasses.

    Reserved for the big moments (fanfare, cheer, explosion) - everything else
    stays dry so the mix does not turn to mush on a phone speaker.
    """
    x = np.asarray(sig, dtype=np.float64)
    x = np.concatenate([x, np.zeros(_nsamp(tail))])
    fb = 0.55 + 0.35 * min(1.0, max(0.0, room))
    wet = np.zeros(len(x))
    for delay, scale in ((0.0297, 1.0), (0.0371, 0.9), (0.0411, 0.82), (0.0437, 0.75)):
        wet += scale * _feedback_comb(x, delay, fb * (0.98 - 0.03 * scale))
    wet = one_pole_lp(wet / 3.5, damp)
    wet = allpass(wet, 0.0050, 0.7)
    wet = allpass(wet, 0.0017, 0.7)
    return x * (1.0 - mix) + wet * mix


def delay_echo(sig: np.ndarray, time_s: float, feedback: float = 0.35,
               mix: float = 0.3, tail: float = 0.6) -> np.ndarray:
    """Simple feedback delay with a tail, for arcade-flavoured risers."""
    x = np.concatenate([np.asarray(sig, float), np.zeros(_nsamp(tail))])
    return x * (1.0 - mix) + _feedback_comb(x, time_s, feedback) * mix


def distort(sig: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """Soft saturation - adds harmonics and glues layers together."""
    return np.tanh(np.asarray(sig, dtype=np.float64) * drive) / math.tanh(drive)


def stereoize(sig: np.ndarray, width: float = 0.4, haas_ms: float = 8.0,
              pan: float = 0.0) -> np.ndarray:
    """Mono -> (n, 2) with a Haas delay and optional constant-power pan."""
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 2:
        return x
    d = max(1, int(haas_ms * 0.001 * SAMPLE_RATE))
    delayed = np.concatenate([np.zeros(d), x[:-d]]) if d < len(x) else x
    left = x * (1.0 - width) + delayed * width
    right = x * (1.0 - width) + np.concatenate([x[d // 2:], np.zeros(d // 2)]) * width
    ang = (min(1.0, max(-1.0, pan)) + 1.0) * math.pi / 4.0
    return np.column_stack((left * math.cos(ang) * 1.414, right * math.sin(ang) * 1.414))


def mix_into(dest: np.ndarray, src: np.ndarray, offset: int, gain: float = 1.0) -> None:
    """Add `src` into `dest` at a sample offset, clipping to the buffer end."""
    if offset >= len(dest):
        return
    start = max(0, offset)
    seg = src[max(0, -offset):]
    end = min(len(dest), start + len(seg))
    if end > start:
        dest[start:end] += seg[:end - start] * gain


def normalize(sig: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale to a target peak. Silent buffers are returned untouched."""
    m = float(np.max(np.abs(sig))) if sig.size else 0.0
    if m < 1e-9:
        return sig
    return sig * (peak / m)


def resample(sig: np.ndarray, ratio: float) -> np.ndarray:
    """Linear-interpolated resample. ratio>1 => higher pitch, shorter buffer."""
    ratio = max(0.25, min(4.0, float(ratio)))
    if abs(ratio - 1.0) < 1e-3:
        return sig
    n = len(sig)
    m = max(2, int(n / ratio))
    src = np.linspace(0.0, n - 1.0, m)
    if sig.ndim == 1:
        return np.interp(src, np.arange(n), sig)
    return np.column_stack([np.interp(src, np.arange(n), sig[:, c])
                            for c in range(sig.shape[1])])


def wrap_tail(sig: np.ndarray, loop_len: int) -> np.ndarray:
    """Fold everything past `loop_len` back onto the head - seamless loops.

    Reverb and delay tails would otherwise leave a hole at the loop point.
    """
    if len(sig) <= loop_len:
        out = np.zeros((loop_len,) + sig.shape[1:])
        out[:len(sig)] = sig
        return out
    head = sig[:loop_len].copy()
    tail = sig[loop_len:]
    for start in range(0, len(tail), loop_len):
        seg = tail[start:start + loop_len]
        head[:len(seg)] += seg
    return head


# --------------------------------------------------------------------------- #
# Sound design - UI
# --------------------------------------------------------------------------- #
#
# Every factory below returns a mono (or stereo) float buffer in [-1, 1].
# They take an rng so the bank is deterministic but the variants differ.

def sfx_ui_tap(rng: np.random.Generator, jitter: float = 1.0) -> np.ndarray:
    n = _nsamp(0.055)
    body = fm(880 * jitter, 2.0, np.linspace(3.0, 0.2, n), n)
    click = one_pole_lp(noise(n, rng) * perc_env(n, 0.004), 5200) * 0.5
    sig = (body * 0.7 + click) * perc_env(n, 0.035)
    return normalize(fade_edges(sig, 2.0), 0.55)


def sfx_ui_back(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.11)
    f = glide(n, 720, 430, 0.7)
    sig = (triangle(f, n) * 0.7 + sine(f * 2, n) * 0.2) * adsr(n, 0.004, 0.03, 0.4, 0.07)
    sig += one_pole_lp(noise(n, rng), 3000) * perc_env(n, 0.006) * 0.25
    return normalize(fade_edges(sig, 2.0), 0.55)


def sfx_ui_toggle(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.09)
    half = n // 2
    sig = np.zeros(n)
    sig[:half] = square(620, half, 0.35) * perc_env(half, 0.02)
    sig[half:] = square(930, n - half, 0.35) * perc_env(n - half, 0.03)
    sig = one_pole_lp(sig, 4200)
    return normalize(fade_edges(sig, 2.0), 0.5)


def sfx_ui_error(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.22)
    # Two detuned square waves a semitone apart = an unmistakable "nope".
    sig = square(196, n, 0.5) * 0.5 + square(196 * 1.06, n, 0.5) * 0.5
    sig *= 1.0 + 0.35 * np.sin(2 * np.pi * 26 * _time(n))     # buzz tremolo
    sig = one_pole_lp(sig, 1500) * adsr(n, 0.004, 0.05, 0.55, 0.1)
    return normalize(distort(sig, 1.6), 0.6)


def sfx_keypad_press(rng: np.random.Generator, jitter: float = 1.0) -> np.ndarray:
    n = _nsamp(0.07)
    wood = fm(360 * jitter, 3.1, np.linspace(4.5, 0.1, n), n) * perc_env(n, 0.022)
    tick = spectral_filter(noise(n, rng), "bandpass", 2400 * jitter, q=1.4)
    sig = wood * 0.8 + tick * perc_env(n, 0.005) * 0.5
    return normalize(fade_edges(sig, 2.0), 0.5)


# --------------------------------------------------------------------------- #
# Sound design - answers & scoring
# --------------------------------------------------------------------------- #

def _bell(base: float, dur: float, index: float = 4.0, ratio: float = 3.5,
          decay: float = 0.22) -> np.ndarray:
    """Struck FM bell with a bright-to-dark index sweep."""
    n = _nsamp(dur)
    idx = np.linspace(index, 0.15, n) ** 1.5
    tone = fm(base, ratio, idx, n)
    tone += 0.35 * fm(base * 2.0, 1.41, idx * 0.6, n)     # inharmonic shimmer
    return tone * perc_env(n, decay)


def sfx_correct(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.42)
    sig = np.zeros(n)
    # Root then fifth, a beat apart - reads as "yes, keep going".
    mix_into(sig, _bell(784.0, 0.30, 4.0, 3.0, 0.16), 0, 0.9)
    mix_into(sig, _bell(1174.7, 0.34, 3.2, 2.0, 0.19), _nsamp(0.075), 0.7)
    sig += additive(1568.0, n, [(1, 0.25), (2.7, 0.1)]) * perc_env(n, 0.06) * 0.3
    sig = reverb(sig, room=0.5, mix=0.18, tail=0.25)
    return normalize(fade_edges(sig, 3.0), 0.72)


def sfx_correct_streak(rng: np.random.Generator) -> np.ndarray:
    """Base buffer for the climbing streak chime - `play(pitch=...)` moves it."""
    n = _nsamp(0.5)
    sig = np.zeros(n)
    mix_into(sig, _bell(659.3, 0.26, 3.6, 3.0, 0.13), 0, 0.85)
    mix_into(sig, _bell(987.8, 0.28, 3.4, 2.0, 0.15), _nsamp(0.06), 0.8)
    mix_into(sig, _bell(1318.5, 0.34, 3.0, 4.0, 0.2), _nsamp(0.12), 0.75)
    sparkle = additive(2637.0, n, [(1, 0.2), (1.5, 0.12), (2.3, 0.08)])
    sig += sparkle * perc_env(n, 0.12) * 0.35
    sig = reverb(sig, room=0.6, mix=0.22, tail=0.3)
    return normalize(fade_edges(sig, 3.0), 0.75)


def sfx_wrong(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.36)
    f = glide(n, 330, 150, 1.3)
    body = supersaw(f, n, voices=3, detune=0.02) * 0.6 + square(f * 0.5, n, 0.45) * 0.4
    body = sweep_filter(body, "lowpass", np.linspace(2200, 420, 16), order=2)
    sig = body * adsr(n, 0.006, 0.09, 0.5, 0.2, curve=2.0)
    sig += one_pole_lp(noise(n, rng), 700) * perc_env(n, 0.05) * 0.25
    return normalize(distort(sig, 1.4), 0.7)


def sfx_combo(rng: np.random.Generator, step: int) -> np.ndarray:
    """combo_1..combo_5: same voice, higher scale degree, thicker each time."""
    n = _nsamp(0.34 + 0.05 * step)
    root = 523.25 * (SEMITONE ** PENTATONIC[min(step, len(PENTATONIC) - 1)])
    sig = np.zeros(n)
    mix_into(sig, _bell(root, 0.26, 3.2 + 0.3 * step, 2.0, 0.14), 0, 0.9)
    if step >= 1:                                     # add a fifth above
        mix_into(sig, _bell(root * 1.5, 0.24, 3.0, 3.0, 0.12), _nsamp(0.03), 0.55)
    if step >= 3:                                     # and an octave sparkle
        mix_into(sig, _bell(root * 2.0, 0.3, 2.6, 4.0, 0.16), _nsamp(0.06), 0.45)
    sig += noise(n, rng) * perc_env(n, 0.01) * 0.12 * (1 + step * 0.15)
    sig = reverb(sig, room=0.45, mix=0.12 + 0.03 * step, tail=0.2)
    return normalize(fade_edges(sig, 3.0), 0.6 + 0.06 * step)


# --------------------------------------------------------------------------- #
# Sound design - the rope
# --------------------------------------------------------------------------- #

def sfx_rope_pull(rng: np.random.Generator, jitter: float = 1.0) -> np.ndarray:
    """Fibrous drag: bandpassed noise sweeping up, over a low body thud."""
    n = _nsamp(0.3)
    fibers = sweep_filter(noise(n, rng), "bandpass",
                          np.linspace(420, 1500, 12) * jitter, q=1.1)
    thud = sine(glide(n, 120 * jitter, 62, 0.8), n) * perc_env(n, 0.07)
    sig = fibers * adsr(n, 0.01, 0.08, 0.5, 0.16) * 0.8 + thud * 0.5
    return normalize(fade_edges(sig, 3.0), 0.7)


def sfx_rope_creak(rng: np.random.Generator) -> np.ndarray:
    """Stick-slip: a resonant band gated by an irregular, wobbling AM."""
    n = _nsamp(0.8)
    t = _time(n)
    raw = noise(n, rng, "pink")
    body = spectral_filter(raw, "bandpass", 680, q=7.0)
    body += 0.6 * spectral_filter(raw, "bandpass", 1420, q=9.0)
    # Irregular grip-and-release modulation.
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 11 * t + 2.5 * np.sin(2 * np.pi * 2.3 * t))
    am *= 0.6 + 0.4 * np.sin(2 * np.pi * 3.7 * t + 1.1)
    sig = body * am * adsr(n, 0.05, 0.2, 0.7, 0.3)
    sig += sine(glide(n, 90, 74, 1.0), n) * 0.15 * adsr(n, 0.08, 0.2, 0.6, 0.3)
    return normalize(fade_edges(sig, 6.0), 0.6)


def sfx_rope_snap(rng: np.random.Generator) -> np.ndarray:
    """Crack: a hard highpassed transient, a whip, and a short room tail."""
    n = _nsamp(0.5)
    crack = one_pole_hp(noise(n, rng), 1800) * perc_env(n, 0.012, attack=0.0006)
    whip = sine(glide(n, 2400, 220, 0.55), n) * perc_env(n, 0.05) * 0.6
    fray = spectral_filter(noise(n, rng), "bandpass", 900, q=2.0) * perc_env(n, 0.12) * 0.4
    sig = crack * 1.0 + whip + fray
    sig = reverb(sig, room=0.55, mix=0.22, tail=0.3)
    return normalize(distort(sig, 1.3), 0.85)


# --------------------------------------------------------------------------- #
# Sound design - referee & countdown
# --------------------------------------------------------------------------- #

def _whistle_body(dur: float, rng: np.random.Generator, f1: float = 2100.0,
                  trill: float = 24.0) -> np.ndarray:
    """Pea whistle: two close tones plus a fast warble and breath noise."""
    n = _nsamp(dur)
    wob = 1.0 + 0.012 * np.sin(2 * np.pi * trill * _time(n))
    tone = sine(f1 * wob, n) * 0.6 + sine(f1 * 1.26 * wob, n) * 0.35
    tone += sine(f1 * 2.0 * wob, n) * 0.12
    breath = spectral_filter(noise(n, rng), "bandpass", f1 * 1.1, q=1.6) * 0.25
    env = adsr(n, 0.012, 0.05, 0.85, min(0.12, dur * 0.4), curve=1.4)
    return (tone + breath) * env


def sfx_whistle_start(rng: np.random.Generator) -> np.ndarray:
    sig = _whistle_body(0.55, rng)
    return normalize(fade_edges(sig, 4.0), 0.62)


def sfx_whistle_end(rng: np.random.Generator) -> np.ndarray:
    """Two short pips then a long blast - the universal "that's time"."""
    n = _nsamp(1.0)
    sig = np.zeros(n)
    mix_into(sig, _whistle_body(0.13, rng, 2150, 26), 0, 0.85)
    mix_into(sig, _whistle_body(0.13, rng, 2150, 26), _nsamp(0.18), 0.85)
    mix_into(sig, _whistle_body(0.52, rng, 2050, 22), _nsamp(0.36), 1.0)
    return normalize(fade_edges(sig, 4.0), 0.66)


def sfx_countdown_tick(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.1)
    block = fm(1180, 2.7, np.linspace(5.0, 0.1, n), n) * perc_env(n, 0.03)
    click = one_pole_hp(noise(n, rng), 4000) * perc_env(n, 0.003) * 0.4
    return normalize(fade_edges(block + click, 2.0), 0.6)


def sfx_countdown_go(rng: np.random.Generator) -> np.ndarray:
    """A quick riser resolving into a bright major stab."""
    n = _nsamp(0.62)
    sig = np.zeros(n)
    rise_n = _nsamp(0.16)
    riser = sweep_filter(noise(rise_n, rng), "bandpass",
                         np.linspace(500, 4200, 10), q=1.0)
    mix_into(sig, riser * np.linspace(0.1, 1.0, rise_n) ** 2, 0, 0.5)
    stab_n = _nsamp(0.46)
    stab = np.zeros(stab_n)
    for mult, amp in ((1.0, 1.0), (1.26, 0.7), (1.5, 0.7), (2.0, 0.5)):
        stab += amp * supersaw(523.25 * mult, stab_n, 4, 0.008)
    stab = sweep_filter(stab, "lowpass", np.linspace(6000, 1800, 8), order=2)
    stab *= adsr(stab_n, 0.006, 0.12, 0.45, 0.3, curve=2.0)
    mix_into(sig, stab, rise_n, 0.55)
    sig = reverb(sig, room=0.5, mix=0.18, tail=0.25)
    return normalize(sig, 0.85)


# --------------------------------------------------------------------------- #
# Sound design - crowd & kids
# --------------------------------------------------------------------------- #

def _kid_voice(dur: float, f0: float, f1: float, rng: np.random.Generator,
               vowel: Sequence[float] = (750, 1500, 2700),
               breath: float = 0.12) -> np.ndarray:
    """Formant-synth child voice: vibrato saw through three resonant peaks."""
    n = _nsamp(dur)
    f = glide(n, f0, f1, 0.8) * vibrato(n, rng.uniform(5.0, 7.5), 0.03)
    src = saw(f, n) * 0.7 + square(f, n, 0.42) * 0.3
    voiced = formant(src, vowel, q=7.0)
    air = spectral_filter(noise(n, rng), "bandpass", 3000, q=0.9) * breath
    return (voiced + air) * adsr(n, 0.03, 0.12, 0.65, dur * 0.4, curve=1.6)


def sfx_kid_effort(rng: np.random.Generator, variant: int = 0) -> np.ndarray:
    """Short grunt/yell - the "hnngh!" of a kid leaning back on the rope."""
    vowels = [(700, 1150, 2500), (620, 1000, 2400), (820, 1300, 2650)]
    n = _nsamp(0.42)
    base = rng.uniform(240, 310)
    v = _kid_voice(0.36, base * 1.25, base * 0.82, rng,
                   vowels[variant % len(vowels)], breath=0.18)
    sig = np.zeros(n)
    mix_into(sig, v, _nsamp(0.02), 1.0)
    sig += one_pole_lp(noise(n, rng), 900) * perc_env(n, 0.03) * 0.12
    return normalize(distort(sig, 1.2), 0.68)


def _crowd_bed(dur: float, rng: np.random.Generator, bright: float = 1.0,
               voices: int = 9) -> np.ndarray:
    """Layered kid voices + filtered noise roar; the raw material for a crowd."""
    n = _nsamp(dur)
    bed = np.zeros(n)
    for i in range(voices):
        d = float(rng.uniform(0.35, 0.75))
        f0 = float(rng.uniform(300, 520) * bright)
        v = _kid_voice(d, f0, f0 * rng.uniform(0.8, 1.35), rng,
                       (rng.uniform(600, 900), rng.uniform(1100, 1700), 2600),
                       breath=0.2)
        mix_into(bed, v, int(rng.uniform(0.0, max(0.01, dur - d)) * SAMPLE_RATE),
                 0.55 / math.sqrt(voices))
    roar = spectral_filter(noise(n, rng, "pink"), "bandpass", 900 * bright, q=0.6)
    roar *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.9 * _time(n) + rng.uniform(0, 6))
    return bed + roar * 0.45


def sfx_crowd_cheer(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(1.9)
    bed = _crowd_bed(1.9, rng, bright=1.15, voices=11) * swell_env(n, 0.28, 2.4)
    claps = np.zeros(n)
    for _ in range(26):                                # scattered applause
        c = one_pole_hp(noise(_nsamp(0.05), rng), 2000) * perc_env(_nsamp(0.05), 0.01)
        mix_into(claps, c, int(rng.uniform(0.05, 1.7) * SAMPLE_RATE), rng.uniform(0.2, 0.5))
    sig = reverb(bed + claps * 0.5, room=0.85, mix=0.32, tail=0.6)
    return normalize(stereoize(sig, 0.5, 11.0), 0.8)


def sfx_crowd_gasp(rng: np.random.Generator) -> np.ndarray:
    """A sharp collective inhale that cuts off - "ooh, they nearly had it"."""
    n = _nsamp(1.1)
    inhale = sweep_filter(noise(n, rng, "pink"), "bandpass",
                          np.linspace(400, 2600, 12), q=0.8)
    inhale *= np.concatenate([np.linspace(0, 1, _nsamp(0.28)) ** 1.5,
                              np.linspace(1, 0, n - _nsamp(0.28)) ** 2.2])
    voices = np.zeros(n)
    for i in range(6):
        d = float(rng.uniform(0.4, 0.6))
        f0 = float(rng.uniform(260, 380))
        v = _kid_voice(d, f0, f0 * 1.18, rng, (450, 900, 2400), breath=0.25)
        mix_into(voices, v, int(rng.uniform(0.0, 0.35) * SAMPLE_RATE), 0.3)
    sig = reverb(inhale * 0.8 + voices, room=0.7, mix=0.25, tail=0.4)
    return normalize(stereoize(sig, 0.45, 9.0), 0.62)


def sfx_crowd_ambient(rng: np.random.Generator) -> np.ndarray:
    """Seamlessly loopable murmur for the arena between beats."""
    loop = _nsamp(4.0)
    xfade = _nsamp(0.25)
    bed = _crowd_bed(4.0 + 0.25, rng, bright=0.9, voices=10) * 0.55
    bed = spectral_filter(bed, "lowpass", 3400, order=2)
    t = _time(len(bed))
    bed *= 0.75 + 0.25 * np.sin(2 * np.pi * 0.25 * t)      # slow breathing
    # Standard loop crossfade: the material just past the loop point is faded
    # into the head, so playback wraps around without a seam.
    looped = bed[:loop].copy()
    ramp = np.linspace(0.0, 1.0, xfade)
    looped[:xfade] = looped[:xfade] * ramp + bed[loop:loop + xfade] * (1.0 - ramp)
    return normalize(stereoize(looped, 0.55, 13.0), 0.4)


# --------------------------------------------------------------------------- #
# Sound design - rewards & progression
# --------------------------------------------------------------------------- #

def sfx_star_earn(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.7)
    sig = np.zeros(n)
    # A quick three-note sparkle up the scale, plus high shimmer dust.
    for i, mult in enumerate((1.0, 1.25, 1.5)):
        mix_into(sig, _bell(1046.5 * mult, 0.32, 3.0, 3.5, 0.14),
                 _nsamp(0.055 * i), 0.8 - 0.12 * i)
    dust = additive(3136.0, n, [(1, 0.2), (1.7, 0.12), (2.6, 0.08), (3.9, 0.05)])
    sig += dust * perc_env(n, 0.18) * 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * _time(n)))
    sig = reverb(sig, room=0.6, mix=0.24, tail=0.35)
    return normalize(fade_edges(sig, 3.0), 0.75)


def sfx_level_up(rng: np.random.Generator) -> np.ndarray:
    """Ascending arpeggio over a pad swell - the biggest "you grew" moment."""
    n = _nsamp(1.35)
    sig = np.zeros(n)
    for i, semi in enumerate((0, 4, 7, 12, 16)):
        f = 392.0 * (SEMITONE ** semi)
        mix_into(sig, _bell(f, 0.5, 3.4, 2.0, 0.2), _nsamp(0.085 * i), 0.75)
    pad_n = _nsamp(1.1)
    pad = sum(supersaw(392.0 * (SEMITONE ** s), pad_n, 5, 0.014)
              for s in (0, 4, 7, 12)) / 4.0
    pad = sweep_filter(pad, "lowpass", np.linspace(700, 4200, 10), order=2)
    mix_into(sig, pad * adsr(pad_n, 0.25, 0.3, 0.5, 0.5, curve=1.5), 0, 0.45)
    sig = reverb(sig, room=0.75, mix=0.28, tail=0.5)
    return normalize(sig, 0.85)


def sfx_coin(rng: np.random.Generator) -> np.ndarray:
    """The classic two-note coin blip, with metallic FM instead of a pure tone."""
    n = _nsamp(0.22)
    sig = np.zeros(n)
    a = _nsamp(0.055)
    sig[:a] = fm(988.0, 2.9, np.linspace(2.5, 0.6, a), a) * perc_env(a, 0.05)
    b = n - a
    sig[a:] = fm(1318.5, 2.9, np.linspace(2.5, 0.2, b), b) * perc_env(b, 0.1)
    sig += additive(2637.0, n, [(1, 0.15), (1.5, 0.08)]) * perc_env(n, 0.04) * 0.35
    return normalize(fade_edges(sig, 2.0), 0.6)


def sfx_unlock(rng: np.random.Generator) -> np.ndarray:
    """Mechanical clunk resolving into a bright chime - a door opening."""
    n = _nsamp(0.85)
    clunk_n = _nsamp(0.22)
    clunk = one_pole_lp(noise(clunk_n, rng), 900) * perc_env(clunk_n, 0.05)
    clunk += sine(glide(clunk_n, 180, 90, 0.6), clunk_n) * perc_env(clunk_n, 0.06) * 0.8
    sig = np.zeros(n)
    mix_into(sig, clunk, 0, 0.8)
    mix_into(sig, _bell(880.0, 0.5, 3.6, 2.0, 0.22), _nsamp(0.16), 0.7)
    mix_into(sig, _bell(1318.5, 0.5, 3.0, 3.0, 0.24), _nsamp(0.24), 0.6)
    sig = reverb(sig, room=0.65, mix=0.24, tail=0.4)
    return normalize(sig, 0.8)


def sfx_victory_fanfare(rng: np.random.Generator) -> np.ndarray:
    """Short brass-ish I-V-I fanfare with a snare roll and a real tail."""
    n = _nsamp(2.4)
    sig = np.zeros(n)
    # (start seconds, duration, semitone offsets) - a rising triad figure.
    figure = [
        (0.00, 0.20, (0, 4, 7)),
        (0.20, 0.20, (0, 4, 7)),
        (0.40, 0.36, (2, 7, 11)),
        (0.78, 0.85, (4, 12, 16)),
    ]
    for start, dur, chord in figure:
        m = _nsamp(dur)
        voice = np.zeros(m)
        for s in chord:
            f = 261.63 * (SEMITONE ** s)
            voice += supersaw(f * vibrato(m, 5.0, 0.006), m, 5, 0.011)
        voice = sweep_filter(voice / len(chord), "lowpass",
                             np.linspace(1200, 5200, 8), order=2)
        voice *= adsr(m, 0.02, 0.1, 0.62, min(0.35, dur * 0.5), curve=1.6)
        mix_into(sig, voice, _nsamp(start), 0.75)
    # Snare-ish roll under the pickup.
    for i in range(14):
        m = _nsamp(0.05)
        hit = one_pole_hp(noise(m, rng), 1400) * perc_env(m, 0.012)
        mix_into(sig, hit, _nsamp(0.05 * i), 0.12 + 0.02 * i)
    # Timpani-style low hits on the downbeats.
    for start in (0.0, 0.4, 0.78):
        m = _nsamp(0.35)
        boom = sine(glide(m, 110, 70, 0.7), m) * perc_env(m, 0.12)
        mix_into(sig, boom, _nsamp(start), 0.5)
    sig = reverb(sig, room=0.85, mix=0.3, tail=0.7)
    return normalize(stereoize(sig, 0.35, 12.0), 0.88)


def sfx_defeat_sting(rng: np.random.Generator) -> np.ndarray:
    """Descending minor figure on detuned, heavily damped saws."""
    n = _nsamp(1.5)
    sig = np.zeros(n)
    for i, semi in enumerate((0, -2, -5)):
        m = _nsamp(0.5 if i < 2 else 0.85)
        f = 349.23 * (SEMITONE ** semi)
        voice = supersaw(f * vibrato(m, 4.0, 0.012), m, 4, 0.02)
        voice += 0.4 * supersaw(f * 0.5, m, 3, 0.015)
        voice = sweep_filter(voice, "lowpass", np.linspace(2400, 500, 8), order=2)
        voice *= adsr(m, 0.03, 0.15, 0.5, 0.3, curve=2.0)
        mix_into(sig, voice, _nsamp(0.26 * i), 0.6)
    drum_n = _nsamp(0.6)
    drum = sine(glide(drum_n, 90, 45, 0.8), drum_n) * perc_env(drum_n, 0.16)
    mix_into(sig, drum, _nsamp(0.52), 0.6)
    sig = reverb(sig, room=0.7, mix=0.24, tail=0.5)
    return normalize(sig, 0.78)


# --------------------------------------------------------------------------- #
# Sound design - power-ups & hazards
# --------------------------------------------------------------------------- #

def sfx_powerup_pickup(rng: np.random.Generator) -> np.ndarray:
    """Fast upward arpeggio + FM shimmer; delay tail sells the "collect"."""
    n = _nsamp(0.55)
    sig = np.zeros(n)
    for i, semi in enumerate((0, 4, 7, 12, 16, 19)):
        m = _nsamp(0.14)
        f = 523.25 * (SEMITONE ** semi)
        blip = fm(f, 2.0, np.linspace(2.4, 0.2, m), m) * perc_env(m, 0.05)
        mix_into(sig, blip, _nsamp(0.045 * i), 0.7)
    sig = delay_echo(sig, 0.09, feedback=0.3, mix=0.28, tail=0.0)[:n]
    return normalize(fade_edges(sig, 3.0), 0.72)


def sfx_freeze(rng: np.random.Generator) -> np.ndarray:
    """Crystalline: ring-modulated shimmer plus a descending icy noise sweep."""
    n = _nsamp(0.95)
    shimmer = np.zeros(n)
    for f in (1568.0, 2093.0, 2637.0, 3136.0):
        shimmer += sine(f * vibrato(n, 7.0, 0.004), n) * 0.25
    shimmer *= sine(31.0, n) * 0.5 + 0.5                   # ring mod = glassiness
    shimmer *= adsr(n, 0.01, 0.25, 0.35, 0.55, curve=2.0)
    ice = sweep_filter(noise(n, rng), "bandpass", np.linspace(6000, 900, 12), q=1.4)
    ice *= perc_env(n, 0.3)
    sig = shimmer * 0.7 + ice * 0.5
    sig += sine(glide(n, 220, 110, 1.0), n) * perc_env(n, 0.08) * 0.3
    sig = reverb(sig, room=0.7, mix=0.26, tail=0.4)
    return normalize(sig, 0.7)


def sfx_bomb_tick(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.08)
    ping = sine(2100, n) * perc_env(n, 0.012)
    snap = one_pole_hp(noise(n, rng), 3500) * perc_env(n, 0.004, attack=0.0004)
    body = fm(520, 4.0, np.linspace(3.0, 0.0, n), n) * perc_env(n, 0.02)
    return normalize(fade_edges(ping * 0.5 + snap * 0.8 + body * 0.5, 1.5), 0.6)


def sfx_bomb_explode(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(1.7)
    boom = sine(glide(n, 150, 28, 0.55), n) * perc_env(n, 0.28)
    blast = sweep_filter(noise(n, rng, "pink"), "lowpass",
                         np.linspace(7000, 260, 16), order=2)
    blast *= perc_env(n, 0.35, attack=0.004)
    crack = one_pole_hp(noise(n, rng), 2500) * perc_env(n, 0.02, attack=0.0005)
    debris = np.zeros(n)
    for _ in range(18):
        m = _nsamp(0.04)
        bit = spectral_filter(noise(m, rng), "bandpass", float(rng.uniform(700, 3500)), q=2.0)
        mix_into(debris, bit * perc_env(m, 0.01),
                 int(rng.uniform(0.1, 1.0) * SAMPLE_RATE), rng.uniform(0.1, 0.35))
    sig = boom * 1.0 + blast * 0.8 + crack * 0.6 + debris * 0.5
    sig = distort(sig, 1.5)
    sig = reverb(sig, room=0.9, mix=0.3, tail=0.6)
    return normalize(stereoize(sig, 0.3, 10.0), 0.95)


# --------------------------------------------------------------------------- #
# Sound design - motion
# --------------------------------------------------------------------------- #

def sfx_swoosh(rng: np.random.Generator) -> np.ndarray:
    n = _nsamp(0.38)
    arc = np.concatenate([np.linspace(500, 3200, 8), np.linspace(3200, 700, 8)])
    sig = sweep_filter(noise(n, rng, "pink"), "bandpass", arc, q=0.9)
    sig *= adsr(n, 0.04, 0.1, 0.7, 0.2, curve=1.8)
    return normalize(fade_edges(sig, 4.0), 0.6)


def sfx_pop(rng: np.random.Generator, jitter: float = 1.0) -> np.ndarray:
    n = _nsamp(0.1)
    body = sine(glide(n, 900 * jitter, 260, 0.5), n) * perc_env(n, 0.022)
    lip = one_pole_lp(noise(n, rng), 2600) * perc_env(n, 0.004) * 0.4
    return normalize(fade_edges(body + lip, 2.0), 0.62)


def sfx_whoosh_transition(rng: np.random.Generator) -> np.ndarray:
    """Long doppler pass with a pan sweep - used between screens."""
    n = _nsamp(0.7)
    arc = np.concatenate([np.linspace(300, 4200, 10), np.linspace(4200, 500, 8)])
    body = sweep_filter(noise(n, rng, "pink"), "bandpass", arc, q=0.7)
    tone = sine(glide(n, 220, 1400, 0.9), n) * 0.18 * swell_env(n, 0.5, 2.0)
    sig = body * swell_env(n, 0.45, 1.6) + tone
    # Hard-panned sweep left -> right.
    pan = np.linspace(-1.0, 1.0, n)
    ang = (pan + 1.0) * math.pi / 4.0
    st = np.column_stack((sig * np.cos(ang) * 1.414, sig * np.sin(ang) * 1.414))
    return normalize(fade_edges(st, 5.0), 0.7)


# --------------------------------------------------------------------------- #
# Music beds
# --------------------------------------------------------------------------- #
#
# Three short loops, each rendered as a stereo buffer with its reverb tail
# wrapped back to the head so the loop point is seamless.

_MINOR_PENT = (0, 3, 5, 7, 10)


def _pluck(freq: float, dur: float, bright: float = 1.0, level: float = 1.0) -> np.ndarray:
    """Karplus-ish pluck done with FM - cheap and it sits well under a pad."""
    n = _nsamp(dur)
    idx = np.linspace(3.0 * bright, 0.05, n) ** 1.4
    v = fm(freq, 2.0, idx, n) * 0.7 + sine(freq, n) * 0.3
    return v * perc_env(n, dur * 0.35) * level


def _kick(dur: float = 0.32, f0: float = 150.0, level: float = 1.0) -> np.ndarray:
    n = _nsamp(dur)
    body = sine(glide(n, f0, 42, 0.5), n) * perc_env(n, 0.1)
    click = np.zeros(n)
    click[:64] = np.linspace(1, 0, 64) * 0.5
    return distort(body + click, 1.6) * level


def _snare(rng: np.random.Generator, dur: float = 0.22, level: float = 1.0) -> np.ndarray:
    n = _nsamp(dur)
    body = sine(glide(n, 320, 180, 0.6), n) * perc_env(n, 0.05) * 0.5
    rattle = one_pole_hp(noise(n, rng), 1200) * perc_env(n, 0.07)
    return (body + rattle) * level


def _hat(rng: np.random.Generator, dur: float = 0.07, level: float = 1.0) -> np.ndarray:
    n = _nsamp(dur)
    return one_pole_hp(noise(n, rng), 6500) * perc_env(n, dur * 0.3) * level


def music_menu(rng: np.random.Generator) -> np.ndarray:
    """Calm arpeggio pad - unhurried, warm, no percussion beyond a shaker."""
    bpm = 84.0
    beat = 60.0 / bpm
    bars, beats_per_bar = 4, 4
    loop = _nsamp(beat * bars * beats_per_bar)
    buf = np.zeros(loop + _nsamp(2.0))

    # Am9 - Fmaj7 - Cmaj9 - Gsus2, one bar each, rooted around A3.
    chords = [(0, 3, 7, 10, 14), (-4, 0, 3, 7, 11), (-9, -2, 3, 7, 10), (-2, 2, 5, 9, 12)]
    root = 220.0
    for bar, chord in enumerate(chords):
        bar_t = bar * beats_per_bar * beat
        # Pad: slow detuned stack with a gentle filter opening.
        pad_n = _nsamp(beat * beats_per_bar * 1.05)
        pad = sum(supersaw(root * (SEMITONE ** s) * 0.5, pad_n, 5, 0.013)
                  for s in chord[:3]) / 3.0
        pad = sweep_filter(pad, "lowpass", np.linspace(500, 1500, 6), order=2)
        pad *= adsr(pad_n, 0.5, 0.4, 0.7, 0.8, curve=1.4)
        mix_into(buf, pad, _nsamp(bar_t), 0.3)
        # Arpeggio: eighth notes walking up and back down the chord.
        pattern = [0, 1, 2, 3, 4, 3, 2, 1]
        for i, deg in enumerate(pattern):
            f = root * (SEMITONE ** chord[deg % len(chord)])
            mix_into(buf, _pluck(f, beat * 0.9, 0.9, 0.5),
                     _nsamp(bar_t + i * beat * 0.5), 0.55)
        # Shaker on the offbeats.
        for i in range(beats_per_bar * 2):
            mix_into(buf, _hat(rng, 0.05, 0.16),
                     _nsamp(bar_t + i * beat * 0.5 + beat * 0.25), 1.0)

    buf = reverb(buf, room=0.8, mix=0.3, tail=1.2)
    return normalize(stereoize(wrap_tail(buf, loop), 0.4, 14.0), 0.62)


def music_match(rng: np.random.Generator) -> np.ndarray:
    """Driving percussive bed for an active match - kick, clap, hats, bass."""
    bpm = 128.0
    beat = 60.0 / bpm
    bars, beats_per_bar = 4, 4
    loop = _nsamp(beat * bars * beats_per_bar)
    buf = np.zeros(loop + _nsamp(1.5))

    bassline = [0, 0, 7, 0, 5, 5, 0, 7]          # semitones over the root
    root = 110.0
    for bar in range(bars):
        bar_t = bar * beats_per_bar * beat
        for b in range(beats_per_bar):
            t = bar_t + b * beat
            mix_into(buf, _kick(0.3, 155, 0.9), _nsamp(t), 1.0)      # four to the floor
            if b % 2 == 1:
                mix_into(buf, _snare(rng, 0.2, 0.55), _nsamp(t), 1.0)
            for h in range(2):
                mix_into(buf, _hat(rng, 0.06, 0.22 if h else 0.14),
                         _nsamp(t + h * beat * 0.5), 1.0)
        for i in range(8):                                            # eighth-note bass
            semi = bassline[(bar * 8 + i) % len(bassline)]
            m = _nsamp(beat * 0.45)
            v = supersaw(root * (SEMITONE ** semi), m, 3, 0.008)
            v = one_pole_lp(v, 420) * adsr(m, 0.006, 0.06, 0.55, 0.1)
            mix_into(buf, v, _nsamp(bar_t + i * beat * 0.5), 0.55)
        if bar % 2 == 1:                                              # answering stab
            m = _nsamp(beat * 0.5)
            stab = sum(supersaw(root * 4 * (SEMITONE ** s), m, 4, 0.012)
                       for s in (0, 3, 7)) / 3.0
            stab = one_pole_lp(stab, 2600) * adsr(m, 0.008, 0.1, 0.3, 0.15)
            mix_into(buf, stab, _nsamp(bar_t + beat * 2.5), 0.3)

    buf = reverb(buf, room=0.45, mix=0.14, tail=0.6)
    return normalize(stereoize(wrap_tail(buf, loop), 0.3, 9.0), 0.72)


def music_tense(rng: np.random.Generator) -> np.ndarray:
    """Same DNA as `match`, faster and darker - the last 15 seconds."""
    bpm = 152.0
    beat = 60.0 / bpm
    bars, beats_per_bar = 4, 4
    loop = _nsamp(beat * bars * beats_per_bar)
    buf = np.zeros(loop + _nsamp(1.5))

    root = 98.0                                    # a whole tone down = darker
    for bar in range(bars):
        bar_t = bar * beats_per_bar * beat
        for b in range(beats_per_bar):
            t = bar_t + b * beat
            mix_into(buf, _kick(0.26, 168, 1.0), _nsamp(t), 1.0)
            mix_into(buf, _kick(0.18, 150, 0.5), _nsamp(t + beat * 0.75), 1.0)
            if b % 2 == 1:
                mix_into(buf, _snare(rng, 0.18, 0.6), _nsamp(t), 1.0)
            for h in range(4):                                       # 16th hats
                mix_into(buf, _hat(rng, 0.045, 0.2 if h % 2 == 0 else 0.11),
                         _nsamp(t + h * beat * 0.25), 1.0)
        for i in range(16):                                          # urgent 16th bass
            semi = (0, 0, 1, 0, 6, 0, 1, 3)[(bar * 16 + i) % 8]
            m = _nsamp(beat * 0.22)
            v = square(root * (SEMITONE ** semi), m, 0.45)
            v = one_pole_lp(v, 520) * adsr(m, 0.004, 0.04, 0.4, 0.06)
            mix_into(buf, v, _nsamp(bar_t + i * beat * 0.25), 0.4)
        # Rising noise tension sweep across every second bar.
        if bar % 2 == 1:
            m = _nsamp(beat * 4)
            sweep = sweep_filter(noise(m, rng, "pink"), "bandpass",
                                 np.linspace(400, 5000, 12), q=0.8)
            mix_into(buf, sweep * np.linspace(0, 1, m) ** 2, _nsamp(bar_t), 0.18)
    # Ominous held drone underneath.
    drone = supersaw(root * 0.5, len(buf), 5, 0.02)
    buf += one_pole_lp(drone, 200) * 0.18

    buf = reverb(buf, room=0.5, mix=0.16, tail=0.6)
    return normalize(stereoize(wrap_tail(buf, loop), 0.28, 7.0), 0.75)


MUSIC_BUILDERS: Dict[str, Callable[[np.random.Generator], np.ndarray]] = {
    "menu": music_menu,
    "match": music_match,
    "tense": music_tense,
}


# --------------------------------------------------------------------------- #
# The SFX registry
# --------------------------------------------------------------------------- #
#
# name -> list of builders. More than one entry means the engine round-robins
# randomly between variants, which keeps repeated taps from sounding robotic.

def _variants(fn: Callable[..., np.ndarray], *argsets) -> List[Callable]:
    return [(lambda rng, a=a: fn(rng, *a)) for a in argsets]


SFX_BUILDERS: Dict[str, List[Callable[[np.random.Generator], np.ndarray]]] = {
    # UI
    "ui_tap": _variants(sfx_ui_tap, (0.94,), (1.0,), (1.07,)),
    "ui_back": [sfx_ui_back],
    "ui_toggle": [sfx_ui_toggle],
    "ui_error": [sfx_ui_error],
    "keypad_press": _variants(sfx_keypad_press, (0.95,), (1.0,), (1.06,)),
    # Answers
    "correct": [sfx_correct],
    "correct_streak": [sfx_correct_streak],
    "wrong": [sfx_wrong],
    # Rope
    "rope_pull": _variants(sfx_rope_pull, (0.92,), (1.0,), (1.1,)),
    "rope_creak": [sfx_rope_creak],
    "rope_snap": [sfx_rope_snap],
    # Referee / countdown
    "whistle_start": [sfx_whistle_start],
    "whistle_end": [sfx_whistle_end],
    "countdown_tick": [sfx_countdown_tick],
    "countdown_go": [sfx_countdown_go],
    # Crowd
    "crowd_cheer": [sfx_crowd_cheer],
    "crowd_gasp": [sfx_crowd_gasp],
    "crowd_ambient_loop": [sfx_crowd_ambient],
    "kid_effort": _variants(sfx_kid_effort, (0,), (1,), (2,)),
    # Rewards
    "star_earn": [sfx_star_earn],
    "level_up": [sfx_level_up],
    "coin": [sfx_coin],
    "unlock": [sfx_unlock],
    "victory_fanfare": [sfx_victory_fanfare],
    "defeat_sting": [sfx_defeat_sting],
    # Power-ups / hazards
    "powerup_pickup": [sfx_powerup_pickup],
    "freeze": [sfx_freeze],
    "bomb_tick": [sfx_bomb_tick],
    "bomb_explode": [sfx_bomb_explode],
    # Motion
    "swoosh": [sfx_swoosh],
    "pop": _variants(sfx_pop, (0.9,), (1.0,), (1.12,)),
    "whoosh_transition": [sfx_whoosh_transition],
}

# combo_1 .. combo_5, generated from one design so they stay a family.
for _i in range(1, 6):
    SFX_BUILDERS[f"combo_{_i}"] = [(lambda rng, s=_i - 1: sfx_combo(rng, s))]

SFX_NAMES: Tuple[str, ...] = tuple(sorted(SFX_BUILDERS))

# Older call sites (main.py) used these names; keep them working.
LEGACY_ALIASES: Dict[str, str] = {
    "click": "ui_tap",
    "tick": "countdown_tick",
    "right": "correct",
    "whistle": "whistle_start",
    "whoosh": "swoosh",
    "scream_left": "crowd_cheer",
    "scream_right": "crowd_gasp",
    "cheer": "crowd_cheer",
    "gasp": "crowd_gasp",
    "explode": "bomb_explode",
}

# Sounds that should loop by default when played.
LOOPING_SFX = frozenset({"crowd_ambient_loop"})

# Long, layered designs that cost real time to render. They are built on the
# background thread with the music, so startup stays snappy on slow hardware -
# none of them can possibly be needed in the first half second of a session.
DEFERRED_SFX: Tuple[str, ...] = (
    "crowd_cheer", "crowd_gasp", "crowd_ambient_loop",
    "victory_fanfare", "defeat_sting", "bomb_explode", "level_up",
)

# Per-sound mix trims (dB-ish, linear). Keeps the bank balanced without having
# to re-tune every synth function.
MIX_TRIM: Dict[str, float] = {
    "ui_tap": 0.55, "ui_back": 0.55, "ui_toggle": 0.55, "keypad_press": 0.5,
    "ui_error": 0.7, "countdown_tick": 0.7, "bomb_tick": 0.55,
    "rope_creak": 0.6, "rope_pull": 0.75, "crowd_ambient_loop": 0.45,
    "crowd_cheer": 0.9, "crowd_gasp": 0.7, "kid_effort": 0.7,
    "victory_fanfare": 0.95, "bomb_explode": 1.0, "pop": 0.6, "swoosh": 0.6,
}


# --------------------------------------------------------------------------- #
# Buffer -> pygame conversion
# --------------------------------------------------------------------------- #

def to_int16(buf: np.ndarray) -> np.ndarray:
    """Float mono/stereo in [-1, 1] -> C-contiguous int16 stereo for pygame."""
    x = np.asarray(buf, dtype=np.float64)
    if x.ndim == 1:
        x = np.column_stack((x, x))
    x = np.clip(x, -1.0, 1.0)
    return np.ascontiguousarray((x * 32767.0).astype(np.int16))


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

class AudioEngine:
    """Synthesized SFX + procedural music, with a silent fallback.

    Typical use::

        audio = AudioEngine()
        audio.play_music("menu")
        audio.play("ui_tap")
        audio.play("correct_streak", pitch=audio.streak_pitch(streak))
        ...
        audio.update(dt)        # once per frame

    Every method is safe to call even when there is no audio device.
    """

    def __init__(self, enabled: bool = True) -> None:
        # --- public-ish state -------------------------------------------- #
        self.available = False          # True once the mixer is live
        self.enabled = bool(enabled)    # user toggle; False = stay quiet
        self.sfx_volume = 0.85
        self.music_volume = 0.55
        self.current_music: Optional[str] = None
        self.init_seconds = 0.0

        # --- internals ---------------------------------------------------- #
        self._buffers: Dict[str, List[np.ndarray]] = {}
        self._sounds: Dict[str, List["pygame.mixer.Sound"]] = {}
        self._pitch_cache: Dict[Tuple[str, int, int], "pygame.mixer.Sound"] = {}
        self._music_buffers: Dict[str, np.ndarray] = {}
        self._music_sounds: Dict[str, "pygame.mixer.Sound"] = {}
        self._pending_sfx: Dict[str, List[np.ndarray]] = {}
        self._music_lock = threading.Lock()
        self._music_thread: Optional[threading.Thread] = None
        # Set by the render worker once every bed is built. Must exist before
        # the thread starts (and even when audio is disabled and it never
        # starts at all), or `music_ready` / `wait_for_music` raise.
        self._music_ready = False
        self._assets_ready = False
        self._pending_music: Optional[Tuple[str, bool, int]] = None
        self._music_channels: List["pygame.mixer.Channel"] = []
        self._music_slot = 0
        self._loop_channels: Dict[str, "pygame.mixer.Channel"] = {}
        self._rng = np.random.default_rng(_SEED)
        self._play_rng = np.random.default_rng(_SEED ^ 0x1234)
        self._duck_gain = 1.0          # current music attenuation (1 = none)
        self._duck_target = 1.0
        self._duck_recover = 0.0       # seconds remaining before recovery
        self._duck_rate = 4.0
        self._rate_ratio = 1.0         # SAMPLE_RATE / actual mixer rate

        if not self.enabled:
            return

        t0 = time.perf_counter()
        if not self._init_mixer():
            return
        try:
            self._build_sfx()
        except Exception:
            # A synthesis bug must never take the game down - go silent.
            self.available = False
            return
        self._start_music_thread()
        self.init_seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ #
    # Boot
    # ------------------------------------------------------------------ #

    def _init_mixer(self) -> bool:
        """Bring up pygame.mixer, adapting to an already-initialized one."""
        if pygame is None:
            return False
        try:
            cur = pygame.mixer.get_init()
            if cur is None:
                pygame.mixer.pre_init(SAMPLE_RATE, BIT_DEPTH, NUM_CHANNELS, MIXER_BUFFER)
                pygame.mixer.init()
            elif cur[0] != SAMPLE_RATE or abs(cur[2]) != NUM_CHANNELS:
                # Someone else opened the device with settings we can't use.
                try:
                    pygame.mixer.quit()
                    pygame.mixer.pre_init(SAMPLE_RATE, BIT_DEPTH, NUM_CHANNELS, MIXER_BUFFER)
                    pygame.mixer.init()
                except Exception:
                    pass
            init = pygame.mixer.get_init()
            if init is None:
                return False
            self._rate_ratio = SAMPLE_RATE / float(init[0] or SAMPLE_RATE)
            pygame.mixer.set_num_channels(POLYPHONY)
            pygame.mixer.set_reserved(MUSIC_CHANNELS)
            self._music_channels = [pygame.mixer.Channel(i) for i in range(MUSIC_CHANNELS)]
            self.available = True
            return True
        except Exception:
            self.available = False
            return False

    def _synth(self, name: str, rng: np.random.Generator) -> List[np.ndarray]:
        """Render every variant of one effect to normalized float buffers."""
        trim = MIX_TRIM.get(name, 0.8)
        bufs: List[np.ndarray] = []
        for builder in SFX_BUILDERS[name]:
            try:
                buf = np.asarray(builder(rng), dtype=np.float64)
                # Block DC: an offset wastes headroom and thumps on small
                # phone speakers. Pulse waves in particular carry one.
                buf = (buf - buf.mean(axis=0)) * trim
                bufs.append(np.clip(np.nan_to_num(buf), -1.0, 1.0).astype(np.float32))
            except Exception:
                continue
        return bufs

    def _register(self, name: str, bufs: List[np.ndarray]) -> None:
        """Turn rendered buffers into Sounds. Main thread only."""
        sounds = [s for s in (self._make_sound(b) for b in bufs) if s is not None]
        if bufs:
            self._buffers[name] = bufs
        if sounds:
            self._sounds[name] = sounds

    def _build_sfx(self) -> None:
        """Synthesize the latency-critical effects; defer the expensive ones."""
        for name in SFX_BUILDERS:
            if name in DEFERRED_SFX:
                continue
            self._register(name, self._synth(name, self._rng))

    def _make_sound(self, buf: np.ndarray) -> Optional["pygame.mixer.Sound"]:
        """float buffer -> pygame Sound, resampling if the device rate differs."""
        if pygame is None:
            return None
        try:
            if abs(self._rate_ratio - 1.0) > 1e-3:
                buf = resample(buf, self._rate_ratio)
            return pygame.sndarray.make_sound(to_int16(buf))
        except Exception:
            return None

    def _start_music_thread(self) -> None:
        """Render the (much longer) music beds off the main thread."""
        def worker() -> None:
            rng = np.random.default_rng(_SEED ^ 0xBEEF)
            for track, builder in MUSIC_BUILDERS.items():
                try:
                    buf = np.clip(np.nan_to_num(builder(rng)), -1.0, 1.0).astype(np.float32)
                except Exception:
                    continue
                with self._music_lock:
                    self._music_buffers[track] = buf

            # The heavy crowd/finale effects render here too. Only the numpy
            # buffers are produced on this thread; turning them into pygame
            # Sounds happens on the main thread in update(), because mixer
            # object creation is not safe to do from a worker.
            sfx_rng = np.random.default_rng(_SEED ^ 0xC0FFEE)
            for name in DEFERRED_SFX:
                if name not in SFX_BUILDERS:
                    continue
                try:
                    bufs = self._synth(name, sfx_rng)
                except Exception:
                    continue
                if bufs:
                    with self._music_lock:
                        self._pending_sfx[name] = bufs

            with self._music_lock:
                self._music_ready = True

        self._music_thread = threading.Thread(target=worker, name="roperush-music",
                                              daemon=True)
        self._music_thread.start()

    def wait_for_music(self, timeout: float = 10.0) -> bool:
        """Block until the music beds finish rendering (tests / loading screens)."""
        if self._music_thread is None:
            return self._music_ready
        self._music_thread.join(timeout)
        with self._music_lock:
            return self._music_ready

    # ------------------------------------------------------------------ #
    # SFX playback
    # ------------------------------------------------------------------ #

    def play(self, name: str, volume: float = 1.0, pitch: float = 1.0,
             loops: int = 0) -> None:
        """Fire a one-shot (or a loop, for `*_loop` names).

        `pitch` resamples the buffer: 2.0 is an octave up, 0.5 an octave down.
        Resampled variants are cached, so climbing a scale costs nothing after
        the first pass.
        """
        if not (self.available and self.enabled):
            return
        name = LEGACY_ALIASES.get(name, name)
        sounds = self._sounds.get(name)
        if not sounds:
            return
        idx = int(self._play_rng.integers(0, len(sounds))) if len(sounds) > 1 else 0
        snd = sounds[idx]
        if abs(pitch - 1.0) > 1e-3:
            snd = self._pitched(name, idx, pitch) or snd
        if loops == 0 and name in LOOPING_SFX:
            loops = -1
        try:
            vol = max(0.0, min(1.0, volume)) * self.sfx_volume
            snd.set_volume(vol)
            ch = snd.play(loops=loops)
            if loops != 0 and ch is not None:
                self._loop_channels[name] = ch
        except Exception:
            pass

    def _pitched(self, name: str, idx: int, pitch: float) -> Optional["pygame.mixer.Sound"]:
        """Return (and memoize) a resampled copy of a bank buffer."""
        key = (name, idx, int(round(pitch * 100)))
        snd = self._pitch_cache.get(key)
        if snd is not None:
            return snd
        if len(self._pitch_cache) > 160:            # cheap bound on the cache
            self._pitch_cache.clear()
        try:
            buf = resample(self._buffers[name][idx], pitch)
            snd = self._make_sound(buf)
        except Exception:
            return None
        if snd is not None:
            self._pitch_cache[key] = snd
        return snd

    def play_streak(self, streak: int, name: str = "correct_streak",
                    volume: float = 1.0) -> None:
        """Convenience: play the streak chime `streak` steps up the scale."""
        self.play(name, volume=volume, pitch=self.streak_pitch(streak))

    @staticmethod
    def streak_pitch(streak: int) -> float:
        """Map a streak count onto a pentatonic pitch ratio (caps at ~2 octaves)."""
        step = PENTATONIC[max(0, min(int(streak), len(PENTATONIC) - 1))]
        return float(SEMITONE ** step)

    def stop(self, name: str, fade_ms: int = 120) -> None:
        """Stop a looping effect started via `play`."""
        name = LEGACY_ALIASES.get(name, name)
        ch = self._loop_channels.pop(name, None)
        if ch is None:
            return
        try:
            ch.fadeout(max(0, int(fade_ms)))
        except Exception:
            pass

    def stop_all(self) -> None:
        """Silence everything, music included."""
        if not self.available:
            return
        try:
            pygame.mixer.stop()
        except Exception:
            pass
        self._loop_channels.clear()
        self.current_music = None

    # ------------------------------------------------------------------ #
    # Music
    # ------------------------------------------------------------------ #

    def play_music(self, track: str, loop: bool = True, fade_ms: int = 800) -> None:
        """Crossfade to a music bed. Queued if the bed is still rendering."""
        if not (self.available and self.enabled):
            return
        if track not in MUSIC_BUILDERS:
            return
        if track == self.current_music and self._music_playing():
            return
        snd = self._music_sound(track)
        if snd is None:
            self._pending_music = (track, loop, fade_ms)   # picked up in update()
            return
        self._pending_music = None
        old = self._music_channels[self._music_slot]
        self._music_slot = (self._music_slot + 1) % len(self._music_channels)
        new = self._music_channels[self._music_slot]
        try:
            if old.get_busy():
                old.fadeout(max(1, int(fade_ms)))
            new.set_volume(self.music_volume * self._duck_gain)
            new.play(snd, loops=-1 if loop else 0, fade_ms=max(0, int(fade_ms)))
            self.current_music = track
        except Exception:
            pass

    def stop_music(self, fade_ms: int = 600) -> None:
        if not self.available:
            return
        self._pending_music = None
        for ch in self._music_channels:
            try:
                if ch.get_busy():
                    ch.fadeout(max(1, int(fade_ms)))
            except Exception:
                pass
        self.current_music = None

    def _music_sound(self, track: str) -> Optional["pygame.mixer.Sound"]:
        """Convert a rendered bed to a Sound (lazily, on the main thread)."""
        snd = self._music_sounds.get(track)
        if snd is not None:
            return snd
        with self._music_lock:
            buf = self._music_buffers.get(track)
        if buf is None:
            return None
        snd = self._make_sound(buf)
        if snd is not None:
            self._music_sounds[track] = snd
        return snd

    def _music_playing(self) -> bool:
        try:
            return any(ch.get_busy() for ch in self._music_channels)
        except Exception:
            return False

    @property
    def music_ready(self) -> bool:
        with self._music_lock:
            return self._music_ready

    # ------------------------------------------------------------------ #
    # Mix control
    # ------------------------------------------------------------------ #

    def set_sfx_volume(self, v: float) -> None:
        self.sfx_volume = max(0.0, min(1.0, float(v)))

    def set_music_volume(self, v: float) -> None:
        self.music_volume = max(0.0, min(1.0, float(v)))
        self._apply_music_volume()

    def set_enabled(self, on: bool) -> None:
        """Master mute. Turning it off stops music and loops immediately."""
        self.enabled = bool(on)
        if not self.enabled:
            self.stop_all()

    def duck(self, amount: float = 0.4, duration: float = 0.5) -> None:
        """Dip the music under a big SFX, then glide back up.

        `amount` is how much to remove (0.4 = down to 60% volume); `duration`
        is how long to hold the dip before recovering.
        """
        amount = max(0.0, min(0.95, float(amount)))
        self._duck_target = 1.0 - amount
        self._duck_gain = min(self._duck_gain, self._duck_target)
        self._duck_recover = max(0.0, float(duration))
        self._apply_music_volume()

    def _apply_music_volume(self) -> None:
        if not self.available:
            return
        vol = self.music_volume * self._duck_gain
        for ch in self._music_channels:
            try:
                ch.set_volume(vol)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Per-frame
    # ------------------------------------------------------------------ #

    def update(self, dt: float) -> None:
        """Call once per frame: recovers ducking, starts queued music."""
        if not self.available:
            return

        # Adopt any deferred effects the render thread has finished. Sounds
        # must be constructed here rather than on the worker.
        if self._pending_sfx:
            with self._music_lock:
                ready = self._pending_sfx
                self._pending_sfx = {}
            for name, bufs in ready.items():
                self._register(name, bufs)
            if not self._pending_sfx:
                self._assets_ready = True

        # A play_music() issued before the bed finished rendering.
        if self._pending_music is not None:
            track, loop, fade = self._pending_music
            if self._music_sound(track) is not None:
                self.play_music(track, loop, fade)
        # Duck envelope: hold, then ramp back to unity.
        if self._duck_gain < 1.0 or self._duck_target < 1.0:
            if self._duck_recover > 0.0:
                self._duck_recover = max(0.0, self._duck_recover - dt)
                if self._duck_gain > self._duck_target:
                    self._duck_gain = max(self._duck_target,
                                          self._duck_gain - self._duck_rate * dt)
            else:
                self._duck_target = 1.0
                self._duck_gain = min(1.0, self._duck_gain + self._duck_rate * 0.5 * dt)
            self._apply_music_volume()

    # ------------------------------------------------------------------ #
    # Introspection (used by tests, the audio settings screen, and tooling)
    # ------------------------------------------------------------------ #

    def sound_names(self) -> Tuple[str, ...]:
        return SFX_NAMES

    def music_names(self) -> Tuple[str, ...]:
        return tuple(MUSIC_BUILDERS)

    def has(self, name: str) -> bool:
        return LEGACY_ALIASES.get(name, name) in self._buffers

    def get_buffer(self, name: str, variant: int = 0) -> Optional[np.ndarray]:
        """Raw float buffer for a sound - for waveform dumps and assertions."""
        bufs = self._buffers.get(LEGACY_ALIASES.get(name, name))
        if not bufs:
            return None
        return bufs[min(variant, len(bufs) - 1)]

    def get_music_buffer(self, track: str) -> Optional[np.ndarray]:
        with self._music_lock:
            return self._music_buffers.get(track)


# Older code (and other agents' modules) may still import SoundBank.
SoundBank = AudioEngine


__all__ = [
    "AudioEngine",
    "SoundBank",
    "SAMPLE_RATE",
    "SFX_NAMES",
    "SFX_BUILDERS",
    "MUSIC_BUILDERS",
    "LEGACY_ALIASES",
    "PENTATONIC",
    "SEMITONE",
    # DSP toolkit, exported so other systems can synthesize one-offs.
    "adsr", "perc_env", "swell_env", "fade_edges",
    "sine", "saw", "square", "triangle", "supersaw", "fm", "additive",
    "noise", "glide", "vibrato",
    "one_pole_lp", "one_pole_hp", "spectral_filter", "sweep_filter", "formant",
    "allpass", "reverb", "delay_echo", "distort", "stereoize",
    "mix_into", "normalize", "resample", "wrap_tail", "to_int16",
]

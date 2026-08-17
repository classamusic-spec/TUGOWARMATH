"""Persistent player profile, XP curve and reward economy for RopeRush.

This module owns *everything the player keeps* between sessions: experience,
level, currency, per-grade and per-skill performance history, unlocks,
equipped cosmetics, achievements, accessibility settings and the daily-streak
calendar. It knows nothing about pygame or rendering - it is a plain data
layer that scenes read from and write to.

Storage
-------
One JSON document, `roperush_save.json`, written next to the game modules.
Writes are atomic (temp file + `os.replace`) so a crash or a battery pull mid
save can never leave a half-written file behind.

Reads are **paranoid**. A save file is untrusted input: it may be missing,
empty, truncated, hand-edited, written by a newer build, or corrupted by the
filesystem. `Profile.load()` never raises - every field is coerced through a
defensive reader and anything unusable falls back to its default. In the worst
case the player gets a fresh profile, which is annoying but recoverable; a
crash on launch is neither.

Versioning
----------
`SAVE_VERSION` is stamped into every document. Older versions are run through
`_MIGRATIONS` step by step until they reach the current schema. A version we
do not recognise (typically a downgrade after the player ran a newer build)
resets to defaults rather than guessing at foreign fields.

Mastery model
-------------
Per-skill mastery is an exponentially weighted moving average of an
*answer quality* score in 0..1 that rewards accuracy first and speed second,
and which **decays toward neutral (0.5) with a 10-day half-life** when a skill
goes unpractised. A skill you crushed a month ago is no longer evidence that
you still know it, and - just as importantly - a skill you bombed a month ago
should not keep you pinned at the bottom of the difficulty band forever.

Public API
----------
``Settings``, ``GradeStats``, ``SkillStats``, ``Profile``, ``xp_for_level``,
``level_for_xp``, ``stars_for_result``, ``save_path``, ``set_save_path``,
``SAVE_VERSION``, ``SAVE_FILENAME``.
"""

from __future__ import annotations

import datetime as _datetime
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import math_engine as ME


# --------------------------------------------------------------------------- #
# Storage location
# --------------------------------------------------------------------------- #

#: Bumped whenever the on-disk schema changes in a way older readers can't
#: handle. Every bump needs an entry in `_MIGRATIONS`.
SAVE_VERSION = 2

SAVE_FILENAME = "roperush_save.json"

#: Tests (and any future "profile picker") can redirect the save file without
#: monkeypatching the module internals.
_save_path_override: Optional[Path] = None


def save_path() -> Path:
    """Absolute path of the save file - next to the game modules by default."""
    if _save_path_override is not None:
        return _save_path_override
    return Path(__file__).resolve().parent / SAVE_FILENAME


def set_save_path(path: Optional[Any]) -> None:
    """Redirect (or, with ``None``, restore) the save file location."""
    global _save_path_override
    _save_path_override = None if path is None else Path(path)


# --------------------------------------------------------------------------- #
# Tuning constants
# --------------------------------------------------------------------------- #

# --- XP curve ---------------------------------------------------------------
# Cumulative XP to *reach* a level. Level 1 sits at 0. The curve is gently
# superlinear (linear term + a ~1.7 power term) so early levels arrive fast
# enough to teach the reward loop, while later ones stay meaningful without
# ever turning into a grind wall.
_XP_LINEAR = 60.0
_XP_POWER_COEFF = 14.0
_XP_POWER = 1.7

#: Levels at which the player is handed a gem on top of the usual coins.
_GEM_LEVEL_INTERVAL = 5

# --- Coin / XP economy ------------------------------------------------------
XP_PER_MATCH = 40           # simply showing up and finishing
XP_PER_STAR = 25
XP_WIN_BONUS = 40
XP_PER_SCORE_POINT = 0.5    # capped, see `_score_xp`
XP_SCORE_CAP = 120

COINS_PER_MATCH = 8
COINS_PER_STAR = 12
COINS_WIN_BONUS = 25

#: Per-mode multiplier applied to both XP and coins. Practice is deliberately
#: cheap so it stays a *learning* space rather than the optimal farm.
MODE_REWARD_MULT: Dict[str, float] = {
    "classic": 1.00,
    "blitz": 1.15,
    "survival": 1.10,
    "boss": 1.50,
    "practice": 0.35,
    "daily": 1.25,
}

# --- Mastery ----------------------------------------------------------------
MASTERY_NEUTRAL = 0.5
MASTERY_HALF_LIFE_DAYS = 10.0

#: "A confident answer takes about this long" per grade, in ms. Used to turn a
#: solve time into a 0..1 speed score. These are generous on purpose - we are
#: rewarding fluency, not punishing a kid who thinks before typing.
GRADE_TARGET_MS: Dict[str, int] = {
    "pre-k": 7000,
    "k": 7000,
    "1st": 6500,
    "2nd": 6500,
    "3rd": 6000,
    "4th": 6000,
}
_DEFAULT_TARGET_MS = 6500

# --- Star thresholds --------------------------------------------------------
# `stars_for_result` blends three normalised inputs into one 0..1 performance
# score. Accuracy dominates: this is a maths game, not a twitch game.
STAR_W_ACCURACY = 0.45
STAR_W_MARGIN = 0.30
STAR_W_SPEED = 0.25
STAR_3_THRESHOLD = 0.82
STAR_2_THRESHOLD = 0.55

#: How many recent answers we keep for the adaptive-difficulty flow channel
#: (see `content.pick_difficulty`). Small on purpose - this is short-term
#: memory; long-term memory lives in the mastery values.
RECENT_WINDOW = 24

#: Modes and cosmetics the player owns before earning anything.
DEFAULT_MODES: Tuple[str, ...] = ("classic", "practice")
DEFAULT_COSMETICS: Tuple[str, ...] = ("headband_rookie", "jersey_home")
DEFAULT_EQUIPPED: Dict[str, str] = {
    "headband": "headband_rookie",
    "jersey": "jersey_home",
}

#: Fallback unlock table used only if `content.py` is unavailable. `add_xp`
#: prefers the live tables in content.py (see `_level_unlock_table`).
_FALLBACK_UNLOCKS: Dict[int, Tuple[str, ...]] = {
    3: ("mode:blitz",),
    4: ("mode:daily",),
    6: ("mode:survival",),
    10: ("mode:boss",),
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _today_key() -> str:
    """Local calendar day as 'YYYYMMDD' - the key format for daily results."""
    return _datetime.date.today().strftime("%Y%m%d")


def _parse_day_key(key: str) -> Optional[_datetime.date]:
    try:
        return _datetime.datetime.strptime(str(key), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


# --- defensive JSON readers -------------------------------------------------
# Every one of these takes *whatever was in the file* and returns something the
# rest of the module can safely use. None of them raise.


def _as_int(v: Any, default: int = 0) -> int:
    try:
        if isinstance(v, bool):
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        if isinstance(v, bool):
            return default
        f = float(v)
    except (TypeError, ValueError):
        return default
    # NaN / inf would poison every downstream comparison.
    if not math.isfinite(f):
        return default
    return f


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _as_str(v: Any, default: str = "") -> str:
    return v if isinstance(v, str) else default


def _as_str_set(v: Any) -> Set[str]:
    if isinstance(v, (list, tuple, set, frozenset)):
        return {s for s in v if isinstance(s, str)}
    return set()


def _as_str_map(v: Any) -> Dict[str, str]:
    if not isinstance(v, dict):
        return {}
    return {k: s for k, s in v.items() if isinstance(k, str) and isinstance(s, str)}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    """Player-controlled audio and accessibility preferences."""

    sfx_volume: float = 0.9
    music_volume: float = 0.6
    haptics: bool = True
    colorblind_mode: bool = False     # shifts EMBER/FROST to a safe pair
    left_handed: bool = False         # mirrors the single-player keypad side
    reduce_motion: bool = False       # damps screen shake / heavy particles
    show_hints: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sfx_volume": round(self.sfx_volume, 4),
            "music_volume": round(self.music_volume, 4),
            "haptics": self.haptics,
            "colorblind_mode": self.colorblind_mode,
            "left_handed": self.left_handed,
            "reduce_motion": self.reduce_motion,
            "show_hints": self.show_hints,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Settings":
        if not isinstance(raw, dict):
            return cls()
        d = cls()
        return cls(
            sfx_volume=_clamp01(_as_float(raw.get("sfx_volume"), d.sfx_volume)),
            music_volume=_clamp01(_as_float(raw.get("music_volume"), d.music_volume)),
            haptics=_as_bool(raw.get("haptics"), d.haptics),
            colorblind_mode=_as_bool(raw.get("colorblind_mode"), d.colorblind_mode),
            left_handed=_as_bool(raw.get("left_handed"), d.left_handed),
            reduce_motion=_as_bool(raw.get("reduce_motion"), d.reduce_motion),
            show_hints=_as_bool(raw.get("show_hints"), d.show_hints),
        )


# --------------------------------------------------------------------------- #
# Per-grade / per-skill statistics
# --------------------------------------------------------------------------- #


@dataclass
class GradeStats:
    """Aggregate performance in one `math_engine.GRADES` band."""

    attempts: int = 0
    correct: int = 0
    total_ms: int = 0
    best_ms: int = 0              # 0 == no timed solve recorded yet
    mastery: float = MASTERY_NEUTRAL

    @property
    def accuracy(self) -> float:
        return (self.correct / self.attempts) if self.attempts else 0.0

    @property
    def avg_ms(self) -> int:
        return int(self.total_ms / self.attempts) if self.attempts else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.attempts,
            "correct": self.correct,
            "total_ms": self.total_ms,
            "best_ms": self.best_ms,
            "mastery": round(self.mastery, 6),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "GradeStats":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            attempts=max(0, _as_int(raw.get("attempts"))),
            correct=max(0, _as_int(raw.get("correct"))),
            total_ms=max(0, _as_int(raw.get("total_ms"))),
            best_ms=max(0, _as_int(raw.get("best_ms"))),
            mastery=_clamp01(_as_float(raw.get("mastery"), MASTERY_NEUTRAL)),
        )


@dataclass
class SkillStats:
    """Aggregate performance in one `math_engine.SKILLS` strand.

    `last_seen_ms` carries a default so the documented five-field positional
    construction still works; it exists because mastery has to decay toward
    neutral over *wall-clock* time, which needs a timestamp to decay from.
    """

    attempts: int = 0
    correct: int = 0
    total_ms: int = 0
    streak: int = 0               # current consecutive-correct run
    mastery: float = MASTERY_NEUTRAL
    last_seen_ms: int = 0         # epoch ms of the most recent answer

    @property
    def accuracy(self) -> float:
        return (self.correct / self.attempts) if self.attempts else 0.0

    @property
    def avg_ms(self) -> int:
        return int(self.total_ms / self.attempts) if self.attempts else 0

    def decayed_mastery(self, now_ms: Optional[int] = None) -> float:
        """Mastery pulled back toward neutral by time since last practice.

        Retention halves every `MASTERY_HALF_LIFE_DAYS`; the *distance from
        neutral* is what decays, so a strong skill fades toward 0.5 from above
        and a weak one recovers toward 0.5 from below.
        """
        if self.attempts <= 0 or self.last_seen_ms <= 0:
            return MASTERY_NEUTRAL
        now = _now_ms() if now_ms is None else now_ms
        days = max(0.0, (now - self.last_seen_ms) / 86_400_000.0)
        retained = 0.5 ** (days / MASTERY_HALF_LIFE_DAYS)
        return _clamp01(MASTERY_NEUTRAL + (self.mastery - MASTERY_NEUTRAL) * retained)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempts": self.attempts,
            "correct": self.correct,
            "total_ms": self.total_ms,
            "streak": self.streak,
            "mastery": round(self.mastery, 6),
            "last_seen_ms": self.last_seen_ms,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "SkillStats":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            attempts=max(0, _as_int(raw.get("attempts"))),
            correct=max(0, _as_int(raw.get("correct"))),
            total_ms=max(0, _as_int(raw.get("total_ms"))),
            streak=max(0, _as_int(raw.get("streak"))),
            mastery=_clamp01(_as_float(raw.get("mastery"), MASTERY_NEUTRAL)),
            last_seen_ms=max(0, _as_int(raw.get("last_seen_ms"))),
        )


# --------------------------------------------------------------------------- #
# XP curve
# --------------------------------------------------------------------------- #


def xp_for_level(level: int) -> int:
    """Total cumulative XP required to *reach* `level`. Level 1 == 0 XP.

    Strictly increasing for every level >= 1, so `level_for_xp` can binary-walk
    it safely. The shape is `60n + 14n^1.7` where `n = level - 1`: near-linear
    for the first handful of levels, then pulling away gently.
    """
    lv = int(level)
    if lv <= 1:
        return 0
    n = float(lv - 1)
    return int(round(_XP_LINEAR * n + _XP_POWER_COEFF * (n ** _XP_POWER)))


def level_for_xp(xp: int) -> Tuple[int, float]:
    """Split a cumulative XP total into `(level, progress)`.

    `progress` is 0..1 through the *current* level, so exactly landing on a
    level threshold yields `(level, 0.0)` - the exact inverse of
    `xp_for_level`. Never returns a level below 1.
    """
    total = max(0, _as_int(xp))
    level = 1
    # The curve is superlinear, so stepping up is bounded and cheap in practice
    # (a few hundred iterations even for absurd XP totals).
    while total >= xp_for_level(level + 1):
        level += 1
    floor_xp = xp_for_level(level)
    ceil_xp = xp_for_level(level + 1)
    span = ceil_xp - floor_xp
    progress = _clamp01((total - floor_xp) / span) if span > 0 else 0.0
    return level, progress


def stars_for_result(margin: float, accuracy: float, time_frac: float) -> int:
    """Rate a finished match 1..3 stars.

    Parameters are all normalised 0..1:

    * `margin`    - how decisively the match was won (0 = photo finish / loss,
                    1 = total rout).
    * `accuracy`  - fraction of problems answered correctly.
    * `time_frac` - fraction of the allotted time consumed; *lower is better*,
                    so it is inverted into a speed score.

    A player never drops below one star: finishing a match is itself worth
    something, and a zero-star result reads as punishment to a seven-year-old.
    """
    m = _clamp01(margin)
    a = _clamp01(accuracy)
    speed = 1.0 - _clamp01(time_frac)
    score = STAR_W_ACCURACY * a + STAR_W_MARGIN * m + STAR_W_SPEED * speed
    if score >= STAR_3_THRESHOLD:
        return 3
    if score >= STAR_2_THRESHOLD:
        return 2
    return 1


def _score_xp(score: int) -> int:
    """Bonus XP from a raw mode score, capped so grinding one mode can't spike."""
    return int(min(XP_SCORE_CAP, max(0, score) * XP_PER_SCORE_POINT))


def _level_unlock_table() -> Dict[int, List[str]]:
    """Level -> unlock event keys, read from content.py when it is importable.

    Imported lazily and defensively: `content.py` imports *this* module for
    typing, so a module-level import here would be a cycle, and progression
    must stay usable on its own (tests, tools, headless stat dumps).
    """
    table: Dict[int, List[str]] = {}
    try:
        import content  # noqa: WPS433 - deliberate deferred import
    except Exception:
        for lv, keys in _FALLBACK_UNLOCKS.items():
            table.setdefault(lv, []).extend(keys)
        return table

    try:
        for key, mode in content.MODES.items():
            lv = int(mode.unlock_level)
            if lv > 1:
                table.setdefault(lv, []).append(f"mode:{key}")
        for cos in content.COSMETICS:
            lv = int(cos.unlock_level)
            # Cost-bearing cosmetics are shop items, not level rewards.
            if lv > 1 and int(cos.cost) <= 0:
                table.setdefault(lv, []).append(f"cosmetic:{cos.key}")
    except Exception:
        return {lv: list(keys) for lv, keys in _FALLBACK_UNLOCKS.items()}
    return table


# --------------------------------------------------------------------------- #
# Profile
# --------------------------------------------------------------------------- #


@dataclass
class Profile:
    """Everything a player keeps between sessions."""

    version: int = SAVE_VERSION

    # --- economy ---
    xp: int = 0
    level: int = 1
    coins: int = 0
    gems: int = 0
    total_stars: int = 0

    # --- lifetime record ---
    matches_played: int = 0
    matches_won: int = 0
    best_streak: int = 0
    problems_solved: int = 0
    problems_missed: int = 0
    fastest_solve_ms: int = 0     # 0 == none recorded yet

    # --- learning history ---
    per_grade_stats: Dict[str, GradeStats] = field(default_factory=dict)
    per_skill_stats: Dict[str, SkillStats] = field(default_factory=dict)

    # --- collection ---
    unlocked_modes: Set[str] = field(default_factory=lambda: set(DEFAULT_MODES))
    unlocked_cosmetics: Set[str] = field(
        default_factory=lambda: set(DEFAULT_COSMETICS)
    )
    equipped: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EQUIPPED))
    earned_achievements: Set[str] = field(default_factory=set)

    # --- preferences ---
    settings: Settings = field(default_factory=Settings)

    # --- calendar ---
    daily_streak: int = 0
    last_played_date: str = ""            # 'YYYYMMDD', "" == never played
    best_survival_seconds: float = 0.0
    daily_results: Dict[str, int] = field(default_factory=dict)

    # --- short-term adaptive memory (additive; defaults keep it optional) ---
    #: Rolling window of `[correct, elapsed_ms, grade]` triples, newest last.
    #: Feeds `content.pick_difficulty`'s flow channel. Capped at RECENT_WINDOW.
    recent_answers: List[List[Any]] = field(default_factory=list)

    # ------------------------------------------------------------ economy -- #

    def add_xp(self, amount: int) -> List[str]:
        """Grant XP, level up as far as it carries, and return unlock keys.

        Returned keys are UI events, in the order they happened:
        ``"level:7"``, ``"mode:blitz"``, ``"cosmetic:trail_comet"``,
        ``"gems:1"``. Mode and cosmetic unlocks are also applied to the
        profile's `unlocked_modes` / `unlocked_cosmetics` sets.
        """
        gain = max(0, _as_int(amount))
        if gain <= 0:
            return []

        before = self.level
        self.xp += gain
        self.level, _ = level_for_xp(self.xp)

        events: List[str] = []
        if self.level <= before:
            return events

        unlocks = _level_unlock_table()
        gems_won = 0
        for lv in range(before + 1, self.level + 1):
            events.append(f"level:{lv}")
            for key in unlocks.get(lv, ()):
                kind, _, name = key.partition(":")
                if kind == "mode":
                    self.unlocked_modes.add(name)
                elif kind == "cosmetic":
                    self.unlocked_cosmetics.add(name)
                events.append(key)
            if lv % _GEM_LEVEL_INTERVAL == 0:
                gems_won += 1

        if gems_won:
            self.gems += gems_won
            events.append(f"gems:{gems_won}")
        return events

    # ------------------------------------------------------------ answers -- #

    def record_answer(
        self, skill: str, grade: str, correct: bool, elapsed_ms: int
    ) -> None:
        """Fold one answered problem into the lifetime and per-strand history.

        Updates the skill and grade EWMAs, the consecutive-correct streaks, the
        fastest-solve record and the short-term flow window.
        """
        skill = _as_str(skill) or "add"
        grade = _as_str(grade) or "k"
        correct = bool(correct)
        # Guard against a stopwatch that was never started, or one that ran
        # while the app was backgrounded for an hour.
        ms = int(_clamp(_as_int(elapsed_ms), 0, 120_000))
        now = _now_ms()

        # --- lifetime ---
        if correct:
            self.problems_solved += 1
            if ms > 0 and (self.fastest_solve_ms <= 0 or ms < self.fastest_solve_ms):
                self.fastest_solve_ms = ms
        else:
            self.problems_missed += 1

        quality = self._answer_quality(grade, correct, ms)

        # --- per-skill ---
        ss = self.per_skill_stats.get(skill)
        if ss is None:
            ss = SkillStats()
            self.per_skill_stats[skill] = ss
        # Decay *before* blending, so a long layoff genuinely resets the prior
        # rather than being immediately overwritten by one fresh answer.
        base = ss.decayed_mastery(now)
        alpha = self._blend_rate(ss.attempts)
        ss.mastery = _clamp01(base + alpha * (quality - base))
        ss.attempts += 1
        ss.total_ms += ms
        ss.last_seen_ms = now
        if correct:
            ss.correct += 1
            ss.streak += 1
            self.best_streak = max(self.best_streak, ss.streak)
        else:
            ss.streak = 0

        # --- per-grade ---
        gs = self.per_grade_stats.get(grade)
        if gs is None:
            gs = GradeStats()
            self.per_grade_stats[grade] = gs
        g_alpha = self._blend_rate(gs.attempts)
        gs.mastery = _clamp01(gs.mastery + g_alpha * (quality - gs.mastery))
        gs.attempts += 1
        gs.total_ms += ms
        if correct:
            gs.correct += 1
            if ms > 0 and (gs.best_ms <= 0 or ms < gs.best_ms):
                gs.best_ms = ms

        # --- short-term flow window ---
        self.recent_answers.append([correct, ms, grade])
        if len(self.recent_answers) > RECENT_WINDOW:
            del self.recent_answers[:-RECENT_WINDOW]

    @staticmethod
    def _answer_quality(grade: str, correct: bool, elapsed_ms: int) -> float:
        """Score one answer 0..1: accuracy first, speed as the tiebreaker.

        A correct answer is worth 0.55 before speed is considered and 1.0 when
        it lands comfortably inside the grade's target time. A miss is worth
        0.05 regardless of how quickly it was submitted - guessing fast is not
        a skill we want to reinforce.
        """
        if not correct:
            return 0.05
        target = GRADE_TARGET_MS.get(grade, _DEFAULT_TARGET_MS)
        if elapsed_ms <= 0:
            return 0.80  # untimed (e.g. practice) - credit, but not a record
        ratio = elapsed_ms / float(target)
        # Full speed credit at <=40% of target, none at >=200%.
        speed = _clamp01((2.0 - ratio) / 1.6)
        return _clamp01(0.55 + 0.45 * speed)

    @staticmethod
    def _blend_rate(attempts: int) -> float:
        """EWMA alpha: move fast while we know little, settle as evidence piles up."""
        return 0.10 + 0.30 * math.exp(-max(0, attempts) / 8.0)

    # ------------------------------------------------------------ matches -- #

    def record_match(
        self,
        won: bool,
        stars: int,
        mode: str,
        grade: str,
        score: int = 0,
    ) -> Dict[str, Any]:
        """Bank the result of a finished match and pay it out.

        Returns a rewards summary the results screen can animate straight from::

            {'xp': int, 'coins': int, 'gems': int, 'stars': int,
             'level_before': int, 'level_after': int, 'leveled_up': bool,
             'unlocks': [str, ...], 'new_daily_best': bool}
        """
        won = bool(won)
        stars = int(_clamp(_as_int(stars), 0, 3))
        mode = _as_str(mode) or "classic"
        grade = _as_str(grade) or "k"
        score = max(0, _as_int(score))
        mult = MODE_REWARD_MULT.get(mode, 1.0)

        self.matches_played += 1
        if won:
            self.matches_won += 1
        self.total_stars += stars

        # Daily challenge keeps a per-day high score.
        new_daily_best = False
        if mode == "daily":
            day = _today_key()
            if score > self.daily_results.get(day, -1):
                self.daily_results[day] = score
                new_daily_best = True

        xp_gain = XP_PER_MATCH + XP_PER_STAR * stars + (XP_WIN_BONUS if won else 0)
        xp_gain = int(round((xp_gain + _score_xp(score)) * mult))
        coin_gain = COINS_PER_MATCH + COINS_PER_STAR * stars + (
            COINS_WIN_BONUS if won else 0
        )
        coin_gain = int(round(coin_gain * mult))

        level_before = self.level
        self.coins += coin_gain
        unlocks = self.add_xp(xp_gain)
        # add_xp already credited any milestone gems.
        gems_gain = sum(
            int(k.split(":", 1)[1]) for k in unlocks if k.startswith("gems:")
        )

        return {
            "xp": xp_gain,
            "coins": coin_gain,
            "gems": gems_gain,
            "stars": stars,
            "level_before": level_before,
            "level_after": self.level,
            "leveled_up": self.level > level_before,
            "unlocks": unlocks,
            "new_daily_best": new_daily_best,
        }

    # ------------------------------------------------------------ queries -- #

    def mastery(self, skill: str) -> float:
        """Time-decayed mastery of one skill, 0..1 (0.5 == no opinion yet)."""
        ss = self.per_skill_stats.get(_as_str(skill))
        if ss is None:
            return MASTERY_NEUTRAL
        return ss.decayed_mastery()

    def grade_mastery(self, grade: str) -> float:
        """Mean decayed mastery across the skills a grade actually teaches."""
        skills = ME.skills_for_grade(grade) or ME.SKILLS
        return sum(self.mastery(s) for s in skills) / float(len(skills))

    def weakest_skills(self, n: int = 3) -> List[str]:
        """The `n` skills most in need of practice, weakest first.

        Practised-and-shaky skills outrank never-tried ones: a skill sitting at
        0.30 is concrete evidence of a gap, whereas an untouched skill only
        sits at neutral. Ties break toward the skill with fewer attempts.
        """
        n = max(0, int(n))
        if n == 0:
            return []
        ranked = sorted(
            ME.SKILLS,
            key=lambda s: (
                self.mastery(s),
                self.per_skill_stats.get(s, SkillStats()).attempts,
                ME.SKILLS.index(s),
            ),
        )
        return ranked[:n]

    def accuracy(self) -> float:
        """Lifetime answer accuracy, 0..1 (0.0 before the first problem)."""
        total = self.problems_solved + self.problems_missed
        return (self.problems_solved / total) if total else 0.0

    def win_rate(self) -> float:
        """Lifetime match win rate, 0..1."""
        return (self.matches_won / self.matches_played) if self.matches_played else 0.0

    def level_progress(self) -> float:
        """Progress 0..1 through the current level, for the XP bar."""
        return level_for_xp(self.xp)[1]

    # ---------------------------------------------------------- calendar -- #

    def touch_daily(self) -> bool:
        """Register a play session today. True if this is a new calendar day.

        Consecutive days extend `daily_streak`; a gap of two or more days
        restarts it at 1. Called once when the player reaches the main menu.
        """
        today = _today_key()
        if self.last_played_date == today:
            return False

        prev = _parse_day_key(self.last_played_date)
        now = _parse_day_key(today)
        if prev is not None and now is not None and (now - prev).days == 1:
            self.daily_streak += 1
        else:
            self.daily_streak = 1
        self.last_played_date = today
        return True

    def record_survival(self, seconds: float) -> bool:
        """Update the survival-mode record. True if it is a new personal best."""
        s = max(0.0, _as_float(seconds))
        if s > self.best_survival_seconds:
            self.best_survival_seconds = s
            return True
        return False

    # ------------------------------------------------------ serialisation -- #

    def to_dict(self) -> Dict[str, Any]:
        """Plain JSON-safe document. Sets are stored sorted for stable diffs."""
        return {
            "version": SAVE_VERSION,
            "xp": self.xp,
            "level": self.level,
            "coins": self.coins,
            "gems": self.gems,
            "total_stars": self.total_stars,
            "matches_played": self.matches_played,
            "matches_won": self.matches_won,
            "best_streak": self.best_streak,
            "problems_solved": self.problems_solved,
            "problems_missed": self.problems_missed,
            "fastest_solve_ms": self.fastest_solve_ms,
            "per_grade_stats": {
                k: v.to_dict() for k, v in sorted(self.per_grade_stats.items())
            },
            "per_skill_stats": {
                k: v.to_dict() for k, v in sorted(self.per_skill_stats.items())
            },
            "unlocked_modes": sorted(self.unlocked_modes),
            "unlocked_cosmetics": sorted(self.unlocked_cosmetics),
            "equipped": dict(self.equipped),
            "earned_achievements": sorted(self.earned_achievements),
            "settings": self.settings.to_dict(),
            "daily_streak": self.daily_streak,
            "last_played_date": self.last_played_date,
            "best_survival_seconds": round(self.best_survival_seconds, 3),
            "daily_results": {
                k: _as_int(v) for k, v in self.daily_results.items()
                if isinstance(k, str)
            },
            "recent_answers": [
                [bool(a[0]), _as_int(a[1]), _as_str(a[2])]
                for a in self.recent_answers
                if isinstance(a, (list, tuple)) and len(a) >= 3
            ],
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "Profile":
        """Rebuild a profile from an already-migrated document.

        Field by field, with a default for anything missing or unusable. This
        is the only place that decides what a broken value means.
        """
        if not isinstance(raw, dict):
            return cls()

        p = cls()
        p.version = SAVE_VERSION
        p.xp = max(0, _as_int(raw.get("xp")))
        p.coins = max(0, _as_int(raw.get("coins")))
        p.gems = max(0, _as_int(raw.get("gems")))
        p.total_stars = max(0, _as_int(raw.get("total_stars")))
        p.matches_played = max(0, _as_int(raw.get("matches_played")))
        p.matches_won = max(0, _as_int(raw.get("matches_won")))
        p.best_streak = max(0, _as_int(raw.get("best_streak")))
        p.problems_solved = max(0, _as_int(raw.get("problems_solved")))
        p.problems_missed = max(0, _as_int(raw.get("problems_missed")))
        p.fastest_solve_ms = max(0, _as_int(raw.get("fastest_solve_ms")))

        # Level is *derived* from XP, never trusted from disk - that keeps a
        # hand-edited save from granting unlocks it never earned.
        p.level = level_for_xp(p.xp)[0]

        grades = raw.get("per_grade_stats")
        if isinstance(grades, dict):
            for k, v in grades.items():
                if isinstance(k, str):
                    p.per_grade_stats[k] = GradeStats.from_dict(v)

        skills = raw.get("per_skill_stats")
        if isinstance(skills, dict):
            for k, v in skills.items():
                if isinstance(k, str):
                    p.per_skill_stats[k] = SkillStats.from_dict(v)

        p.unlocked_modes = _as_str_set(raw.get("unlocked_modes")) | set(DEFAULT_MODES)
        p.unlocked_cosmetics = (
            _as_str_set(raw.get("unlocked_cosmetics")) | set(DEFAULT_COSMETICS)
        )
        p.equipped = dict(DEFAULT_EQUIPPED)
        p.equipped.update(_as_str_map(raw.get("equipped")))
        p.earned_achievements = _as_str_set(raw.get("earned_achievements"))
        p.settings = Settings.from_dict(raw.get("settings"))

        p.daily_streak = max(0, _as_int(raw.get("daily_streak")))
        p.last_played_date = _as_str(raw.get("last_played_date"))
        p.best_survival_seconds = max(
            0.0, _as_float(raw.get("best_survival_seconds"))
        )

        dailies = raw.get("daily_results")
        if isinstance(dailies, dict):
            for k, v in dailies.items():
                if isinstance(k, str):
                    p.daily_results[k] = _as_int(v)

        recent = raw.get("recent_answers")
        if isinstance(recent, list):
            for entry in recent[-RECENT_WINDOW:]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    p.recent_answers.append(
                        [_as_bool(entry[0]), max(0, _as_int(entry[1])),
                         _as_str(entry[2])]
                    )
        return p

    # ------------------------------------------------------------- disk --- #

    def save(self) -> None:
        """Atomically write the profile. Never raises on a failed write.

        A game that crashes because it could not save is strictly worse than a
        game that quietly loses one session of progress, so I/O errors are
        swallowed here; the caller has no useful recovery anyway.
        """
        path = save_path()
        tmp = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=1, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    @classmethod
    def load(cls) -> "Profile":
        """Read the save file, or hand back a fresh profile. Never raises."""
        path = save_path()
        try:
            if not path.is_file():
                return cls()
            text = path.read_text(encoding="utf-8")
        except Exception:
            return cls()

        if not text.strip():
            return cls()

        try:
            raw = json.loads(text)
        except Exception:
            # Truncated or otherwise malformed JSON.
            return cls()

        if not isinstance(raw, dict):
            return cls()

        migrated = _migrate(raw)
        if migrated is None:
            return cls()

        try:
            return cls.from_dict(migrated)
        except Exception:
            return cls()


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #


def _migrate_v1_to_v2(raw: Dict[str, Any]) -> Dict[str, Any]:
    """v1 -> v2.

    v1 shipped before the daily challenge and survival mode existed, and named
    the star total `stars`. Everything else carried over unchanged.
    """
    out = dict(raw)
    if "total_stars" not in out and "stars" in out:
        out["total_stars"] = out.get("stars")
    out.setdefault("daily_results", {})
    out.setdefault("best_survival_seconds", 0.0)
    out.setdefault("recent_answers", [])
    out["version"] = 2
    return out


#: from_version -> migration step producing `from_version + 1`.
_MIGRATIONS = {
    1: _migrate_v1_to_v2,
}


def _migrate(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Bring a save document up to `SAVE_VERSION`, or return None to reset.

    None means "this document is not something we can honestly interpret" -
    a version from the future (the player downgraded the app) or one whose
    version field is missing/garbage. Resetting is the safe answer: guessing
    at an unknown schema risks silently corrupting real progress.
    """
    version = raw.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        return None
    if version > SAVE_VERSION or version < 1:
        return None

    doc = raw
    while version < SAVE_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:
            return None
        try:
            doc = step(doc)
        except Exception:
            return None
        version += 1
    return doc

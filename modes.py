"""Game-mode rules engine for RopeRush.

This module owns *what the rules are*: how much rope a correct answer is worth,
what a wrong answer costs, when a match is over, who won, and how many stars it
was worth. It is deliberately free of pygame, rendering, audio and file I/O -
main.py drives it and owns every pixel.

The split
---------
* ``math_engine`` decides **what a kid is asked**.
* ``modes``      decides **what happens when they answer**.
* ``main``       decides **what it looks like**.

The contract main.py uses
-------------------------
::

    rules = make_mode("classic", {"grade": "2nd", "ai_skill": 0.55})
    state = rules.new_state()
    rules.on_start(state)

    problem = rules.next_problem(state, "left")
    ...
    result = rules.on_correct(state, "left", problem, elapsed=2.4)
    # -> PullResult(delta=-0.11, combo=3, points=17, events=['correct', ...])

    rules.update(state, dt)          # clocks, power-ups, boss AI, hazards
    outcome = rules.check_end(state) # None until the match is actually over

Sign convention
---------------
``MatchState.rope`` runs from -1 (left / EMBER has pulled it all the way over)
to +1 (right / FROST has). Zero is dead centre. Every ``PullResult.delta`` is
already signed for the side that earned it, so main.py only ever does
``state.rope += result.delta`` - which the engine has already done for it.

Solo modes
----------
Survival, Boss, Practice and Daily are single-player. The human is
``rules.player_side`` (default ``'left'``) and the opposing side is driven by
the mode itself rather than by a second calculator.

Public API
----------
``MatchState``, ``PullResult``, ``MatchResult``, ``ModeRules``, ``ClassicMode``,
``BlitzMode``, ``SurvivalMode``, ``BossMode``, ``PracticeMode``,
``DailyChallenge``, ``make_mode``, ``MODE_KEYS``, ``MODE_INFO``, ``PowerUp``,
``POWERUPS``, ``maybe_spawn_powerup``, ``apply_powerup``, ``active_powerups``,
``has_powerup``, ``shield_charges``, ``is_frozen``, ``drain_events``,
``AIOpponent``.
"""

from __future__ import annotations

import datetime as _datetime
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from math_engine import (
    GRADES,
    Problem,
    ProblemStream,
    daily_seed,
    grade_label,
    grade_skill_weights,
    make_daily_set,
    skill_label,
    skills_for_grade,
)


__all__ = [
    "MatchState",
    "PullResult",
    "MatchResult",
    "ModeRules",
    "ClassicMode",
    "BlitzMode",
    "SurvivalMode",
    "BossMode",
    "PracticeMode",
    "DailyChallenge",
    "make_mode",
    "MODE_KEYS",
    "MODE_INFO",
    "PowerUp",
    "POWERUPS",
    "maybe_spawn_powerup",
    "apply_powerup",
    "active_powerups",
    "has_powerup",
    "shield_charges",
    "is_frozen",
    "drain_events",
    "AIOpponent",
    "SIDES",
]


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

#: The two ends of the rope. 'left' is EMBER, 'right' is FROST (see theme.py).
SIDES: Tuple[str, str] = ("left", "right")

#: Rope never leaves this band, in any mode, under any arithmetic.
ROPE_MIN = -1.0
ROPE_MAX = 1.0

_GRADE_ALIASES: Dict[str, str] = {
    "prek": "pre-k", "pre-k": "pre-k", "pk": "pre-k",
    "k": "k", "kindergarten": "k", "kinder": "k",
    "1": "1st", "1st": "1st", "first": "1st",
    "2": "2nd", "2nd": "2nd", "second": "2nd",
    "3": "3rd", "3rd": "3rd", "third": "3rd",
    "4": "4th", "4th": "4th", "fourth": "4th",
}


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _clamp01(x: float) -> float:
    return _clamp(float(x), 0.0, 1.0)


def _finite(x: float, fallback: float = 0.0) -> float:
    """Guarantee a real number. NaN/inf can never reach the rope."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _clamp_rope(x: float) -> float:
    """The single choke point every rope write goes through."""
    return _clamp(_finite(x, 0.0), ROPE_MIN, ROPE_MAX)


def _norm_grade(grade: Any) -> str:
    """Canonical grade key. Mirrors math_engine's tolerance without its privates."""
    if isinstance(grade, int):
        return GRADES[max(0, min(grade, len(GRADES) - 1))]
    g = str(grade).strip().lower().replace("_", "-").replace(" ", "-")
    if g in GRADES:
        return g
    return _GRADE_ALIASES.get(g, "k")


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


def _sign(side: str) -> float:
    """Which way this side drags the rope. Left pulls negative."""
    return -1.0 if side == "left" else 1.0


# --------------------------------------------------------------------------- #
# Core data types
# --------------------------------------------------------------------------- #


@dataclass
class MatchState:
    """Everything mutable about one match in progress.

    This is the only object that travels between main.py and the rules. It is
    plain data on purpose: it can be logged, diffed, snapshotted for a replay,
    or handed to a test harness without dragging any engine state along.
    """

    rope: float = 0.0                                  # -1..+1, negative = left
    time_left: float = 0.0                             # seconds; 0 when untimed
    elapsed: float = 0.0                               # seconds since on_start
    scores: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    streaks: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    best_streaks: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    lockouts: Dict[str, float] = field(default_factory=lambda: {"left": 0.0, "right": 0.0})
    correct: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    wrong: Dict[str, int] = field(default_factory=lambda: {"left": 0, "right": 0})
    modifiers: Dict[str, Any] = field(default_factory=dict)
    finished: bool = False

    def opponent(self, side: str) -> str:
        """The other end of the rope."""
        return _other(side)

    # ------------------------------------------------------------- convenience #
    def accuracy(self, side: str) -> float:
        attempts = self.correct[side] + self.wrong[side]
        return (self.correct[side] / attempts) if attempts else 0.0

    def leader(self, band: float = 0.0) -> Optional[str]:
        """Who is ahead right now, or None if inside `band` of dead centre."""
        if abs(self.rope) <= band:
            return None
        return "left" if self.rope < 0 else "right"


@dataclass
class PullResult:
    """What one answer did to the match.

    ``delta`` has *already been applied* to ``MatchState.rope`` by the time this
    is returned - main.py reads it purely to drive VFX (screen shake magnitude,
    particle count, rope snap direction).
    """

    delta: float = 0.0            # signed change applied to rope
    combo: int = 0                # streak length after this answer
    points: int = 0               # score added to this side
    events: List[str] = field(default_factory=list)

    @property
    def magnitude(self) -> float:
        """Unsigned strength of the pull - handy for shake/particle scaling."""
        return abs(self.delta)


@dataclass
class MatchResult:
    """The verdict, once and for all. Returned by `check_end` when done."""

    winner: Optional[str] = None   # 'left' | 'right' | None for a tie
    reason: str = "timeout"        # 'threshold' | 'timeout' | 'boss_defeated' | ...
    stars: int = 0                 # 0..3
    score: int = 0                 # the player-facing headline number
    stats: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Power-ups
# --------------------------------------------------------------------------- #


@dataclass
class PowerUp:
    """One collectable effect.

    ``icon`` is always a key that exists in ``icons.ICONS`` so main.py can draw
    it without a lookup failure. ``weight`` biases the spawn roll; ``duration``
    is how long the effect stays active (instant effects use a short duration
    purely so the HUD has something to flash).
    """

    key: str
    name: str
    icon: str
    duration: float
    weight: float


#: The full drop table. Keys are stable - saves and analytics reference them.
POWERUPS: Dict[str, PowerUp] = {
    "double_pull": PowerUp(
        key="double_pull", name="Double Pull", icon="zap",
        duration=6.0, weight=1.4,
    ),
    "freeze_opponent": PowerUp(
        key="freeze_opponent", name="Deep Freeze", icon="snowflake",
        duration=2.5, weight=1.0,
    ),
    "shield": PowerUp(
        key="shield", name="Shield", icon="shield",
        duration=12.0, weight=1.2,
    ),
    "time_bonus": PowerUp(
        key="time_bonus", name="Time Bonus", icon="timer",
        duration=0.8, weight=1.0,
    ),
    "skill_swap": PowerUp(
        key="skill_swap", name="Skill Swap", icon="refresh",
        duration=8.0, weight=0.9,
    ),
}

#: Seconds added by `time_bonus` (or of drag pressure rewound, in Survival).
TIME_BONUS_SECONDS = 6.0

#: How many wrong answers one `shield` pickup soaks up.
SHIELD_CHARGES = 1


# ------------------------------------------------------------ query helpers -- #


def active_powerups(state: MatchState, side: str) -> Dict[str, float]:
    """``{key: seconds_remaining}`` for the timed effects running on `side`."""
    return state.modifiers.setdefault("powerups", {"left": {}, "right": {}}).setdefault(
        side, {}
    )


def has_powerup(state: MatchState, side: str, key: str) -> bool:
    """True if `side` currently has the named timed effect running."""
    return active_powerups(state, side).get(key, 0.0) > 0.0


def shield_charges(state: MatchState, side: str) -> int:
    """How many wrong answers `side` can still absorb for free."""
    return int(state.modifiers.setdefault("shields", {"left": 0, "right": 0}).get(side, 0))


def is_frozen(state: MatchState, side: str) -> bool:
    """True while `side` is frozen out (its pulls do not land)."""
    return state.modifiers.setdefault("frozen", {"left": 0.0, "right": 0.0}).get(
        side, 0.0
    ) > 0.0


def drain_events(state: MatchState) -> List[str]:
    """Take the ambient events raised by `update()` (expiries, boss attacks).

    Answer-driven events arrive on `PullResult.events`; these are the ones that
    happen *between* answers. main.py should drain this once per frame.
    """
    events = state.modifiers.get("events", [])
    state.modifiers["events"] = []
    return list(events)


def _raise_event(state: MatchState, name: str) -> None:
    state.modifiers.setdefault("events", []).append(name)


# --------------------------------------------------------- spawn & apply ----- #


def maybe_spawn_powerup(state: MatchState, rng: random.Random) -> Optional[str]:
    """Roll for a power-up drop, respecting the mode's cooldown.

    Returns the power-up key that spawned (also parked on
    ``state.modifiers['pending_powerup']`` so main.py can float a pickup on the
    rope) or None. Cooldown is ticked down by `ModeRules.update`, so calling
    this on every correct answer is safe and cheap.
    """
    mods = state.modifiers
    if state.finished or not mods.get("powerups_enabled", True):
        return None
    if mods.get("pending_powerup"):
        return None                              # one drop in flight at a time
    if mods.get("spawn_cd", 0.0) > 0.0:
        return None
    if rng.random() > float(mods.get("spawn_chance", 0.35)):
        return None

    keys = list(POWERUPS.keys())
    weights = [POWERUPS[k].weight for k in keys]
    total = sum(weights)
    roll = rng.random() * total
    acc = 0.0
    chosen = keys[-1]
    for key, w in zip(keys, weights):
        acc += w
        if roll <= acc:
            chosen = key
            break

    mods["pending_powerup"] = chosen
    mods["spawn_cd"] = float(mods.get("spawn_cooldown", 9.0))
    return chosen


def apply_powerup(state: MatchState, side: str, key: str) -> None:
    """Grant `key` to `side`. Unknown keys are ignored rather than raising.

    Timed effects stack their duration onto whatever is already running so a
    second pickup always feels like a reward. Instant effects fire immediately.
    """
    spec = POWERUPS.get(key)
    if spec is None or side not in SIDES:
        return

    if key == "freeze_opponent":
        # Lands on the *opponent*, not the collector.
        frozen = state.modifiers.setdefault("frozen", {"left": 0.0, "right": 0.0})
        victim = _other(side)
        frozen[victim] = max(frozen.get(victim, 0.0), 0.0) + spec.duration
    elif key == "shield":
        shields = state.modifiers.setdefault("shields", {"left": 0, "right": 0})
        shields[side] = shields.get(side, 0) + SHIELD_CHARGES

    if key == "time_bonus":
        # Handled by the mode, because "more time" means different things in
        # Classic (clock) and Survival (drag pressure).
        state.modifiers.setdefault("time_bonus_pending", []).append(side)

    # Every power-up gets a timed entry so the HUD can show a countdown pill.
    active = active_powerups(state, side)
    active[key] = max(active.get(key, 0.0), 0.0) + spec.duration

    if state.modifiers.get("pending_powerup") == key:
        state.modifiers["pending_powerup"] = None
    state.modifiers.setdefault("collected", {"left": [], "right": []}).setdefault(
        side, []
    ).append(key)
    _raise_event(state, f"powerup_pickup:{side}:{key}")


# --------------------------------------------------------------------------- #
# AI opponent
# --------------------------------------------------------------------------- #


class AIOpponent:
    """A simulated player that answers on a timer.

    The goal is *plausibility*, not optimality. A metronome opponent reads as a
    machine and feels unfair; a person is bursty. So:

    * think time is log-normal - a long right tail, never a negative delay;
    * every so often the AI **hesitates** (distraction, re-reading the prompt);
    * the error rate falls super-linearly as `skill` rises, so a low-skill AI is
      genuinely beatable and a high-skill one is genuinely scary;
    * skill also tightens the *variance*, so strong opponents feel metronomic
      and weak ones feel scattered - which is how real players actually differ.

    `update(dt)` returns None on most frames, and True/False on the frame where
    the AI commits to an answer. main.py feeds that straight into
    `ModeRules.on_correct` / `on_wrong`.
    """

    #: Median think time at skill 0 for each grade's content, in seconds.
    BASE_SECONDS: Dict[str, float] = {
        "pre-k": 5.0,
        "k": 5.4,
        "1st": 6.0,
        "2nd": 7.0,
        "3rd": 7.6,
        "4th": 8.4,
    }

    #: Think time is clamped into this band so nothing ever stalls or spams.
    MIN_DELAY = 0.55
    MAX_DELAY = 22.0

    def __init__(
        self,
        grade: str,
        skill: float,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.grade = _norm_grade(grade)
        self.rng = rng if rng is not None else random.Random()
        self.skill = _clamp01(skill)

        # Running record, purely for HUD / post-match copy.
        self.answered = 0
        self.correct = 0
        self.wrong = 0

        self._frozen = 0.0
        self._lockout = 0.0
        self._timer = self._roll_delay()

    # ------------------------------------------------------------------ tuning #
    def set_skill(self, skill: float) -> None:
        """Retune mid-match (rubber-banding, boss phases, difficulty ramps).

        The pending answer is left alone - yanking the current timer would read
        as the opponent teleporting.
        """
        self.skill = _clamp01(skill)

    @property
    def error_rate(self) -> float:
        """Chance the next answer is wrong. Falls off fast as skill rises."""
        return _clamp(0.03 + 0.44 * (1.0 - self.skill) ** 1.5, 0.02, 0.55)

    @property
    def median_delay(self) -> float:
        """Median think time at the current skill - what the AI *feels* like."""
        base = self.BASE_SECONDS.get(self.grade, 6.0)
        return base * (1.0 - 0.58 * self.skill)

    # ----------------------------------------------------------------- timing #
    def _roll_delay(self) -> float:
        """Log-normal think time with an occasional hesitation on top."""
        median = max(0.4, self.median_delay)
        # sigma shrinks with skill: experts are consistent, novices are not.
        sigma = 0.52 - 0.24 * self.skill
        try:
            delay = self.rng.lognormvariate(math.log(median), sigma)
        except (ValueError, OverflowError):  # pragma: no cover - defensive
            delay = median
        # Hesitation: a beat of "wait, what?" that novices get much more often.
        if self.rng.random() < 0.10 + 0.16 * (1.0 - self.skill):
            delay += self.rng.uniform(0.5, 2.4) * (1.4 - 0.7 * self.skill)
        return _clamp(_finite(delay, median), self.MIN_DELAY, self.MAX_DELAY)

    def freeze(self, seconds: float) -> None:
        """Stop the AI thinking for a while (the `freeze_opponent` power-up)."""
        self._frozen = max(self._frozen, max(0.0, _finite(seconds, 0.0)))

    def lockout(self, seconds: float) -> None:
        """Apply the same post-wrong penalty a human gets."""
        self._lockout = max(self._lockout, max(0.0, _finite(seconds, 0.0)))

    # ------------------------------------------------------------------ update #
    def update(self, dt: float) -> Optional[bool]:
        """Advance the AI. Returns True/False on the frame it answers."""
        dt = max(0.0, _finite(dt, 0.0))
        if dt <= 0.0:
            return None

        if self._frozen > 0.0:
            self._frozen -= dt
            return None
        if self._lockout > 0.0:
            self._lockout -= dt
            return None

        self._timer -= dt
        if self._timer > 0.0:
            return None

        # Carry the overshoot into the next problem so fast frames and slow
        # frames produce the same long-run answer rate.
        self._timer = max(0.05, self._timer + self._roll_delay())
        ok = self.rng.random() >= self.error_rate
        self.answered += 1
        if ok:
            self.correct += 1
        else:
            self.wrong += 1
        return ok

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<AIOpponent {self.grade} skill={self.skill:.2f} "
            f"median={self.median_delay:.1f}s err={self.error_rate:.0%}>"
        )


# --------------------------------------------------------------------------- #
# Base rules
# --------------------------------------------------------------------------- #


class ModeRules:
    """Shared machinery for every mode.

    Subclasses override the tuning constants and, where a mode genuinely
    differs, the `_update_mode` / `check_end` / `hud_extras` hooks. Everything
    else - streaks, combos, lockouts, power-up lifecycle, score, star maths -
    lives here exactly once.
    """

    # ------------------------------------------------------------- identity -- #
    key: str = "base"
    name: str = "Base"
    solo: bool = False
    icon: str = "rope"
    blurb: str = ""

    # -------------------------------------------------------------- tuning --- #
    duration: float = 90.0          # match clock, seconds
    timed: bool = True              # False -> time_left is ignored entirely
    threshold: float = 1.0          # |rope| that ends the match
    tie_band: float = 0.05          # |rope| under this at timeout == a draw
    base_pull: float = 0.085        # rope per correct answer, before modifiers
    wrong_gain: float = 0.060       # rope handed to the opponent on a miss
    lockout_seconds: float = 1.4    # input freeze after a miss
    combo_cap: int = 6              # streak length where the multiplier caps
    combo_step: float = 0.12        # multiplier gained per streak step
    max_seconds: float = 600.0      # hard safety cap; no mode may outlive it

    # ------------------------------------------------------------ power-ups -- #
    powerups_enabled: bool = True
    powerup_chance: float = 0.35    # roll per eligible correct answer
    powerup_cooldown: float = 9.0   # seconds between drops
    streak_spawn_every: int = 3     # a streak of this many rolls for a drop

    # ------------------------------------------------------------------------ #
    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        self.config = cfg

        self.grade: str = _norm_grade(cfg.get("grade", "2nd"))
        self.difficulty: float = _clamp01(cfg.get("difficulty", 0.5))
        self.player_side: str = cfg.get("player_side", "left")
        if self.player_side not in SIDES:
            self.player_side = "left"
        self.ai_skill: float = _clamp01(cfg.get("ai_skill", 0.5))
        self.weak_skills: List[str] = list(cfg.get("weak_skills", []) or [])

        # Per-instance overrides for the tuning table above.
        for name in ("duration", "threshold", "base_pull", "wrong_gain",
                     "lockout_seconds", "max_seconds", "powerup_chance",
                     "powerup_cooldown", "tie_band"):
            if name in cfg:
                setattr(self, name, float(cfg[name]))
        if "combo_cap" in cfg:
            self.combo_cap = max(1, int(cfg["combo_cap"]))
        if "powerups_enabled" in cfg:
            self.powerups_enabled = bool(cfg["powerups_enabled"])

        self.seed: Optional[int] = cfg.get("seed")
        self.rng = random.Random(self.seed)

        # One adaptive stream per side; solo modes simply never pull the other.
        stream_seed = self.seed
        self.streams: Dict[str, ProblemStream] = {
            side: ProblemStream(
                self.grade,
                self.difficulty,
                weak_skills=self.weak_skills,
                seed=(None if stream_seed is None else stream_seed + i),
            )
            for i, side in enumerate(SIDES)
        }

        self._result: Optional[MatchResult] = None

    # ------------------------------------------------------------- lifecycle - #
    def new_state(self) -> MatchState:
        """A fresh, fully-initialised match. Safe to call repeatedly."""
        self._result = None
        state = MatchState(
            rope=0.0,
            time_left=float(self.duration) if self.timed else 0.0,
            elapsed=0.0,
            modifiers={
                "mode": self.key,
                "solo": self.solo,
                "player_side": self.player_side,
                "grade": self.grade,
                # power-up bookkeeping
                "powerups": {"left": {}, "right": {}},
                "shields": {"left": 0, "right": 0},
                "frozen": {"left": 0.0, "right": 0.0},
                "swap_pending": {"left": False, "right": False},
                "collected": {"left": [], "right": []},
                "pending_powerup": None,
                "powerups_enabled": self.powerups_enabled,
                "spawn_chance": self.powerup_chance,
                "spawn_cooldown": self.powerup_cooldown,
                "spawn_cd": self.powerup_cooldown * 0.5,
                "time_bonus_pending": [],
                "events": [],
            },
        )
        self._init_mode(state)
        return state

    def _init_mode(self, state: MatchState) -> None:
        """Hook: seed any mode-specific entries in `state.modifiers`."""

    def on_start(self, state: MatchState) -> None:
        """Called once when the countdown finishes and play actually begins."""
        state.finished = False
        state.elapsed = 0.0
        _raise_event(state, "match_start")

    # -------------------------------------------------------------- content -- #
    def next_problem(self, state: MatchState, side: Optional[str] = None) -> Problem:
        """Serve the next problem for `side` (defaults to the human player).

        Modes bend the skill mix here: Practice hunts weaknesses, Boss follows
        its phase table, Daily replays a fixed deterministic list.
        """
        side = side or self.player_side
        stream = self.streams.get(side) or self.streams[self.player_side]
        return stream.next(self.skills_for(state, side))

    def skills_for(self, state: MatchState, side: str) -> Optional[List[str]]:
        """Hook: restrict the skill pool for the next draw. None == no filter."""
        return None

    def consume_swap(self, state: MatchState, side: str) -> bool:
        """True (once) if `skill_swap` wants main.py to re-roll `side`'s problem."""
        pending = state.modifiers.setdefault(
            "swap_pending", {"left": False, "right": False}
        )
        if pending.get(side):
            pending[side] = False
            return True
        return False

    # --------------------------------------------------------------- answers - #
    def on_correct(
        self,
        state: MatchState,
        side: str,
        problem: Problem,
        elapsed: float,
    ) -> PullResult:
        """A correct answer from `side`, `elapsed` seconds after it was served."""
        if state.finished or side not in SIDES:
            return PullResult(0.0, state.streaks.get(side, 0), 0, ["ignored"])
        if state.lockouts.get(side, 0.0) > 0.0:
            # main.py blocks input during a lockout; this is belt-and-braces.
            return PullResult(0.0, state.streaks[side], 0, ["locked"])

        events: List[str] = ["correct"]
        elapsed = max(0.0, _finite(elapsed, 0.0))

        # --- speed: answering well inside the budget is worth ~2.2x a crawl ---
        budget = max(0.5, _finite(getattr(problem, "time_budget", 8.0), 8.0))
        pace = _clamp(elapsed / budget, 0.0, 2.0)
        speed = _clamp(1.45 - 0.75 * pace, 0.65, 1.45)

        # --- combo -----------------------------------------------------------
        state.streaks[side] += 1
        combo = state.streaks[side]
        state.best_streaks[side] = max(state.best_streaks[side], combo)
        mult = 1.0 + self.combo_step * min(combo - 1, self.combo_cap)
        if combo > 1:
            events.append("streak_up")
        if combo >= 3 and combo % self.streak_spawn_every == 0:
            events.append("shockwave")

        # --- weight: harder content moves more rope --------------------------
        weight = _clamp(_finite(getattr(problem, "weight", 1.0), 1.0), 0.5, 2.0)

        pull = self.base_pull * speed * mult * weight
        pull = self._scale_pull(state, side, problem, pull, events)

        if has_powerup(state, side, "double_pull"):
            pull *= 2.0
            events.append("double_pull")

        # Frozen sides still *score* - they simply cannot move the rope.
        if is_frozen(state, side):
            pull = 0.0
            events.append("frozen")

        delta = _finite(pull, 0.0) * _sign(side)
        before = state.rope
        state.rope = _clamp_rope(state.rope + delta)
        delta = state.rope - before          # report what actually landed

        points = max(1, int(round(10.0 * weight * speed * mult)))
        state.scores[side] += points
        state.correct[side] += 1

        stream = self.streams.get(side)
        if stream is not None:
            stream.report(problem, True, elapsed)

        self._on_correct_extra(state, side, problem, elapsed, abs(delta), events)

        if self._eligible_for_drop(state, side, combo):
            if maybe_spawn_powerup(state, self.rng):
                events.append("powerup_spawn")

        return PullResult(delta=delta, combo=combo, points=points, events=events)

    def on_wrong(self, state: MatchState, side: str, problem: Problem) -> PullResult:
        """A wrong answer from `side`: lockout, lost streak, ground given up."""
        if state.finished or side not in SIDES:
            return PullResult(0.0, state.streaks.get(side, 0), 0, ["ignored"])

        events: List[str] = ["wrong"]
        state.wrong[side] += 1

        stream = self.streams.get(side)
        if stream is not None:
            stream.report(problem, False, getattr(problem, "time_budget", 8.0))

        # --- shield: eats the whole penalty, streak included ------------------
        shields = state.modifiers.setdefault("shields", {"left": 0, "right": 0})
        if shields.get(side, 0) > 0:
            shields[side] -= 1
            if shields[side] <= 0:
                active_powerups(state, side).pop("shield", None)
            events.append("shield_block")
            self._on_wrong_extra(state, side, problem, events)
            return PullResult(0.0, state.streaks[side], 0, events)

        # --- skill_swap: mercy rule, the streak survives one miss -------------
        if has_powerup(state, side, "skill_swap"):
            events.append("streak_saved")
        else:
            state.streaks[side] = 0

        lock = self._lockout_for(state, side)
        if lock > 0.0:
            state.lockouts[side] = max(state.lockouts.get(side, 0.0), lock)
            events.append("lockout")

        gain = self._wrong_gain_for(state, side)
        delta = 0.0
        if gain > 0.0:
            before = state.rope
            state.rope = _clamp_rope(state.rope + gain * _sign(_other(side)))
            delta = state.rope - before
            if delta:
                events.append("opponent_gain")

        self._on_wrong_extra(state, side, problem, events)
        return PullResult(delta=delta, combo=0, points=0, events=events)

    # ------------------------------------------------------------ mode hooks - #
    def _scale_pull(
        self,
        state: MatchState,
        side: str,
        problem: Problem,
        pull: float,
        events: List[str],
    ) -> float:
        """Hook: last chance to bend the raw pull before power-ups apply."""
        return pull

    def _on_correct_extra(
        self,
        state: MatchState,
        side: str,
        problem: Problem,
        elapsed: float,
        magnitude: float,
        events: List[str],
    ) -> None:
        """Hook: boss damage, survival relief, practice bookkeeping."""

    def _on_wrong_extra(
        self,
        state: MatchState,
        side: str,
        problem: Problem,
        events: List[str],
    ) -> None:
        """Hook: hints, hazard resets, boss counter-attacks."""

    def _lockout_for(self, state: MatchState, side: str) -> float:
        return float(self.lockout_seconds)

    def _wrong_gain_for(self, state: MatchState, side: str) -> float:
        return float(self.wrong_gain)

    def _eligible_for_drop(self, state: MatchState, side: str, combo: int) -> bool:
        """Drops are earned by streaks, not handed out on every answer."""
        return (
            self.powerups_enabled
            and combo >= self.streak_spawn_every
            and combo % self.streak_spawn_every == 0
        )

    def _grant_time(self, state: MatchState, side: str, seconds: float) -> None:
        """How `time_bonus` cashes out. Overridden by the untimed modes."""
        if self.timed:
            state.time_left = max(0.0, _finite(state.time_left, 0.0) + seconds)
            _raise_event(state, f"time_bonus:{side}")

    # ----------------------------------------------------------------- update #
    def update(self, state: MatchState, dt: float) -> None:
        """Advance every clock the rules own. Call once per frame."""
        if state.finished:
            return
        dt = max(0.0, _finite(dt, 0.0))
        if dt <= 0.0:
            return

        state.elapsed = _finite(state.elapsed, 0.0) + dt
        if self.timed:
            state.time_left = max(0.0, _finite(state.time_left, 0.0) - dt)

        # --- lockouts ---------------------------------------------------------
        for side in SIDES:
            if state.lockouts.get(side, 0.0) > 0.0:
                state.lockouts[side] = max(0.0, state.lockouts[side] - dt)
                if state.lockouts[side] <= 0.0:
                    _raise_event(state, f"lockout_end:{side}")

        # --- freezes ----------------------------------------------------------
        frozen = state.modifiers.setdefault("frozen", {"left": 0.0, "right": 0.0})
        for side in SIDES:
            if frozen.get(side, 0.0) > 0.0:
                frozen[side] = max(0.0, frozen[side] - dt)
                if frozen[side] <= 0.0:
                    _raise_event(state, f"unfrozen:{side}")

        # --- instant effects queued by apply_powerup --------------------------
        pending_time = state.modifiers.get("time_bonus_pending") or []
        if pending_time:
            state.modifiers["time_bonus_pending"] = []
            for side in pending_time:
                self._grant_time(state, side, TIME_BONUS_SECONDS)

        # --- timed effects tick down and expire -------------------------------
        for side in SIDES:
            active = active_powerups(state, side)
            for key in list(active.keys()):
                active[key] -= dt
                if active[key] <= 0.0:
                    del active[key]
                    if key == "shield":
                        # An unused shield lapses with its pill.
                        state.modifiers["shields"][side] = 0
                    if key == "skill_swap":
                        state.modifiers["swap_pending"][side] = False
                    _raise_event(state, f"powerup_end:{side}:{key}")
                elif key == "skill_swap":
                    state.modifiers["swap_pending"][side] = True

        # --- drop cooldown ----------------------------------------------------
        if state.modifiers.get("spawn_cd", 0.0) > 0.0:
            state.modifiers["spawn_cd"] = max(0.0, state.modifiers["spawn_cd"] - dt)

        # --- mode-specific simulation ----------------------------------------
        self._update_mode(state, dt)

        state.rope = _clamp_rope(state.rope)
        self.check_end(state)

    def _update_mode(self, state: MatchState, dt: float) -> None:
        """Hook: survival drag, boss attacks, hazards."""

    # ------------------------------------------------------------------- end - #
    def check_end(self, state: MatchState) -> Optional[MatchResult]:
        """The verdict, or None while the match is still live. Idempotent."""
        if self._result is not None:
            return self._result

        winner, reason = self._end_condition(state)
        if reason is None:
            return None

        state.finished = True
        self._result = self._build_result(state, winner, reason)
        _raise_event(state, f"match_end:{reason}")
        return self._result

    def _end_condition(
        self, state: MatchState
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(winner, reason)``; a reason of None means 'keep playing'."""
        if state.rope <= -self.threshold:
            return "left", "threshold"
        if state.rope >= self.threshold:
            return "right", "threshold"
        if self.timed and state.time_left <= 0.0:
            return state.leader(self.tie_band), "timeout"
        if state.elapsed >= self.max_seconds:
            return state.leader(self.tie_band), "timeout"
        return None, None

    def _build_result(
        self, state: MatchState, winner: Optional[str], reason: str
    ) -> MatchResult:
        side = self.player_side
        score = int(state.scores.get(side, 0))
        return MatchResult(
            winner=winner,
            reason=reason,
            stars=self._stars(state, winner, reason),
            score=score,
            stats=self._stats(state, winner, reason),
        )

    def _stars(
        self, state: MatchState, winner: Optional[str], reason: str
    ) -> int:
        """0-3 stars from the human's perspective: won, accurate, dominant."""
        side = self.player_side
        stars = 0
        if winner == side:
            stars += 1
        acc = state.accuracy(side)
        if acc >= 0.70 and state.correct[side] >= 3:
            stars += 1
        if acc >= 0.90 and state.best_streaks[side] >= 5:
            stars += 1
        return int(_clamp(stars, 0, 3))

    def _stats(
        self, state: MatchState, winner: Optional[str], reason: str
    ) -> Dict[str, Any]:
        """The end-of-match panel's data. Every value is JSON-safe."""
        side = self.player_side
        stats: Dict[str, Any] = {
            "mode": self.key,
            "mode_name": self.name,
            "solo": self.solo,
            "grade": self.grade,
            "grade_label": grade_label(self.grade),
            "player_side": side,
            "winner": winner,
            "reason": reason,
            "elapsed": round(state.elapsed, 2),
            "rope": round(state.rope, 4),
            "scores": dict(state.scores),
            "correct": dict(state.correct),
            "wrong": dict(state.wrong),
            "best_streaks": dict(state.best_streaks),
            "accuracy": round(state.accuracy(side), 3),
            "powerups_used": list(
                state.modifiers.get("collected", {}).get(side, [])
            ),
        }
        stream = self.streams.get(side)
        if stream is not None:
            stats["stream"] = stream.summary()
        return stats

    # ------------------------------------------------------------------- HUD - #
    def hud_extras(self, state: MatchState) -> dict:
        """Everything main.py needs to draw the mode-specific chrome."""
        return {
            "mode": self.key,
            "mode_name": self.name,
            "icon": self.icon,
            "solo": self.solo,
            "timed": self.timed,
            "player_side": self.player_side,
            "rope": round(state.rope, 4),
            "time_left": round(state.time_left, 2) if self.timed else None,
            "threshold": self.threshold,
            "combo_mult": {
                s: round(
                    1.0
                    + self.combo_step
                    * min(max(state.streaks[s] - 1, 0), self.combo_cap),
                    3,
                )
                for s in SIDES
            },
            "powerups": {
                s: {k: round(v, 2) for k, v in active_powerups(state, s).items()}
                for s in SIDES
            },
            "shields": {s: shield_charges(state, s) for s in SIDES},
            "frozen": {s: is_frozen(state, s) for s in SIDES},
            "lockouts": {s: round(state.lockouts[s], 2) for s in SIDES},
            "pending_powerup": state.modifiers.get("pending_powerup"),
            "pending_icon": (
                POWERUPS[state.modifiers["pending_powerup"]].icon
                if state.modifiers.get("pending_powerup")
                else None
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.key} grade={self.grade}>"


# --------------------------------------------------------------------------- #
# 1. Classic
# --------------------------------------------------------------------------- #


class ClassicMode(ModeRules):
    """The headline mode: 90 seconds, two calculators, one rope.

    Correct answers pull toward you and are scaled by how fast you solved and
    how long your streak is. A miss locks you out for a beat and hands the
    other side ground - which is what makes a streak worth protecting.
    """

    key = "classic"
    name = "Classic"
    solo = False
    icon = "rope"
    blurb = "90 seconds. Two teams. One rope."

    duration = 90.0
    base_pull = 0.085
    wrong_gain = 0.060
    lockout_seconds = 1.4
    combo_cap = 6
    combo_step = 0.12
    max_seconds = 240.0

    def _stars(self, state: MatchState, winner, reason) -> int:
        """A win is table stakes; the extra stars are for *how* you won.

        Deliberately not gated on `reason`: winning on the clock is still a
        win, and a scrappy 60%-accuracy victory should read as one star rather
        than being lumped in with a clean sweep.
        """
        side = self.player_side
        if winner != side:
            return 0
        stars = 1
        if state.accuracy(side) >= 0.70:
            stars += 1
        if state.best_streaks[side] >= 6 and state.accuracy(side) >= 0.85:
            stars += 1
        return int(_clamp(stars, 0, 3))


# --------------------------------------------------------------------------- #
# 2. Blitz
# --------------------------------------------------------------------------- #


class BlitzMode(ModeRules):
    """One minute, no lockout, tiny pulls, enormous combos.

    Classic rewards accuracy under pressure. Blitz rewards *rate*: a miss costs
    you nothing but the streak you were building, so the only losing move is
    stopping to think. The combo ceiling is doubled to make a long clean run
    genuinely explosive.
    """

    key = "blitz"
    name = "Blitz"
    solo = False
    icon = "bolt"
    blurb = "60 seconds. No penalties. Pure speed."

    duration = 60.0
    base_pull = 0.052
    wrong_gain = 0.030
    lockout_seconds = 0.0          # the whole point of the mode
    combo_cap = 12
    combo_step = 0.10
    tie_band = 0.04
    max_seconds = 180.0
    powerup_chance = 0.5
    powerup_cooldown = 6.0
    streak_spawn_every = 4

    def _stars(self, state: MatchState, winner, reason) -> int:
        """Blitz stars are about throughput, not the scoreline."""
        side = self.player_side
        stars = 1 if winner == side else 0
        if state.correct[side] >= 12:
            stars += 1
        if state.best_streaks[side] >= 10:
            stars += 1
        return int(_clamp(stars, 0, 3))


# --------------------------------------------------------------------------- #
# 3. Survival
# --------------------------------------------------------------------------- #


class SurvivalMode(ModeRules):
    """Solo. Something on the other end pulls, forever, harder and harder.

    There is no winning - only lasting. The drag rate climbs linearly with
    elapsed time, so a player's answer rate eventually *cannot* match it and
    the rope goes over. That guarantee is what makes the mode terminate.

    Score is whole seconds survived; `best_seconds` persists on the mode
    instance so main.py can show "personal best" without a save round-trip
    (pass the stored value in as ``config['best_seconds']``).
    """

    key = "survival"
    name = "Survival"
    solo = True
    icon = "heart"
    blurb = "Hold the line. It only gets stronger."

    timed = False
    duration = 0.0
    base_pull = 0.115
    wrong_gain = 0.055
    lockout_seconds = 1.0
    combo_cap = 8
    combo_step = 0.11
    max_seconds = 900.0            # absolute backstop; the drag wins long before

    #: Rope units per second the AI drags at t=0, and how much that grows per
    #: second of pressure. Tuned so a competent player lasts 60-150s.
    base_drag = 0.020
    drag_growth = 0.0016
    #: How much buffer a player may bank on their own side. Without a floor a
    #: strong player could park at -1.0 and stall the mode forever.
    player_floor = -0.60

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.best_seconds: float = float(self.config.get("best_seconds", 0.0) or 0.0)

    # ------------------------------------------------------------------------ #
    @property
    def _ai_side(self) -> str:
        return _other(self.player_side)

    def _init_mode(self, state: MatchState) -> None:
        state.modifiers["pressure"] = 0.0        # the "difficulty clock"
        state.modifiers["drag_rate"] = self.base_drag
        state.modifiers["best_seconds"] = self.best_seconds
        state.modifiers["survived"] = 0.0

    def _grant_time(self, state: MatchState, side: str, seconds: float) -> None:
        """No clock to extend - rewind the drag pressure instead."""
        state.modifiers["pressure"] = max(
            0.0, float(state.modifiers.get("pressure", 0.0)) - seconds
        )
        _raise_event(state, f"pressure_relief:{side}")

    def _update_mode(self, state: MatchState, dt: float) -> None:
        # Pressure only builds while the player is actually playing.
        pressure = float(state.modifiers.get("pressure", 0.0)) + dt
        state.modifiers["pressure"] = pressure
        state.modifiers["survived"] = state.elapsed

        rate = self.base_drag + self.drag_growth * pressure
        state.modifiers["drag_rate"] = rate

        # Freezing the dragger is the only way to buy a breather.
        if is_frozen(state, self._ai_side):
            return

        state.rope = _clamp_rope(state.rope + rate * dt * _sign(self._ai_side))
        # Keep the player from banking an unbounded (mode-stalling) buffer.
        if self.player_side == "left":
            state.rope = max(state.rope, self.player_floor)
        else:
            state.rope = min(state.rope, -self.player_floor)

    def _end_condition(self, state):
        """Only one way out: the rope is dragged over the player's line."""
        ai_side = self._ai_side
        if _sign(ai_side) * state.rope >= self.threshold:
            return ai_side, "dragged_over"
        if state.elapsed >= self.max_seconds:
            return ai_side, "timeout"
        return None, None

    def _build_result(self, state, winner, reason) -> MatchResult:
        seconds = state.elapsed
        record = seconds > self.best_seconds
        self.best_seconds = max(self.best_seconds, seconds)
        result = super()._build_result(state, winner, reason)
        result.score = int(seconds)
        result.stats["seconds_survived"] = round(seconds, 2)
        result.stats["best_seconds"] = round(self.best_seconds, 2)
        result.stats["new_record"] = bool(record)
        result.stats["final_drag_rate"] = round(
            float(state.modifiers.get("drag_rate", 0.0)), 4
        )
        return result

    def _stars(self, state, winner, reason) -> int:
        """Stars are pure endurance milestones."""
        seconds = state.elapsed
        stars = 0
        for gate in (30.0, 60.0, 100.0):
            if seconds >= gate:
                stars += 1
        return int(_clamp(stars, 0, 3))

    def hud_extras(self, state: MatchState) -> dict:
        extras = super().hud_extras(state)
        ai_side = self._ai_side
        # 0 at safe, 1 at the losing line - drives the red vignette.
        danger = _clamp01((_sign(ai_side) * state.rope + 0.4) / 1.4)
        extras.update(
            {
                "survived": round(state.elapsed, 1),
                "best_seconds": round(self.best_seconds, 1),
                "drag_rate": round(float(state.modifiers.get("drag_rate", 0.0)), 4),
                "pressure": round(float(state.modifiers.get("pressure", 0.0)), 1),
                "danger": round(danger, 3),
                "ai_side": ai_side,
            }
        )
        return extras


# --------------------------------------------------------------------------- #
# 4. Boss
# --------------------------------------------------------------------------- #


@dataclass
class _BossPhase:
    """One third of a boss fight: its look, its pace and its hazard."""

    index: int
    name: str
    icon: str
    attack_period: float      # seconds between attacks
    attack_power: float       # rope taken per landed attack
    hazard_limit: float       # seconds to answer before the hazard bites
    hazard_penalty: float     # rope lost when it does
    damage_scale: float       # how well the player's answers land this phase


class BossMode(ModeRules):
    """Solo. A three-phase boss with an HP bar, a hazard, and a tell.

    Correct answers damage the boss *and* pull rope. The boss answers back on a
    timer: it winds up for ``telegraph_lead`` seconds before every attack, and
    that wind-up is published through `hud_extras` as ``telegraph`` /
    ``telegraph_t`` so main.py can flash a warning ring and shake the camera a
    beat before the hit actually lands.

    Each phase also runs a **hazard**: the current problem must be answered
    inside ``hazard_limit`` seconds or the player loses ground. The clock resets
    on every answer, and main.py can reset it explicitly via `on_problem`.

    Phases shift the skill mix too - the fight opens on the grade's bread and
    butter and closes on everything, weighted toward what the player is weak at.
    """

    key = "boss"
    name = "Boss Battle"
    solo = True
    icon = "crown"
    blurb = "Three phases. One very rude opponent."

    duration = 0.0
    timed = False
    base_pull = 0.070
    wrong_gain = 0.045
    lockout_seconds = 1.2
    combo_cap = 8
    combo_step = 0.13
    max_seconds = 300.0
    powerup_chance = 0.45
    powerup_cooldown = 8.0

    #: Boss HP and how much a clean answer takes off it.
    #:
    #: Tuned against a 60fps simulation at a realistic ~3s answer cadence.
    #: The original 100 HP / 4.6 dmg made this the *easiest* mode in the game
    #: - a coin-flip player beat it 62% of the time - because the fight ended
    #: long before the later phases could apply any pressure. A boss should be
    #: the hardest thing on the menu, so it now takes roughly twice as many
    #: clean answers and the attacks bite harder as the phases escalate.
    max_hp = 185.0
    base_damage = 4.4
    #: How long the wind-up warning is visible before an attack connects.
    telegraph_lead = 1.1

    PHASES: Tuple[_BossPhase, ...] = (
        _BossPhase(1, "Warm-Up", "flame", 6.0, 0.090, 10.0, 0.065, 1.00),
        _BossPhase(2, "Pressure", "bolt", 4.4, 0.115, 7.5, 0.085, 0.90),
        _BossPhase(3, "Final Form", "crown", 3.2, 0.145, 5.5, 0.110, 0.80),
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.max_hp = float(self.config.get("boss_hp", self.max_hp))
        # Phase skill mixes, widening as the fight escalates.
        ordered = sorted(
            grade_skill_weights(self.grade).items(), key=lambda kv: -kv[1]
        )
        names = [k for k, _w in ordered] or skills_for_grade(self.grade) or ["add"]
        n = len(names)
        self._phase_skills: Tuple[List[str], ...] = (
            names[: max(2, n // 3)],
            names[: max(3, (2 * n) // 3)],
            names,
        )

    # ------------------------------------------------------------------------ #
    @property
    def _boss_side(self) -> str:
        return _other(self.player_side)

    def phase(self, state: MatchState) -> _BossPhase:
        """The phase currently running (HP-driven, never goes backwards)."""
        idx = int(_clamp(state.modifiers.get("boss_phase", 1), 1, len(self.PHASES)))
        return self.PHASES[idx - 1]

    def _init_mode(self, state: MatchState) -> None:
        state.modifiers.update(
            {
                "boss_hp": self.max_hp,
                "boss_max_hp": self.max_hp,
                "boss_phase": 1,
                "attack_in": self.PHASES[0].attack_period,
                "telegraph": False,
                "hazard_left": self.PHASES[0].hazard_limit,
                "attacks_landed": 0,
                "attacks_blocked": 0,
                "hazard_fails": 0,
            }
        )

    # ---------------------------------------------------------------- content #
    def skills_for(self, state: MatchState, side: str) -> Optional[List[str]]:
        phase = self.phase(state)
        skills = list(self._phase_skills[phase.index - 1])
        if phase.index == 3:
            # The last phase leans on whatever the player has been fluffing.
            stream = self.streams.get(side)
            if stream is not None:
                skills += [s for s in stream.weak_skills(2) if s not in skills]
        return skills or None

    def on_problem(self, state: MatchState, side: str, problem: Problem) -> None:
        """Optional hook: main.py calls this when a new problem goes on screen.

        It restarts the hazard clock. If main.py never calls it the hazard still
        works - `on_correct`/`on_wrong` reset the clock too - it just measures
        from the last *answer* rather than the last *serve*.
        """
        state.modifiers["hazard_left"] = self.phase(state).hazard_limit

    def _reset_hazard(self, state: MatchState) -> None:
        state.modifiers["hazard_left"] = self.phase(state).hazard_limit

    # ---------------------------------------------------------------- answers #
    def _scale_pull(self, state, side, problem, pull, events) -> float:
        if side != self.player_side:
            return pull
        return pull * self.phase(state).damage_scale

    def _on_correct_extra(
        self, state, side, problem, elapsed, magnitude, events
    ) -> None:
        self._reset_hazard(state)
        if side != self.player_side:
            return

        phase = self.phase(state)
        budget = max(0.5, _finite(getattr(problem, "time_budget", 8.0), 8.0))
        speed = _clamp(1.5 - 0.8 * _clamp(elapsed / budget, 0.0, 2.0), 0.6, 1.5)
        mult = 1.0 + self.combo_step * min(
            max(state.streaks[side] - 1, 0), self.combo_cap
        )
        damage = self.base_damage * speed * mult * phase.damage_scale
        if has_powerup(state, side, "double_pull"):
            damage *= 2.0

        hp = max(0.0, float(state.modifiers.get("boss_hp", self.max_hp)) - damage)
        state.modifiers["boss_hp"] = hp
        state.modifiers["last_damage"] = round(damage, 2)
        events.append("boss_hit")

        self._sync_phase(state, events)

    def _on_wrong_extra(self, state, side, problem, events) -> None:
        self._reset_hazard(state)

    def _sync_phase(self, state: MatchState, events: List[str]) -> None:
        """Advance the phase when HP crosses a third. Never regresses."""
        frac = float(state.modifiers.get("boss_hp", 0.0)) / max(1e-6, self.max_hp)
        target = 1 if frac > 2.0 / 3.0 else (2 if frac > 1.0 / 3.0 else 3)
        current = int(state.modifiers.get("boss_phase", 1))
        if target > current:
            state.modifiers["boss_phase"] = target
            phase = self.PHASES[target - 1]
            state.modifiers["attack_in"] = phase.attack_period
            state.modifiers["hazard_left"] = phase.hazard_limit
            state.modifiers["telegraph"] = False
            events.append("boss_phase")
            _raise_event(state, f"boss_phase:{target}")

    # ----------------------------------------------------------------- update #
    def _update_mode(self, state: MatchState, dt: float) -> None:
        phase = self.phase(state)
        player = self.player_side
        boss = self._boss_side

        # --- hazard: answer inside the window or give up ground ---------------
        hazard = float(state.modifiers.get("hazard_left", phase.hazard_limit)) - dt
        if hazard <= 0.0:
            state.modifiers["hazard_fails"] = (
                int(state.modifiers.get("hazard_fails", 0)) + 1
            )
            hazard = phase.hazard_limit
            state.streaks[player] = 0
            state.rope = _clamp_rope(
                state.rope + phase.hazard_penalty * _sign(boss)
            )
            _raise_event(state, "hazard_fail")
        state.modifiers["hazard_left"] = hazard

        # --- attack: telegraph, then land -------------------------------------
        attack_in = float(state.modifiers.get("attack_in", phase.attack_period)) - dt
        telegraphing = 0.0 < attack_in <= self.telegraph_lead
        if telegraphing and not state.modifiers.get("telegraph"):
            _raise_event(state, "boss_telegraph")
        state.modifiers["telegraph"] = telegraphing

        if attack_in <= 0.0:
            attack_in += phase.attack_period
            state.modifiers["telegraph"] = False
            if is_frozen(state, boss):
                _raise_event(state, "boss_attack_frozen")
            else:
                shields = state.modifiers.setdefault(
                    "shields", {"left": 0, "right": 0}
                )
                if shields.get(player, 0) > 0:
                    shields[player] -= 1
                    if shields[player] <= 0:
                        active_powerups(state, player).pop("shield", None)
                    state.modifiers["attacks_blocked"] = (
                        int(state.modifiers.get("attacks_blocked", 0)) + 1
                    )
                    _raise_event(state, "boss_attack_blocked")
                else:
                    state.rope = _clamp_rope(
                        state.rope + phase.attack_power * _sign(boss)
                    )
                    state.modifiers["attacks_landed"] = (
                        int(state.modifiers.get("attacks_landed", 0)) + 1
                    )
                    _raise_event(state, "boss_attack")
        state.modifiers["attack_in"] = max(0.0, attack_in)

    # -------------------------------------------------------------------- end #
    def _end_condition(self, state):
        if float(state.modifiers.get("boss_hp", 1.0)) <= 0.0:
            return self.player_side, "boss_defeated"
        boss = self._boss_side
        if _sign(boss) * state.rope >= self.threshold:
            return boss, "threshold"
        if state.elapsed >= self.max_seconds:
            return boss, "timeout"
        return None, None

    def _build_result(self, state, winner, reason) -> MatchResult:
        result = super()._build_result(state, winner, reason)
        hp = float(state.modifiers.get("boss_hp", 0.0))
        result.stats.update(
            {
                "boss_hp_left": round(hp, 1),
                "boss_hp_frac": round(hp / max(1e-6, self.max_hp), 3),
                "phase_reached": int(state.modifiers.get("boss_phase", 1)),
                "attacks_landed": int(state.modifiers.get("attacks_landed", 0)),
                "attacks_blocked": int(state.modifiers.get("attacks_blocked", 0)),
                "hazard_fails": int(state.modifiers.get("hazard_fails", 0)),
            }
        )
        return result

    def _stars(self, state, winner, reason) -> int:
        if reason != "boss_defeated":
            return 0
        stars = 1
        side = self.player_side
        if state.accuracy(side) >= 0.75:
            stars += 1
        if int(state.modifiers.get("hazard_fails", 0)) == 0 and (
            state.best_streaks[side] >= 5
        ):
            stars += 1
        return int(_clamp(stars, 0, 3))

    def hud_extras(self, state: MatchState) -> dict:
        extras = super().hud_extras(state)
        phase = self.phase(state)
        hp = float(state.modifiers.get("boss_hp", self.max_hp))
        attack_in = float(state.modifiers.get("attack_in", phase.attack_period))
        telegraph = bool(state.modifiers.get("telegraph", False))
        # 0 -> just started winding up, 1 -> impact this instant.
        telegraph_t = (
            _clamp01(1.0 - attack_in / max(1e-6, self.telegraph_lead))
            if telegraph
            else 0.0
        )
        extras.update(
            {
                "boss_side": self._boss_side,
                "boss_hp": round(hp, 1),
                "boss_max_hp": self.max_hp,
                "boss_hp_frac": round(hp / max(1e-6, self.max_hp), 4),
                "phase": phase.index,
                "phase_name": phase.name,
                "phase_icon": phase.icon,
                "phase_count": len(self.PHASES),
                "telegraph": telegraph,
                "telegraph_t": round(telegraph_t, 3),
                "telegraph_lead": self.telegraph_lead,
                "attack_in": round(attack_in, 2),
                "attack_power": phase.attack_power,
                "hazard_left": round(float(state.modifiers.get("hazard_left", 0.0)), 2),
                "hazard_limit": phase.hazard_limit,
                "hazard_t": round(
                    _clamp01(
                        1.0
                        - float(state.modifiers.get("hazard_left", 0.0))
                        / max(1e-6, phase.hazard_limit)
                    ),
                    3,
                ),
                "hazard_fails": int(state.modifiers.get("hazard_fails", 0)),
                "phase_skills": [
                    skill_label(s) for s in self._phase_skills[phase.index - 1]
                ],
            }
        )
        return extras


# --------------------------------------------------------------------------- #
# 5. Practice
# --------------------------------------------------------------------------- #


class PracticeMode(ModeRules):
    """Solo, untimed, unloseable. The place a kid goes to actually learn.

    There is no clock, no opponent and no fail state. The stream is pinned to
    whatever the player is currently worst at, a wrong answer surfaces the
    problem's own hint instead of a penalty, and the session ends when the
    target is met or the player taps out - never because they ran out of
    anything.

    The rope still moves, purely as feedback; `check_end` ignores it entirely.
    """

    key = "practice"
    name = "Practice"
    solo = True
    icon = "book"
    blurb = "No clock. No pressure. Just the tricky ones."

    timed = False
    duration = 0.0
    base_pull = 0.060
    wrong_gain = 0.0               # nothing is taken from you
    lockout_seconds = 0.0
    combo_cap = 6
    combo_step = 0.10
    powerups_enabled = False       # rewards would be meaningless here
    max_seconds = 3600.0

    #: Correct answers needed to complete a session.
    default_target = 10

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.target: int = max(1, int(self.config.get("target", self.default_target)))

    def _init_mode(self, state: MatchState) -> None:
        state.modifiers.update(
            {
                "target": self.target,
                "hint": None,
                "hint_for": None,
                "done": False,
                "focus_skills": list(self.weak_skills),
            }
        )

    # ---------------------------------------------------------------- content #
    #: Accuracy at or above which a skill counts as mastered and drops out of
    #: the practice rotation (provided something genuinely weaker is left).
    mastered_at = 0.85

    def skills_for(self, state: MatchState, side: str) -> Optional[List[str]]:
        """Pin the draw to the player's weakest skills, once we know any.

        `ProblemStream.weak_skills` returns the N worst *whatever their score*,
        so on a short session it will happily hand back a skill the child is
        acing. Practice time is scarce, so anything already mastered is dropped
        - unless that would leave nothing to work on.
        """
        stream = self.streams.get(side)
        focus: List[str] = []
        if stream is not None:
            ranked = list(stream.weak_skills(limit=4))
            per_skill = stream.summary().get("per_skill", {})
            # Unseen / seeded-weak skills have no record yet: keep them.
            weak = [
                s for s in ranked
                if per_skill.get(s, {}).get("accuracy", 0.0) < self.mastered_at
            ]
            focus = (weak or ranked)[:3]
        if not focus:
            focus = [s for s in self.weak_skills if s][:3]
        if not focus:
            return None                      # no evidence yet - draw the full mix
        state.modifiers["focus_skills"] = focus
        return focus

    # ---------------------------------------------------------------- answers #
    def _on_correct_extra(self, state, side, problem, elapsed, magnitude, events):
        state.modifiers["hint"] = None
        state.modifiers["hint_for"] = None
        if state.correct[side] >= self.target:
            state.modifiers["done"] = True
            events.append("practice_complete")

    def _on_wrong_extra(self, state, side, problem, events) -> None:
        """A miss is a teaching moment, so hand back the hint and the answer."""
        hint = getattr(problem, "hint", None) or "Take another look at the numbers."
        state.modifiers["hint"] = hint
        state.modifiers["hint_for"] = getattr(problem, "prompt", "")
        state.modifiers["answer_label"] = getattr(problem, "answer_label", "")
        events.append("hint")

    def request_end(self, state: MatchState) -> None:
        """The player tapped 'done'. The only other way a session finishes."""
        state.modifiers["done"] = True

    # -------------------------------------------------------------------- end #
    def _end_condition(self, state):
        if state.modifiers.get("done"):
            return self.player_side, "complete"
        if state.elapsed >= self.max_seconds:
            return self.player_side, "complete"
        return None, None

    def _stars(self, state, winner, reason) -> int:
        """Effort first, then accuracy. Practice never awards zero for trying."""
        side = self.player_side
        if state.correct[side] <= 0:
            return 0
        stars = 1
        acc = state.accuracy(side)
        if acc >= 0.70:
            stars += 1
        if acc >= 0.90 and state.correct[side] >= self.target:
            stars += 1
        return int(_clamp(stars, 0, 3))

    def _build_result(self, state, winner, reason) -> MatchResult:
        result = super()._build_result(state, winner, reason)
        result.stats.update(self.session_summary(state))
        return result

    def session_summary(self, state: MatchState) -> Dict[str, Any]:
        """The teach-back panel: what was practised and how it went."""
        side = self.player_side
        stream = self.streams.get(side)
        summary = stream.summary() if stream is not None else {}
        per_skill = summary.get("per_skill", {})
        # Rank by accuracy so main.py can show "keep working on..." in order.
        ranked = sorted(per_skill.items(), key=lambda kv: kv[1]["accuracy"])
        return {
            "target": self.target,
            "completed": state.correct[side] >= self.target,
            "attempted": state.correct[side] + state.wrong[side],
            "practiced": [
                {"skill": s, "label": d["label"], "accuracy": d["accuracy"],
                 "attempts": d["attempts"]}
                for s, d in ranked
            ],
            "improved": [s for s, d in ranked if d["accuracy"] >= 0.8],
            "still_tricky": [s for s, d in ranked if d["accuracy"] < 0.6],
            "focus_skills": list(state.modifiers.get("focus_skills", [])),
        }

    def hud_extras(self, state: MatchState) -> dict:
        extras = super().hud_extras(state)
        side = self.player_side
        extras.update(
            {
                "target": self.target,
                "done_count": state.correct[side],
                "remaining": max(0, self.target - state.correct[side]),
                "progress": round(_clamp01(state.correct[side] / self.target), 3),
                "attempted": state.correct[side] + state.wrong[side],
                "accuracy": round(state.accuracy(side), 3),
                "hint": state.modifiers.get("hint"),
                "focus_skills": [
                    skill_label(s) for s in state.modifiers.get("focus_skills", [])
                ],
                "session_clock": round(state.elapsed, 1),
            }
        )
        return extras


# --------------------------------------------------------------------------- #
# 6. Daily Challenge
# --------------------------------------------------------------------------- #


class DailyChallenge(ModeRules):
    """One fixed set of problems per calendar day, identical for every player.

    Determinism is the whole product here: the seed is ``YYYYMMDD`` as an int,
    the problem list is built once in `__init__` from `make_daily_set`, and
    `next_problem` walks that list in order. Two installs on the same date and
    grade see the same prompts in the same order, which is what makes the
    leaderboard mean anything.

    One attempt: once `check_end` has produced a result the mode refuses to
    hand out a second state until `reset_attempt()` is called explicitly.
    """

    key = "daily"
    name = "Daily Challenge"
    solo = True
    icon = "star"
    blurb = "Same problems for everyone. One shot."

    duration = 120.0
    timed = True
    base_pull = 0.075
    wrong_gain = 0.0               # solo run - nobody to hand ground to
    lockout_seconds = 1.2
    combo_cap = 8
    combo_step = 0.12
    powerups_enabled = False       # a level playing field is the point
    max_seconds = 300.0

    #: Problems in a day's set.
    default_count = 15

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        date = cfg.get("date")
        if isinstance(date, str):
            try:
                date = _datetime.date.fromisoformat(date)
            except ValueError:
                date = None
        if date is None:
            date = _datetime.date.today()
        self.date: _datetime.date = date

        # The seed is fixed by the calendar, never by the caller's rng.
        cfg.setdefault("seed", daily_seed(self.date))
        super().__init__(cfg)

        self.count: int = max(1, int(self.config.get("count", self.default_count)))
        self.daily_seed: int = daily_seed(self.date)
        #: The immutable problem list for this date + grade.
        self.problems: List[Problem] = make_daily_set(
            self.grade, self.count, self.date, self.difficulty
        )
        self.attempted: bool = False

    # ------------------------------------------------------------- lifecycle - #
    def new_state(self) -> MatchState:
        state = super().new_state()
        state.modifiers["locked_out_attempt"] = self.attempted
        return state

    def on_start(self, state: MatchState) -> None:
        super().on_start(state)
        self.attempted = True

    def reset_attempt(self) -> None:
        """Clear the single-attempt lock (debug builds / a new calendar day)."""
        self.attempted = False
        self._result = None

    def _init_mode(self, state: MatchState) -> None:
        state.modifiers.update(
            {
                "daily_seed": self.daily_seed,
                "date": self.date.isoformat(),
                "index": 0,
                "count": self.count,
            }
        )

    # ---------------------------------------------------------------- content #
    def next_problem(self, state: MatchState, side: Optional[str] = None) -> Problem:
        """Walk the fixed list in order. The last problem repeats if overrun."""
        idx = int(state.modifiers.get("index", 0))
        problem = self.problems[min(idx, len(self.problems) - 1)]
        state.modifiers["index"] = idx + 1
        return problem

    def problem_sequence(self) -> List[str]:
        """The prompts, in order - the thing a determinism test compares."""
        return [p.prompt for p in self.problems]

    # ---------------------------------------------------------------- answers #
    def _on_correct_extra(self, state, side, problem, elapsed, magnitude, events):
        if state.correct[side] + state.wrong[side] >= self.count:
            state.modifiers["set_complete"] = True
            events.append("set_complete")

    def _on_wrong_extra(self, state, side, problem, events) -> None:
        if state.correct[side] + state.wrong[side] >= self.count:
            state.modifiers["set_complete"] = True
            events.append("set_complete")

    # -------------------------------------------------------------------- end #
    def _end_condition(self, state):
        side = self.player_side
        if state.correct[side] + state.wrong[side] >= self.count:
            return side, "set_complete"
        # NB: no rope threshold here. Everyone must answer the same number of
        # problems or the leaderboard compares different amounts of work - a
        # strong player would cap the rope early and never reach the last few.
        # The rope is progress feedback in this mode, nothing more.
        if self.timed and state.time_left <= 0.0:
            return (side if state.correct[side] > state.wrong[side] else None), "timeout"
        if state.elapsed >= self.max_seconds:
            return None, "timeout"
        return None, None

    def _build_result(self, state, winner, reason) -> MatchResult:
        result = super()._build_result(state, winner, reason)
        side = self.player_side
        # Leaderboard score: raw points, plus a bonus for finishing with time
        # to spare, plus a streak bonus. Deterministic given the same play.
        # The time weight is kept modest on purpose - the board should rank
        # accuracy first and speed second, not the other way round.
        time_bonus = int(max(0.0, state.time_left) * 2) if self.timed else 0
        streak_bonus = state.best_streaks[side] * 15
        result.score = int(state.scores[side] + time_bonus + streak_bonus)
        result.stats.update(
            {
                "daily_seed": self.daily_seed,
                "date": self.date.isoformat(),
                "answered": state.correct[side] + state.wrong[side],
                "count": self.count,
                "time_bonus": time_bonus,
                "streak_bonus": streak_bonus,
                "base_points": state.scores[side],
                "leaderboard_score": int(
                    state.scores[side] + time_bonus + streak_bonus
                ),
            }
        )
        return result

    def _stars(self, state, winner, reason) -> int:
        side = self.player_side
        answered = state.correct[side] + state.wrong[side]
        if answered <= 0:
            return 0
        acc = state.correct[side] / answered
        stars = 0
        if state.correct[side] >= max(1, self.count // 3):
            stars += 1
        if acc >= 0.70:
            stars += 1
        if acc >= 0.90 and state.correct[side] >= self.count - 1:
            stars += 1
        return int(_clamp(stars, 0, 3))

    def hud_extras(self, state: MatchState) -> dict:
        extras = super().hud_extras(state)
        side = self.player_side
        answered = state.correct[side] + state.wrong[side]
        extras.update(
            {
                "date": self.date.isoformat(),
                "daily_seed": self.daily_seed,
                "index": min(answered + 1, self.count),
                "count": self.count,
                "progress": round(_clamp01(answered / self.count), 3),
                "single_attempt": True,
                "accuracy": round(state.accuracy(side), 3),
            }
        )
        return extras


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_MODE_CLASSES: Dict[str, type] = {
    ClassicMode.key: ClassicMode,
    BlitzMode.key: BlitzMode,
    SurvivalMode.key: SurvivalMode,
    BossMode.key: BossMode,
    PracticeMode.key: PracticeMode,
    DailyChallenge.key: DailyChallenge,
}

#: Menu order - Classic first because it is the mode the game teaches you.
MODE_KEYS: List[str] = ["classic", "blitz", "survival", "boss", "practice", "daily"]

#: Everything the mode-select carousel needs without instantiating anything.
MODE_INFO: Dict[str, Dict[str, Any]] = {
    key: {
        "key": key,
        "name": _MODE_CLASSES[key].name,
        "icon": _MODE_CLASSES[key].icon,
        "solo": _MODE_CLASSES[key].solo,
        "blurb": _MODE_CLASSES[key].blurb,
        "timed": _MODE_CLASSES[key].timed,
        "duration": _MODE_CLASSES[key].duration,
    }
    for key in MODE_KEYS
}

_MODE_ALIASES: Dict[str, str] = {
    "normal": "classic",
    "standard": "classic",
    "speed": "blitz",
    "endless": "survival",
    "boss_battle": "boss",
    "train": "practice",
    "daily_challenge": "daily",
}


def make_mode(key: str, config: Optional[dict] = None) -> ModeRules:
    """Build the rules object for `key`.

    Raises ValueError on an unknown key rather than silently falling back - a
    typo in a mode key should fail loudly at the menu, not halfway into a match.
    """
    k = str(key).strip().lower().replace("-", "_")
    k = _MODE_ALIASES.get(k, k)
    cls = _MODE_CLASSES.get(k)
    if cls is None:
        raise ValueError(
            f"unknown game mode {key!r}; expected one of {', '.join(MODE_KEYS)}"
        )
    return cls(config)

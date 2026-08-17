"""Content metadata for RopeRush: modes, grade bands, cosmetics, achievements.

This module is the game's *catalogue*. It holds no state and does no I/O - it
declares what exists (the six play modes, the six curriculum bands, the shop,
the achievement list) and provides the two decisions that depend on the
catalogue plus the player's history: which achievements just fired, and how
hard the next problem should be.

Separation of concerns
----------------------
`progression.py` owns what the player *has*. This module owns what there *is*
to have. The dependency runs one way at import time - content imports
progression only for typing, and progression reaches back into content lazily
inside `add_xp` - so neither module can deadlock the other on import.

Grade and skill keys come straight from `math_engine.GRADES` / `SKILLS`; the
band table is asserted against them at import so a typo fails loudly here
rather than silently generating the wrong curriculum at runtime.

Icons are names from `icons.ICONS` and nothing else; `_ICON` validates every
one at import for the same reason.

Public API
----------
``ModeDef``, ``MODES``, ``MODE_ORDER``, ``GradeBand``, ``GRADE_BANDS``,
``Cosmetic``, ``COSMETICS``, ``Achievement``, ``ACHIEVEMENTS``,
``check_achievements``, ``pick_difficulty``, ``mode_for``, ``band_for``,
``cosmetics_for_slot``, ``achievement_for``, ``GRADE_DIFFICULTY_BANDS``,
``SLOTS``, ``RARITIES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import math_engine as ME
from icons import ICONS

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from progression import Profile


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def _ICON(name: str) -> str:
    """Return `name` if it is a real icon, else raise at import time."""
    if name not in ICONS:
        raise ValueError(f"content.py references unknown icon {name!r}")
    return name


def _GRADE(key: str) -> str:
    """Return `key` if it is a real math_engine grade, else raise at import."""
    if key not in ME.GRADES:
        raise ValueError(f"content.py references unknown grade {key!r}")
    return key


# --------------------------------------------------------------------------- #
# Play modes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModeDef:
    """One entry on the mode-select carousel."""

    key: str
    name: str
    tagline: str          # one line, shown under the name on the card
    icon: str             # a name from icons.ICONS
    unlock_level: int     # 1 == available from a fresh save
    description: str      # full paragraph for the mode's detail sheet
    solo: bool            # True == single player vs. the game, no rival lane


MODES: Dict[str, ModeDef] = {
    "classic": ModeDef(
        key="classic",
        name="Classic",
        tagline="First team to pull the rope home.",
        icon=_ICON("rope"),
        unlock_level=1,
        description=(
            "The main event. Two teams, one rope, and a stream of problems. "
            "Every correct answer heaves the rope your way and every miss "
            "gives a little back. Solve faster than your rival and the marker "
            "crosses their line."
        ),
        solo=False,
    ),
    "blitz": ModeDef(
        key="blitz",
        name="Blitz",
        tagline="Ninety seconds. Pull as hard as you can.",
        icon=_ICON("bolt"),
        unlock_level=3,
        description=(
            "Same rope, half the patience. The clock runs for ninety seconds "
            "and problems arrive quicker than you can second-guess them. "
            "Whoever has the rope when the buzzer goes takes it."
        ),
        solo=False,
    ),
    "survival": ModeDef(
        key="survival",
        name="Survival",
        tagline="Three misses and the rope is gone.",
        icon=_ICON("heart"),
        unlock_level=6,
        description=(
            "You against an opponent who never tires and never stops "
            "speeding up. You start with three hearts; every miss costs one. "
            "Hold the line as long as you can - your best time is the score."
        ),
        solo=True,
    ),
    "boss": ModeDef(
        key="boss",
        name="Boss Match",
        tagline="A champion who answers back.",
        icon=_ICON("crown"),
        unlock_level=10,
        description=(
            "A named rival with a real personality and a real weakness. "
            "Bosses pull harder on the skills you are shakiest at, so the way "
            "through is to shore up the gaps rather than out-tap them."
        ),
        solo=False,
    ),
    "practice": ModeDef(
        key="practice",
        name="Practice",
        tagline="No clock, no rival, no pressure.",
        icon=_ICON("pencil"),
        unlock_level=1,
        description=(
            "Pick a skill and work it with hints on and the timer off. "
            "Nothing here can be lost, and answers still count toward mastery "
            "- just for fewer coins, because the point is learning."
        ),
        solo=True,
    ),
    "daily": ModeDef(
        key="daily",
        name="Daily Pull",
        tagline="One shared puzzle set, every day.",
        icon=_ICON("flag"),
        unlock_level=4,
        description=(
            "Everybody gets the same problems today, seeded from the date. "
            "One scored run per day; come back tomorrow for a fresh set and "
            "keep your streak alive."
        ),
        solo=True,
    ),
}

#: Presentation order for the mode carousel (dicts preserve it, but being
#: explicit means the UI never depends on literal order in this file).
MODE_ORDER: Tuple[str, ...] = (
    "classic", "blitz", "survival", "boss", "practice", "daily",
)


def mode_for(key: str) -> Optional[ModeDef]:
    """Look up a mode by key, or None."""
    return MODES.get(key)


# --------------------------------------------------------------------------- #
# Grade bands
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GradeBand:
    """A curriculum band, wrapping one `math_engine.GRADES` key for the UI."""

    key: str
    label: str
    min_age: int
    max_age: int
    skills: List[str]     # math_engine skill keys actually taught in this band
    blurb: str            # parent-facing "what your child will practise"


def _band(key: str, min_age: int, max_age: int, blurb: str) -> GradeBand:
    """Build a band, taking its label and skill list from math_engine."""
    k = _GRADE(key)
    return GradeBand(
        key=k,
        label=ME.grade_label(k),
        min_age=min_age,
        max_age=max_age,
        skills=list(ME.skills_for_grade(k)),
        blurb=blurb,
    )


GRADE_BANDS: List[GradeBand] = [
    _band(
        "pre-k", 3, 5,
        "Counting to ten, spotting how many without counting, first shapes, "
        "and telling which pile is bigger.",
    ),
    _band(
        "k", 5, 6,
        "Ten-frames and number bonds, adding and taking away within ten, "
        "and comparing small numbers.",
    ),
    _band(
        "1st", 6, 7,
        "Adding and subtracting within twenty, tens and ones, o'clock and "
        "half past, and naming 2D shapes.",
    ),
    _band(
        "2nd", 7, 8,
        "Two-digit addition and subtraction with regrouping, arrays as early "
        "multiplication, coins and dollars, and time to five minutes.",
    ),
    _band(
        "3rd", 8, 9,
        "Times tables and division facts, fractions on a number line, "
        "rounding, elapsed time, and area and perimeter.",
    ),
    _band(
        "4th", 9, 11,
        "Multi-digit multiplication and division, equivalent fractions, "
        "decimals and money, factors, and measuring angles.",
    ),
]

# The band table must cover exactly math_engine's grades, in the same order.
assert [b.key for b in GRADE_BANDS] == list(ME.GRADES), (
    "GRADE_BANDS must mirror math_engine.GRADES"
)


def band_for(grade_key: str) -> Optional[GradeBand]:
    """Look up a grade band by math_engine grade key, or None."""
    for band in GRADE_BANDS:
        if band.key == grade_key:
            return band
    return None


# --------------------------------------------------------------------------- #
# Cosmetics
# --------------------------------------------------------------------------- #

SLOTS: Tuple[str, ...] = ("headband", "jersey", "trail", "emote")
RARITIES: Tuple[str, ...] = ("common", "rare", "epic", "legendary")


@dataclass(frozen=True)
class Cosmetic:
    """A purely visual item. Nothing here ever changes the maths or the pull.

    `cost` of 0 means the item is not sold - it is either owned from the start
    or handed out by a level unlock (see `progression._level_unlock_table`).
    """

    key: str
    name: str
    slot: str             # one of SLOTS
    cost: int             # coins; 0 == not purchasable
    unlock_level: int     # level gate; 1 == no gate
    rarity: str           # one of RARITIES


COSMETICS: List[Cosmetic] = [
    # --- headbands ---------------------------------------------------------
    Cosmetic("headband_rookie", "Rookie Band", "headband", 0, 1, "common"),
    Cosmetic("headband_stripe", "Twin Stripe", "headband", 120, 1, "common"),
    Cosmetic("headband_flame", "Emberburn", "headband", 0, 3, "rare"),
    Cosmetic("headband_frostbite", "Frostbite", "headband", 320, 5, "rare"),
    Cosmetic("headband_circuit", "Live Circuit", "headband", 700, 12, "epic"),
    Cosmetic("headband_laurel", "Gold Laurel", "headband", 1500, 20, "legendary"),

    # --- jerseys -----------------------------------------------------------
    Cosmetic("jersey_home", "Home Kit", "jersey", 0, 1, "common"),
    Cosmetic("jersey_away", "Away Kit", "jersey", 140, 1, "common"),
    Cosmetic("jersey_chevron", "Chevron", "jersey", 300, 4, "rare"),
    Cosmetic("jersey_dusk", "Dusk Fade", "jersey", 0, 8, "rare"),
    Cosmetic("jersey_aurora", "Aurora", "jersey", 850, 15, "epic"),
    Cosmetic("jersey_champion", "Champion's Kit", "jersey", 1800, 24, "legendary"),

    # --- rope trails -------------------------------------------------------
    Cosmetic("trail_dust", "Chalk Dust", "trail", 100, 1, "common"),
    Cosmetic("trail_spark", "Sparkstream", "trail", 260, 3, "common"),
    Cosmetic("trail_comet", "Comet Tail", "trail", 0, 7, "rare"),
    Cosmetic("trail_glacier", "Glacier", "trail", 620, 11, "epic"),
    Cosmetic("trail_supernova", "Supernova", "trail", 1600, 22, "legendary"),

    # --- emotes ------------------------------------------------------------
    Cosmetic("emote_thumbsup", "Nice One", "emote", 80, 1, "common"),
    Cosmetic("emote_flex", "Flex", "emote", 180, 2, "common"),
    Cosmetic("emote_facepalm", "Oof", "emote", 240, 5, "rare"),
    Cosmetic("emote_taunt", "Is That All?", "emote", 0, 9, "rare"),
    Cosmetic("emote_mic_drop", "Mic Drop", "emote", 900, 18, "epic"),
    Cosmetic("emote_crown", "Bow to Me", "emote", 2000, 25, "legendary"),
]

# Catalogue sanity - a bad slot or rarity would break the shop's grouping.
for _c in COSMETICS:
    assert _c.slot in SLOTS, f"bad cosmetic slot {_c.slot!r}"
    assert _c.rarity in RARITIES, f"bad cosmetic rarity {_c.rarity!r}"
assert len({c.key for c in COSMETICS}) == len(COSMETICS), "duplicate cosmetic key"


def cosmetics_for_slot(slot: str) -> List[Cosmetic]:
    """Every cosmetic in one slot, cheapest and earliest first."""
    return sorted(
        (c for c in COSMETICS if c.slot == slot),
        key=lambda c: (c.unlock_level, c.cost, c.key),
    )


# --------------------------------------------------------------------------- #
# Achievements
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Achievement:
    """A one-shot badge.

    `check` is a pure predicate over a `Profile`. It must be safe to call at
    any time and on any profile - `check_achievements` is the only thing that
    decides *when* a badge is actually awarded, and it awards each one once.
    """

    key: str
    name: str
    description: str
    icon: str                              # a name from icons.ICONS
    check: Callable[["Profile"], bool]


def _mastered_count(p: "Profile", threshold: float = 0.85) -> int:
    """How many skills sit above a mastery threshold right now."""
    return sum(1 for s in ME.SKILLS if p.mastery(s) >= threshold)


ACHIEVEMENTS: List[Achievement] = [
    # --- first steps -------------------------------------------------------
    Achievement(
        "first_pull", "First Pull", "Finish your first match.",
        _ICON("rope"), lambda p: p.matches_played >= 1,
    ),
    Achievement(
        "first_win", "Rope Burn", "Win your first match.",
        _ICON("trophy"), lambda p: p.matches_won >= 1,
    ),
    Achievement(
        "hundred_solved", "Century", "Solve 100 problems.",
        _ICON("check"), lambda p: p.problems_solved >= 100,
    ),
    Achievement(
        "thousand_solved", "Four Digits", "Solve 1,000 problems.",
        _ICON("chart"), lambda p: p.problems_solved >= 1000,
    ),

    # --- streaks & speed ---------------------------------------------------
    Achievement(
        "streak_10", "On a Roll", "Answer 10 in a row correctly.",
        _ICON("flame"), lambda p: p.best_streak >= 10,
    ),
    Achievement(
        "streak_25", "Unbreakable", "Answer 25 in a row correctly.",
        _ICON("shield"), lambda p: p.best_streak >= 25,
    ),
    Achievement(
        "quickdraw", "Quickdraw", "Solve a problem in under 1.5 seconds.",
        _ICON("zap"),
        lambda p: 0 < p.fastest_solve_ms <= 1500,
    ),
    Achievement(
        "sharpshooter", "Sharpshooter",
        "Reach 90% lifetime accuracy over 200 problems.",
        _ICON("target"),
        lambda p: (p.problems_solved + p.problems_missed) >= 200
        and p.accuracy() >= 0.90,
    ),

    # --- match record --------------------------------------------------------
    Achievement(
        "win_10", "Ten Down", "Win 10 matches.",
        _ICON("medal"), lambda p: p.matches_won >= 10,
    ),
    Achievement(
        "win_50", "Anchor", "Win 50 matches.",
        _ICON("crown"), lambda p: p.matches_won >= 50,
    ),
    Achievement(
        "stars_30", "Constellation", "Collect 30 stars.",
        _ICON("star"), lambda p: p.total_stars >= 30,
    ),
    Achievement(
        "stars_100", "Star Chart", "Collect 100 stars.",
        _ICON("sparkle"), lambda p: p.total_stars >= 100,
    ),

    # --- progression -------------------------------------------------------
    Achievement(
        "level_10", "Contender", "Reach level 10.",
        _ICON("bolt"), lambda p: p.level >= 10,
    ),
    Achievement(
        "level_25", "Rope Master", "Reach level 25.",
        _ICON("trophy"), lambda p: p.level >= 25,
    ),
    Achievement(
        "coin_hoard", "Coin Hoard", "Hold 2,000 coins at once.",
        _ICON("coin"), lambda p: p.coins >= 2000,
    ),
    Achievement(
        "collector", "Collector", "Unlock 10 cosmetics.",
        _ICON("gem"), lambda p: len(p.unlocked_cosmetics) >= 10,
    ),

    # --- learning ----------------------------------------------------------
    Achievement(
        "skill_master_3", "Triple Threat", "Master 3 different skills.",
        _ICON("book"), lambda p: _mastered_count(p) >= 3,
    ),
    Achievement(
        "skill_master_all", "Full Curriculum",
        "Master every skill at once.",
        _ICON("globe"), lambda p: _mastered_count(p) >= len(ME.SKILLS),
    ),
    Achievement(
        "all_grades", "Well Rounded",
        "Play a match in every grade band.",
        _ICON("grid"),
        lambda p: all(
            p.per_grade_stats.get(g) is not None
            and p.per_grade_stats[g].attempts > 0
            for g in ME.GRADES
        ),
    ),

    # --- habit -------------------------------------------------------------
    Achievement(
        "daily_7", "Week Streak", "Play 7 days in a row.",
        _ICON("clock"), lambda p: p.daily_streak >= 7,
    ),
    Achievement(
        "daily_30", "Month Streak", "Play 30 days in a row.",
        _ICON("bell"), lambda p: p.daily_streak >= 30,
    ),
    Achievement(
        "survivor", "Survivor", "Last 2 minutes in Survival.",
        _ICON("heart"), lambda p: p.best_survival_seconds >= 120.0,
    ),
    Achievement(
        "all_modes", "Tour of Duty", "Unlock every mode.",
        _ICON("users"),
        lambda p: all(k in p.unlocked_modes for k in MODES),
    ),
]

assert len({a.key for a in ACHIEVEMENTS}) == len(ACHIEVEMENTS), (
    "duplicate achievement key"
)

_ACHIEVEMENT_BY_KEY: Dict[str, Achievement] = {a.key: a for a in ACHIEVEMENTS}


def achievement_for(key: str) -> Optional[Achievement]:
    """Look up an achievement by key, or None."""
    return _ACHIEVEMENT_BY_KEY.get(key)


def check_achievements(profile: "Profile") -> List[Achievement]:
    """Award any newly satisfied achievements and return **only** the new ones.

    Earned keys are written into `profile.earned_achievements` before this
    returns, so a second call with an unchanged profile yields an empty list.
    That makes it safe to call after every match without the results screen
    having to track what it already showed.

    A `check` that raises is treated as "not yet earned" rather than being
    allowed to take down the results screen.
    """
    earned = profile.earned_achievements
    newly: List[Achievement] = []
    for ach in ACHIEVEMENTS:
        if ach.key in earned:
            continue
        try:
            hit = bool(ach.check(profile))
        except Exception:
            hit = False
        if hit:
            earned.add(ach.key)
            newly.append(ach)
    return newly


# --------------------------------------------------------------------------- #
# Adaptive difficulty
# --------------------------------------------------------------------------- #

#: Per-grade clamp on `pick_difficulty`. A struggling 4th grader still gets
#: 4th-grade-shaped problems (never a Pre-K one), and a prodigy in Pre-K never
#: gets handed long division. Difficulty is *within* a grade, by design - see
#: math_engine's module docstring.
GRADE_DIFFICULTY_BANDS: Dict[str, Tuple[float, float]] = {
    "pre-k": (0.05, 0.55),
    "k": (0.08, 0.62),
    "1st": (0.10, 0.70),
    "2nd": (0.12, 0.80),
    "3rd": (0.15, 0.90),
    "4th": (0.18, 1.00),
}
_DEFAULT_BAND: Tuple[float, float] = (0.10, 0.80)

# Flow-channel tuning.
_FLOW_WINDOW = 8          # answers of short-term memory consulted
_FAST_MS = 4500           # "answered without hesitating"
_HOT_RUN_MIN = 2          # trailing fast-correct answers before we ramp up
_HOT_STEP = 0.055         # per extra fast-correct answer in the run
_HOT_CAP = 0.22
_COLD_LOOKBACK = 5        # window in which repeated misses count
_COLD_STEP = 0.11         # per miss - deliberately ~2x the ramp-up rate
_COLD_CAP = 0.30
_SINGLE_MISS_STEP = 0.05  # gentle nudge for one fresh miss


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def pick_difficulty(profile: "Profile", grade_key: str) -> float:
    """Choose the next problem's difficulty (0..1 *within* `grade_key`).

    The heuristic keeps the player in a flow channel - hard enough to stay
    interesting, never hard enough to stall - by combining a slow signal with
    a fast one:

    1. **Baseline from mastery (slow).** The mean time-decayed mastery of the
       skills this grade actually teaches places the player inside the grade's
       band. Mastery already blends accuracy and speed and already fades
       toward neutral after a layoff, so a returning player is eased back in
       rather than dropped straight back at their old ceiling.

    2. **Flow adjustment from the last few answers (fast).** Looking only at
       recent answers *in this grade*:

       * a trailing run of two or more fast correct answers ramps difficulty
         up by `_HOT_STEP` per extra answer, capped at `+0.22`;
       * two or more misses inside the last five answers ramps it down by
         `_COLD_STEP` per miss, capped at `-0.30`;
       * a single fresh miss gets a small `-0.05` nudge.

       The ramp down is roughly twice as steep as the ramp up and needs less
       evidence to trigger. That asymmetry is deliberate: being bored for one
       problem costs a moment of interest, while being overwhelmed costs a kid
       their willingness to keep playing.

    3. **Clamp to the grade band** so difficulty always stays curriculum
       appropriate for the selected grade.

    A profile with no history lands a little below the middle of its band,
    which is where a first-time player should start.
    """
    lo, hi = GRADE_DIFFICULTY_BANDS.get(grade_key, _DEFAULT_BAND)

    # --- 1. slow signal: mastery places us inside the band ------------------
    skills = ME.skills_for_grade(grade_key) or list(ME.SKILLS)
    try:
        mastery = sum(profile.mastery(s) for s in skills) / float(len(skills))
    except Exception:
        mastery = 0.5
    mastery = _clamp(mastery, 0.0, 1.0)
    # Neutral mastery (0.5) sits at ~57% of the band; a fully mastered player
    # rides the top, a fully lost one sits just above the floor.
    base = lo + (hi - lo) * _clamp(0.15 + 0.85 * mastery, 0.0, 1.0)

    # --- 2. fast signal: the recent flow window -----------------------------
    recent = getattr(profile, "recent_answers", None) or []
    window = [
        entry for entry in recent
        if isinstance(entry, (list, tuple)) and len(entry) >= 3
        and entry[2] == grade_key
    ][-_FLOW_WINDOW:]

    adjust = 0.0
    if window:
        # Hot: how many fast correct answers are we riding right now?
        run = 0
        for correct, ms, _g in reversed(window):
            if correct and 0 < int(ms) <= _FAST_MS:
                run += 1
            else:
                break
        if run >= _HOT_RUN_MIN:
            adjust += min(_HOT_CAP, _HOT_STEP * (run - 1))

        # Cold: misses cluster harder than a single slip.
        tail = window[-_COLD_LOOKBACK:]
        misses = sum(1 for correct, _ms, _g in tail if not correct)
        if misses >= 2:
            adjust -= min(_COLD_CAP, _COLD_STEP * misses)
        elif misses == 1 and not window[-1][0]:
            adjust -= _SINGLE_MISS_STEP

    # --- 3. clamp to the grade's curriculum band ----------------------------
    return _clamp(base + adjust, lo, hi)

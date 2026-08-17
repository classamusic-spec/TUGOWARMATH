"""Curriculum-accurate math content engine for RopeRush.

This module is the single source of truth for *what a kid is asked*. It knows
nothing about pygame, rendering or game rules - it only produces `Problem`
objects. Rendering of the optional manipulative (`Problem.visual`) lives in
main.py; this module just describes what should be drawn.

Design goals
------------
1.  **Correctness above everything.** A wrong answer key is the worst possible
    bug in a kids' educational product, so every generator derives its answer
    from the very same integers it used to build the prompt string. No answer
    is ever "guessed", re-parsed or floating-point-accumulated - fraction and
    decimal work goes through `fractions.Fraction` or integer hundredths.
2.  **Curriculum shape, not just arithmetic.** Pre-K counts and subitizes, K
    works ten-frames and number bonds, 1st does place value and o'clock time,
    2nd does money/arrays/regrouping, 3rd does facts + fractions on a number
    line + area, 4th does multi-digit ops, equivalent fractions, decimals,
    factors and angles.
3.  **Difficulty genuinely changes the problem.** `difficulty` (0..1, *within*
    a grade) scales operand magnitude, forces regrouping/borrowing, adds a
    second step, widens the times-tables, and moves fractions from "same
    denominator" to "genuinely unlike denominators". It is not a label.

Answer encoding
---------------
`Problem.answer` is always a float so one comparison path serves everything.
Most problems are directly typeable on the in-game calculator
(`input_mode == 'number'`). A few concepts have no natural typed form, so they
are multiple-choice only (`input_mode == 'choice'`) and carry `choice_labels`
parallel to `choices`:

* symbolic comparison  -> answer is CMP_LESS / CMP_EQUAL / CMP_GREATER
                          (-1.0 / 0.0 / 1.0), labels "<", "=", ">"
* clock reading        -> answer is `time_code(h, m)` == h * 100 + m,
                          labels "3:30"
* non-terminating      -> e.g. 1/3; labels "1/3"
  fraction results
* angle classification -> answer is an index into ANGLE_TYPES, labels
                          "acute" / "right" / "obtuse" / "straight"

Everything else is a plain number the child can type.

VisualSpec data schemas
-----------------------
`VisualSpec.kind` plus a kind-specific `data` dict. main.py owns the renderers.

* ``dots``        ``{'count': int, 'shape': str, 'layout': 'scatter'|'dice'|'row',
                     'groups': [int, ...] | None, 'labels': [str, ...] | None}``
* ``tenframe``    ``{'count': int, 'frames': int}``
* ``numberline``  ``{'min': float, 'max': float, 'step': float,
                     'denominator': int | None, 'mark': float | None,
                     'mystery': float | None, 'labels': bool}``
* ``fraction_bar````{'bars': [{'parts': int, 'filled': int, 'label': str}, ...]}``
* ``clock``       ``{'hour': int, 'minute': int, 'minute_hand': bool}``
* ``coins``       ``{'coins': [{'name': str, 'value': int}, ...], 'unit': 'cents'|'dollars'}``
* ``shapes``      ``{'shape': str, 'sides': int, 'corners': int,
                     'ask': 'sides'|'corners'}``
* ``array``       ``{'rows': int, 'cols': int, 'label': str | None}``
* ``base_ten``    ``{'hundreds': int, 'tens': int, 'ones': int}``

Public API
----------
``GRADES``, ``SKILLS``, ``Problem``, ``VisualSpec``, ``make_problem``,
``make_problem_set``, ``grade_label``, ``skill_label``, ``skills_for_grade``,
``ProblemStream``.
"""

from __future__ import annotations

import datetime as _datetime
import math
import random
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

GRADES: List[str] = ["pre-k", "k", "1st", "2nd", "3rd", "4th"]

SKILLS: List[str] = [
    "count",
    "add",
    "sub",
    "mul",
    "div",
    "fraction",
    "compare",
    "place_value",
    "time",
    "money",
    "shape",
    "word",
]

_GRADE_LABELS: Dict[str, str] = {
    "pre-k": "Pre-K",
    "k": "Kindergarten",
    "1st": "1st Grade",
    "2nd": "2nd Grade",
    "3rd": "3rd Grade",
    "4th": "4th Grade",
}

#: Compact form for tight UI (grade pills, HUD chips).
GRADE_SHORT: Dict[str, str] = {
    "pre-k": "Pre-K",
    "k": "K",
    "1st": "1st",
    "2nd": "2nd",
    "3rd": "3rd",
    "4th": "4th",
}

_SKILL_LABELS: Dict[str, str] = {
    "count": "Counting",
    "add": "Addition",
    "sub": "Subtraction",
    "mul": "Multiplication",
    "div": "Division",
    "fraction": "Fractions",
    "compare": "Comparing",
    "place_value": "Place Value",
    "time": "Telling Time",
    "money": "Money",
    "shape": "Shapes",
    "word": "Word Problems",
}

#: Operator glyphs. Both live in every font in theme.FONT_STACK_*, and both are
#: what US classrooms actually put in front of these kids.
MUL_SIGN = "x"
DIV_SIGN = "÷"  # ÷

#: Symbolic-comparison answer encoding.
CMP_LESS = -1.0
CMP_EQUAL = 0.0
CMP_GREATER = 1.0
CMP_LABELS = ["<", "=", ">"]

#: Angle-classification answer encoding (answer == index).
ANGLE_TYPES = ["acute", "right", "obtuse", "straight"]

#: Incremented if a generator ever raises. Tests assert this stays at zero;
#: at runtime we degrade to a safe fallback problem instead of crashing a match.
GENERATOR_FAILURES = 0


def grade_label(grade: str) -> str:
    """Human-readable name for a grade key ('1st' -> '1st Grade')."""
    return _GRADE_LABELS.get(_norm_grade(grade), _norm_grade(grade))


def skill_label(skill: str) -> str:
    """Human-readable name for a skill key ('place_value' -> 'Place Value')."""
    return _SKILL_LABELS.get(skill, skill.replace("_", " ").title())


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #


@dataclass
class VisualSpec:
    """Instructions for main.py to render a manipulative next to the prompt.

    The renderers live in main.py; this module only describes *what* to draw.
    See the module docstring for the per-kind `data` schemas.
    """

    kind: str
    data: dict = field(default_factory=dict)


@dataclass
class Problem:
    """One question, its answer key, and everything needed to present it."""

    prompt: str
    answer: float
    skill: str = "add"
    grade: str = "k"
    difficulty: float = 0.5
    tolerance: float = 1e-6
    choices: Optional[List[float]] = None
    visual: Optional[VisualSpec] = None
    hint: Optional[str] = None
    time_budget: float = 8.0

    # --- additive extras (safe for callers that ignore them) ---------------- #
    #: Parallel to `choices`. When present, main.py should show these strings
    #: on the choice buttons instead of the raw float.
    choice_labels: Optional[List[str]] = None
    #: 'number' -> the child may type the answer on the calculator.
    #: 'choice'  -> the answer has no natural typed form; force the choice UI.
    input_mode: str = "number"
    #: Short unit suffix for display, e.g. "cents", "min", "cm", "°".
    unit: str = ""

    # ---------------------------------------------------------------- checks #
    def check(self, value: float) -> bool:
        """True if a submitted numeric answer counts as correct."""
        try:
            return abs(float(value) - self.answer) <= self.tolerance
        except (TypeError, ValueError):
            return False

    @property
    def weight(self) -> float:
        """Pull weight - harder problems are worth a bit more rope.

        Kept for compatibility with the original `math_problems.Problem`, which
        main.py already multiplies into its impulse.
        """
        base = 0.8 + 0.14 * _grade_index(self.grade)
        return round(base * (0.85 + 0.35 * _clamp01(self.difficulty)), 3)

    @property
    def answer_label(self) -> str:
        """How the answer should be *spoken* in a solution reveal."""
        if self.choices and self.choice_labels:
            for value, label in zip(self.choices, self.choice_labels):
                if abs(value - self.answer) <= max(self.tolerance, 1e-9):
                    return label
        return format_number(self.answer)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def _norm_grade(grade: str) -> str:
    """Accept 'K', 'Pre-K', 'pre_k', 3 (index) ... and return a canonical key."""
    if isinstance(grade, int):
        return GRADES[max(0, min(grade, len(GRADES) - 1))]
    g = str(grade).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "prek": "pre-k",
        "pre-k": "pre-k",
        "pk": "pre-k",
        "k": "k",
        "kindergarten": "k",
        "kinder": "k",
        "1": "1st",
        "1st": "1st",
        "first": "1st",
        "2": "2nd",
        "2nd": "2nd",
        "second": "2nd",
        "3": "3rd",
        "3rd": "3rd",
        "third": "3rd",
        "4": "4th",
        "4th": "4th",
        "fourth": "4th",
    }
    return aliases.get(g, "k")


def _grade_index(grade: str) -> int:
    g = _norm_grade(grade)
    return GRADES.index(g) if g in GRADES else 1


def _span(d: float, lo: int, hi: int) -> int:
    """Interpolate an integer bound across the difficulty ramp."""
    return int(round(lo + (hi - lo) * _clamp01(d)))


def _spanf(d: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * _clamp01(d)


def format_number(value: float) -> str:
    """Render a float the way a kid expects to see it (no trailing '.0')."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text


def time_code(hour: int, minute: int) -> float:
    """Encode a clock time as a single float: 3:30 -> 330.0."""
    return float(hour * 100 + minute)


def time_label(hour: int, minute: int) -> str:
    return f"{hour}:{minute:02d}"


def _terminating(fr: Fraction) -> bool:
    """True if the fraction has an exact finite decimal expansion."""
    den = fr.denominator
    for p in (2, 5):
        while den % p == 0:
            den //= p
    return den == 1


def _frac_label(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{fr.numerator}/{fr.denominator}"


def _int_choices(
    rng: random.Random,
    answer: float,
    count: int = 4,
    lo: int = 0,
    hi: Optional[int] = None,
    spread: int = 3,
) -> List[float]:
    """Plausible near-miss distractors around an integer answer.

    Always returns a list that *contains* the answer, with no duplicates.
    """
    target = int(round(answer))
    distractors: set = set()
    for _ in range(400):
        if len(distractors) >= count - 1:
            break
        delta = rng.randint(1, max(1, spread))
        cand = target + (delta if rng.random() < 0.5 else -delta)
        if cand == target or cand < lo:
            continue
        if hi is not None and cand > hi:
            continue
        distractors.add(cand)
    # Deterministic widening if the local window was too tight to fill.
    step = 1
    while len(distractors) < count - 1 and step <= 2000:
        for cand in (target + step, target - step):
            if len(distractors) >= count - 1:
                break
            if cand != target and cand >= lo and (hi is None or cand <= hi):
                distractors.add(cand)
        step += 1
    out = [float(target)] + [float(v) for v in sorted(distractors)[: count - 1]]
    rng.shuffle(out)
    return out


def _labeled_choices(
    rng: random.Random,
    pairs: Sequence[Tuple[float, str]],
    tolerance: float = 1e-6,
    count: int = 4,
) -> Tuple[List[float], List[str]]:
    """Build (values, labels) from (value, label) pairs; pairs[0] is the answer.

    Duplicates - by value within `tolerance`, or by label - are dropped so a
    child never sees the same option twice or two options that are both right.
    Callers should pass a generous candidate list; the first `count` survivors
    (always including the answer) are kept and shuffled.
    """
    kept: List[Tuple[float, str]] = []
    for value, label in pairs:
        if len(kept) >= max(2, count):
            break
        clash = any(
            abs(value - v) <= max(tolerance, 1e-9) * 2 or label == lb for v, lb in kept
        )
        if not clash:
            kept.append((float(value), label))
    order = list(range(len(kept)))
    rng.shuffle(order)
    return [kept[i][0] for i in order], [kept[i][1] for i in order]


def _fraction_distractors(
    rng: random.Random,
    answer: Fraction,
    denom_pool: Sequence[int] = (2, 3, 4, 5, 6, 8, 10),
    upper: Optional[Fraction] = Fraction(1, 1),
) -> List[Tuple[float, str]]:
    """A big shuffled pool of plausible wrong fractions near `answer`.

    Includes same-denominator neighbours first (the most instructive lures),
    then fractions built on other common denominators.
    """
    same: List[Fraction] = []
    other: List[Fraction] = []
    den = answer.denominator
    limit = den if upper is None else int(upper * den) + den
    for k in range(1, max(2, limit) + 1):
        cand = Fraction(k, den)
        if cand != answer and cand > 0 and (upper is None or cand <= upper):
            same.append(cand)
    for od in denom_pool:
        if od == den:
            continue
        top = od if upper is None else int(upper * od) + od
        for k in range(1, max(2, top) + 1):
            cand = Fraction(k, od)
            if cand != answer and cand > 0 and (upper is None or cand <= upper):
                other.append(cand)
    rng.shuffle(same)
    rng.shuffle(other)
    return [(float(f), _frac_label(f)) for f in (same + other)]


def _round_half_up(n: int, place: int) -> int:
    """Round a non-negative int to the nearest `place` (10/100/1000), half up."""
    q, r = divmod(n, place)
    return (q + 1) * place if r * 2 >= place else q * place


def _factors(n: int) -> List[int]:
    out = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            out.append(i)
            if i != n // i:
                out.append(n // i)
        i += 1
    return sorted(out)


def _weighted_pick(rng: random.Random, items: Sequence, weights: Sequence[float]):
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(list(items))
    roll = rng.random() * total
    acc = 0.0
    for item, w in zip(items, weights):
        acc += w
        if roll <= acc:
            return item
    return items[-1]


# --------------------------------------------------------------------------- #
# Shared content tables
# --------------------------------------------------------------------------- #

_COUNT_OBJECTS = [
    ("star", "stars"),
    ("apple", "apples"),
    ("balloon", "balloons"),
    ("fish", "fish"),
    ("flower", "flowers"),
    ("bug", "bugs"),
    ("heart", "hearts"),
    ("ball", "balls"),
]

_SHAPES_EASY = [("triangle", 3), ("square", 4), ("rectangle", 4)]
_SHAPES_MID = [("pentagon", 5), ("hexagon", 6), ("rhombus", 4), ("trapezoid", 4)]
_SHAPES_HARD = [("octagon", 8), ("heptagon", 7)]

_NAMES = [
    "Ava", "Ben", "Mia", "Leo", "Zoe", "Kai",
    "Nina", "Omar", "Ruby", "Sam", "Iris", "Theo",
]

_THINGS = [
    ("sticker", "stickers"),
    ("marble", "marbles"),
    ("crayon", "crayons"),
    ("cookie", "cookies"),
    ("card", "cards"),
    ("shell", "shells"),
    ("bead", "beads"),
    ("acorn", "acorns"),
]

_CONTAINERS = [
    ("box", "boxes"),
    ("bag", "bags"),
    ("basket", "baskets"),
    ("shelf", "shelves"),
    ("row", "rows"),
    ("pack", "packs"),
]

_COIN_TABLE = [("penny", 1), ("nickel", 5), ("dime", 10), ("quarter", 25)]


# --------------------------------------------------------------------------- #
# Pre-K generators
# --------------------------------------------------------------------------- #
#
# Counting to 10, subitizing, more/less, growing patterns, shape recognition,
# and the very first "put two small groups together" addition.


def _pk_count(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    n = rng.randint(1, top)
    singular, plural = rng.choice(_COUNT_OBJECTS)
    return Problem(
        prompt=f"How many {plural}?",
        answer=float(n),
        skill="count",
        choices=_int_choices(rng, n, 4, lo=1, hi=max(10, n + 3), spread=2),
        visual=VisualSpec("dots", {"count": n, "shape": singular, "layout": "scatter"}),
        hint="Touch each one as you count: 1, 2, 3...",
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _pk_subitize(rng: random.Random, d: float) -> Problem:
    n = rng.randint(1, _span(d, 3, 6))
    singular, _plural = rng.choice(_COUNT_OBJECTS)
    return Problem(
        prompt="How many? Quick!",
        answer=float(n),
        skill="count",
        choices=_int_choices(rng, n, 4, lo=1, hi=8, spread=2),
        visual=VisualSpec("dots", {"count": n, "shape": singular, "layout": "dice"}),
        hint="You can see it without counting - it makes a picture.",
        time_budget=_spanf(d, 6.0, 4.5),
    )


def _pk_more_less(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    a = rng.randint(1, top)
    b = rng.randint(1, top)
    # Keep the gap obvious at low difficulty, allow a 1-apart gap when harder.
    min_gap = 2 if d < 0.5 else 1
    tries = 0
    while abs(a - b) < min_gap and tries < 40:
        b = rng.randint(1, top)
        tries += 1
    if a == b:
        b = a + 1 if a < top else a - 1
    ask_more = rng.random() < 0.6
    answer = max(a, b) if ask_more else min(a, b)
    word = "more" if ask_more else "fewer"
    singular, plural = rng.choice(_COUNT_OBJECTS)
    values = [float(a), float(b)]
    rng.shuffle(values)
    return Problem(
        prompt=f"Which group has {word} {plural}?",
        answer=float(answer),
        skill="compare",
        choices=values,
        visual=VisualSpec(
            "dots",
            {
                "count": a + b,
                "shape": singular,
                "layout": "row",
                "groups": [a, b],
                "labels": [str(a), str(b)],
            },
        ),
        hint=f"Count each group, then pick the one with {word}.",
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _pk_pattern(rng: random.Random, d: float) -> Problem:
    step = 1 if d < 0.35 else rng.choice([1, 2, 2, 5, 10][: 2 if d < 0.7 else 5])
    if d > 0.75 and rng.random() < 0.3:
        # Counting *down* is the hard version of the same idea.
        start = rng.randint(4 * step, 4 * step + _span(d, 2, 8))
        seq = [start - i * step for i in range(4)]
    else:
        start = rng.randint(1, _span(d, 3, 9))
        seq = [start + i * step for i in range(4)]
    answer = seq[3]
    shown = ",  ".join(str(v) for v in seq[:3])
    return Problem(
        prompt=f"What comes next?\n{shown},  __",
        answer=float(answer),
        skill="count",
        choices=_int_choices(rng, answer, 4, lo=0, hi=answer + 4 * max(step, 2), spread=max(1, step)),
        hint="Look at how much it jumps each time.",
        time_budget=_spanf(d, 10.0, 8.0),
    )


def _pk_shape(rng: random.Random, d: float) -> Problem:
    pool = list(_SHAPES_EASY)
    if d >= 0.45:
        pool += _SHAPES_MID
    if d >= 0.8:
        pool += _SHAPES_HARD
    name, sides = rng.choice(pool)
    ask_corners = rng.random() < 0.4
    word = "corners" if ask_corners else "sides"
    return Problem(
        prompt=f"How many {word} does a {name} have?",
        answer=float(sides),  # for these polygons corners == sides
        skill="shape",
        choices=_int_choices(rng, sides, 4, lo=3, hi=9, spread=2),
        visual=VisualSpec(
            "shapes",
            {"shape": name, "sides": sides, "corners": sides,
             "ask": "corners" if ask_corners else "sides"},
        ),
        hint=f"Trace around the {name} and count each {word[:-1]}.",
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _pk_add_tiny(rng: random.Random, d: float) -> Problem:
    top = _span(d, 2, 4)
    a = rng.randint(1, top)
    b = rng.randint(1, top)
    total = a + b
    singular, _plural = rng.choice(_COUNT_OBJECTS)
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(total),
        skill="add",
        choices=_int_choices(rng, total, 4, lo=0, hi=12, spread=2),
        visual=VisualSpec(
            "dots",
            {"count": total, "shape": singular, "layout": "row",
             "groups": [a, b], "labels": [str(a), str(b)]},
        ),
        hint="Put the two groups together and count them all.",
        time_budget=_spanf(d, 10.0, 8.0),
    )


# --------------------------------------------------------------------------- #
# Kindergarten generators
# --------------------------------------------------------------------------- #
#
# Add/subtract within 10 (never negative), ten-frames, number bonds, ordering
# and 2D shapes.


def _k_add(rng: random.Random, d: float) -> Problem:
    total_cap = _span(d, 5, 10)
    a = rng.randint(0, total_cap)
    b = rng.randint(0, total_cap - a)
    total = a + b
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(total),
        skill="add",
        choices=_int_choices(rng, total, 4, lo=0, hi=12, spread=2),
        visual=VisualSpec("tenframe", {"count": total, "frames": 1}),
        hint="Start at the bigger number and count on.",
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _k_sub(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    a = rng.randint(1, top)
    b = rng.randint(0, a)  # never negative in K
    diff = a - b
    return Problem(
        prompt=f"{a} - {b} = ?",
        answer=float(diff),
        skill="sub",
        choices=_int_choices(rng, diff, 4, lo=0, hi=12, spread=2),
        visual=VisualSpec("tenframe", {"count": a, "frames": 1}),
        hint="Start at the big number and count back.",
        time_budget=_spanf(d, 10.0, 8.0),
    )


def _k_tenframe(rng: random.Random, d: float) -> Problem:
    if d < 0.6:
        n = rng.randint(1, 10)
        frames = 1
    else:
        n = rng.randint(11, 20)
        frames = 2
    return Problem(
        prompt=("How many dots are in the ten frame?" if frames == 1
                else "How many dots are in the ten frames?"),
        answer=float(n),
        skill="count",
        choices=_int_choices(rng, n, 4, lo=0, hi=20, spread=2),
        visual=VisualSpec("tenframe", {"count": n, "frames": frames}),
        hint="A full row is 5. A full frame is 10.",
        time_budget=_spanf(d, 8.0, 6.5),
    )


def _k_bond(rng: random.Random, d: float) -> Problem:
    total = rng.randint(_span(d, 4, 7), _span(d, 6, 10))
    part = rng.randint(0, total)
    missing = total - part
    if rng.random() < 0.5:
        prompt = f"{part} and __ make {total}"
    else:
        prompt = f"__ and {part} make {total}"
    return Problem(
        prompt=prompt,
        answer=float(missing),
        skill="add",
        choices=_int_choices(rng, missing, 4, lo=0, hi=10, spread=2),
        visual=VisualSpec("tenframe", {"count": part, "frames": 1}),
        hint=f"Fill the frame up to {total}. How many more do you need?",
        time_budget=_spanf(d, 11.0, 8.0),
    )


def _k_order(rng: random.Random, d: float) -> Problem:
    top = _span(d, 10, 20)
    mode = rng.choice(["after", "before", "largest", "smallest"])
    if mode in ("after", "before"):
        n = rng.randint(2, top - 1)
        answer = n + 1 if mode == "after" else n - 1
        prompt = f"What number comes {mode} {n}?"
        hint = "Say the counting words in order."
    else:
        values = rng.sample(range(1, top + 1), 3)
        answer = max(values) if mode == "largest" else min(values)
        shown = ",  ".join(str(v) for v in values)
        prompt = f"Which is {mode}?\n{shown}"
        hint = "Line them up on the number line in your head."
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="compare",
        choices=_int_choices(rng, answer, 4, lo=0, hi=top + 2, spread=2),
        hint=hint,
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _k_compare(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    a = rng.randint(1, top)
    b = rng.randint(1, top)
    while a == b:
        b = rng.randint(1, top)
    greater = rng.random() < 0.6
    answer = max(a, b) if greater else min(a, b)
    word = "greater" if greater else "less"
    values = [float(a), float(b)]
    rng.shuffle(values)
    return Problem(
        prompt=f"Which number is {word}?\n{a}   or   {b}",
        answer=float(answer),
        skill="compare",
        choices=values,
        hint="The bigger number is further along when you count.",
        time_budget=_spanf(d, 8.0, 6.0),
    )


# --------------------------------------------------------------------------- #
# 1st grade generators
# --------------------------------------------------------------------------- #
#
# Add/subtract within 20 (crossing ten at higher difficulty), missing addends,
# tens-and-ones place value, o'clock / half past, comparing two-digit numbers.


def _g1_add(rng: random.Random, d: float) -> Problem:
    cap = _span(d, 10, 20)
    a = rng.randint(1, max(1, cap - 1))
    b = rng.randint(1, max(1, cap - a))
    if d >= 0.55:
        # Force a "cross the ten" fact, which is the real 1st-grade target.
        for _ in range(20):
            a = rng.randint(4, 9)
            b = rng.randint(max(2, 11 - a), min(9, cap - a))
            if b >= 1 and a + b > 10:
                break
        b = max(1, min(b, cap - a))
    total = a + b
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(total),
        skill="add",
        choices=_int_choices(rng, total, 4, lo=0, hi=25, spread=3),
        visual=VisualSpec("tenframe", {"count": min(total, 20), "frames": 2}),
        hint="Make a ten first, then add what is left over.",
        time_budget=_spanf(d, 9.0, 7.0),
    )


def _g1_sub(rng: random.Random, d: float) -> Problem:
    cap = _span(d, 10, 20)
    a = rng.randint(2, cap)
    if d >= 0.55 and a > 10:
        b = rng.randint(a - 9, min(9, a))  # answer stays under 10 -> crosses ten
    else:
        b = rng.randint(0, a)
    b = max(0, min(b, a))
    diff = a - b
    return Problem(
        prompt=f"{a} - {b} = ?",
        answer=float(diff),
        skill="sub",
        choices=_int_choices(rng, diff, 4, lo=0, hi=25, spread=3),
        hint="Count back, or think 'what plus the small number makes the big one?'",
        time_budget=_spanf(d, 10.0, 7.5),
    )


def _g1_missing_addend(rng: random.Random, d: float) -> Problem:
    total = rng.randint(_span(d, 6, 12), _span(d, 10, 20))
    missing = rng.randint(1, total - 1)
    known = total - missing
    if rng.random() < 0.5:
        prompt = f"{known} + ? = {total}"
    else:
        prompt = f"? + {known} = {total}"
    return Problem(
        prompt=prompt,
        answer=float(missing),
        skill="add",
        choices=_int_choices(rng, missing, 4, lo=0, hi=20, spread=3),
        hint=f"How far is it from {known} up to {total}?",
        time_budget=_spanf(d, 12.0, 9.0),
    )


def _g1_place_value(rng: random.Random, d: float) -> Problem:
    mode = rng.choice(["build", "how_many_tens", "value_of", "ten_more"])
    tens = rng.randint(1, 9)
    ones = rng.randint(0, 9)
    number = tens * 10 + ones
    if mode == "build":
        return Problem(
            prompt=f"{tens} tens and {ones} ones make __",
            answer=float(number),
            skill="place_value",
            choices=_int_choices(rng, number, 4, lo=0, hi=99, spread=11),
            visual=VisualSpec("base_ten", {"hundreds": 0, "tens": tens, "ones": ones}),
            hint="Each rod is 10. Each little cube is 1.",
            time_budget=_spanf(d, 11.0, 8.0),
        )
    if mode == "how_many_tens":
        return Problem(
            prompt=f"How many tens are in {number}?",
            answer=float(tens),
            skill="place_value",
            choices=_int_choices(rng, tens, 4, lo=0, hi=9, spread=2),
            visual=VisualSpec("base_ten", {"hundreds": 0, "tens": tens, "ones": ones}),
            hint="The first digit tells you the tens.",
            time_budget=_spanf(d, 9.0, 7.0),
        )
    if mode == "value_of" and d >= 0.4:
        return Problem(
            prompt=f"What is the value of the {tens} in {number}?",
            answer=float(tens * 10),
            skill="place_value",
            choices=_int_choices(rng, tens * 10, 4, lo=0, hi=99, spread=11),
            hint="It is sitting in the tens place, so it is worth that many tens.",
            time_budget=_spanf(d, 11.0, 8.5),
        )
    step = 10 if d < 0.7 else rng.choice([10, 10, 20])
    up = rng.random() < 0.6
    base = rng.randint(step, 89) if not up else rng.randint(1, 99 - step)
    answer = base + step if up else base - step
    return Problem(
        prompt=f"What is {step} {'more' if up else 'less'} than {base}?",
        answer=float(answer),
        skill="place_value",
        choices=_int_choices(rng, answer, 4, lo=0, hi=120, spread=11),
        hint="Only the tens digit changes.",
        time_budget=_spanf(d, 10.0, 8.0),
    )


def _clock_problem(
    rng: random.Random,
    d: float,
    minutes_allowed: Sequence[int],
    time_budget: float,
) -> Problem:
    """Shared 'read the clock' generator; multiple-choice by construction."""
    hour = rng.randint(1, 12)
    minute = rng.choice(list(minutes_allowed))
    answer = time_code(hour, minute)
    pairs = [(answer, time_label(hour, minute))]
    guard = 0
    while len(pairs) < 4 and guard < 60:
        guard += 1
        dh = rng.choice([-1, 0, 0, 1])
        alt_hour = ((hour - 1 + dh) % 12) + 1
        alt_minute = rng.choice(list(minutes_allowed))
        if alt_hour == hour and alt_minute == minute:
            continue
        pairs.append((time_code(alt_hour, alt_minute), time_label(alt_hour, alt_minute)))
    values, labels = _labeled_choices(rng, pairs)
    return Problem(
        prompt="What time does the clock show?",
        answer=answer,
        skill="time",
        choices=values,
        choice_labels=labels,
        input_mode="choice",
        visual=VisualSpec(
            "clock", {"hour": hour, "minute": minute, "minute_hand": True}
        ),
        hint="The short hand is the hour. The long hand counts the minutes by 5s.",
        time_budget=time_budget,
    )


def _g1_time(rng: random.Random, d: float) -> Problem:
    minutes = [0, 30] if d < 0.6 else [0, 0, 30, 30]
    return _clock_problem(rng, d, minutes, _spanf(d, 12.0, 9.0))


def _g1_compare(rng: random.Random, d: float) -> Problem:
    top = _span(d, 40, 99)
    a = rng.randint(10, top)
    b = rng.randint(10, top)
    if d >= 0.6 and rng.random() < 0.5:
        b = a // 10 * 10 + rng.randint(0, 9)  # same tens digit -> must read ones
    if rng.random() < 0.12:
        b = a  # equality really does show up
    symbol = CMP_GREATER if a > b else (CMP_LESS if a < b else CMP_EQUAL)
    values, labels = _labeled_choices(
        rng, [(symbol, CMP_LABELS[int(symbol) + 1])]
        + [(v, CMP_LABELS[int(v) + 1]) for v in (CMP_LESS, CMP_EQUAL, CMP_GREATER)]
    )
    return Problem(
        prompt=f"{a}  ?  {b}",
        answer=symbol,
        skill="compare",
        choices=values,
        choice_labels=labels,
        input_mode="choice",
        hint="Compare the tens first. If they match, compare the ones.",
        time_budget=_spanf(d, 10.0, 7.0),
    )


# --------------------------------------------------------------------------- #
# 2nd grade generators
# --------------------------------------------------------------------------- #
#
# Add/subtract within 100 (regrouping forced as difficulty rises), arrays and
# repeated addition, coins, time to 5 minutes, place value to 1000, and the
# first real word problems.


def _g2_add(rng: random.Random, d: float) -> Problem:
    cap = _span(d, 50, 99)
    need_regroup = d >= 0.45
    a = b = 0
    for _ in range(60):
        a = rng.randint(10, cap)
        b = rng.randint(10, max(11, cap - a) if a < cap else 11)
        b = max(1, min(b, 99 - a)) if a < 99 else 1
        regroups = (a % 10) + (b % 10) >= 10
        if regroups == need_regroup or not need_regroup:
            break
    if need_regroup and (a % 10) + (b % 10) < 10:
        # Nudge the ones digit until it genuinely carries.
        bump = 10 - ((a % 10) + (b % 10))
        if b + bump + a <= 99:
            b += bump
        elif a - bump >= 10:
            a -= 0
            b = min(99 - a, b + bump)
    total = a + b
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(total),
        skill="add",
        choices=_int_choices(rng, total, 4, lo=0, hi=220, spread=11),
        hint="Add the ones. If you get 10 or more, carry a ten.",
        time_budget=_spanf(d, 12.0, 9.0),
    )


def _g2_sub(rng: random.Random, d: float) -> Problem:
    need_borrow = d >= 0.45
    a = rng.randint(20, 99)
    b = rng.randint(1, a)
    for _ in range(60):
        a = rng.randint(_span(d, 20, 40), 99)
        b = rng.randint(1, a)
        borrows = (a % 10) < (b % 10)
        if borrows == need_borrow:
            break
    diff = a - b
    return Problem(
        prompt=f"{a} - {b} = ?",
        answer=float(diff),
        skill="sub",
        choices=_int_choices(rng, diff, 4, lo=0, hi=120, spread=11),
        hint="If the top ones digit is too small, borrow a ten.",
        time_budget=_spanf(d, 13.0, 10.0),
    )


def _g2_array(rng: random.Random, d: float) -> Problem:
    rows = rng.randint(2, _span(d, 4, 6))
    cols = rng.randint(2, _span(d, 5, 9))
    total = rows * cols
    style = rng.random()
    if style < 0.45:
        prompt = f"{rows} rows of {cols}. How many in all?"
        visual = VisualSpec("array", {"rows": rows, "cols": cols, "label": None})
        hint = f"Count by {cols}s, {rows} times."
    elif style < 0.75:
        chain = " + ".join([str(cols)] * rows)
        prompt = f"{chain} = ?"
        visual = VisualSpec("array", {"rows": rows, "cols": cols, "label": None})
        hint = f"That is {rows} groups of {cols}."
    else:
        prompt = f"{rows} {MUL_SIGN} {cols} = ?"
        visual = VisualSpec("array", {"rows": rows, "cols": cols, "label": None})
        hint = f"Skip-count by {cols}: that is {rows} jumps."
    return Problem(
        prompt=prompt,
        answer=float(total),
        skill="mul",
        choices=_int_choices(rng, total, 4, lo=0, hi=80, spread=max(2, cols)),
        visual=visual,
        hint=hint,
        time_budget=_spanf(d, 13.0, 9.5),
    )


def _g2_money(rng: random.Random, d: float) -> Problem:
    pool = _COIN_TABLE[: 2] if d < 0.25 else (_COIN_TABLE[:3] if d < 0.55 else _COIN_TABLE)
    count = rng.randint(2, _span(d, 3, 6))
    coins = [rng.choice(pool) for _ in range(count)]
    total = sum(v for _n, v in coins)
    # Keep 2nd-grade totals under a dollar so the answer stays in whole cents.
    while total > 99:
        coins.pop()
        total = sum(v for _n, v in coins)
    if not coins:
        coins = [("dime", 10)]
        total = 10
    return Problem(
        prompt="How many cents?",
        answer=float(total),
        skill="money",
        unit="cents",
        choices=_int_choices(rng, total, 4, lo=1, hi=120, spread=6),
        visual=VisualSpec(
            "coins",
            {"coins": [{"name": n, "value": v} for n, v in coins], "unit": "cents"},
        ),
        hint="Start with the biggest coins, then count on.",
        time_budget=_spanf(d, 14.0, 10.0),
    )


def _g2_time(rng: random.Random, d: float) -> Problem:
    minutes = [0, 15, 30, 45] if d < 0.5 else list(range(0, 60, 5))
    return _clock_problem(rng, d, minutes, _spanf(d, 13.0, 9.5))


def _g2_place_value(rng: random.Random, d: float) -> Problem:
    hundreds = rng.randint(1, 9)
    tens = rng.randint(0, 9)
    ones = rng.randint(0, 9)
    number = hundreds * 100 + tens * 10 + ones
    mode = rng.choice(["build", "expanded", "value_of", "hundred_more"])
    if mode == "build":
        return Problem(
            prompt=f"{hundreds} hundreds, {tens} tens and {ones} ones make __",
            answer=float(number),
            skill="place_value",
            choices=_int_choices(rng, number, 4, lo=0, hi=999, spread=101),
            visual=VisualSpec(
                "base_ten", {"hundreds": hundreds, "tens": tens, "ones": ones}
            ),
            hint="Flats are 100, rods are 10, cubes are 1.",
            time_budget=_spanf(d, 13.0, 9.0),
        )
    if mode == "expanded":
        parts = [f"{hundreds * 100}"]
        if tens:
            parts.append(f"{tens * 10}")
        if ones:
            parts.append(f"{ones}")
        return Problem(
            prompt=" + ".join(parts) + " = ?",
            answer=float(number),
            skill="place_value",
            choices=_int_choices(rng, number, 4, lo=0, hi=999, spread=101),
            hint="Stack them up: hundreds, tens, ones.",
            time_budget=_spanf(d, 12.0, 9.0),
        )
    if mode == "value_of":
        place = rng.choice(["hundreds", "tens", "ones"])
        digit = {"hundreds": hundreds, "tens": tens, "ones": ones}[place]
        worth = {"hundreds": 100, "tens": 10, "ones": 1}[place]
        return Problem(
            prompt=f"In {number}, what is the value of the {place} digit?",
            answer=float(digit * worth),
            skill="place_value",
            choices=_int_choices(rng, digit * worth, 4, lo=0, hi=999, spread=max(2, worth)),
            hint=f"The {place} digit is {digit}, and each one is worth {worth}.",
            time_budget=_spanf(d, 13.0, 10.0),
        )
    step = rng.choice([10, 100]) if d >= 0.5 else 100
    up = rng.random() < 0.6
    if up:
        base = rng.randint(100, 999 - step)
        answer = base + step
    else:
        base = rng.randint(100 + step, 999)
        answer = base - step
    return Problem(
        prompt=f"What is {step} {'more' if up else 'less'} than {base}?",
        answer=float(answer),
        skill="place_value",
        choices=_int_choices(rng, answer, 4, lo=0, hi=1200, spread=max(11, step)),
        hint="Only one digit changes.",
        time_budget=_spanf(d, 12.0, 9.0),
    )


def _g2_compare(rng: random.Random, d: float) -> Problem:
    top = _span(d, 99, 999)
    a = rng.randint(10, top)
    b = rng.randint(10, top)
    if rng.random() < 0.12:
        b = a
    symbol = CMP_GREATER if a > b else (CMP_LESS if a < b else CMP_EQUAL)
    values, labels = _labeled_choices(
        rng,
        [(symbol, CMP_LABELS[int(symbol) + 1])]
        + [(v, CMP_LABELS[int(v) + 1]) for v in (CMP_LESS, CMP_EQUAL, CMP_GREATER)],
    )
    return Problem(
        prompt=f"{a}  ?  {b}",
        answer=symbol,
        skill="compare",
        choices=values,
        choice_labels=labels,
        input_mode="choice",
        hint="Compare the biggest place value first.",
        time_budget=_spanf(d, 10.0, 7.5),
    )


# --------------------------------------------------------------------------- #
# Word problems (2nd grade and up)
# --------------------------------------------------------------------------- #
#
# Every template computes its answer from the same integers it prints, and each
# is capped so the arithmetic stays inside the grade band.


def _wp_join(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _s, plural = rng.choice(_THINGS)
    a = rng.randint(max(2, mag // 3), mag)
    b = rng.randint(2, max(3, mag // 2))
    return (
        f"{name} has {a} {plural}. A friend gives {b} more. "
        f"How many {plural} now?",
        a + b,
        "Put the two amounts together.",
    )


def _wp_take_from(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _s, plural = rng.choice(_THINGS)
    a = rng.randint(max(4, mag // 2), mag)
    b = rng.randint(1, a - 1)
    return (
        f"{name} had {a} {plural} and gave away {b}. How many are left?",
        a - b,
        "Take the second amount off the first.",
    )


def _wp_compare(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    n1, n2 = rng.sample(_NAMES, 2)
    _s, plural = rng.choice(_THINGS)
    a = rng.randint(max(4, mag // 2), mag)
    b = rng.randint(1, a - 1)
    return (
        f"{n1} has {a} {plural}. {n2} has {b}. How many more does {n1} have?",
        a - b,
        "'How many more' means subtract.",
    )


def _wp_missing_addend(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _s, plural = rng.choice(_THINGS)
    start = rng.randint(max(2, mag // 3), mag)
    added = rng.randint(2, max(3, mag // 2))
    total = start + added
    return (
        f"{name} had some {plural}, found {added} more, and now has {total}. "
        f"How many did {name} start with?",
        start,
        "Work backwards: take the new ones back off the total.",
    )


def _wp_equal_groups(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    _cs, cplural = rng.choice(_CONTAINERS)
    _s, plural = rng.choice(_THINGS)
    groups = rng.randint(2, max(3, min(9, mag // 4)))
    each = rng.randint(2, max(3, min(10, mag // 3)))
    return (
        f"There are {groups} {cplural} with {each} {plural} in each. "
        f"How many {plural} in all?",
        groups * each,
        f"Equal groups: skip-count by {each}.",
    )


def _wp_share(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    _s, plural = rng.choice(_THINGS)
    groups = rng.randint(2, max(3, min(9, mag // 4)))
    each = rng.randint(2, max(3, min(10, mag // 3)))
    total = groups * each
    return (
        f"{total} {plural} are shared equally between {groups} kids. "
        f"How many does each kid get?",
        each,
        "Sharing equally means divide.",
    )


def _wp_money(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _s, plural = rng.choice(_THINGS)
    price = rng.choice([5, 10, 15, 20, 25, 30, 40, 50])
    qty = rng.randint(2, 5)
    return (
        f"One {_s} costs {price} cents. {name} buys {qty} of them. "
        f"How many cents does {name} spend?",
        price * qty,
        f"{qty} lots of {price} cents.",
    )


def _wp_two_step_add_mul(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _cs, cplural = rng.choice(_CONTAINERS)
    _s, plural = rng.choice(_THINGS)
    start = rng.randint(mag // 3, mag)
    groups = rng.randint(2, 6)
    each = rng.randint(3, 9)
    return (
        f"{name} has {start} {plural} and buys {groups} {cplural} "
        f"with {each} in each. How many {plural} now?",
        start + groups * each,
        "First multiply the new groups, then add what was already there.",
    )


def _wp_two_step_sub_mul(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    _s, plural = rng.choice(_THINGS)
    groups = rng.randint(2, 6)
    each = rng.randint(3, 9)
    used = groups * each
    total = used + rng.randint(2, max(3, mag // 2))
    return (
        f"There were {total} {plural}. {groups} kids each took {each}. "
        f"How many {plural} are left?",
        total - used,
        "Multiply what was taken, then subtract from the total.",
    )


def _wp_two_step_add_sub(rng: random.Random, mag: int) -> Tuple[str, int, str]:
    name = rng.choice(_NAMES)
    _s, plural = rng.choice(_THINGS)
    a = rng.randint(mag // 3, mag)
    b = rng.randint(5, max(6, mag // 2))
    c = rng.randint(2, a + b - 1)
    return (
        f"{name} had {a} {plural}, got {b} more, then gave {c} away. "
        f"How many are left?",
        a + b - c,
        "Do it in order: add first, then subtract.",
    )


#: (template, minimum grade index, minimum difficulty, step count)
_WORD_TEMPLATES: List[Tuple[Callable, int, float, int]] = [
    (_wp_join, 3, 0.0, 1),
    (_wp_take_from, 3, 0.0, 1),
    (_wp_compare, 3, 0.2, 1),
    (_wp_missing_addend, 3, 0.5, 1),
    (_wp_equal_groups, 3, 0.4, 1),
    (_wp_share, 4, 0.0, 1),
    (_wp_money, 3, 0.2, 1),
    (_wp_two_step_add_mul, 4, 0.55, 2),
    (_wp_two_step_sub_mul, 4, 0.55, 2),
    (_wp_two_step_add_sub, 4, 0.35, 2),
]


def _make_word_problem(rng: random.Random, d: float, grade: str) -> Problem:
    gi = _grade_index(grade)
    mag = {3: _span(d, 12, 40), 4: _span(d, 20, 90), 5: _span(d, 40, 200)}.get(
        gi, _span(d, 10, 30)
    )
    pool = [t for t in _WORD_TEMPLATES if gi >= t[1] and d >= t[2]]
    if not pool:
        pool = [_WORD_TEMPLATES[0]]
    template, _mg, _md, steps = rng.choice(pool)
    prompt, answer, hint = template(rng, mag)
    unit = "cents" if template is _wp_money else ""
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="word",
        unit=unit,
        choices=_int_choices(
            rng, answer, 4, lo=0, hi=max(20, int(answer * 2) + 10), spread=max(2, answer // 6)
        ),
        hint=hint,
        time_budget=_spanf(d, 18.0, 13.0) + (6.0 if steps > 1 else 0.0),
    )


def _g2_word(rng: random.Random, d: float) -> Problem:
    return _make_word_problem(rng, d, "2nd")


def _g3_word(rng: random.Random, d: float) -> Problem:
    return _make_word_problem(rng, d, "3rd")


def _g4_word(rng: random.Random, d: float) -> Problem:
    return _make_word_problem(rng, d, "4th")


# --------------------------------------------------------------------------- #
# 3rd grade generators
# --------------------------------------------------------------------------- #
#
# Multiplication and division facts to 10, fractions on a number line, unit
# fractions of a set, area and perimeter, rounding, elapsed time.


def _g3_mul(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    a = rng.randint(2, top)
    b = rng.randint(2, top)
    if d >= 0.6:
        # Bias toward the facts kids actually stumble on (6-9 tables).
        a = rng.randint(max(3, top - 4), top)
    product = a * b
    return Problem(
        prompt=f"{a} {MUL_SIGN} {b} = ?",
        answer=float(product),
        skill="mul",
        choices=_int_choices(rng, product, 4, lo=0, hi=140, spread=max(2, min(a, b))),
        visual=VisualSpec("array", {"rows": a, "cols": b, "label": None})
        if d < 0.4
        else None,
        hint=f"Skip-count by {b}, {a} times.",
        time_budget=_spanf(d, 10.0, 7.0),
    )


def _g3_div(rng: random.Random, d: float) -> Problem:
    top = _span(d, 5, 10)
    divisor = rng.randint(2, top)
    quotient = rng.randint(2, top)
    dividend = divisor * quotient
    return Problem(
        prompt=f"{dividend} {DIV_SIGN} {divisor} = ?",
        answer=float(quotient),
        skill="div",
        choices=_int_choices(rng, quotient, 4, lo=0, hi=15, spread=2),
        hint=f"How many {divisor}s fit inside {dividend}?",
        time_budget=_spanf(d, 12.0, 8.5),
    )


def _g3_fraction_numberline(rng: random.Random, d: float) -> Problem:
    denom = rng.choice([2, 4] if d < 0.35 else ([2, 3, 4, 6] if d < 0.7 else [3, 4, 6, 8]))
    num = rng.randint(1, denom - 1) if denom > 1 else 1
    fr = Fraction(num, denom)
    value = float(fr)
    exact = _terminating(fr)
    # Same-denominator neighbours first, then wrong-denominator lures - both
    # are real 3rd-grade misconceptions.
    pairs: List[Tuple[float, str]] = [(value, _frac_label(fr))]
    pairs += _fraction_distractors(rng, fr, upper=Fraction(1, 1))
    values, labels = _labeled_choices(rng, pairs, tolerance=0.01, count=4)
    return Problem(
        prompt="What fraction is the dot sitting on?",
        answer=value,
        skill="fraction",
        tolerance=1e-6 if exact else 0.005,
        choices=values,
        choice_labels=labels,
        input_mode="number" if exact else "choice",
        visual=VisualSpec(
            "numberline",
            {
                "min": 0.0,
                "max": 1.0,
                "step": 1.0 / denom,
                "denominator": denom,
                "mark": value,
                "mystery": value,
                "labels": True,
            },
        ),
        hint=f"The line from 0 to 1 is cut into {denom} equal pieces.",
        time_budget=_spanf(d, 15.0, 11.0),
    )


def _g3_unit_fraction(rng: random.Random, d: float) -> Problem:
    denom = rng.choice([2, 3, 4] if d < 0.5 else [2, 3, 4, 5, 6, 8])
    per = rng.randint(2, _span(d, 4, 9))
    total = denom * per
    numer = 1 if d < 0.55 else rng.randint(1, denom - 1) if denom > 1 else 1
    answer = per * numer
    label = _frac_label(Fraction(numer, denom))
    return Problem(
        prompt=f"What is {label} of {total}?",
        answer=float(answer),
        skill="fraction",
        choices=_int_choices(rng, answer, 4, lo=0, hi=total + 6, spread=max(2, per)),
        visual=VisualSpec(
            "fraction_bar",
            {"bars": [{"parts": denom, "filled": numer, "label": label}]},
        ),
        hint=f"Split {total} into {denom} equal groups first.",
        time_budget=_spanf(d, 16.0, 11.0),
    )


def _g3_area_perimeter(rng: random.Random, d: float) -> Problem:
    w = rng.randint(2, _span(d, 6, 12))
    h = rng.randint(2, _span(d, 5, 10))
    want_area = rng.random() < 0.55
    if want_area:
        answer = w * h
        prompt = f"A rectangle is {w} cm long and {h} cm wide.\nWhat is its area?"
        hint = "Area = length x width (count the unit squares)."
        unit = "sq cm"
    else:
        answer = 2 * (w + h)
        prompt = f"A rectangle is {w} cm long and {h} cm wide.\nWhat is its perimeter?"
        hint = "Perimeter is the walk all the way around: two lengths + two widths."
        unit = "cm"
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="shape",
        unit=unit,
        choices=_int_choices(rng, answer, 4, lo=1, hi=answer * 2 + 12, spread=max(2, min(w, h))),
        visual=VisualSpec("array", {"rows": h, "cols": w, "label": unit}),
        hint=hint,
        time_budget=_spanf(d, 16.0, 12.0),
    )


def _g3_rounding(rng: random.Random, d: float) -> Problem:
    place = 10 if d < 0.5 else rng.choice([10, 100])
    lo, hi = (11, 99) if place == 10 else (101, 999)
    n = rng.randint(lo, hi)
    while n % place == 0:  # rounding an already-round number is a non-question
        n = rng.randint(lo, hi)
    answer = _round_half_up(n, place)
    word = "ten" if place == 10 else "hundred"
    return Problem(
        prompt=f"Round {n} to the nearest {word}.",
        answer=float(answer),
        skill="place_value",
        choices=_int_choices(
            rng, answer, 4, lo=0, hi=answer + place * 3, spread=place
        ),
        visual=VisualSpec(
            "numberline",
            {
                "min": float((n // place) * place),
                "max": float((n // place + 1) * place),
                "step": float(place),
                "denominator": None,
                "mark": float(n),
                "mystery": None,
                "labels": True,
            },
        ),
        hint=f"Which {word} is {n} closer to? 5 or more rounds up.",
        time_budget=_spanf(d, 14.0, 10.0),
    )


def _g3_elapsed(rng: random.Random, d: float) -> Problem:
    start_h = rng.randint(1, 11)
    start_m = rng.choice(list(range(0, 60, 5)))
    gap = rng.choice([5, 10, 15, 20, 30] if d < 0.5 else [5, 10, 15, 20, 25, 35, 40, 45, 50])
    total = start_h * 60 + start_m + gap
    end_h = (total // 60 - 1) % 12 + 1
    end_m = total % 60
    return Problem(
        prompt=(
            f"Recess starts at {time_label(start_h, start_m)} and ends at "
            f"{time_label(end_h, end_m)}.\nHow many minutes long is it?"
        ),
        answer=float(gap),
        skill="time",
        unit="min",
        choices=_int_choices(rng, gap, 4, lo=5, hi=60, spread=10),
        hint="Count on by 5s from the start time.",
        time_budget=_spanf(d, 18.0, 13.0),
    )


def _g3_compare_fraction(rng: random.Random, d: float) -> Problem:
    if d < 0.5:
        denom = rng.choice([3, 4, 5, 6, 8])
        n1 = rng.randint(1, denom - 1)
        n2 = rng.randint(1, denom - 1)
        f1, f2 = Fraction(n1, denom), Fraction(n2, denom)
        hint = "Same size pieces, so just compare how many you have."
    else:
        # Unit fractions: more pieces means each piece is smaller.
        d1, d2 = rng.sample([2, 3, 4, 5, 6, 8, 10], 2)
        f1, f2 = Fraction(1, d1), Fraction(1, d2)
        hint = "The bigger the bottom number, the smaller each piece is."
    symbol = CMP_GREATER if f1 > f2 else (CMP_LESS if f1 < f2 else CMP_EQUAL)
    values, labels = _labeled_choices(
        rng,
        [(symbol, CMP_LABELS[int(symbol) + 1])]
        + [(v, CMP_LABELS[int(v) + 1]) for v in (CMP_LESS, CMP_EQUAL, CMP_GREATER)],
    )
    return Problem(
        prompt=f"{_frac_label(f1)}  ?  {_frac_label(f2)}",
        answer=symbol,
        skill="compare",
        choices=values,
        choice_labels=labels,
        input_mode="choice",
        visual=VisualSpec(
            "fraction_bar",
            {
                "bars": [
                    {"parts": f1.denominator, "filled": f1.numerator, "label": _frac_label(f1)},
                    {"parts": f2.denominator, "filled": f2.numerator, "label": _frac_label(f2)},
                ]
            },
        ),
        hint=hint,
        time_budget=_spanf(d, 14.0, 10.0),
    )


def _g3_add(rng: random.Random, d: float) -> Problem:
    top = _span(d, 99, 999)
    a = rng.randint(20, top)
    b = rng.randint(20, top)
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(a + b),
        skill="add",
        choices=_int_choices(rng, a + b, 4, lo=0, hi=2200, spread=101),
        hint="Line up the places and carry when a column hits 10.",
        time_budget=_spanf(d, 14.0, 10.0),
    )


def _g3_sub(rng: random.Random, d: float) -> Problem:
    top = _span(d, 99, 999)
    a = rng.randint(40, top)
    b = rng.randint(10, a)
    return Problem(
        prompt=f"{a} - {b} = ?",
        answer=float(a - b),
        skill="sub",
        choices=_int_choices(rng, a - b, 4, lo=0, hi=1100, spread=101),
        hint="Borrow from the next place when the top digit is too small.",
        time_budget=_spanf(d, 15.0, 11.0),
    )


# --------------------------------------------------------------------------- #
# 4th grade generators
# --------------------------------------------------------------------------- #
#
# Multi-digit multiplication, long division with remainders, equivalent
# fractions, fraction add/sub with like and unlike denominators, decimals to
# hundredths, factors and multiples, and angle basics.


def _g4_mul(rng: random.Random, d: float) -> Problem:
    if d < 0.35:
        a = rng.randint(11, 49)
        b = rng.randint(2, 9)
    elif d < 0.7:
        a = rng.randint(100, 999)
        b = rng.randint(2, 9)
    else:
        a = rng.randint(11, 99)
        b = rng.randint(11, 49)
    product = a * b
    return Problem(
        prompt=f"{a} {MUL_SIGN} {b} = ?",
        answer=float(product),
        skill="mul",
        choices=_int_choices(
            rng, product, 4, lo=0, hi=product * 2 + 50, spread=max(3, b * 2)
        ),
        hint="Break the big number into hundreds, tens and ones, then add.",
        time_budget=_spanf(d, 20.0, 14.0),
    )


def _g4_div(rng: random.Random, d: float) -> Problem:
    divisor = rng.randint(2, _span(d, 6, 12))
    quotient = rng.randint(_span(d, 3, 12), _span(d, 12, 90))
    remainder = 0 if d < 0.3 else rng.randint(0, divisor - 1)
    dividend = divisor * quotient + remainder
    if remainder and rng.random() < 0.45:
        return Problem(
            prompt=f"{dividend} {DIV_SIGN} {divisor}\nWhat is the remainder?",
            answer=float(remainder),
            skill="div",
            choices=_int_choices(rng, remainder, 4, lo=0, hi=max(3, divisor - 1), spread=2),
            hint=f"How much is left over after all the whole {divisor}s?",
            time_budget=_spanf(d, 20.0, 15.0),
        )
    label = "" if remainder == 0 else "\n(whole number part only)"
    return Problem(
        prompt=f"{dividend} {DIV_SIGN} {divisor} = ?{label}",
        answer=float(quotient),
        skill="div",
        choices=_int_choices(
            rng, quotient, 4, lo=0, hi=quotient * 2 + 10, spread=max(2, quotient // 5)
        ),
        hint="Divide one place at a time, left to right.",
        time_budget=_spanf(d, 20.0, 15.0),
    )


def _g4_equivalent_fraction(rng: random.Random, d: float) -> Problem:
    denom = rng.choice([2, 3, 4, 5] if d < 0.5 else [2, 3, 4, 5, 6, 8, 10])
    numer = rng.randint(1, denom - 1)
    factor = rng.randint(2, _span(d, 3, 6))
    if rng.random() < 0.55:
        prompt = f"{numer}/{denom} = ?/{denom * factor}"
        answer = numer * factor
        hint = f"The bottom was multiplied by {factor}, so the top must be too."
        hi = denom * factor
    else:
        prompt = f"{numer * factor}/{denom * factor} = {numer}/?"
        answer = denom
        hint = "Simplify: divide the top and the bottom by the same number."
        hi = denom * factor
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="fraction",
        choices=_int_choices(rng, answer, 4, lo=1, hi=max(hi + 4, answer + 4), spread=2),
        visual=VisualSpec(
            "fraction_bar",
            {
                "bars": [
                    {"parts": denom, "filled": numer, "label": f"{numer}/{denom}"},
                    {"parts": denom * factor, "filled": numer * factor,
                     "label": f"?/{denom * factor}"},
                ]
            },
        ),
        hint=hint,
        time_budget=_spanf(d, 18.0, 13.0),
    )


def _g4_fraction_addsub(rng: random.Random, d: float) -> Problem:
    if d < 0.4:
        # Like denominators.
        denom = rng.choice([3, 4, 5, 6, 8])
        n1 = rng.randint(1, denom - 1)
        n2 = rng.randint(1, denom - 1)
        f1, f2 = Fraction(n1, denom), Fraction(n2, denom)
        hint = "Same bottom number: just add or subtract the tops."
    elif d < 0.72:
        # One denominator is a multiple of the other.
        base = rng.choice([2, 3, 4, 5])
        mult = rng.choice([2, 3, 4])
        f1 = Fraction(rng.randint(1, base - 1) if base > 1 else 1, base)
        f2 = Fraction(rng.randint(1, base * mult - 1), base * mult)
        hint = "Rename the first fraction so both have the same bottom number."
    else:
        # Genuinely unlike denominators.
        d1, d2 = rng.sample([2, 3, 4, 5, 6, 8], 2)
        f1 = Fraction(rng.randint(1, d1 - 1), d1)
        f2 = Fraction(rng.randint(1, d2 - 1), d2)
        hint = "Find a common denominator for both, then add or subtract."
    subtract = rng.random() < 0.45
    if subtract and f1 < f2:
        f1, f2 = f2, f1
    result = (f1 - f2) if subtract else (f1 + f2)
    if subtract and result == 0:
        subtract = False
        result = f1 + f2
    op = "-" if subtract else "+"
    value = float(result)
    exact = _terminating(result)
    # Build fraction-labelled distractors, including the classic
    # "added the bottoms too" error.
    pairs: List[Tuple[float, str]] = [(value, _frac_label(result))]
    wrong_den = f1.denominator + f2.denominator
    wrong_num = f1.numerator + f2.numerator if not subtract else abs(
        f1.numerator - f2.numerator
    )
    if wrong_den and wrong_num:
        lure = Fraction(wrong_num, wrong_den)
        pairs.append((float(lure), _frac_label(lure)))
    for _ in range(6):
        jitter = Fraction(rng.randint(1, 5), rng.choice([2, 3, 4, 5, 6, 8]))
        cand = result + jitter if rng.random() < 0.5 else result - jitter
        if cand > 0:
            pairs.append((float(cand), _frac_label(cand)))
    tol = 1e-6 if exact else 0.005
    values, labels = _labeled_choices(rng, pairs[:5], tolerance=max(tol, 0.01))
    return Problem(
        prompt=f"{_frac_label(f1)} {op} {_frac_label(f2)} = ?",
        answer=value,
        skill="fraction",
        tolerance=tol,
        choices=values,
        choice_labels=labels,
        input_mode="number" if exact else "choice",
        visual=VisualSpec(
            "fraction_bar",
            {
                "bars": [
                    {"parts": f1.denominator, "filled": f1.numerator, "label": _frac_label(f1)},
                    {"parts": f2.denominator, "filled": f2.numerator, "label": _frac_label(f2)},
                ]
            },
        ),
        hint=hint,
        time_budget=_spanf(d, 22.0, 16.0),
    )


def _g4_decimal(rng: random.Random, d: float) -> Problem:
    # All decimal work is done in integer hundredths so nothing ever drifts.
    step = 10 if d < 0.4 else 1  # tenths first, then true hundredths
    a = rng.randrange(step, _span(d, 100, 400), step)
    b = rng.randrange(step, _span(d, 100, 300), step)
    subtract = rng.random() < 0.45
    if subtract and b > a:
        a, b = b, a
    cents = a - b if subtract else a + b
    op = "-" if subtract else "+"
    money = rng.random() < 0.35
    if money:
        prompt = f"${a / 100:.2f} {op} ${b / 100:.2f} = ?"
        hint = "Line up the decimal points, then add or subtract like whole cents."
    else:
        prompt = f"{a / 100:.2f} {op} {b / 100:.2f} = ?"
        hint = "Line up the decimal points. Think in hundredths."
    answer = cents / 100.0
    pairs = [(answer, f"{answer:.2f}")]
    for delta in (10, -10, 1, -1, 100, -100, 9, -9):
        alt = cents + delta
        if alt > 0:
            pairs.append((alt / 100.0, f"{alt / 100.0:.2f}"))
    values, labels = _labeled_choices(rng, pairs[:5], tolerance=0.001)
    return Problem(
        prompt=prompt,
        answer=answer,
        skill="money" if money else "add" if not subtract else "sub",
        tolerance=1e-6,
        choices=values,
        choice_labels=labels,
        hint=hint,
        time_budget=_spanf(d, 20.0, 14.0),
    )


def _g4_factors(rng: random.Random, d: float) -> Problem:
    mode = rng.choice(["gcf", "lcm", "factor_count", "next_multiple", "is_prime_ish"])
    if mode == "gcf":
        a = rng.randint(4, _span(d, 24, 60))
        b = rng.randint(4, _span(d, 24, 60))
        answer = math.gcd(a, b)
        prompt = f"What is the greatest common factor of {a} and {b}?"
        hint = "List what divides into both, then take the biggest."
        hi = min(a, b)
    elif mode == "lcm":
        a = rng.randint(2, _span(d, 6, 12))
        b = rng.randint(2, _span(d, 6, 12))
        answer = a * b // math.gcd(a, b)
        prompt = f"What is the least common multiple of {a} and {b}?"
        hint = "Count by each number until you hit the same one."
        hi = a * b + 4
    elif mode == "factor_count":
        n = rng.randint(_span(d, 8, 20), _span(d, 24, 60))
        answer = len(_factors(n))
        prompt = f"How many factors does {n} have?"
        hint = "Pair them up: 1 x n, 2 x ..., and count them all."
        hi = 16
    elif mode == "next_multiple":
        b = rng.randint(3, _span(d, 6, 12))
        k = rng.randint(3, _span(d, 8, 20))
        n = b * k + rng.randint(1, b - 1)
        answer = b * (k + 1)
        prompt = f"What is the next multiple of {b} after {n}?"
        hint = f"Keep counting by {b}s until you pass {n}."
        hi = answer + b * 3
    else:
        n = rng.choice([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 4, 6, 8, 9, 12, 15, 21, 25])
        answer = len(_factors(n))
        prompt = f"How many factors does {n} have?\n(A prime has exactly 2.)"
        hint = "Count every number that divides in evenly, including 1 and itself."
        hi = 12
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="place_value",
        choices=_int_choices(rng, answer, 4, lo=1, hi=max(hi, answer + 4), spread=3),
        hint=hint,
        time_budget=_spanf(d, 22.0, 16.0),
    )


def _g4_angles(rng: random.Random, d: float) -> Problem:
    mode = rng.choice(["turn", "complement", "supplement", "classify"])
    if mode == "turn" or d < 0.3:
        quarters = rng.randint(1, 4)
        answer = 90 * quarters
        label = {1: "1/4", 2: "1/2", 3: "3/4", 4: "a full"}[quarters]
        return Problem(
            prompt=f"How many degrees is {label} turn?",
            answer=float(answer),
            skill="shape",
            unit="°",
            choices=_int_choices(rng, answer, 4, lo=45, hi=360, spread=90),
            hint="A full turn all the way around is 360 degrees.",
            time_budget=_spanf(d, 16.0, 12.0),
        )
    if mode == "complement":
        a = rng.randrange(5, 86, 5)
        answer = 90 - a
        prompt = f"Two angles make a right angle. One is {a}°.\nWhat is the other?"
        hint = "A right angle is 90 degrees in total."
        hi = 90
    elif mode == "supplement":
        a = rng.randrange(5, 176, 5)
        answer = 180 - a
        prompt = f"Two angles make a straight line. One is {a}°.\nWhat is the other?"
        hint = "A straight line is 180 degrees in total."
        hi = 180
    else:
        a = rng.choice(
            [rng.randint(5, 89), 90, rng.randint(91, 179), 180]
        )
        idx = 0 if a < 90 else (1 if a == 90 else (2 if a < 180 else 3))
        values, labels = _labeled_choices(
            rng,
            [(float(idx), ANGLE_TYPES[idx])]
            + [(float(i), ANGLE_TYPES[i]) for i in range(len(ANGLE_TYPES))],
        )
        return Problem(
            prompt=f"An angle measures {a}°. What kind of angle is it?",
            answer=float(idx),
            skill="shape",
            choices=values,
            choice_labels=labels,
            input_mode="choice",
            hint="Under 90 is acute, exactly 90 is right, over 90 is obtuse.",
            time_budget=_spanf(d, 15.0, 11.0),
        )
    return Problem(
        prompt=prompt,
        answer=float(answer),
        skill="shape",
        unit="°",
        choices=_int_choices(rng, answer, 4, lo=0, hi=hi, spread=10),
        hint=hint,
        time_budget=_spanf(d, 18.0, 13.0),
    )


def _g4_rounding(rng: random.Random, d: float) -> Problem:
    place = rng.choice([100, 1000] if d >= 0.45 else [10, 100])
    lo = place + 1
    hi = place * 10 - 1 if place >= 100 else 999
    n = rng.randint(lo, max(lo + 10, hi))
    while n % place == 0:
        n = rng.randint(lo, max(lo + 10, hi))
    answer = _round_half_up(n, place)
    word = {10: "ten", 100: "hundred", 1000: "thousand"}[place]
    return Problem(
        prompt=f"Round {n} to the nearest {word}.",
        answer=float(answer),
        skill="place_value",
        choices=_int_choices(rng, answer, 4, lo=0, hi=answer + place * 3, spread=place),
        hint=f"Look at the digit just right of the {word}s place.",
        time_budget=_spanf(d, 15.0, 11.0),
    )


def _g4_compare(rng: random.Random, d: float) -> Problem:
    if rng.random() < 0.5:
        # Decimals: the "0.7 vs 0.65" trap.
        a = rng.randrange(5, 200, 5) / 100.0
        b = rng.randrange(5, 200, 1) / 100.0
        left, right = f"{a:.2f}", f"{b:.2f}"
        symbol = CMP_GREATER if a > b else (CMP_LESS if a < b else CMP_EQUAL)
        hint = "More digits does not mean bigger. Compare tenths first."
    else:
        d1, d2 = rng.sample([2, 3, 4, 5, 6, 8, 10, 12], 2)
        f1 = Fraction(rng.randint(1, d1 - 1), d1)
        f2 = Fraction(rng.randint(1, d2 - 1), d2)
        left, right = _frac_label(f1), _frac_label(f2)
        symbol = CMP_GREATER if f1 > f2 else (CMP_LESS if f1 < f2 else CMP_EQUAL)
        hint = "Give both fractions the same bottom number, then compare."
    values, labels = _labeled_choices(
        rng,
        [(symbol, CMP_LABELS[int(symbol) + 1])]
        + [(v, CMP_LABELS[int(v) + 1]) for v in (CMP_LESS, CMP_EQUAL, CMP_GREATER)],
    )
    return Problem(
        prompt=f"{left}  ?  {right}",
        answer=symbol,
        skill="compare",
        choices=values,
        choice_labels=labels,
        input_mode="choice",
        hint=hint,
        time_budget=_spanf(d, 15.0, 11.0),
    )


def _g4_add(rng: random.Random, d: float) -> Problem:
    top = _span(d, 999, 99999)
    a = rng.randint(100, top)
    b = rng.randint(100, top)
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(a + b),
        skill="add",
        choices=_int_choices(rng, a + b, 4, lo=0, hi=(a + b) * 2, spread=1001),
        hint="Stack them up and carry between places.",
        time_budget=_spanf(d, 16.0, 12.0),
    )


def _g4_sub(rng: random.Random, d: float) -> Problem:
    top = _span(d, 999, 99999)
    a = rng.randint(200, top)
    b = rng.randint(100, a)
    return Problem(
        prompt=f"{a} - {b} = ?",
        answer=float(a - b),
        skill="sub",
        choices=_int_choices(rng, a - b, 4, lo=0, hi=a + 100, spread=1001),
        hint="Borrow across the zeros carefully.",
        time_budget=_spanf(d, 18.0, 13.0),
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
#
# Each entry is (skill, weight, generator). Weight controls how often a skill
# shows up in a well-mixed stream for that grade.

Generator = Callable[[random.Random, float], Problem]

_REGISTRY: Dict[str, List[Tuple[str, float, Generator]]] = {
    "pre-k": [
        ("count", 3.0, _pk_count),
        ("count", 1.6, _pk_subitize),
        ("count", 1.4, _pk_pattern),
        ("compare", 2.0, _pk_more_less),
        ("shape", 1.6, _pk_shape),
        ("add", 1.4, _pk_add_tiny),
    ],
    "k": [
        ("add", 2.6, _k_add),
        ("sub", 2.2, _k_sub),
        ("count", 1.8, _k_tenframe),
        ("add", 1.6, _k_bond),
        ("compare", 1.4, _k_order),
        ("compare", 1.2, _k_compare),
        ("shape", 1.2, _pk_shape),
    ],
    "1st": [
        ("add", 2.6, _g1_add),
        ("sub", 2.4, _g1_sub),
        ("add", 1.6, _g1_missing_addend),
        ("place_value", 2.0, _g1_place_value),
        ("time", 1.4, _g1_time),
        ("compare", 1.4, _g1_compare),
        ("count", 0.9, _k_tenframe),
        ("shape", 0.9, _pk_shape),
    ],
    "2nd": [
        ("add", 2.4, _g2_add),
        ("sub", 2.4, _g2_sub),
        ("mul", 2.0, _g2_array),
        ("money", 1.6, _g2_money),
        ("time", 1.3, _g2_time),
        ("place_value", 1.8, _g2_place_value),
        ("compare", 1.2, _g2_compare),
        ("word", 2.0, _g2_word),
    ],
    "3rd": [
        ("mul", 2.8, _g3_mul),
        ("div", 2.4, _g3_div),
        ("fraction", 1.8, _g3_fraction_numberline),
        ("fraction", 1.6, _g3_unit_fraction),
        ("shape", 1.5, _g3_area_perimeter),
        ("place_value", 1.4, _g3_rounding),
        ("time", 1.2, _g3_elapsed),
        ("compare", 1.2, _g3_compare_fraction),
        ("add", 1.2, _g3_add),
        ("sub", 1.2, _g3_sub),
        ("word", 2.0, _g3_word),
    ],
    "4th": [
        ("mul", 2.6, _g4_mul),
        ("div", 2.4, _g4_div),
        ("fraction", 1.8, _g4_equivalent_fraction),
        ("fraction", 1.8, _g4_fraction_addsub),
        ("money", 1.4, _g4_decimal),
        ("place_value", 1.4, _g4_factors),
        ("place_value", 1.2, _g4_rounding),
        ("shape", 1.4, _g4_angles),
        ("compare", 1.2, _g4_compare),
        ("add", 1.0, _g4_add),
        ("sub", 1.0, _g4_sub),
        ("word", 2.2, _g4_word),
    ],
}


def skills_for_grade(grade: str) -> List[str]:
    """Distinct skills that actually appear in a grade, in curriculum order."""
    g = _norm_grade(grade)
    seen: List[str] = []
    for skill, _w, _fn in _REGISTRY.get(g, []):
        if skill not in seen:
            seen.append(skill)
    return sorted(seen, key=SKILLS.index)


def grade_skill_weights(grade: str) -> Dict[str, float]:
    """Total generator weight per skill for a grade (used to mix a stream)."""
    out: Dict[str, float] = {}
    for skill, w, _fn in _REGISTRY.get(_norm_grade(grade), []):
        out[skill] = out.get(skill, 0.0) + w
    return out


# --------------------------------------------------------------------------- #
# Public generation API
# --------------------------------------------------------------------------- #


def _safe_fallback(grade: str, difficulty: float, rng: random.Random) -> Problem:
    """Last-resort problem so a live match can never crash on content."""
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    return Problem(
        prompt=f"{a} + {b} = ?",
        answer=float(a + b),
        skill="add",
        grade=_norm_grade(grade),
        difficulty=_clamp01(difficulty),
        choices=_int_choices(rng, a + b, 4, lo=0, hi=20, spread=2),
        hint="Count on from the bigger number.",
        time_budget=8.0,
    )


def make_problem(
    grade: str,
    difficulty: float = 0.5,
    skills: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Problem:
    """Return one fresh problem for `grade` at `difficulty` (0..1 within grade).

    `skills` optionally restricts the draw to a subset of `SKILLS`; unknown or
    unavailable skills are ignored rather than raising, so a caller can always
    pass a wish-list.
    """
    global GENERATOR_FAILURES
    rng = rng or random
    g = _norm_grade(grade)
    d = _clamp01(difficulty)

    entries = _REGISTRY.get(g, _REGISTRY["k"])
    if skills:
        wanted = set(skills)
        filtered = [e for e in entries if e[0] in wanted]
        if filtered:
            entries = filtered

    for _attempt in range(4):
        skill, _w, fn = _weighted_pick(rng, entries, [e[1] for e in entries])
        try:
            problem = fn(rng, d)
        except Exception:  # pragma: no cover - defensive, must never fire
            GENERATOR_FAILURES += 1
            continue
        problem.grade = g
        problem.difficulty = d
        if not problem.skill:
            problem.skill = skill
        return problem
    return _safe_fallback(g, d, rng)


def make_problem_set(
    grade: str,
    count: int,
    difficulty: float = 0.5,
    skills: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> List[Problem]:
    """A batch of `count` problems, de-duplicated by prompt where possible."""
    rng = rng or random
    out: List[Problem] = []
    seen: set = set()
    attempts = 0
    while len(out) < max(0, int(count)) and attempts < max(20, count * 12):
        attempts += 1
        p = make_problem(grade, difficulty, skills, rng)
        if p.prompt in seen:
            continue
        seen.add(p.prompt)
        out.append(p)
    # If the space was genuinely too small to fill (e.g. Pre-K shapes only),
    # top up with repeats rather than returning short.
    while len(out) < max(0, int(count)):
        out.append(make_problem(grade, difficulty, skills, rng))
    return out


# --------------------------------------------------------------------------- #
# Adaptive stream
# --------------------------------------------------------------------------- #


class ProblemStream:
    """An endless, well-mixed stream of problems for one player.

    Responsibilities:

    * **Mix.** Draws skills using the grade's curriculum weights, biased toward
      the skills this player is actually weak at.
    * **Adapt.** Nudges difficulty up on fast correct answers and down on
      wrong ones, so the stream tracks the child rather than the clock.
    * **No repeats.** Never serves the same prompt twice in a row (and avoids
      the last few where it can).
    """

    #: How many recent prompts to avoid re-serving.
    RECENT_WINDOW = 6
    #: Difficulty is clamped to this band so a match never becomes trivial or
    #: impossible.
    MIN_DIFFICULTY = 0.05
    MAX_DIFFICULTY = 0.98

    def __init__(
        self,
        grade: str,
        difficulty: float = 0.5,
        weak_skills: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.grade = _norm_grade(grade)
        self._difficulty = _clamp01(difficulty)
        self.rng = random.Random(seed)
        self._recent: deque = deque(maxlen=self.RECENT_WINDOW)
        self._base_weights = grade_skill_weights(self.grade)

        # Per-skill running record: attempts, correct, cumulative pace ratio.
        self._stats: Dict[str, Dict[str, float]] = {
            s: {"n": 0.0, "ok": 0.0, "pace": 0.0} for s in self._base_weights
        }
        # Seeded weakness gets an immediate boost before any evidence arrives.
        self._seed_weak = {s for s in (weak_skills or []) if s in self._base_weights}

        self.served = 0
        self.total_correct = 0
        self.total_wrong = 0
        self.streak = 0
        self.best_streak = 0

    # ------------------------------------------------------------- difficulty #
    @property
    def difficulty(self) -> float:
        return self._difficulty

    @difficulty.setter
    def difficulty(self, value: float) -> None:
        self._difficulty = min(
            self.MAX_DIFFICULTY, max(self.MIN_DIFFICULTY, _clamp01(value))
        )

    # ------------------------------------------------------------------ mixing #
    def _skill_weight(self, skill: str) -> float:
        base = self._base_weights.get(skill, 1.0)
        st = self._stats.get(skill, {"n": 0.0, "ok": 0.0})
        n = st["n"]
        if n <= 0:
            # Unseen skills get a mild boost so the mix explores early.
            boost = 1.35 if skill in self._seed_weak else 1.1
            return base * boost
        accuracy = st["ok"] / n
        # 100% accurate -> 0.7x, 0% accurate -> 2.6x.
        weakness = 1.0 - accuracy
        weight = base * (0.7 + 1.9 * weakness)
        if skill in self._seed_weak:
            weight *= 1.25
        return max(0.15, weight)

    def weak_skills(self, limit: int = 3, min_attempts: int = 3) -> List[str]:
        """The skills this player is currently worst at (lowest accuracy)."""
        scored = [
            (st["ok"] / st["n"], s)
            for s, st in self._stats.items()
            if st["n"] >= min_attempts
        ]
        scored.sort()
        out = [s for _acc, s in scored[:limit]]
        for s in self._seed_weak:
            if s not in out:
                out.append(s)
        return out[:limit] if out else sorted(self._seed_weak)

    # ------------------------------------------------------------------ serving #
    def next(self, skills: Optional[List[str]] = None) -> Problem:
        """Serve the next problem (optionally forcing a skill subset)."""
        pool = list(skills) if skills else list(self._base_weights.keys())
        pool = [s for s in pool if s in self._base_weights] or list(
            self._base_weights.keys()
        )
        problem: Optional[Problem] = None
        for _ in range(8):
            skill = _weighted_pick(
                self.rng, pool, [self._skill_weight(s) for s in pool]
            )
            # A little jitter keeps consecutive problems from feeling identical.
            d = _clamp01(self._difficulty + self.rng.uniform(-0.08, 0.08))
            candidate = make_problem(self.grade, d, [skill], self.rng)
            if candidate.prompt not in self._recent:
                problem = candidate
                break
            problem = candidate
        assert problem is not None
        self._recent.append(problem.prompt)
        self.served += 1
        return problem

    # ---------------------------------------------------------------- feedback #
    def report(self, problem: Problem, correct: bool, elapsed: float) -> None:
        """Record an outcome and adapt difficulty + skill mix."""
        skill = problem.skill
        st = self._stats.setdefault(skill, {"n": 0.0, "ok": 0.0, "pace": 0.0})
        budget = max(0.5, float(problem.time_budget))
        pace = max(0.0, float(elapsed)) / budget
        st["n"] += 1.0
        st["ok"] += 1.0 if correct else 0.0
        st["pace"] += pace
        if skill not in self._base_weights:
            self._base_weights[skill] = 1.0

        if correct:
            self.total_correct += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            if pace < 0.45:
                gain = 0.050          # nailed it, and fast
            elif pace < 0.85:
                gain = 0.030
            else:
                gain = 0.012          # right, but slow - barely nudge
            self.difficulty = self._difficulty + gain
            if skill in self._seed_weak and st["n"] >= 4 and st["ok"] / st["n"] > 0.75:
                self._seed_weak.discard(skill)  # they fixed it
        else:
            self.total_wrong += 1
            self.streak = 0
            self.difficulty = self._difficulty - 0.055
            self._seed_weak.add(skill)

    # ----------------------------------------------------------------- summary #
    def summary(self) -> dict:
        """Session report suitable for an end-of-match panel."""
        per_skill = {}
        for skill, st in self._stats.items():
            if st["n"] <= 0:
                continue
            per_skill[skill] = {
                "label": skill_label(skill),
                "attempts": int(st["n"]),
                "correct": int(st["ok"]),
                "accuracy": round(st["ok"] / st["n"], 3),
                "avg_pace": round(st["pace"] / st["n"], 3),
            }
        attempted = self.total_correct + self.total_wrong
        return {
            "grade": self.grade,
            "grade_label": grade_label(self.grade),
            "served": self.served,
            "attempted": attempted,
            "correct": self.total_correct,
            "wrong": self.total_wrong,
            "accuracy": round(self.total_correct / attempted, 3) if attempted else 0.0,
            "best_streak": self.best_streak,
            "difficulty": round(self._difficulty, 3),
            "weak_skills": self.weak_skills(),
            "per_skill": per_skill,
        }


# --------------------------------------------------------------------------- #
# Deterministic helpers (used by the Daily Challenge)
# --------------------------------------------------------------------------- #


def daily_seed(date: Optional[_datetime.date] = None) -> int:
    """YYYYMMDD as an int - every player gets the same problems on a given day."""
    day = date or _datetime.date.today()
    return day.year * 10000 + day.month * 100 + day.day


def make_daily_set(
    grade: str,
    count: int = 20,
    date: Optional[_datetime.date] = None,
    difficulty: float = 0.5,
) -> List[Problem]:
    """A reproducible problem set keyed to a calendar date and grade."""
    seed = daily_seed(date) * 31 + _grade_index(grade)
    rng = random.Random(seed)
    out: List[Problem] = []
    # Ramp difficulty across the set so it opens gently and finishes spicy.
    for i in range(max(1, int(count))):
        t = i / max(1, count - 1)
        d = _clamp01(difficulty * 0.6 + 0.55 * t)
        out.append(make_problem(grade, d, None, rng))
    return out

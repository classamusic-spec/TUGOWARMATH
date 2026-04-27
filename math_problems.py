"""Math problem generator for Tug-of-War Math.

Each grade band returns a Problem with a prompt string and a numeric answer.
Answers are floats so fractions can be supported uniformly; integer problems
compare with a small epsilon.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, List


GRADES = ["Pre-K", "K", "1st", "2nd", "3rd", "4th"]


@dataclass
class Problem:
    prompt: str
    answer: float
    # Tolerance used when comparing typed answers (handles fractions / decimals)
    tolerance: float = 1e-6
    # Pull weight: harder problems are worth a bit more rope pull
    weight: float = 1.0

    def check(self, value: float) -> bool:
        return abs(value - self.answer) <= self.tolerance


def _pre_k() -> Problem:
    kind = random.choice(["count", "bigger", "add_tiny"])
    if kind == "count":
        n = random.randint(1, 5)
        dots = "* " * n
        return Problem(f"How many stars?\n{dots.strip()}", float(n), weight=0.8)
    if kind == "bigger":
        a, b = random.sample(range(1, 6), 2)
        # Answer is the bigger number itself
        return Problem(f"Which is bigger?\n{a}  or  {b}", float(max(a, b)), weight=0.8)
    a, b = random.randint(1, 3), random.randint(1, 2)
    return Problem(f"{a} + {b} = ?", float(a + b), weight=0.8)


def _kindergarten() -> Problem:
    op = random.choice(["+", "-"])
    if op == "+":
        a, b = random.randint(0, 5), random.randint(0, 5)
        return Problem(f"{a} + {b} = ?", float(a + b))
    a = random.randint(2, 5)
    b = random.randint(0, a)
    return Problem(f"{a} - {b} = ?", float(a - b))


def _first() -> Problem:
    op = random.choice(["+", "-", "+", "-", "miss"])
    if op == "+":
        a, b = random.randint(1, 19), random.randint(1, 20 - 1)
        b = min(b, 20 - a)
        return Problem(f"{a} + {b} = ?", float(a + b), weight=1.1)
    if op == "-":
        a = random.randint(2, 20)
        b = random.randint(0, a)
        return Problem(f"{a} - {b} = ?", float(a - b), weight=1.1)
    # missing addend
    a = random.randint(1, 9)
    b = random.randint(1, 9)
    return Problem(f"{a} + ? = {a + b}", float(b), weight=1.2)


def _second() -> Problem:
    op = random.choice(["+", "-", "x"])
    if op == "+":
        a, b = random.randint(10, 80), random.randint(10, 80)
        return Problem(f"{a} + {b} = ?", float(a + b), weight=1.2)
    if op == "-":
        a = random.randint(20, 99)
        b = random.randint(1, a)
        return Problem(f"{a} - {b} = ?", float(a - b), weight=1.2)
    a, b = random.randint(2, 5), random.randint(2, 5)
    return Problem(f"{a} x {b} = ?", float(a * b), weight=1.3)


def _third() -> Problem:
    kind = random.choice(["mul", "div", "add", "sub"])
    if kind == "mul":
        a, b = random.randint(2, 10), random.randint(2, 10)
        return Problem(f"{a} x {b} = ?", float(a * b), weight=1.4)
    if kind == "div":
        b = random.randint(2, 10)
        ans = random.randint(2, 10)
        a = b * ans
        return Problem(f"{a} / {b} = ?", float(ans), weight=1.4)
    if kind == "add":
        a, b = random.randint(20, 99), random.randint(20, 99)
        return Problem(f"{a} + {b} = ?", float(a + b), weight=1.3)
    a = random.randint(30, 99)
    b = random.randint(10, a)
    return Problem(f"{a} - {b} = ?", float(a - b), weight=1.3)


def _fourth() -> Problem:
    kind = random.choice(["mul", "mul", "div", "frac", "addbig", "subbig"])
    if kind == "mul":
        a, b = random.randint(2, 12), random.randint(2, 12)
        return Problem(f"{a} x {b} = ?", float(a * b), weight=1.5)
    if kind == "div":
        b = random.randint(2, 12)
        ans = random.randint(2, 12)
        a = b * ans
        return Problem(f"{a} / {b} = ?", float(ans), weight=1.5)
    if kind == "frac":
        # 1/2 + 1/4 etc., answer in decimal with tolerance
        choices = [
            (Fraction(1, 2), Fraction(1, 4), "1/2 + 1/4"),
            (Fraction(1, 4), Fraction(1, 4), "1/4 + 1/4"),
            (Fraction(1, 2), Fraction(1, 2), "1/2 + 1/2"),
            (Fraction(3, 4), Fraction(1, 4), "3/4 + 1/4"),
            (Fraction(2, 3), Fraction(1, 3), "2/3 + 1/3"),
        ]
        a, b, label = random.choice(choices)
        ans = float(a + b)
        return Problem(
            f"{label} = ?\n(decimal ok)",
            ans,
            tolerance=0.02,
            weight=1.6,
        )
    if kind == "addbig":
        a, b = random.randint(100, 999), random.randint(100, 999)
        return Problem(f"{a} + {b} = ?", float(a + b), weight=1.5)
    a = random.randint(200, 999)
    b = random.randint(50, a - 1)
    return Problem(f"{a} - {b} = ?", float(a - b), weight=1.5)


_GENERATORS: List[Callable[[], Problem]] = [
    _pre_k,
    _kindergarten,
    _first,
    _second,
    _third,
    _fourth,
]


def make_problem(grade_index: int) -> Problem:
    """Return a fresh problem appropriate for grade index 0..5."""
    grade_index = max(0, min(grade_index, len(_GENERATORS) - 1))
    return _GENERATORS[grade_index]()


def grade_label(grade_index: int) -> str:
    return GRADES[max(0, min(grade_index, len(GRADES) - 1))]

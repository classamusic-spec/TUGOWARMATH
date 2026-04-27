# Tug-of-War Math

A two-player Python game. Two teams of stick-figure kids stand on each side of
the screen and pull on a rope. Each team gets its own math problem; whoever
solves theirs faster yanks the flag toward their side. Wrong answers lock you
out for a moment and let the other team gain ground. First team to drag the
flag past their victory line wins; if the 90 s timer runs out first, whichever
side has the flag wins.

Built with `pygame` for graphics and `numpy` to synthesize all sound effects at
startup, so there are no audio asset files to ship.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Controls

### Menu
- `Left` / `Right` (or `A` / `D`) — pick a grade band (Pre-K, K, 1st, 2nd, 3rd, 4th)
- `Space` or `Enter` — start the round
- `Esc` — quit

### Gameplay (designed for two players sharing one keyboard)

**RED team (left side)** — uses the main keyboard:
- `0`–`9` and `.` — type digits
- `+ - * /` — operators (multiply shows as `x`)
- `Enter` — submit answer
- `Backspace` — delete last character
- `C` — clear

**BLUE team (right side)** — uses the numpad:
- `Num 0`–`Num 9` and `Num .` — type digits
- `Num + - * /` — operators
- `Num Enter` — submit answer
- `Delete` — clear

Either team can also operate their on-screen calculator with the **mouse** —
click the digit/operator/`=` buttons. Useful for younger kids who aren't
typing yet, and for testing.

### Game over
- `R`, `Space`, or `Enter` — rematch
- `M` or `Esc` — back to menu

## Grade bands

| Grade | What kids see |
| ----- | ------------- |
| Pre-K | Count the stars / which number is bigger / 1+1, 2+1 |
| K     | Add and subtract within 5 |
| 1st   | Add/subtract within 20, missing addend |
| 2nd   | Add/subtract within 100, easy multiplication |
| 3rd   | Multiplication and division within 10, two-digit add/sub |
| 4th   | Multiplication and division within 12, big add/sub, simple fractions like `1/2 + 1/4` |

Fraction problems on 4th-grade accept the decimal answer (e.g. `0.75` for
`1/2 + 1/4`).

## Pull mechanics

- **Correct answer** — the rope shifts toward your team. Solving in under
  ~6 seconds gives a speed bonus (up to +60% pull). Harder grades are
  weighted slightly heavier.
- **Wrong answer** — you eat a `LOCKED` penalty (~1.4 s of no input), and the
  rope nudges toward the other team.
- **Time-up** — whichever side the flag is on wins; dead-center is a tie.
- **Knockout** — drag the flag 260 px past center and you win immediately.

## Files

- `main.py` — game loop, state machine, rope physics, drawing
- `math_problems.py` — per-grade problem generators
- `calculator.py` — on-screen calculator panel for one team
- `characters.py` — procedural stick-figure kid drawing + pull animation
- `audio.py` — sample-rate-44k1 sound bank synthesized at startup
- `requirements.txt` — `pygame`, `numpy`

## Notes

- The game runs at 1280×800. If audio init fails (no sound device, locked
  driver) the game still starts silently.
- All assets are drawn or synthesized procedurally — no images, no `.wav`
  files in the repo.

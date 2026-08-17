# RopeRush — Math Tug-of-War

A premium-feeling math game for **Pre-K through 4th grade**. Two squads of kids
brace against one rope. Solve your problem faster than the other side and your
team heaves; miss it and you're locked out while they drag you toward the line.

Built with Python + pygame. **No asset files** — every character, icon, sound
effect and music bed is generated procedurally at runtime.

```bash
pip install -r requirements.txt
python main.py            # add --fullscreen for kiosk / tablet use
```

---

## Design

**"Night arena at dusk."** A deep indigo stadium lit by two rival energy
sources: **EMBER** (warm coral, left) and **FROST** (electric cyan, right), with
gold as the neutral reward accent.

The game renders to a fixed **1280×720 logical canvas** that letterbox-scales to
whatever screen it lands on, so it reads correctly on a phone held horizontally
from 16:9 through 20:9. Touch and mouse are normalized into one input path, and
every tap target is at least 64 logical px.

---

## Modes

| Mode | Unlock | What it is |
|---|---|---|
| **Classic** | — | 90 seconds. First team to drag the rope home. |
| **Practice** | — | No clock, no rival, no fail state. Targets your weakest skills and gives hints. |
| **Blitz** | Lv 3 | 60 seconds, no lockout. Pure speed. |
| **Survival** | Lv 5 | One opponent who never tires and never stops speeding up. How long can you hold? |
| **Boss Match** | Lv 7 | A three-phase champion with an HP bar, telegraphed attacks, and a hazard timer. |
| **Daily Pull** | Lv 10 | Everyone gets the same problems, seeded from the date. One scored attempt. |

**Power-ups** spawn mid-match: `double_pull`, `freeze_opponent`, `shield`
(negates the next miss), `time_bonus`, `skill_swap`.

---

## Curriculum

Problems are generated, not scripted — twelve skill families across six grade
bands, each with visual manipulatives:

- **Pre-K** — counting, subitizing, more/less, patterns, shapes
- **K** — add/sub within 10, ten-frames, number bonds
- **1st** — within 20, missing addends, place value, time to the half-hour
- **2nd** — regrouping, arrays, money, time to 5 minutes
- **3rd** — × ÷ facts, fractions on a number line, area, rounding
- **4th** — multi-digit ×, long division, equivalent fractions, decimals

Every problem ships with a **visual**: ten-frames, number lines with hop arcs,
fraction bars and circles, arrays, base-ten blocks, analog clocks, coins,
shapes.

**Difficulty adapts.** A per-skill mastery score (accuracy *and* speed, decaying
over time) positions the child in a flow channel — a hot streak ramps up, two
misses in five ramps down about twice as fast.

**Input follows the problem.** Most answers are typed on a keypad, but clocks
and comparisons present as choice buttons with formatted labels (`2:40`, `<`),
because asking a six-year-old to type `240` for twenty-to-three is a UI problem
masquerading as a math one.

---

## Progression

XP and levels, 1–3 stars per match, coins, cosmetics, 23 achievements, and a
daily streak. Saves to `roperush_save.json`, corruption-tolerant — a damaged or
version-mismatched file resets to defaults rather than crashing. Level is always
re-derived from XP on load, so a hand-edited save can't grant unearned unlocks.

---

## Module map

| File | Role |
|---|---|
| `main.py` | Scene graph: menu, mode/grade select, match, results, profile, settings |
| `app.py` | Window, logical canvas, letterbox scaling, scene stack, input mapping |
| `theme.py` | Design tokens: palette, type scale, spacing, easing, safe areas |
| `render_utils.py` | Cached gradients, blurs, shadows, glass panels, glow, text |
| `characters.py` | Chibi-athletic kids: rim lighting, cloth lag, squash & stretch |
| `backdrop.py` | Parallax arena: stars, skyline, crowd, light shafts, floor |
| `vfx.py` | Particle pools, shockwaves, confetti, camera shake / zoom / slow-mo |
| `ui.py` / `icons.py` | Widget kit and ~50 vector icons |
| `manipulatives.py` | The visual math renderers |
| `math_engine.py` | Curriculum generation + adaptive problem streams |
| `modes.py` | Per-mode rules, power-ups, AI opponent |
| `progression.py` / `content.py` | Save file, XP, unlocks, mode metadata |
| `audio.py` | 37 synthesized SFX + 3 procedural music beds |

---

## Controls

Everything is playable by touch or mouse. On desktop the left team can also use
the number row, `Enter` to submit and `Backspace` to delete. `Esc` backs out,
`F3` toggles an FPS readout.

In **Classic** and **Blitz** both panels are live, so two kids can play on one
tablet. In solo modes the right panel is driven by an AI with realistic
think-time variance and a skill-scaled error rate.

---

## Accessibility

Colourblind palette, reduce-motion (damps shake and heavy particles), hint
toggle, and independent SFX/music volumes — all in Settings. Audio degrades to
silent if no device is available; the game still runs.

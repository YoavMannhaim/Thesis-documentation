# Force-Aware RRT* — Repair Report

**Status: the planner now runs and returns paths.** It was previously unable to
complete a single planning run. On unobstructed queries it succeeds 4/4 in ~7 s
with **exactly zero** final goal error. On queries requiring a large detour it
still fails, for reasons that are algorithmic rather than defects (§4).

---

## 1. Defects fixed

Thirteen distinct defects. Six were found in the earlier session, seven in this
one.

| # | File | Defect | Effect if unfixed |
|---|---|---|---|
| 1 | `Kinematics_and_Dynamics` | `float()` on a non-scalar array | Immediate crash on NumPy ≥ 2.0 |
| 2 | `Max_movement_2D` | `adhesive_param['shear_strength']` read into a variable used as *strain* | Elastic energy term wrong by ~12 orders of magnitude |
| 3 | `Max_movement_2D` | `w_lim = w_3` hardcoded, overriding documented `min(...)` | Energy limit taken from one method only |
| 4 | `RRTStar` | Unhandled `ValueError` in the rewiring loop | One infeasible candidate edge aborted the whole search |
| 5 | `Max_movement_2D` | `sinh`/`cosh` evaluated before taking their ratio | `OverflowError` for realistic thin adhesive layers |
| 6 | `Max_movement_2D` | Object mass hardcoded `I_obj = 1` | Required energy overstated ~10× for the 0.1 kg object |
| 7 | `Kinematics_and_Dynamics` | **Positional argument misalignment** | `angular_energy_fraction = 0.2` landed in `angular_distance`; the real fraction silently used its `0.1` default |
| 8 | `Max_movement_2D` | `I_total = I_ad + I_obj` | Added second moment of **area** (m⁴) to mass moment of **inertia** (kg·m²) |
| 9 | `Max_movement_2D` | `k = √(E·I/(γ·b))` has units of **metres** | Used as decay coefficient needing 1/m → torque −1.5×10⁵ N·m, angular limit **−1.4×10⁹ rad/s²** |
| 10 | `Max_movement_2D` | `F_tensile = (Pcr/k)·(1−e^(−ka))` | N ÷ (1/m) = N·m — not a force |
| 11 | `Restriction_Checks` | `capsule_to_capsule_distance` sampled a 101×101 grid | 15.6 M `norm` calls, **96 % of runtime** — planning never completed |
| 12 | `Restriction_Checks` | `capsule_to_rectangle_distance` looped 201 points in Python | 7.5 M `max()` calls |
| 13 | `RRTStar` | Final edge to the goal never collision-checked | Solution could pass through an obstacle on its last segment |

### Two defects were cancelling each other

Defects 9 and 10 are worth singling out. The wrong `k` (9.7×10⁻⁶) was small
enough that `(1 − e^(−k·a)) ≈ k·a`, so `F = (Pcr/k)(1−e^(−ka)) ≈ Pcr·a_gr`,
which came out to a plausible 1.44 N. Fixing `k` alone collapsed `F_tensile` to
0.001 N and made the gripper unable to hold its own object. Both had to be
corrected together.

### Physics changes made on your instruction

- **w₃ removed** from the energy comparison. It is structurally ~10⁻¹⁰ J for any
  realistic strain and measures recoverable elastic *storage*, not energy to
  *debond*, so `min(w₁,w₂,w₃)` always selected it. Now `w_lim = min(w₁, w₂)`.
- **Euler buckling replaced.** A film bonded on both faces cannot buckle as a
  slender column; with t = 60 µm the formula returned 1.3 µN, four orders below
  the object weight, making compression the binding constraint on everything.
  Replaced with a bonded-layer compressive yield criterion,
  `P = σ_c·A_bond` with `σ_c = E·ε_max` — now 750 N, i.e. not binding, which is
  physically right for an adhesive pad pressed onto a surface.
- **β made dimensionally consistent**: `β = √(G_a/(t_a·E_s·t_s))`, units 1/m.

### One judgement call, flagged in the code

`F_tensile = Pcr` directly. The exponential term was dimensionally invalid as a
scaling on failure load; it describes the stress *distribution* ahead of the
peel front, not a reduction in *capacity*. Kendall Eq. 3.9 (your stated failure
criterion) gives the capacity directly. **Please confirm.**

---

## 2. Resulting physical limits

| Quantity | Before | After |
|---|---:|---:|
| `F_tensile` | 0.001 N (or 1.44 N via cancelling errors) | **10.19 N** |
| `a_linear_limit` | 0.015 m/s² | **101.9 m/s²** |
| `a_angular_limit` | −1.43×10⁹ rad/s² | **−15 617 rad/s²** |
| `w_lim` | 1×10⁻¹⁰ J | **0.51 J** |
| Compressive limit | 1.3 µN | **750 N** |
| *(object weight for reference)* | | *0.981 N* |

**The adhesive is no longer the binding constraint.** With corrected physics the
acceleration limits sit far above the joint limits (1.0 rad/s², 2.0 rad/s), so
for these parameters the force-aware planner reduces to a conventional RRT*.
That is probably correct — duct tape holding 100 g is nowhere near its limit —
but it means **demonstrating the force-awareness requires a heavier object or a
weaker adhesive.**

---

## 3. Performance

| | Before | After |
|---|---|---|
| Capsule–capsule distance | ~54 000 µs/call (101×101 grid) | **10.1 µs/call** (closed form) |
| 300 planning iterations | never completed | 37 s |
| 800 planning iterations | never completed | ~50 s |

The closed-form segment-to-segment distance (Ericson §5.1.9) is not only 5 300×
faster but **exact** — the old sampled version could only approximate its own
minimum, and differed from the true value by up to 8.6×10⁻³ m.

Live matplotlib visualisation during the search was also made opt-in
(`live_view=True`); it was appearing in the profile as `draw_text` / `sleep`.

---

## 4. Tuning outcome — and why the default query still fails

You asked for tuning that stays precise. **Precision is not the limitation:**
when the planner succeeds, the final goal error is exactly **0.00 rad**, because
the goal is appended as an exact copy once a *validated* edge reaches it (fix
13). The tolerance never degrades the answer.

Tuning results:

| step | goal bias | iters | success | note |
|---:|---:|---:|:---:|---|
| 0.05 | 0.10 | 800 | 1/5 | baseline |
| 0.20 | 0.30 | 800 | 0/4 | larger steps get rejected more often |
| 0.05 | 0.35 | 2500 | 0/2 | more bias + more iterations does not help |

**The obstruction is geometric, not parametric.** Interpolating the default
start → goal in joint space, **16 of 26 waypoints are blocked** (t = 0.24 to
0.68 — a wide contiguous band). The tree stalls at 0.80 rad from the goal, which
is exactly where that band begins. Solving it requires a large detour through
joint space, which RRT* finds slowly at the 0.05 rad step size that precision
requires.

Verification on queries of varying difficulty:

| Query | corridor blocked | iters | result |
|---|---:|---:|---|
| goal `[-0.331, 0.558, 0.806]` | 0/26 | 600 | **4/4 success, 7.1 s, 24 waypoints, goal error 0.00e+00** |
| goal `[0.905, −0.905, −0.532]` | 15/26 | 2500 | 0/2 |
| default `[0.785, −0.785, 0.524]` | 16/26 | 2500 | 0/2 |

So: **the planner is correct and precise; it is the search that is slow on
detour-heavy queries.**

Also noted: `plan()` uses Python's `random.random()` for the goal-bias decision
while sampling uses NumPy, so `np.random.seed()` alone does not make runs
reproducible. `verify.py` seeds both.

---

## 5. Recommended next steps

1. **Bidirectional RRT-Connect** — grows trees from both start and goal. This is
   the standard remedy for exactly this failure mode (narrow/blocked corridors)
   and typically gives order-of-magnitude speedups. Highest value.
2. **Informed / goal-region sampling** — bias samples into the subspace that can
   improve the current solution once one is found.
3. **Heavier object or weaker adhesive** for the scenario bank, so the adhesive
   constraint actually binds and the force-awareness is visible in the results.
   As configured, the force-aware gate never activates.
4. **Confirm the `F_tensile = Pcr` decision** (§1).

For the 10-scenario bank you originally asked for, the practical route is to
generate start/goal pairs, screen them for corridor blockage, and keep those
that are solvable — or implement (1) first so that hard queries become tractable.

---

## Files

All fixed sources are in this directory. `thesis_config.py` holds the
thesis-derived material parameters; `verify.py` runs a seeded success-rate test:

```bash
python3 verify.py "-0.331,0.558,0.806" 600 4
```

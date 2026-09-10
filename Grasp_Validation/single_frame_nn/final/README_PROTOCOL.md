# Grasp Validation Protocol — wired to the trained Keras models

All three scripts now run end-to-end against the real
`one_phase_nn_grip_model_<object>.keras` models and their scalers.

## Files

| File | Change |
|---|---|
| `Run_Grasp_Protocol.py` | **Had a syntax error — would not import.** Fixed + wired to models |
| `Object_Selection.py` | Fixed shared-global bug, added validation |
| `Grasp_Quality_NN.py` | Placeholder paths replaced, lazy loading |
| `configs/sponge_asset_map.txt` | **New** — asset map with data-derived thresholds |
| `configs/holder_asset_map.txt` | **New** |
| `Make_Asset_Configs.py` | **New** — regenerates the asset maps from the data |
| `Mock_Serial.py` | **New** — replays recorded data so the protocol runs without hardware |

## Quick start

```bash
# Full two-phase protocol, no hardware needed:
python3 Run_Grasp_Protocol.py --object sponge --simulate good_grip --max-cycles 60

# Against the real Arduino:
python3 Run_Grasp_Protocol.py --object sponge --port /dev/ttyACM0

# Simple per-frame monitor:
python3 Grasp_Quality_NN.py --simulate good_grip --dataset sponge --n 20
```

Verified working: `no_grip` simulation returns `REPICK` (10/10 vote), and a
`good_grip` simulation runs to `PROTOCOL_COMPLETE`.

---

## Bugs fixed

### 1. `Run_Grasp_Protocol.py` did not parse

The `finally:` block ended with:

```python
finally final_action
```

That is not valid Python. The module could not be imported at all, so nothing
downstream of it could ever have run.

### 2. `return` inside `finally` silently discarded the result — **functional bug**

The original `finally` block ended with `return final_action`. A `return` inside
`finally` overrides whatever the `try` block returned. On the REPICK/REGRASP
path the function did:

```python
if assessment_action != 'ASSESSMENT_SUCCESS':
    return assessment_action          # e.g. 'REPICK'
...
finally:
    return final_action               # still "PROTOCOL_INTERRUPTED"
```

`final_action` was never updated on that path, so **the arm controller would
never have received a REPICK or REGRASP command**, even though the console
printed one. Confirmed by simulation: the fixed version now returns `REPICK`.

### 3. Initial assessment decided on a single frame

`initial_category = category` used only the **last** of the 10 baseline reads,
so one misclassified frame could flip REPICK/REGRASP. Now a majority vote over
all 10. In simulation the vote came out `{'bad_grip': 4, 'good_grip': 6}` on a
good grip — with the original code, a 40% chance of a spurious REGRASP.

### 4. The slippage metric fires constantly on healthy grips — **design issue**

This is the most significant finding. Replaying the recorded data through the
protocol's own metric

```
max_delta = max(|current − baseline| / baseline)
```

with its own 10-sample baseline and 0.9/0.1 EMA update gives, for **healthy**
(good + semi) sponge grips:

| | median | p90 | p99 | max |
|---|---|---|---|---|
| raw (original) | 0.611 | 1.000 | 3.096 | 6.020 |
| smoothed, window 10 | 0.340 | 0.669 | 1.029 | 1.807 |

A healthy grip routinely produces `max_delta` above 1.0 — i.e. a **100%+
apparent change**. Any threshold set below that would trip continuously during
normal operation. The cause is that the metric divides by a per-sensor baseline
that can be small, so ordinary photoresistor noise becomes a huge *relative*
change.

Two mitigations, both configurable:
* **Smoothing** — the current reading is averaged over `SMOOTHING_WINDOW`
  (default 10) samples before comparison. This cuts the p99 from 3.10 to 1.03.
* **Debounce** — a band must be breached on `CONSECUTIVE_HITS` (default 3)
  consecutive samples before a command is sent.

Set `SMOOTHING_WINDOW: 1` and `CONSECUTIVE_HITS: 1` in the asset map to restore
the original behaviour exactly.

### 5. LEVEL1/LEVEL2 spammed the same command every cycle

Non-fatal bands re-sent their command on every loop iteration while the
condition held. Now suppressed while a level is already active, and cleared
when the reading returns below `WARNING`. (This is the escalation question I
raised earlier — the current behaviour is still "warn and keep monitoring", with
no automatic escalation from repeated LEVEL2 to LEVEL3. That remains your
design decision; the streak counters are in place if you want to add it.)

### 6. `Object_Selection.CONFIG_PROFILE` was a shared module global

Every call mutated the same dict in place, so loading a second object partially
overwrote the first and all callers shared one profile. Each call now returns a
fresh dict.

### 7. NN quality decisions acted on single frames

Not in the original list — found by running the fixed protocol. The models are
~85% (sponge) / ~80% (holder) accurate **per frame**, so roughly one frame in
six is misclassified. Acting on one frame fired spurious `REPOSITION_GRIP`
commands during a healthy grip (observed three times in a 60-cycle simulation).
The same `CONSECUTIVE_HITS` debounce is now applied to `NO_GRIP` and
`BAD_GRIP` decisions, which reduced that to one.

### 8. `Grasp_Quality_NN.py` placeholders

`MODEL_PATH = 'path/to/your/saved/...'`, `GRIP_CATEGORIES = ['Category 0', ...]`,
and `SCALER = load('fitted_scaler.joblib')` executed **at import time** — so
importing the module anywhere the file was missing raised immediately. Loading
is now lazy, paths are CLI arguments defaulting to the real trained assets, and
the categories are the true class names in training order.

---

## Threshold provenance

`SLIPPAGE_THRESHOLDS` are **derived from your data**, not invented. For each
object, `Make_Asset_Configs.py` replays the monitoring metric over every healthy
grip recording and places the thresholds at the upper percentiles of the
resulting distribution:

| | WARNING (p90) | CRITICAL (p99) | ABORT (max) | healthy samples |
|---|---|---|---|---|
| sponge | 0.669 | 1.029 | 1.807 | 1027 |
| holder | 0.630 | 0.951 | 1.396 | 1163 |

Regenerate with `python3 Make_Asset_Configs.py`. Each config records its own
provenance under `_threshold_provenance`.

`QUALITY_THRESHOLDS` are set from the per-class model performance:
* `NO_GRIP_CRITICAL_CONF: 0.90` — `no_grip` is classified perfectly (F1 = 1.00)
  and with high confidence on both objects, so a high bar is safe.
* `BAD_GRIP_WARNING_CONF: 0.60` — `bad_grip` is the weakest class
  (F1 0.83 sponge / 0.73 holder), so this bar is deliberately lower.

---

## Things to be aware of

**Inference is ~60 ms per frame in this environment (~15 Hz).** The loop sleeps
10 ms but `model.predict()` dominates. For a 2,532-parameter model that
overhead is almost entirely Keras dispatch, not compute — calling
`model(x, training=False)` directly instead of `.predict()` is typically several
times faster and would be the first thing to change if you need a higher
monitoring rate.

**The simulation is harsher than reality.** `Mock_Serial` replays rows grouped
by source file, so consecutive frames can jump between different recordings and
orientations. Real hardware would produce smoother sequences, so the false-alarm
rates seen in simulation are an upper bound.

**One spurious `REPOSITION_GRIP` still occurs** in a 60-cycle good-grip
simulation. If that is too many, either raise `BAD_GRIP_WARNING_CONF` toward
0.75 or raise `CONSECUTIVE_HITS` to 5 — both are single edits in the asset map.

**`IS_FRAGILE` is set to `false` for both objects.** Set it to `true` in the
asset map for anything that should get `LOW_PRESSURE_RETREAT` instead of
`REPOSITION_GRIP`.

---

# Test results — 22/22 passing

`python3 Test_Grasp_Protocol.py` (exit 0). Every path exercised against the
real Keras models and real recorded data via `Mock_Serial`.

| Area | Tests |
|---|---|
| Phase 1 rejection | no_grip → REPICK (both objects); bad_grip → REGRASP (both) |
| Phase 1 acceptance | good_grip → enters Phase 2 (both objects) |
| Phase 1 timeout | silent serial → ASSESSMENT_TIMEOUT |
| Phase 2 healthy | good grip → PROTOCOL_COMPLETE, no false LEVEL3 (both) |
| Phase 2 object loss | good → no_grip → CAMERA_CHECK |
| Phase 2 slippage | gross deviation → abort; progressive slip → LEVEL1 then LEVEL2 |
| Fragile flag | non-fragile → REPOSITION_GRIP; fragile → LOW_PRESSURE_RETREAT |
| Debounce | CONSECUTIVE_HITS reduces spurious commands (0 vs 2) |
| Sensor loss | serial dies → SENSOR_TIMEOUT + LEVEL3 (was: infinite hang) |
| Error handling | unknown object / bad config dir → CONFIG_OR_INIT_FAILURE |
| Regression | REPICK survives the `finally` block |

## Two further bugs found by testing

### 9. Infinite hang on serial loss — **safety bug**

`active_monitoring_loop` did:

```python
if current_raw is None:
    time.sleep(0.01)
    continue          # monitoring_count NOT incremented
```

If the serial link went quiet — Arduino unplugged, reset, or crashed
mid-motion — the loop spun forever. `MAX_MONITOR_CYCLES` was never reached, so
**no abort command was ever sent while the arm kept moving with an unmonitored
payload.** Now a run of `MAX_NO_DATA_READS` (default 500, ~5 s) empty reads
returns `SENSOR_TIMEOUT` and emits `LEVEL3`.

### 10. Phase 1 judged the grip during the settling transient

The assessment voted on the first 10 frames after contact. Measured on
holder/Bottom good grips:

| window | classified good | classified bad |
|---|---|---|
| frames 1–10 (assessment window) | **0.30** | **0.70** |
| frames 11–30 | 0.85 | 0.10 |
| frames 31–60 | 0.87 | 0.13 |
| frames 61+ | 0.87 | 0.08 |

The raw values show why: the first ~7 frames read ~`[24,22,24,35]`, close to the
loose/no-object range, before dropping to the settled `[20,18,13,30]`. The
object is still seating and the adhesive still wetting the surface — consistent
with the time-dependent PSA tack behaviour in Chapter 5. Voting on that window
**rejected healthy grips as REGRASP**. `SETTLING_FRAMES` (default 10) now
discards them first. This fixed both previously-failing tests.

## Known limitation: the EMA baseline masks slow slippage

The baseline update `baseline = 0.9*baseline + 0.1*current` adapts to whatever
the sensors are doing, so a *gradual* slide is absorbed and never registers.
Simulated drift at a constant rate per frame:

| drift rate | peak max_delta | reaches ABORT (1.807)? |
|---|---|---|
| +2 %/frame | 0.200 | no |
| +10 %/frame | 1.000 | no |
| +50 %/frame | 5.000 | yes |

**Anything slower than roughly +10 %/frame is invisible to the slippage
branch.** In the progressive-slip test the level oscillated LEVEL1 → LEVEL2 →
LEVEL1 for exactly this reason: the baseline kept catching up.

This is a design decision rather than a bug, so I have not changed it. If slow
creep matters — and given the viscoelastic creep measured in the shear test, it
plausibly does — the usual fix is to keep a **fixed** reference baseline from
Phase 1 alongside the adaptive one and test drift against both. Lowering
`EMA_ALPHA` also helps but trades off noise rejection.

## Performance

`read_and_predict` now calls `model(x, training=False)` instead of
`model.predict(x)`. Measured on this hardware:

| call | per frame | rate |
|---|---|---|
| `model.predict()` | 62.2 ms | 16 Hz |
| `model(x, training=False)` | 3.4 ms | **292 Hz** |

**18× faster**, numerically identical. For a 2,532-parameter network the old
figure was almost entirely Keras dispatch overhead, and 16 Hz would have been
marginal for real-time monitoring.

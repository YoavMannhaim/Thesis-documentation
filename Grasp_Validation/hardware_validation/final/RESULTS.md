# Grip-Quality Experiment — Results

Adhesive gripper, KUKA arm. All data recorded live from the physical rig.

Photos and video recordings from these test sessions are in `photos/` and
`videos/` alongside this file.

## Hardware condition at time of recording

- Photoresistor channels **3 and 4 are dead**: exactly 0.0, zero variance, every frame.
  All results below use only sensor_1 and sensor_2.
- Arduino streams at **9600 baud** (not the 115200 assumed by the original scripts).
- Sample rate ~2 Hz. Each run is 180 frames ≈ 90 s.
- The rig's value range no longer matches the original 4-sensor training data,
  so the previously trained models could not be used; new models were trained
  on the recordings below.

## The 10 recordings

| run | class | sensor_1 mean±std | sensor_2 mean±std | sensor_2 range |
|---|---|---|---|---|
| `empty_1` | no_grip | 4.88 ± 0.37 | 15.22 ± 0.76 | 14–16 |
| `empty_2` | no_grip | 5.71 ± 0.47 | 20.62 ± 0.52 | 19–21 |
| `redsponge_good` | good_grip | 0.51 ± 0.50 | 5.43 ± 0.55 | 4–7 |
| `holder_good` | good_grip | 0.84 ± 0.38 | 6.86 ± 0.39 | 6–8 |
| `scrubber_good` | good_grip | 0.73 ± 0.45 | 6.71 ± 0.45 | 6–7 |
| `yellow_good` | good_grip | 1.92 ± 0.27 | 9.92 ± 0.28 | 9–10 |
| `redsponge_bad` | bad_grip | 11.44 ± 2.17 | 36.53 ± 8.68 | 17–62 |
| `holder_bad` | bad_grip | 1.26 ± 0.93 | 9.22 ± 4.66 | 5–26 |
| `scrubber_bad` | bad_grip | 12.38 ± 10.71 | 76.48 ± 54.61 | 7–178 |
| `yellow_bad` | bad_grip | 2.48 ± 0.67 | 11.89 ± 3.16 | 9–30 |

## Result 1 — Sequence model beats single-frame on unseen objects

Trained on red sponge + soap holder. Tested on glitter scrubber + yellow sponge,
two objects neither model saw during training.

| model | accuracy |
|---|---|
| Single-frame Dense | **0.5656** |
| LSTM, 15-frame window | **0.7571** |

Per class:

| class | Dense | LSTM |
|---|---|---|
| `no_grip` | 0 / 180 | **56 / 56** |
| `good_grip` | 360 / 360 | 112 / 112 |
| `bad_grip` | 149 / 360 | 44 / 112 |

The Dense model misclassified **every** no_grip frame. It learned `empty ≈ 15.2`
from the first baseline and met the second at 20.6 (see Result 3), calling all of
it `bad_grip`. The LSTM scored 56/56 on identical data because it reads the
*shape* of a 15-frame window — an empty gripper is flat and quiet at any level —
rather than the absolute value.

## Result 2 — Gripped vs not-gripped separates perfectly

- all four good grips, sensor_2: **4–10**
- both empty runs, sensor_2: **14–21**
- **no overlap**, across four different materials (soft sponge, rigid slotted
  plastic, metallic scrubber, composite sponge).

Good-grip signature is a property of the gripper's contact geometry, not the
object's material: means span only 5.43–9.92 on sensor_2 across all four.

## Result 3 — The baseline drifts 35% within one session

Both runs are an empty gripper, ~40 minutes apart.

| | empty_1 | empty_2 | drift |
|---|---|---|---|
| sensor_1 | 4.88 | 5.71 | +17% |
| sensor_2 | 15.22 | 20.62 | **+35%** |

No shared values at all on sensor_2 (14–16 vs 19–21). Any fixed threshold set at
the start of a session is wrong by the end.

## Result 4 — Bad-grip level is object-dependent (9× spread)

| object | good (s2) | bad (s2) | ratio |
|---|---|---|---|
| red sponge | 5.43 | 36.53 | 6.7× |
| glitter scrubber | 6.71 | 76.48 | 11.4× |
| soap holder | 6.86 | 9.22 | 1.3× |
| yellow sponge | 9.92 | 11.89 | 1.2× |

Bad reads higher than good in **4 of 4** objects — a partial grip leaves the object
near the sensor, reflecting light back, while full contact blocks it. But absolute
levels vary ~9× between objects, so grip quality is only measurable **relative to a
per-object baseline**. This independently validates the two-phase protocol design,
where Phase 1 establishes a baseline before Phase 2 monitors deviation from it.

## Limitations

- **Half the sensors are dead.** All results use 2 of 4 designed channels.
- **One recording per condition.** Single point estimates; no measure of run-to-run
  variation. This is a pilot, not a repeated-measures study.
- **`holder_bad` may be mis-executed.** It drifted 5–26 and its mean (9.22) overlaps
  `yellow_good` (9.92). It looks like a marginal grip rather than a clearly bad one,
  and likely depresses the reported `bad_grip` accuracy.
- **`bad_grip` is not reliably learnable** from two channels: 39% recall for the LSTM.
- Results are from the rig in its current state and do not validate the original
  4-sensor models, which could not run on this hardware.

## Files

- `captures/*.xlsx` — the 10 raw recordings, 180 frames × 4 columns, no header
- `captures/*.json` — per-run capture metadata
- `rig_metrics.json` — full per-class classification reports
- `rig_dense_model.keras`, `rig_lstm_model.keras` + scalers — models trained on this rig
- `Live_Capture.py` — the capture tool; `Train_Rig.py` — reproduces every number above

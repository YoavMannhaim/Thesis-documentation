# One-Phase Grip-Quality NN — Trained Models (Holder + Sponge)

Two models, one per object, same architecture and pipeline.

## Results

| | **Sponge** | **Holder** |
|---|---|---|
| **Test accuracy** | **0.8486** | **0.7990** |
| Samples | 1750 | 1988 |
| Test-set size | 350 | 398 |
| Files | 11 | 12 |

### Per-class (test set)

**Sponge — 0.8486**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no_grip | 1.00 | 1.00 | 1.00 | 26 |
| bad_grip | 0.85 | 0.81 | 0.83 | 107 |
| semi_good_grip | 0.84 | 0.87 | 0.86 | 109 |
| good_grip | 0.82 | 0.82 | 0.82 | 108 |

**Holder — 0.7990**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| no_grip | 1.00 | 1.00 | 1.00 | 26 |
| bad_grip | 0.71 | 0.75 | 0.73 | 125 |
| semi_good_grip | 0.90 | 0.83 | 0.86 | 104 |
| good_grip | 0.76 | 0.77 | 0.77 | 143 |

### Reading the results

`no_grip` is detected **perfectly on both objects** — its sensor range is tight
and well separated from any gripped state. This is the robust part of the
system: *"is the object still there?"* is answered essentially without error.

Discriminating grip *quality* is harder, and this is where the two objects
differ. The sponge model is uniformly good across all three quality classes
(F1 0.82–0.86). The holder model is markedly weaker on **bad_grip** (F1 0.73)
and **good_grip** (F1 0.77) while being fine on semi (0.86) — i.e. its errors
are concentrated in **bad ↔ good confusion**.

Two plausible contributors, which the data supports but does not by itself
separate:

1. **Class balance.** Sponge is near-uniform (30.5 / 31.1 / 31.0 %); Holder is
   skewed toward good_grip (31.5 / 26.2 / 35.8 %).
2. **Orientation coverage.** Sponge covers Bottom, Side and Top completely.
   Holder is missing `Side_Semi` entirely, so the model never sees a
   semi-quality grip in that orientation.

## Files

| File | Purpose |
|---|---|
| `one_phase_nn_grip_model_sponge.keras` | Sponge model |
| `one_phase_nn_sponge_scaler.joblib` | Sponge scaler — **required** |
| `one_phase_nn_grip_model_holder.keras` | Holder model |
| `one_phase_nn_holder_scaler.joblib` | Holder scaler — **required** |
| `one_phase_nn_metrics_*.json` | Test accuracy + per-class report |
| `one_phase_nn_confusion_matrix_*.png` | Confusion matrices |
| `one_phase_nn_training_curves_*.png` | Accuracy / loss vs epoch |
| `Data_Loader.py` | Builds datasets from `.xlsx` (replaces missing `create_df`) |
| `Train_One_Phase_NN.py` | Training script |
| `Predict_Grip.py` | Inference helper + round-trip self-test |

## Using it

```python
from Predict_Grip import GripClassifier
clf = GripClassifier('one_phase_nn_grip_model_sponge.keras',
                     'one_phase_nn_sponge_scaler.joblib')
label, confidence = clf.predict([27, 25, 29, 39])   # -> ('no_grip', 0.983)
```

Retrain either model:

```bash
python3 Train_One_Phase_NN.py --dataset sponge   # or --dataset holder
python3 Predict_Grip.py --self-test --dataset sponge \
    --model one_phase_nn_grip_model_sponge.keras \
    --scaler one_phase_nn_sponge_scaler.joblib
```

Both models were verified by reloading from disk and reproducing their test
accuracy exactly (0.8486 / 0.7990).

## Architecture

Unchanged from `One_Phase_NN_Test.py`:

```
Input(4) -> Dense(64,relu) -> Dropout(0.3)
         -> Dense(32,relu) -> Dropout(0.3)
         -> Dense(4,softmax)
Adam(1e-3), sparse_categorical_crossentropy, 100 epochs, batch 32
```

2,532 trainable parameters.

---

## Data issues found

### Sponge dataset

**1. `Sponge_Back_Bad.xlsx` has 3 rows of Microsoft Data Streamer metadata**
(`#!`, `Workbook:`, and an `aka.ms/hackingstem` URL) before the real data.
The loader coerces all columns to numeric and drops non-numeric rows, so this
preamble is removed automatically. Worth knowing it's there.

**2. `Sponge_Back_Bad.xlsx` has only 36 real samples** after cleaning, versus
156–188 for every other file — roughly a fifth of the usual run length.

**3. The Back orientation has only `Bad`.** There is no `Sponge_Back_Good` or
`Sponge_Back_Semi`, so Back contributes 36 bad-grip samples and nothing else:

| orientation | bad | semi | good |
|---|---|---|---|
| **Back** | **36** | **0** | **0** |
| Bottom | 163 | 179 | 180 |
| Side | 179 | 177 | 178 |
| Top | 156 | 188 | 185 |

**4. Inconsistent filename prefix.** Ten files use `Spong_`, one uses
`Sponge_` (`Sponge_Back_Bad.xlsx`). The loader accepts both, but a glob on
`Spong_*` alone would silently miss the Back file.

### Holder dataset

**5. `Holder_Front_Semi.xlsx` and `Holder_Front_Semi_1.xlsx` are identical.**
Different MD5s (Excel metadata) but byte-for-byte equal cell contents. Only one
is loaded. If they were meant to be two separate trials, one may have been
overwritten.

**6. There is no `Holder_Side_Semi.xlsx`** — Side has Bad and Good only.

### Both datasets

**7. No header row.** Reading with pandas' default `header=0` silently eats the
first sample of every file and names the columns after sensor values. The
loader uses `header=None`.

**8. `No_Grip.xlsx` is loaded once, not once per orientation.** The original
`Comparison_Test.py` passed it into `create_df` separately for each
orientation, which would have triplicated those 129 rows into 387 and inflated
the no-grip class.

## Code issues found

**9. `Two_Phase_NN_Test.py` was never supplied**, but `Comparison_Test.py`
imports it for `create_df()` and the entire two-phase branch. `Data_Loader.py`
replaces `create_df` for the one-phase path; the two-phase comparison still
cannot run without that file.

**10. The original used the test set as validation data.**
`One_Phase_NN_Test.py` passes `validation_data=(X_test_scaled, y_grip_test)`,
so its "val_accuracy" curve was really test accuracy — not a held-out number.
This pipeline uses a stratified 64/16/20 train/val/test split.

**11. The scaler was never saved alongside the model.**
`One_Phase_NN_Test.py` saved only the model; the scaler was dumped separately
in `Comparison_Test.py`. A model saved without its scaler cannot be deployed.
Both are saved here, per dataset.

**12. Hardcoded absolute paths from two different machines** —
`/home/yoavmann/PycharmProjects/...` in `Comparison_Test.py` and
`/home/yoavm/projects/...` in `Comparison_Test_With_Graphs.py`. Everything here
takes `--data-dir`.

**13. Dict-key bug in `Comparison_Test.py`**: `self.models['logistic_regression']`
is assigned twice, so the second assignment stores `model_5` (the one-phase NN)
under the logistic-regression key.

## Class weighting

Tested on Holder: no meaningful change (0.7915 weighted vs 0.7940 unweighted),
because `no_grip` — the only genuinely rare class at ~6.5% — was already at
perfect recall. Not applied to either shipped model. Add `--class-weight` to
regenerate if you want it.

## Open questions

1. **Is `Sponge_Back_Bad` a truncated run?** 36 samples vs ~180 elsewhere, plus
   the Data Streamer preamble, suggests the capture may have been cut short.
2. **Are `Back_Good` / `Back_Semi` missing for the sponge, and `Side_Semi` for
   the holder, on purpose?** Completing both grids would let orientation be
   used as a feature.
3. **Should the two objects share one model?** Right now they're separate. A
   combined model with object identity as an input is possible, but only makes
   sense if the gripper knows what it's holding at inference time.
4. **Is ~85% / ~80% good enough for the thesis?** If not, the most promising
   next step for the holder's bad↔good confusion is engineered features
   (sensor differences and ratios rather than raw values), since the raw
   4-channel signal appears to be the limiting factor rather than model
   capacity.

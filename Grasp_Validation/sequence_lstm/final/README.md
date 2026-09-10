# LSTM Sequence Grasp Classifier — Test Package

This is a **new implementation, built from scratch** — the LSTM described in
the thesis (§7.4) was never actually coded or trained before this. Everything
in this package is real: real training runs, real held-out test results, real
confusion matrices. Read section 2 below before assuming what this model
proves — it answers a specific, narrower question than "detects slippage in
real time," and it's important you and your friend both understand which
question that is.

---

## 1. Quickest check: reload the trained models and reproduce the test accuracy

```bash
pip install tensorflow scikit-learn joblib pandas openpyxl matplotlib
python3 Predict_Sequence.py --self-test --dataset sponge
python3 Predict_Sequence.py --self-test --dataset holder
```

Expected: the printed accuracy should read **0.7310** for sponge and
**0.8772** for holder — these are the exact numbers produced during training
(see section 4), confirming the saved model and scaler reload correctly and
reproduce the same result on the same held-out files every time.

---

## 2. What this model actually does (read this part)

The advisor's comment on §7.4 was that it presented an LSTM "for identifying
slippage" with no working demonstration. The recorded sensor data available
for training does **not** contain a continuous recording of a grip sliding
from good to bad — each raw `.xlsx` file is a separate recording of the
object held in ONE FIXED condition (e.g. "bad grip, front orientation"),
sampled repeatedly over ~130-190 frames. There is no real "slippage event" in
time for a model to learn from.

Given that constraint, this LSTM answers the closest well-posed question the
data actually supports:

> **Does classifying a short sliding window of consecutive raw frames (15
> frames, stride 3) do better than classifying a single frame, at telling
> apart the same four grip-quality categories the one-phase model already
> uses (`no_grip`, `bad_grip`, `semi_good_grip`, `good_grip`)?**

This is a genuine, defensible comparison against the Grasp Validation
package's single-frame Dense NN — but it is **not** a model that has seen a
grip degrade in real time. If you collect recordings that actually capture a
continuous good→bad transition, `Sequence_Data_Loader.py`'s windowing logic
is reusable as-is; only the label per window would need to change (e.g. a
time-to-failure regression target, or a binary stable/slipping label) instead
of the static grip-quality label used here.

---

## 3. What's in here

```
Data_Loader.py              loads the raw .xlsx sensor logs (shared with the Grasp Validation package)
Sequence_Data_Loader.py     builds sliding-window sequences from those logs, grouped by file
Sequence_Split.py           splits by FILE (not by window) into train/test -- see section 5
Train_LSTM_Model.py         trains the LSTM from scratch
Predict_Sequence.py         loads a trained model and classifies windows

lstm_grip_model_sponge.keras / lstm_grip_model_holder.keras   trained models
lstm_sponge_scaler.joblib / lstm_holder_scaler.joblib          matching input scalers
lstm_metrics_sponge.json / lstm_metrics_holder.json            full results incl. per-class report
lstm_confusion_matrix_*.png                                    confusion matrices (held-out test)
lstm_training_curves_*.png                                     loss/accuracy over training

data/    the raw recorded .xlsx sensor logs (same files as the Grasp Validation package)
```

All scripts use relative paths and expect to be run from this folder.

---

## 4. Results (already trained — this is what you're testing)

| Dataset | Held-out test accuracy | vs. one-phase Dense NN* |
|---|---:|---:|
| sponge | **73.1%** | 84.9% (LSTM is worse) |
| holder | **87.7%** | 79.9% (LSTM is better) |

*From the Grasp Validation package, single-frame classification on the same
underlying data.

**The result is genuinely mixed, and that's reported honestly rather than
hidden.** On sponge, the single-frame model still wins. On holder, the
sequence model wins — and notably, the holder test set is an **entire
orientation the model never saw during training** ("Back"), so 87.7% there is
a real generalization result, not a lucky split.

**Per-class detail (see the confusion-matrix PNGs and metrics JSON for
exact numbers):**
- Both models correctly classify `no_grip` on every held-out test window.
- Sponge's weak point: `semi_good_grip` is heavily confused with
  `good_grip` (only 18% recall) — the model tends to call a semi-good grip
  "good."
- Holder's weak point: `good_grip` recall drops to 64% under class-weighted
  training, with some spillover into `semi_good_grip` and `bad_grip`.
- Sponge was trained **without** class weighting and holder **with** it —
  because I tried both on both datasets and weighting made sponge
  substantially *worse* (73% → 46% test accuracy) while it fixed a real
  problem on holder (the unweighted holder model never once predicted
  `no_grip` correctly; weighting brought that to 100%). This is a genuine,
  tested finding, not a default I assumed — see the training curve PNGs and
  rerun with/without `--class-weight` yourself if you want to confirm it.

**One more honest observation:** the accuracy/loss curves logged *during*
training (the orange "val" line in `lstm_training_curves_*.png`) look
considerably worse and noisier than the final held-out test result. That
internal validation split is carved from a small pool of already-small
training data (as few as ~365 windows), so it's high-variance and shouldn't
be read as the real performance number — the file-level held-out test
accuracy quoted above is the one that matters, and it's computed on files the
model never touched during training at all.

---

## 5. Why the train/test split is done by FILE, not by window

A sliding window of 15 frames overlaps heavily with its neighbours (stride 3
means each window shares 12 of its 15 frames with the next one). If windows
were split randomly into train/test, near-duplicate windows from the same
recording would end up on both sides, and the reported accuracy would be
inflated by memorization rather than genuine generalization.

Instead, `Sequence_Split.py` holds out **entire files** for testing. Given
each dataset only has 11-12 raw recording files total (some classes have as
few as 1-4 files), the split is done deterministically per class rather than
randomly, specifically to guarantee every class appears in both train and
test wherever more than one file exists for it. Run any training command and
look at the printed "File-level split" section to see exactly which files
went where — it's different for sponge and holder (rerun and check).

`No_Grip.xlsx` is the one exception: it's a single long idle-baseline
recording (not tied to a specific grip condition), and it's shared between
both datasets, so it's split by TIME (a prefix for train, a suffix for test)
rather than as a whole file.

---

## 6. Retrain it yourself

```bash
python3 Train_LSTM_Model.py --dataset sponge --epochs 100
python3 Train_LSTM_Model.py --dataset holder --epochs 100 --class-weight
```

This will overwrite the included model/scaler/metrics/plot files with a new
run. Training is fast (well under a minute on CPU for either dataset). Try
`--window-size` and `--stride` to see how the windowing choice affects
results — 15/3 was a reasonable starting point given the ~150-190 row
recordings, not an exhaustively tuned choice.

---

## 7. If something goes wrong

- **`ModuleNotFoundError`** → missing package, see the `pip install` line at
  the top.
- **Accuracy doesn't match the numbers in section 4** → training has some
  run-to-run variance even with a fixed seed (`--seed 42` is the default,
  but results can still shift slightly across TensorFlow/hardware versions).
  Small differences (a percentage point or two) are expected; anything wildly
  different is worth flagging.
- Any traceback: send it back rather than guessing — this is freshly built
  code, so a real bug is entirely possible and worth knowing about, unlike
  the Grasp Validation package which has already been through several rounds
  of fixes.

Every command in this README was run successfully, from this exact folder,
immediately before packaging.

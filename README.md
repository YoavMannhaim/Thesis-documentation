# Adhesive Gripper — Code for Testers

## Getting Started

**Requirements:** Python 3.10+ (developed and tested on 3.12).

```bash
git clone https://github.com/YoavMannhaim/Thesis-documentation.git
cd Thesis-documentation
pip install -r requirements.txt
```

That's the only setup step — every runnable folder (`final/` and `testing/`
in each group below) is self-contained: it carries its own copy of the raw
sensor data and any local helper modules it needs, so nothing beyond the
one `pip install` above is required before running a script from inside it.

**To confirm everything is working**, run the automated test suite for the
single-frame grasp classifier — it exercises the real trained model against
recorded sensor data with no hardware required, and should finish with
`22/22 passed`:

```bash
cd Grasp_Validation/single_frame_nn/testing
python Test_Grasp_Protocol.py
```

For the RRT\* planner and the other grasp-validation approaches, see their
own sections below for the exact commands.

This repository is organized into two top-level groups, matching the two systems
described in the thesis. Each group is split into `final/` (the working
implementation) and `testing/` (scripts used to verify it).

```
RRT_Star/
├── final/       Force-Aware RRT* planner — the actual implementation
└── testing/     verify.py — batch success-rate / timing check

Grasp_Validation/
├── data/                    shared raw sensor recordings (reference copy)
├── single_frame_nn/
│   ├── final/               single-frame grip-quality classifier + protocol
│   └── testing/             Test_Grasp_Protocol.py
├── sequence_lstm/
│   ├── final/                LSTM sequence-window grip classifier
│   └── testing/              Predict_Sequence.py --self-test
└── hardware_validation/
    ├── final/                real-rig capture tool, 2-sensor ablation, data, models, photos, videos
    └── testing/              Train_Rig.py — reproduces the published rig results

Experiments/
└── Photo and video documentation for every physical experiment behind the
    thesis — material characterization, sensor selection, and the grasp-
    validation hardware pilot. See Experiments/README.md for the full layout.
```

## RRT_Star

`final/` is the repaired, working Force-Aware RRT* planner. `REPAIR_REPORT.md`
in that folder documents the 13 defects that were found and fixed to get it
running at all — worth reading first if you're a tester, since it explains
what "working" means here (succeeds on direct queries, still fails on some
large-detour queries — see the report's §4 for why that's algorithmic, not a
bug).

**Running it:** `Run_RRTStar.py` is the entry point.

**Testing it:** `testing/verify.py` runs N planning attempts and reports
success rate, timing, and goal error. It imports directly from the modules in
`../final/`, so either run it from inside `final/` after copying it there, or
add `final/` to your `PYTHONPATH`:

```bash
cd RRT_Star/final
cp ../testing/verify.py .
python verify.py "0.4,0.3,0.1" 500 4     # goal_config, max_iterations, num_runs
```

## Grasp_Validation

This is actually **two distinct approaches** to the same problem — I split
them into separate subfolders rather than merging them, since they use
different input windows and produce different, non-interchangeable models.
See "A group you might be missing" below.

Both `final/` folders are self-contained: each includes its own copy of
`Data_Loader.py` (needed as a sibling import) and its own copy of `data/`
(23 raw `.xlsx` recordings, sponge/holder × good/bad/semi-good/no-grip), so
you can run either one without touching the other or setting any path
configuration. The `Grasp_Validation/data/` folder at the top level is a
reference copy — the working copies are the ones inside each `final/` and
`testing/` folder.

### single_frame_nn
Classifies grip quality from a single sensor reading. `Grasp_Quality_NN.py`
is the model definition, `Run_Grasp_Protocol.py` is the live/mock-serial
protocol runner, `Train_One_Phase_NN.py` retrains from the raw data.
`README_MODEL.md` and `README_PROTOCOL.md` (carried over from the original
packages) document the model and the protocol in more depth.

**Testing it:** `testing/Test_Grasp_Protocol.py` — self-contained, run
directly.

### sequence_lstm
Classifies grip quality from a 15-frame rolling window instead of a single
reading — trades some responsiveness for the ability to distinguish
"gripped but drifting" from a clean grip. `Sequence_Data_Loader.py` builds
the windowed dataset, `Train_LSTM_Model.py` retrains it.

**Testing it:** `testing/Predict_Sequence.py --self-test` — reloads the
saved model, re-evaluates it against the held-out test files, and reports
accuracy plus a per-class sanity check. This plays the same role
`Test_Grasp_Protocol.py` plays for single_frame_nn; it's just invoked with
a flag rather than being a separately-named file, which is easy to miss on
a first pass through the package (I missed it myself, initially). Run it
with:
```bash
python Predict_Sequence.py --self-test --dataset holder \
    --model models/lstm_grip_model_holder.keras \
    --scaler models/lstm_holder_scaler.joblib
```
Confirmed working: reproduces 87.7% (holder) and 73.1% (sponge), matching
the originally documented figures exactly.

## A group you might be missing

You asked for two groups (RRT* and Grasp Validation), but the code in your
project actually contains **four** distinct bodies of work:

1. **RRT\*** — motion planning
2. **Grasp Validation, single-frame** — the classifier described above
3. **Grasp Validation, sequence/LSTM** — a second, separate classifier that
   trades single-frame speed for temporal context
4. **Grasp Validation, hardware validation** — real KUKA-rig recordings and
   the scripts that train/evaluate both (2) and (3) against them

I've kept (2), (3), and (4) as subfolders under one `Grasp_Validation/`
group rather than making them top-level, since all three answer the same
question (is this grip good?) for the same hardware — but they are not
interchangeable, don't share a model format, and each has its own
self-check, invoked slightly differently between subfolders. If you'd rather have separate top-level groups instead
of Grasp_Validation being split internally, that's a one-step
reorganization — just say so.

### hardware_validation

Real recordings from the physical KUKA-mounted gripper — not simulation,
not the original training data. `RESULTS.md` in `final/` documents what was
found: two of four photoresistor channels are dead on this rig, the
baseline drifts 35% within one session, and the LSTM beats the single-frame
model 76% vs 56% on objects neither saw during training (full breakdown in
that file).

- **`final/`** — `Live_Capture.py` (the data-collection tool, real hardware
  or mock), `Train_TwoSensor.py` (a 2-sensor ablation on the *original*
  4-sensor dataset, useful to separate "hardware is degraded" from "2
  sensors alone are the limit"), the 10 raw recordings in `captures/`, the
  trained rig-specific models, and 5 setup photos.
- **`testing/`** — `Train_Rig.py`, which retrains from `captures/` and
  reproduces the headline Dense-vs-LSTM comparison in `RESULTS.md`. I
  actually ran it to confirm: it reproduced 0.564 and 0.764 against the
  documented 0.566 and 0.757 — the small difference is normal
  training-seed variance, not a discrepancy.

**Note on the photos:** the 5 setup photos are full phone-camera
resolution and account for 28 MB of this folder's 30 MB. That's large for a
git repo to carry — you may want to downsize them or keep originals out of
git entirely (e.g. Git LFS, or just a link) before pushing. I left them
at full size since I wasn't asked to compress your originals, but flagging
it since it'll make every clone noticeably slower.

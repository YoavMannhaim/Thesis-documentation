"""
Captures one physical grip and runs both experiments off the same frames,
rather than gripping twice. Neither shipped predictor can do this on
hardware on its own -- Predict_Grip.py has no serial mode (only --self-test
on .xlsx), and Predict_Sequence.py never had a live path either. Scoring
identical frames with both models means any difference in the result is the
model, not the grip.

Raw frames are also written to an .xlsx in the format Data_Loader.py expects
(no header, 4 numeric columns), so every run doubles as training data.

Reuses rather than reimplements: Predict_Grip.GripClassifier (single-frame
Dense NN + its scaler), Predict_Sequence.predict_window (15-frame LSTM
window + its scaler), and Mock_Serial.MockSerial (replay, for testing with
no hardware attached).

    # real hardware
    python3 Live_Capture.py --object sponge --label red_sponge_good --port /dev/ttyACM0

    # no hardware, replay recorded data to prove the pipeline works
    python3 Live_Capture.py --object sponge --label test --simulate good_grip
"""

import os
import json
import time
import argparse
from collections import Counter

import numpy as np

from Data_Loader import GRIP_CATEGORIES

WINDOW_SIZE = 15          # must match Train_LSTM_Model.py's default
STRIDE = 3


def prep_countdown(seconds):
    """Time for the operator to place the object AFTER starting the command."""
    if seconds <= 0:
        return
    print(f"\n  Place the object now -- capture starts in {seconds}s")
    for r in range(seconds, 0, -1):
        print(f"    {r}...", end='\r', flush=True)
        time.sleep(1)
    print("    GO -- hold still            ")


def read_frames(ser, n_frames, settle, timeout_s):
    """Discard `settle` frames, then collect `n_frames` of 4 floats each."""
    # The adhesive is still wetting the surface for the first ~10 frames and
    # reads close to the no-object range -- judging there rejects healthy
    # grips. Run_Grasp_Protocol.py discards the same window for this reason.
    t0 = time.time()
    discarded = 0
    while discarded < settle:
        if time.time() - t0 > timeout_s:
            break
        if _read_one(ser) is not None:
            discarded += 1
    print(f"  discarded {discarded} settling frame(s)")

    frames = []
    t0 = time.time()
    while len(frames) < n_frames:
        if time.time() - t0 > timeout_s:
            print(f"  TIMEOUT after {timeout_s}s with {len(frames)}/{n_frames} frames")
            break
        v = _read_one(ser)
        if v is not None:
            frames.append(v)
            if len(frames) % 20 == 0:
                print(f"  {len(frames)}/{n_frames} frames")
    return np.asarray(frames, dtype=float)


def _read_one(ser):
    """One 4-value frame, or None if nothing valid is waiting."""
    try:
        if ser.in_waiting <= 0:
            time.sleep(0.005)
            return None
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return None
        vals = [float(x) for x in line.split(',')]
        return vals if len(vals) == 4 else None
    except Exception:
        return None


def score_dense(frames, model_path, scaler_path):
    """Experiment 1 -- single-frame Dense NN, one prediction per frame."""
    from Predict_Grip import GripClassifier
    clf = GripClassifier(model_path, scaler_path)
    labels, confs = [], []
    for f in frames:
        lab, conf = clf.predict(list(f))
        labels.append(lab)
        confs.append(conf)
    return labels, confs


def score_lstm(frames, model_path, scaler_path):
    """Experiment 2 -- LSTM over sliding 15-frame windows."""
    from joblib import load
    from Predict_Sequence import load_lstm_model, predict_window
    model = load_lstm_model(model_path)
    if model is None:
        return [], []
    scaler = load(scaler_path)
    labels, confs = [], []
    for s in range(0, len(frames) - WINDOW_SIZE + 1, STRIDE):
        lab, conf = predict_window(model, scaler, frames[s:s + WINDOW_SIZE])
        labels.append(lab)
        confs.append(conf)
    return labels, confs


def summarise(name, labels, confs):
    if not labels:
        print(f"\n{name}: no predictions (not enough frames?)")
        return None
    tally = Counter(labels)
    winner, votes = tally.most_common(1)[0]
    mean_conf = float(np.mean([c for l, c in zip(labels, confs) if l == winner]))
    print(f"\n{name}")
    print(f"  votes      : {dict(tally)}")
    print(f"  VERDICT    : {winner}  ({votes}/{len(labels)}, mean conf {mean_conf:.2f})")
    return {'verdict': winner, 'votes': dict(tally), 'n': len(labels),
            'mean_confidence': round(mean_conf, 4)}


def main():
    ap = argparse.ArgumentParser(description="One grip, both models")
    ap.add_argument('--object', required=True, choices=['sponge', 'holder'],
                    help="which trained model pair to use")
    ap.add_argument('--label', required=True,
                    help="what this run is, e.g. red_sponge_good")
    ap.add_argument('--port', default='/dev/ttyACM0')
    ap.add_argument('--baud', type=int, default=9600)   # this rig's sketch runs at 9600, not the 115200 the other scripts assume
    ap.add_argument('--frames', type=int, default=180,
                    help="frames to capture (matches the recorded runs)")
    ap.add_argument('--settle', type=int, default=10)
    ap.add_argument('--timeout', type=float, default=None,
                    help="seconds before giving up (default: scaled to "
                         "--frames, assuming a slow ~2 Hz stream)")
    ap.add_argument('--prep', type=int, default=10,
                    help="countdown seconds to place the object before capture")
    ap.add_argument('--simulate', metavar='GRIP_LABEL',
                    help="replay recorded data instead of hardware")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='captures')
    ap.add_argument('--record-only', action='store_true',
                    help="just capture and save frames; skip scoring. Use this "
                         "when the shipped models don't match the rig's range.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.simulate:
        from Mock_Serial import MockSerial
        ser = MockSerial.from_dataset(args.data_dir, dataset=args.object,
                                      grip_label=args.simulate, loop=True)
        print(f"[SIMULATION] replaying '{args.simulate}' -- no hardware in use")
    else:
        import serial
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
        time.sleep(2)          # Arduino resets when the port opens
        print(f"Serial open on {args.port}")

    print(f"\n=== {args.label}  (model: {args.object}) ===")
    if not args.simulate:
        prep_countdown(args.prep)
    timeout = args.timeout if args.timeout else max(60.0, args.frames * 2.0)
    frames = read_frames(ser, args.frames, args.settle, timeout)

    if not args.simulate:
        ser.close()

    if len(frames) < WINDOW_SIZE:
        print(f"\nFAILED: got {len(frames)} frames, need at least {WINDOW_SIZE}. "
              f"Check the Arduino is streaming 'v1,v2,v3,v4' lines.")
        return

    print(f"\nCaptured {len(frames)} frames. "
          f"Mean per sensor: {np.round(frames.mean(axis=0), 1)}")

    xlsx = os.path.join(args.out_dir, f"{args.label}.xlsx")
    try:
        import pandas as pd
        pd.DataFrame(frames).to_excel(xlsx, header=False, index=False)
        print(f"Raw frames -> {xlsx}")
    except Exception as e:
        print(f"Could not write xlsx ({e}); writing csv instead")
        xlsx = os.path.join(args.out_dir, f"{args.label}.csv")
        np.savetxt(xlsx, frames, delimiter=',', fmt='%g')

    if args.record_only:
        out = {'label': args.label, 'object_model': args.object,
               'n_frames': int(len(frames)), 'simulated': bool(args.simulate),
               'sensor_means': [round(float(x), 2) for x in frames.mean(axis=0)],
               'record_only': True, 'raw_file': os.path.basename(xlsx)}
        js = os.path.join(args.out_dir, f"{args.label}.json")
        with open(js, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nRECORD-ONLY: {len(frames)} frames saved, no scoring.")
        print(f"Results -> {js}")
        return

    d_lab, d_conf = score_dense(frames,
                                f'one_phase_nn_grip_model_{args.object}.keras',
                                f'one_phase_nn_{args.object}_scaler.joblib')
    l_lab, l_conf = score_lstm(frames,
                               f'lstm_grip_model_{args.object}.keras',
                               f'lstm_{args.object}_scaler.joblib')

    r1 = summarise("EXPERIMENT 1 -- Dense NN (single frame)", d_lab, d_conf)
    r2 = summarise("EXPERIMENT 2 -- LSTM (15-frame window)", l_lab, l_conf)

    if r1 and r2:
        agree = "AGREE" if r1['verdict'] == r2['verdict'] else "DISAGREE"
        print(f"\n  the two models {agree}")

    out = {'label': args.label, 'object_model': args.object,
           'n_frames': int(len(frames)), 'simulated': bool(args.simulate),
           'sensor_means': [round(float(x), 2) for x in frames.mean(axis=0)],
           'experiment_1_dense': r1, 'experiment_2_lstm': r2,
           'raw_file': os.path.basename(xlsx)}
    js = os.path.join(args.out_dir, f"{args.label}.json")
    with open(js, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Results -> {js}")


if __name__ == '__main__':
    main()

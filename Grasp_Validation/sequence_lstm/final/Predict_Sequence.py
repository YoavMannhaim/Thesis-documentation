"""
Loads a trained LSTM sequence model and classifies windows of raw sensor
readings. Mirrors Predict_Grip.py from the Grasp Validation package, but
takes a window of consecutive frames as input instead of a single frame.

Round-trip self-test against the held-out test files:
    python3 Predict_Sequence.py --self-test --dataset sponge
    python3 Predict_Sequence.py --self-test --dataset holder
"""
import argparse
import numpy as np
from joblib import load

from Data_Loader import GRIP_CATEGORIES
from Sequence_Data_Loader import build_sequence_dataset
from Sequence_Split import split_by_file


def load_lstm_model(path):
    from tensorflow.keras.models import load_model
    try:
        model = load_model(path)
        print(f"Model loaded successfully: {path}")
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None


def predict_window(model, scaler, window):
    """window: (window_size, 4) raw sensor values -> (category, confidence)."""
    w = np.asarray(window, dtype=np.float32)
    n, f = w.shape
    scaled = scaler.transform(w.reshape(-1, f)).reshape(1, n, f)
    probs = model.predict(scaled, verbose=0)[0]
    idx = int(np.argmax(probs))
    return GRIP_CATEGORIES[idx], float(probs[idx])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None,
                    help="Defaults to lstm_grip_model_<dataset>.keras")
    ap.add_argument('--scaler', default=None,
                    help="Defaults to lstm_<dataset>_scaler.joblib")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--dataset', default='sponge', choices=['sponge', 'holder'])
    ap.add_argument('--window-size', type=int, default=15)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--test-fraction', type=float, default=0.25)
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()

    model_path = a.model or f'lstm_grip_model_{a.dataset}.keras'
    scaler_path = a.scaler or f'lstm_{a.dataset}_scaler.joblib'

    model = load_lstm_model(model_path)
    scaler = load(scaler_path)
    print(f"Scaler loaded successfully: {scaler_path}")

    if a.self_test:
        X, y, groups, _ = build_sequence_dataset(
            a.data_dir, dataset=a.dataset, window_size=a.window_size,
            stride=a.stride, verbose=False)
        _, _, X_test, y_test, _, test_files = split_by_file(
            X, y, groups, test_fraction=a.test_fraction)

        correct = 0
        for i in range(len(X_test)):
            cat, conf = predict_window(model, scaler, X_test[i])
            true_cat = GRIP_CATEGORIES[y_test[i]]
            correct += (cat == true_cat)
        acc = correct / len(X_test)
        print(f"\nReloaded model accuracy on held-out test files {test_files}: "
              f"{acc:.4f}  ({correct}/{len(X_test)})")

        print("\nSingle-window sanity checks (first window of each test class):")
        seen = set()
        for i in range(len(X_test)):
            true_cat = GRIP_CATEGORIES[y_test[i]]
            if true_cat in seen:
                continue
            seen.add(true_cat)
            cat, conf = predict_window(model, scaler, X_test[i])
            flag = "OK" if cat == true_cat else "MISS"
            print(f"  [{flag}] true={true_cat:<16} -> predicted={cat:<16} (conf {conf:.3f})")

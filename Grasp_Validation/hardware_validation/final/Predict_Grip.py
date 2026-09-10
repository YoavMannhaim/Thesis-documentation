"""
Loads the saved one-phase NN and its scaler, and classifies raw
photoresistor readings. This is the piece you drop into the real-time loop.

Usage as a module:
    from Predict_Grip import GripClassifier
    clf = GripClassifier('one_phase_nn_grip_model.keras',
                         'one_phase_nn_scaler.joblib')
    label, confidence = clf.predict([27, 25, 29, 39])

Usage from the shell (round-trip self-test against the training data):
    python3 Predict_Grip.py --self-test --data-dir /path/to/xlsx
"""

import os
import argparse
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model

GRIP_CATEGORIES = ['no_grip', 'bad_grip', 'semi_good_grip', 'good_grip']


class GripClassifier:
    def __init__(self, model_path, scaler_path):
        self.model = load_model(model_path)
        self.scaler = load(scaler_path)

    def predict(self, sensor_values):
        """
        sensor_values : sequence of 4 raw photoresistor readings.
        Returns (label, confidence).
        """
        x = np.asarray(sensor_values, dtype='float32').reshape(1, -1)
        if x.shape[1] != 4:
            raise ValueError(f"expected 4 sensor values, got {x.shape[1]}")
        xs = self.scaler.transform(x)
        p = self.model.predict(xs, verbose=0)[0]
        idx = int(np.argmax(p))
        return GRIP_CATEGORIES[idx], float(p[idx])

    def predict_batch(self, sensor_matrix):
        x = np.asarray(sensor_matrix, dtype='float32')
        xs = self.scaler.transform(x)
        p = self.model.predict(xs, verbose=0)
        return np.argmax(p, axis=1), p.max(axis=1)


def _self_test(model_path, scaler_path, data_dir, dataset='holder'):
    """Reload from disk and confirm predictions match the in-memory results."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from Data_Loader import build_dataset, SENSOR_COLS

    df = build_dataset(data_dir, dataset=dataset, verbose=False)
    X = df[SENSOR_COLS].values.astype('float32')
    y = df['grip_quality'].values.astype('int32')
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    clf = GripClassifier(model_path, scaler_path)
    y_pred, conf = clf.predict_batch(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Reloaded model test accuracy : {acc:.4f}")
    print(f"Mean confidence              : {conf.mean():.3f}")
    print("\nSingle-sample sanity checks:")
    for raw, expect in [([27, 25, 29, 39], 'no_grip (tight baseline range)'),
                        (X_test[0].tolist(), GRIP_CATEGORIES[y_test[0]]),
                        (X_test[1].tolist(), GRIP_CATEGORIES[y_test[1]])]:
        lbl, c = clf.predict(raw)
        print(f"  {str([round(v,1) for v in raw]):<28} -> {lbl:<16} "
              f"(conf {c:.3f})   [expected: {expect}]")
    return acc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='one_phase_nn_grip_model_holder.keras')
    ap.add_argument('--scaler', default='one_phase_nn_holder_scaler.joblib')
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--dataset', default='holder', choices=['holder','sponge'])
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        _self_test(a.model, a.scaler, a.data_dir, a.dataset)
    else:
        clf = GripClassifier(a.model, a.scaler)
        print(clf.predict([27, 25, 29, 39]))

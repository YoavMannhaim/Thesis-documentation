"""
Retrains both models using only sensor_1 and sensor_2.

This exists because on the physical rig, photoresistor channels 3 and 4 are
dead -- they read exactly 0.0 with zero variance across every frame
(confirmed live: 24/24 frames, one distinct value each, while channels 1
and 2 swung normally). The shipped models take 4 inputs, so feeding them
two real channels and two constants produces confident but meaningless
predictions.

These 2-input models are trained on the SAME recordings, with the SAME splits
and hyperparameters as the 4-sensor originals -- only the input width changes.
That keeps the accuracy drop attributable to the lost channels rather than to
a different training procedure.

Outputs (per dataset):
    two_sensor_nn_grip_model_<ds>.keras     + two_sensor_nn_<ds>_scaler.joblib
    two_sensor_lstm_grip_model_<ds>.keras   + two_sensor_lstm_<ds>_scaler.joblib
    two_sensor_metrics_<ds>.json
"""

import os, json, argparse
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from Data_Loader import build_dataset, GRIP_CATEGORIES
from Sequence_Data_Loader import build_sequence_dataset
from Sequence_Split import split_by_file

SENSORS = ['sensor_1', 'sensor_2']       # the two channels still alive
RANDOM_STATE = 42                        # same as Train_One_Phase_NN.py


def build_dense(n_features):
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(n_features,), name='sensor_input')
    h = Dense(64, activation='relu')(inp)
    h = Dropout(0.3)(h)
    h = Dense(32, activation='relu')(h)
    h = Dropout(0.3)(h)
    out = Dense(4, activation='softmax', name='grip_output')(h)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def build_lstm(window, n_features):
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(window, n_features))
    x = LSTM(32)(inp)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(4, activation='softmax')(x)
    m = Model(inp, out)
    m.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return m


def train_dense(ds, data_dir, epochs):
    df = build_dataset(data_dir, dataset=ds, verbose=False)
    X = df[SENSORS].to_numpy(np.float32)
    y = df['grip_quality'].to_numpy()

    # identical 64/16/20 stratified split to the 4-sensor original
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tmp)

    sc = StandardScaler().fit(X_train)
    m = build_dense(len(SENSORS))
    m.fit(sc.transform(X_train), y_train,
          validation_data=(sc.transform(X_val), y_val),
          epochs=epochs, batch_size=32, verbose=0)

    pred = np.argmax(m.predict(sc.transform(X_test), verbose=0), axis=1)
    acc = float(accuracy_score(y_test, pred))
    rep = classification_report(y_test, pred, target_names=GRIP_CATEGORIES,
                                output_dict=True, zero_division=0)
    m.save(f'two_sensor_nn_grip_model_{ds}.keras')
    dump(sc, f'two_sensor_nn_{ds}_scaler.joblib')
    return acc, rep, len(y_test)


def train_lstm(ds, data_dir, epochs, window=15, stride=3):
    X, y, groups, _ = build_sequence_dataset(data_dir, dataset=ds,
                                             window_size=window, stride=stride,
                                             verbose=False)
    X = X[:, :, :2]                       # keep only the two live channels
    Xtr, ytr, Xte, yte, trf, tef = split_by_file(X, y, groups)

    n, w, f = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(-1, f))
    Xtr_s = sc.transform(Xtr.reshape(-1, f)).reshape(n, w, f)
    Xte_s = sc.transform(Xte.reshape(-1, f)).reshape(Xte.shape[0], w, f)

    m = build_lstm(w, f)
    m.fit(Xtr_s, ytr, epochs=epochs, batch_size=32, verbose=0,
          validation_split=0.2)
    pred = np.argmax(m.predict(Xte_s, verbose=0), axis=1)
    acc = float(accuracy_score(yte, pred))
    rep = classification_report(yte, pred, target_names=GRIP_CATEGORIES,
                                output_dict=True, zero_division=0)
    m.save(f'two_sensor_lstm_grip_model_{ds}.keras')
    dump(sc, f'two_sensor_lstm_{ds}_scaler.joblib')
    return acc, rep, len(yte), tef


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--epochs', type=int, default=100)
    args = ap.parse_args()

    for ds in ['sponge', 'holder']:
        print(f"\n{'='*58}\n{ds.upper()}  --  sensors {SENSORS}\n{'='*58}")
        d_acc, d_rep, d_n = train_dense(ds, args.data_dir, args.epochs)
        print(f"  Dense (2-sensor) test accuracy : {d_acc:.4f}   (n={d_n})")
        l_acc, l_rep, l_n, tef = train_lstm(ds, args.data_dir, args.epochs)
        print(f"  LSTM  (2-sensor) test accuracy : {l_acc:.4f}   (n={l_n})")
        print(f"  LSTM held-out files: {tef}")

        json.dump({'dataset': ds, 'sensors_used': SENSORS,
                   'dense_test_accuracy': d_acc, 'dense_report': d_rep,
                   'lstm_test_accuracy': l_acc, 'lstm_report': l_rep,
                   'lstm_test_files': tef, 'epochs': args.epochs,
                   'random_state': RANDOM_STATE},
                  open(f'two_sensor_metrics_{ds}.json', 'w'), indent=2)
        print(f"  metrics -> two_sensor_metrics_{ds}.json")

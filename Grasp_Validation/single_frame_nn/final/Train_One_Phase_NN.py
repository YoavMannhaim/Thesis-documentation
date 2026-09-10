"""
Trains the one-phase grip-quality neural network on the photoresistor data
and saves a deployable .keras model plus the fitted scaler.

Architecture is unchanged from One_Phase_NN_Test.py:
    Input(4) -> Dense(64,relu) -> Dropout(0.3)
             -> Dense(32,relu) -> Dropout(0.3)
             -> Dense(4,softmax)
    Adam(lr=1e-3), sparse_categorical_crossentropy, 100 epochs, batch 32

This version differs from the original script in a few ways (all flagged
in README_MODEL.md): the split is a proper three-way 64/16/20 train/val/test,
stratified on grip quality -- the original passed the test set in as
validation_data, so its "val_accuracy" curve was really test accuracy.
Figures save to disk instead of plt.show(), since this runs headless. The
fitted StandardScaler is saved alongside the model, which the original only
did in Comparison_Test.py, even though inference is wrong without it.
Class weighting is available as an option, since no_grip is only ~6.5% of
the data but is the most safety-critical class (object lost).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from joblib import dump

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Data_Loader import build_dataset, SENSOR_COLS, GRIP_CATEGORIES

RANDOM_STATE = 42


def build_model():
    inp = Input(shape=(4,), name='sensor_input')
    h = Dense(64, activation='relu')(inp)
    h = Dropout(0.3)(h)
    h = Dense(32, activation='relu')(h)
    h = Dropout(0.3)(h)
    out = Dense(4, activation='softmax', name='grip_output')(h)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_history(history, path, title_suffix=''):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'], label='Train', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', lw=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.set_title('One-Phase NN Accuracy' + title_suffix)
    ax1.grid(alpha=0.3); ax1.legend()
    ax2.plot(history.history['loss'], label='Train', lw=2)
    ax2.plot(history.history['val_loss'], label='Validation', lw=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.set_title('One-Phase NN Loss' + title_suffix)
    ax2.grid(alpha=0.3); ax2.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(y_true, y_pred, path, title_suffix=''):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(GRIP_CATEGORIES)), GRIP_CATEGORIES, rotation=30, ha='right')
    ax.set_yticks(range(len(GRIP_CATEGORIES)), GRIP_CATEGORIES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('One-Phase NN Grip Classification' + title_suffix)
    fig.colorbar(im)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--dataset', default='holder', choices=['holder', 'sponge'],
                    help='Which object dataset to train on')
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--class-weight', action='store_true',
                    help='Weight classes inversely to frequency')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    print("=" * 62)
    print("LOADING DATA")
    print("=" * 62)
    df = build_dataset(args.data_dir, dataset=args.dataset)

    X = df[SENSOR_COLS].values.astype('float32')
    y = df['grip_quality'].values.astype('int32')

    # --- 3-way stratified split: 64 / 16 / 20 ---
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_tmp)
    print(f"\nSplit -> train {len(X_train)} | val {len(X_val)} | test {len(X_test)}")

    # --- Scale (fit on TRAIN only) ---
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    scaler_path = os.path.join(args.out_dir, f'one_phase_nn_{args.dataset}_scaler.joblib')
    dump(scaler, scaler_path)
    print(f"Scaler saved -> {scaler_path}")

    cw = None
    if args.class_weight:
        classes = np.unique(y_train)
        w = compute_class_weight('balanced', classes=classes, y=y_train)
        cw = {int(c): float(wi) for c, wi in zip(classes, w)}
        print("Class weights:", cw)

    print("\n" + "=" * 62)
    print("TRAINING")
    print("=" * 62)
    model = build_model()
    model.summary()

    history = model.fit(X_train_s, y_train,
                        validation_data=(X_val_s, y_val),
                        epochs=args.epochs, batch_size=32,
                        class_weight=cw, verbose=0)

    # --- Evaluate on the held-out TEST set ---
    probs = model.predict(X_test_s, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nFinal train acc : {history.history['accuracy'][-1]:.4f}")
    print(f"Final val acc   : {history.history['val_accuracy'][-1]:.4f}")
    print(f"TEST accuracy   : {acc:.4f}")
    print("\nTest classification report:")
    report = classification_report(y_test, y_pred, labels=[0, 1, 2, 3],
                                   target_names=GRIP_CATEGORIES, zero_division=0)
    print(report)

    suffix = f' [{args.dataset}]' + (' (class-weighted)' if args.class_weight else '')
    tag = f'_{args.dataset}' + ('_weighted' if args.class_weight else '')
    plot_history(history, os.path.join(args.out_dir, f'one_phase_nn_training_curves{tag}.png'), suffix)
    plot_confusion(y_test, y_pred, os.path.join(args.out_dir, f'one_phase_nn_confusion_matrix{tag}.png'), suffix)

    model_path = os.path.join(args.out_dir, f'one_phase_nn_grip_model{tag}.keras')
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    meta = {
        'dataset': args.dataset,
        'test_accuracy': float(acc),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'n_samples': int(len(df)),
        'split': {'train': int(len(X_train)), 'val': int(len(X_val)), 'test': int(len(X_test))},
        'grip_categories': GRIP_CATEGORIES,
        'sensor_columns': SENSOR_COLS,
        'class_weighted': bool(args.class_weight),
        'epochs': args.epochs,
        'random_state': RANDOM_STATE,
        'scaler': os.path.basename(scaler_path),
        'per_class_report': classification_report(
            y_test, y_pred, labels=[0, 1, 2, 3], target_names=GRIP_CATEGORIES,
            zero_division=0, output_dict=True),
    }
    meta_path = os.path.join(args.out_dir, f'one_phase_nn_metrics{tag}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metrics saved -> {meta_path}")
    return acc


if __name__ == '__main__':
    main()

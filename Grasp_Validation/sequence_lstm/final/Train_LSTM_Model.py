"""
Trains an LSTM sequence classifier on sliding windows of raw photoresistor
readings, and evaluates it against a file-level held-out test set (see
Sequence_Split.py for why file-level, not window-level, splitting is used).

See Sequence_Data_Loader.py's module docstring for the honest framing of
what this model does and does not demonstrate.

Usage:
    python3 Train_LSTM_Model.py --dataset sponge
    python3 Train_LSTM_Model.py --dataset holder --window-size 15 --stride 3

Outputs (into --out-dir, default '.'):
    lstm_grip_model_<dataset>.keras
    lstm_<dataset>_scaler.joblib
    lstm_metrics_<dataset>.json
    lstm_confusion_matrix_<dataset>.png
    lstm_training_curves_<dataset>.png
"""
import argparse
import json
import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from Sequence_Data_Loader import build_sequence_dataset, GRIP_CATEGORIES
from Sequence_Split import split_by_file, report as split_report


def scale_windows(X_train, X_test):
    """Fit a StandardScaler on TRAIN windows only, apply to both."""
    n_tr, w, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, f))
    X_train_s = scaler.transform(X_train.reshape(-1, f)).reshape(n_tr, w, f)
    X_test_s = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape[0], w, f)
    return X_train_s, X_test_s, scaler


def build_model(window_size, n_features, n_classes, lstm_units=32, dense_units=16,
                dropout=0.3):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    inp = Input(shape=(window_size, n_features))
    x = LSTM(lstm_units)(inp)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(n_classes, activation='softmax')(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--dataset', default='sponge', choices=['sponge', 'holder'])
    ap.add_argument('--out-dir', default='.')
    ap.add_argument('--window-size', type=int, default=15)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--test-fraction', type=float, default=0.25)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--class-weight', action='store_true',
                    help="Balance the loss for the minority no_grip class")
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    print("=" * 70)
    print("BUILDING WINDOWED SEQUENCE DATASET")
    print("=" * 70)
    X, y, groups, file_report = build_sequence_dataset(
        args.data_dir, dataset=args.dataset,
        window_size=args.window_size, stride=args.stride)

    print("\n" + "=" * 70)
    print("FILE-LEVEL TRAIN/TEST SPLIT")
    print("=" * 70)
    X_train, y_train, X_test, y_test, train_files, test_files = split_by_file(
        X, y, groups, test_fraction=args.test_fraction)
    split_report(train_files, test_files, y_train, y_test, GRIP_CATEGORIES)

    X_train, X_test, scaler = scale_windows(X_train, X_test)
    scaler_path = f'{args.out_dir}/lstm_{args.dataset}_scaler.joblib'
    dump(scaler, scaler_path)
    print(f"\nScaler saved -> {scaler_path}")

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    model = build_model(args.window_size, X.shape[2], len(GRIP_CATEGORIES))
    model.summary()

    class_weight = None
    if args.class_weight:
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes.tolist(), weights.tolist()))
        print(f"\nUsing class weights: {class_weight}")

    import tensorflow as tf
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,   # carved from TRAIN windows only, for early stopping
        epochs=args.epochs, batch_size=16, class_weight=class_weight,
        callbacks=[early_stop], verbose=2)

    model_path = f'{args.out_dir}/lstm_grip_model_{args.dataset}.keras'
    model.save(model_path)
    print(f"\nModel saved -> {model_path}")

    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT (FILE-LEVEL) TEST SET")
    print("=" * 70)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {test_acc:.4f}")
    present = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    target_names = [GRIP_CATEGORIES[i] for i in present]
    report_txt = classification_report(y_test, y_pred, labels=present,
                                       target_names=target_names, zero_division=0)
    print(report_txt)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(GRIP_CATEGORIES))))

    metrics = {
        'dataset': args.dataset,
        'window_size': args.window_size,
        'stride': args.stride,
        'test_accuracy': float(test_acc),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'epochs_trained': len(history.history['accuracy']),
        'n_train_windows': int(len(y_train)),
        'n_test_windows': int(len(y_test)),
        'train_files': train_files,
        'test_files': test_files,
        'classification_report': classification_report(
            y_test, y_pred, labels=present, target_names=target_names,
            zero_division=0, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': GRIP_CATEGORIES,
    }
    metrics_path = f'{args.out_dir}/lstm_metrics_{args.dataset}.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # --- plots ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(GRIP_CATEGORIES))); ax.set_xticklabels(GRIP_CATEGORIES, rotation=45, ha='right')
    ax.set_yticks(range(len(GRIP_CATEGORIES))); ax.set_yticklabels(GRIP_CATEGORIES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'LSTM confusion matrix -- {args.dataset}\n(file-level held-out test, acc={test_acc:.3f})')
    for i in range(len(GRIP_CATEGORIES)):
        for j in range(len(GRIP_CATEGORIES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    cm_path = f'{args.out_dir}/lstm_confusion_matrix_{args.dataset}.png'
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot -> {cm_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val (within-train split)')
    axes[0].set_title('Loss'); axes[0].set_xlabel('epoch'); axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='train')
    axes[1].plot(history.history['val_accuracy'], label='val (within-train split)')
    axes[1].axhline(test_acc, color='red', ls='--', label=f'held-out test ({test_acc:.3f})')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('epoch'); axes[1].legend()
    plt.suptitle(f'LSTM training curves -- {args.dataset}')
    plt.tight_layout()
    curves_path = f'{args.out_dir}/lstm_training_curves_{args.dataset}.png'
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"Training curves plot -> {curves_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()

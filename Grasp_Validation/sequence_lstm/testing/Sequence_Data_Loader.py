"""
Builds sliding-window sequences from the same raw photoresistor .xlsx logs
used by Data_Loader.py, for training a sequence model (LSTM) instead of a
single-frame classifier.

Worth reading before using the model: each raw file (e.g.
`Holder_Front_Bad.xlsx`) is a continuous recording of ~130-190 consecutive
sensor frames taken while the object was held in one fixed grip condition
(e.g. "bad grip, front orientation"). The recordings do not contain a
continuous transition from a good grip sliding into a bad one -- each file
is a single static condition sampled repeatedly over time.

That means the LSTM built here answers a specific, honestly-scoped question:

    "Does a short temporal window of raw sensor frames classify grip quality
     more accurately than a single frame?" (the comparison already made by the
     one-phase Dense NN in the Grasp Validation package)

It is NOT trained on, and should NOT be presented as, a model that has seen a
grip degrading in real time and learned to recognise the moment of slippage --
that would require recordings that actually contain such a transition, which
does not exist in the available data. If/when such recordings are collected,
this module's windowing logic can be reused as-is; only the label assigned to
each window would need to change (e.g. "time-to-failure" or a binary
slipping/stable label per window) instead of the static grip-quality label
used here.

Windows are built PER SOURCE FILE, so a window never mixes frames from two
different recordings (which would be temporally meaningless), and the
train/test split in Train_LSTM_Model.py is done PER FILE for the same reason
-- adjacent windows from one recording are highly correlated, so splitting at
the window level would leak information between train and test.

Usage:
    from Sequence_Data_Loader import build_sequence_dataset
    X, y, groups = build_sequence_dataset(data_dir, dataset='sponge',
                                          window_size=15, stride=3)
    # X: (n_windows, window_size, 4)   y: (n_windows,) int labels
    # groups: (n_windows,) source_file per window, for grouped splitting
"""

import numpy as np
from Data_Loader import build_dataset, SENSOR_COLS, GRIP_LABEL_MAP, GRIP_CATEGORIES


def _windows_from_group(arr, window_size, stride):
    """arr: (n_rows, 4) in original time order -> (n_windows, window_size, 4)."""
    n = len(arr)
    if n < window_size:
        return np.empty((0, window_size, arr.shape[1]))
    starts = range(0, n - window_size + 1, stride)
    return np.stack([arr[s:s + window_size] for s in starts], axis=0)


def build_sequence_dataset(data_dir, dataset='holder', window_size=15, stride=3,
                           verbose=True):
    """
    Returns:
        X       : (n_windows, window_size, 4) float32 raw sensor values
        y       : (n_windows,) int labels (see GRIP_LABEL_MAP)
        groups  : (n_windows,) str, the source_file each window came from
                  (use for a GroupShuffleSplit / GroupKFold so windows from
                  the same recording never span train and test)
        file_report : dict[source_file] -> n_windows produced, for sanity
                  checking that no file was silently dropped for being
                  shorter than window_size
    """
    df = build_dataset(data_dir, dataset=dataset, verbose=verbose)

    X_list, y_list, g_list = [], [], []
    file_report = {}

    for fname, g in df.groupby('source_file', sort=False):
        arr = g[SENSOR_COLS].to_numpy(dtype=np.float32)
        label = GRIP_LABEL_MAP[g['grip_label'].iloc[0]]
        windows = _windows_from_group(arr, window_size, stride)
        file_report[fname] = len(windows)
        if len(windows) == 0:
            if verbose:
                print(f"  [skip]  {fname}: only {len(arr)} rows, "
                      f"shorter than window_size={window_size} -- 0 windows")
            continue
        X_list.append(windows)
        y_list.append(np.full(len(windows), label, dtype=np.int64))
        g_list.append(np.full(len(windows), fname, dtype=object))

    if not X_list:
        raise RuntimeError(
            f"No windows produced for dataset='{dataset}' with "
            f"window_size={window_size}. Try a smaller window_size.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    if verbose:
        print(f"\nSequence dataset '{dataset}': {len(X)} windows "
              f"(window_size={window_size}, stride={stride}) "
              f"from {len(file_report)} files")
        print("\nWindow count per class:")
        for i, name in enumerate(GRIP_CATEGORIES):
            n = int((y == i).sum())
            print(f"  {name:<16} {n:>5}  ({100.0 * n / len(y):5.1f}%)")

    return X, y, groups, file_report


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    ds = sys.argv[2] if len(sys.argv) > 2 else 'holder'
    build_sequence_dataset(d, ds)

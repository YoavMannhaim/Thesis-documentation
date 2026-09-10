"""
File-level train/test split for the windowed sequence dataset.

A plain random GroupShuffleSplit doesn't really work here: each object
dataset has only 11-12 raw recording files total, and as few as
1-4 files per grip-quality class (e.g. sponge has exactly ONE no_grip file,
shared with the holder dataset). A random group split at this scale can
easily produce a test set missing a class entirely, or a test set with only
one file's worth of windows for the whole split. Rather than rely on chance,
files are assigned to train/test deterministically per class, so every class
with more than one file contributes at least one file to each split.

No_Grip.xlsx is a single long idle-baseline recording, not tied to a specific
object/orientation grip condition -- there is only one condition (no_grip)
throughout it. It is therefore split WITHIN the file (a temporal prefix for
train, a temporal suffix for test) rather than being assigned whole to one
side, which would otherwise put the entire no_grip class on one side of the
split for BOTH datasets (since the file is shared).
"""

import numpy as np
from collections import defaultdict


def split_by_file(X, y, groups, test_fraction=0.25, seed=13):
    """
    Returns (X_train, y_train, X_test, y_test, train_files, test_files).

    Deterministic: files for each class are sorted by name and every
    ceil(1/test_fraction)-th one goes to test, guaranteeing at least one
    train AND one test file per class whenever a class has 2+ files.
    """
    rng = np.random.default_rng(seed)
    unique_files = sorted(set(groups))

    # No_Grip.xlsx: split WITHIN the file by time (see module docstring).
    no_grip_mask_all = np.array([f == 'No_Grip.xlsx' for f in groups])
    train_idx, test_idx = [], []

    if no_grip_mask_all.any():
        ng_idx = np.where(no_grip_mask_all)[0]
        # windows were built in time order within the file, so a temporal
        # prefix/suffix split has no leakage between train and test frames
        cut = int(len(ng_idx) * (1 - test_fraction))
        train_idx.extend(ng_idx[:cut])
        test_idx.extend(ng_idx[cut:])

    # All other files: assign whole files per class.
    file_to_label = {}
    for f, lab in zip(groups, y):
        file_to_label.setdefault(f, lab)
    by_label_files = defaultdict(list)
    for f, lab in file_to_label.items():
        if f == 'No_Grip.xlsx':
            continue
        by_label_files[lab].append(f)

    train_files, test_files = ['No_Grip.xlsx (train portion)'], ['No_Grip.xlsx (test portion)']
    for lab, files in by_label_files.items():
        files = sorted(files)
        if len(files) == 1:
            # only one file for this class: keep it in TRAIN (a class with a
            # single recording cannot be evaluated out-of-file at all; this
            # is reported explicitly by build_report() below).
            train_files.extend(files)
            continue
        n_test = max(1, round(len(files) * test_fraction))
        n_test = min(n_test, len(files) - 1)  # always leave >=1 file in train
        test_set = set(files[:n_test])
        for f in files:
            (test_files if f in test_set else train_files).append(f)

    file_side = {}
    for f in train_files:
        file_side[f.replace(' (train portion)', '')] = 'train'
    for f in test_files:
        file_side[f.replace(' (test portion)', '')] = 'test'

    for i, f in enumerate(groups):
        if f == 'No_Grip.xlsx':
            continue
        (train_idx if file_side.get(f) == 'train' else test_idx).append(i)

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))

    return (X[train_idx], y[train_idx], X[test_idx], y[test_idx],
            sorted(train_files), sorted(test_files))


def report(train_files, test_files, y_train, y_test, categories):
    print("\nFile-level split:")
    print(f"  train files ({len(train_files)}): {train_files}")
    print(f"  test  files ({len(test_files)}): {test_files}")
    print(f"\n  train windows: {len(y_train)}   test windows: {len(y_test)}")
    print("\n  class coverage:")
    for i, name in enumerate(categories):
        ntr = int((y_train == i).sum())
        nte = int((y_test == i).sum())
        flag = "  <-- NOT in test set" if nte == 0 else ("  <-- NOT in train set" if ntr == 0 else "")
        print(f"    {name:<16} train={ntr:>4}  test={nte:>4}{flag}")

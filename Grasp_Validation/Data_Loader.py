"""
Builds the training dataset from the raw photoresistor .xlsx logs.

Replaces Two_Phase_NN_Test.create_df(), which Comparison_Test.py depends on
but which was never supplied.

Supports multiple object datasets, selected by name:

    build_dataset(data_dir, dataset='holder')   # Holder_*.xlsx
    build_dataset(data_dir, dataset='sponge')   # Spong_*.xlsx + Sponge_*.xlsx

A couple of things worth knowing about the raw files. They have no header
row -- the first row is real sensor data, so pandas' default header=0 would
silently eat one sample per file, which is why this uses header=None
instead. Some files (e.g. Sponge_Back_Bad.xlsx) also carry leading
Microsoft Data Streamer metadata rows ('#!', 'Workbook:', a URL); these get
dropped automatically since every column is coerced to numeric first.
Known duplicate files are excluded via _SKIP_FILES.
"""

import os
import glob
import pandas as pd

SENSOR_COLS = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']

GRIP_LABEL_MAP = {
    'no_grip': 0,
    'bad_grip': 1,
    'semi_good_grip': 2,
    'good_grip': 3,
}
GRIP_CATEGORIES = ['no_grip', 'bad_grip', 'semi_good_grip', 'good_grip']

_QUALITY_TOKENS = {
    'bad': 'bad_grip',
    'semi': 'semi_good_grip',
    'good': 'good_grip',
}

# Filename prefixes belonging to each object dataset.
DATASETS = {
    'holder': ['Holder_'],
    'sponge': ['Spong_', 'Sponge_'],   # note the inconsistent spelling in the raw files
}

# Files excluded as exact duplicates (verified by content comparison).
_SKIP_FILES = {
    'Holder_Front_Semi_1.xlsx',   # byte-identical to Holder_Front_Semi.xlsx
}


def _parse_filename(fname):
    """'Spong_Bottom_Bad.xlsx' -> ('Bottom', 'bad_grip')."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None
    orientation = parts[1]
    quality = _QUALITY_TOKENS.get(parts[2].lower())
    return orientation, quality


def _read_sensor_xlsx(path):
    """
    Read a sensor log, tolerating leading metadata rows and missing headers.
    Returns a DataFrame with exactly the 4 sensor columns, numeric.
    """
    d = pd.read_excel(path, header=None)
    d = d.iloc[:, :4]
    d.columns = SENSOR_COLS
    for c in SENSOR_COLS:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    n_before = len(d)
    d = d.dropna(subset=SENSOR_COLS).reset_index(drop=True)
    return d, n_before - len(d)


def build_dataset(data_dir, dataset='holder', verbose=True):
    """
    Returns a DataFrame with columns:
        sensor_1..sensor_4, grip_quality (int), grip_label (str),
        orientation (str), source_file (str)
    """
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {list(DATASETS)}")
    prefixes = DATASETS[dataset]

    frames = []

    # --- No-grip baseline, loaded ONCE ---
    # (The original Comparison_Test.py passed No_Grip.xlsx into create_df once
    #  per orientation, which would have triplicated these rows.)
    no_grip_path = os.path.join(data_dir, 'No_Grip.xlsx')
    if os.path.exists(no_grip_path):
        d, skipped = _read_sensor_xlsx(no_grip_path)
        d['grip_label'] = 'no_grip'
        d['orientation'] = 'none'
        d['source_file'] = 'No_Grip.xlsx'
        frames.append(d)
        if verbose and skipped:
            print(f"  [clean] No_Grip.xlsx: dropped {skipped} non-numeric row(s)")

    # --- Object files ---
    paths = []
    for pfx in prefixes:
        paths.extend(glob.glob(os.path.join(data_dir, f'{pfx}*.xlsx')))
    for path in sorted(set(paths)):
        base = os.path.basename(path)
        if base in _SKIP_FILES:
            if verbose:
                print(f"  [skip]  {base} (identical duplicate)")
            continue
        orientation, quality = _parse_filename(base)
        if quality is None:
            if verbose:
                print(f"  [skip]  {base} (unrecognised quality token)")
            continue
        d, skipped = _read_sensor_xlsx(path)
        if verbose and skipped:
            print(f"  [clean] {base}: dropped {skipped} non-numeric row(s) "
                  f"(metadata preamble)")
        d['grip_label'] = quality
        d['orientation'] = orientation
        d['source_file'] = base
        frames.append(d)

    if not frames:
        raise RuntimeError(f"No files found for dataset '{dataset}' in {data_dir}")

    df = pd.concat(frames, ignore_index=True)
    df['grip_quality'] = df['grip_label'].map(GRIP_LABEL_MAP)

    if verbose:
        print(f"\nDataset '{dataset}': {len(df)} samples from "
              f"{df['source_file'].nunique()} files")
        print("\nClass balance:")
        for name in GRIP_CATEGORIES:
            n = int((df['grip_label'] == name).sum())
            print(f"  {name:<16} {n:>5}  ({100.0 * n / len(df):5.1f}%)")
        print("\nOrientation coverage (samples per orientation x quality):")
        print(pd.crosstab(df['orientation'], df['grip_label']).to_string())

    return df


if __name__ == '__main__':
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else '.'
    ds = sys.argv[2] if len(sys.argv) > 2 else 'holder'
    build_dataset(d, ds)

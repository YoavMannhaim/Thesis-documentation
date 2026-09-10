"""
Loads the per-object asset map (model, scaler, thresholds) used by the
grasp validation protocol.

This differs from the original in a few ways (see README_PROTOCOL.md):
`CONFIG_PROFILE` used to be a module-level dict that every call mutated in
place, so loading a second object partially overwrote the first and every
caller shared one profile -- each call now builds and returns its own dict.
The config directory is resolvable via the ASSET_CONFIG_DIR env var or the
`config_dir` argument, instead of a hardcoded "./configs". Required keys
are validated up front with a clear error, rather than raising a bare
KeyError deep inside the monitoring loop. A few optional keys
(SMOOTHING_WINDOW, EMA_ALPHA, CONSECUTIVE_HITS, MAX_MONITOR_CYCLES) fall
back to sane defaults if absent, so older asset maps still load.
"""

import os
import json
from joblib import load
from tensorflow.keras.models import load_model

# Overridable via env var or the config_dir argument.
ASSET_CONFIG_DIR = os.environ.get('ASSET_CONFIG_DIR', './configs')

_REQUIRED_KEYS = [
    'MODEL_PATH', 'SCALER_PATH', 'CATEGORIES',
    'IS_FRAGILE', 'SLIPPAGE_THRESHOLDS', 'QUALITY_THRESHOLDS',
]

_DEFAULTS = {
    'SMOOTHING_WINDOW': 10,
    'EMA_ALPHA': 0.9,
    'CONSECUTIVE_HITS': 3,
    'MAX_MONITOR_CYCLES': 5000,
    'SETTLING_FRAMES': 10,      # frames discarded before Phase 1 voting
    'MAX_NO_DATA_READS': 500,   # consecutive empty reads before failing safe
}


def load_asset_map(object_name, config_dir=None):
    """Load and validate the object-specific asset map JSON."""
    cdir = config_dir or ASSET_CONFIG_DIR
    file_path = os.path.join(cdir, f"{object_name}_asset_map.txt")

    try:
        with open(file_path, 'r') as f:
            asset_map = json.load(f)
    except FileNotFoundError:
        available = []
        if os.path.isdir(cdir):
            available = sorted(x.replace('_asset_map.txt', '')
                               for x in os.listdir(cdir)
                               if x.endswith('_asset_map.txt'))
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}. "
            f"Available objects in {cdir}: {available or '(none)'}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding {file_path}: {e}")

    missing = [k for k in _REQUIRED_KEYS if k not in asset_map]
    if missing:
        raise KeyError(f"{file_path} is missing required key(s): {missing}")

    for k, v in _DEFAULTS.items():
        asset_map.setdefault(k, v)

    return asset_map


def load_configuration_profile(object_name, config_dir=None):
    """
    Load the model, scaler and configuration for the chosen object.

    Returns a NEW dict per call (the original mutated a shared module global).
    """
    config = load_asset_map(object_name, config_dir=config_dir)

    print(f"Loading assets for: {object_name}...")

    model_path = config["MODEL_PATH"]
    scaler_path = config["SCALER_PATH"]
    for label, p in (("Model", model_path), ("Scaler", scaler_path)):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} not found for '{object_name}': {p}")

    profile = {
        'object_name': object_name,
        'model': load_model(model_path),
        'scaler': load(scaler_path),
        'categories': config["CATEGORIES"],
        'is_fragile': config["IS_FRAGILE"],
        'SLIPPAGE_THRESHOLDS': config["SLIPPAGE_THRESHOLDS"],
        'QUALITY_THRESHOLDS': config["QUALITY_THRESHOLDS"],
        'SMOOTHING_WINDOW': config["SMOOTHING_WINDOW"],
        'EMA_ALPHA': config["EMA_ALPHA"],
        'CONSECUTIVE_HITS': config["CONSECUTIVE_HITS"],
        'MAX_MONITOR_CYCLES': config["MAX_MONITOR_CYCLES"],
        'SETTLING_FRAMES': config["SETTLING_FRAMES"],
        'MAX_NO_DATA_READS': config["MAX_NO_DATA_READS"],
    }

    # Sanity check: the protocol indexes categories[0] as "no grip" and
    # categories[1] as "bad grip". Fail loudly now rather than mis-acting later.
    cats = profile['categories']
    if len(cats) < 2:
        raise ValueError(f"CATEGORIES must have at least 2 entries, got {cats}")
    if 'no_grip' not in cats[0]:
        print(f"  WARNING: categories[0] is '{cats[0]}', expected the no-grip "
              f"class. The protocol treats index 0 as no-grip.")

    print(f"Configuration profile loaded successfully for {object_name}.")
    return profile

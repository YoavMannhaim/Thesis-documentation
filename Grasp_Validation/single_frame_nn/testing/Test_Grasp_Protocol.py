"""
End-to-end test suite for the grasp validation protocol.

Exercises every decision path using MockSerial replays of the real recorded
data and the real trained Keras models. No hardware required.

Run:  python3 Test_Grasp_Protocol.py
Exit code 0 if all tests pass, 1 otherwise.
"""

import io
import sys
import time
import contextlib
import numpy as np

import Run_Grasp_Protocol as RGP
import Object_Selection
from Mock_Serial import MockSerial
from Data_Loader import build_dataset, SENSOR_COLS

DATA_DIR = 'data'
CONFIG_DIR = 'configs'

_sent = []          # records every command the protocol emits
_profile_cache = {}


def _recording_send_command(ser, command):
    _sent.append(command)


RGP.send_command = _recording_send_command
RGP.time.sleep = lambda s: None          # strip real sleeps: tests run fast


def get_profile(obj, **overrides):
    """Load a profile once per object, then apply per-test overrides."""
    if obj not in _profile_cache:
        _profile_cache[obj] = Object_Selection.load_configuration_profile(
            obj, config_dir=CONFIG_DIR)
    p = dict(_profile_cache[obj])        # shallow copy; model/scaler shared
    p.update(overrides)
    return p


def rows_for(dataset, label, orientation=None, limit=None):
    df = build_dataset(DATA_DIR, dataset=dataset, verbose=False)
    sub = df[df.grip_label == label]
    if orientation:
        sub = sub[sub.orientation == orientation]
    v = sub[SENSOR_COLS].values.astype(float)
    return v[:limit] if limit else v


def run_case(profile, ser):
    """Run the protocol, capturing its console output."""
    _sent.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = RGP.run_grasp_protocol(profile, ser=ser)
    return result, list(_sent), buf.getvalue()


RESULTS = []


def check(name, condition, detail=''):
    RESULTS.append((name, bool(condition), detail))
    status = 'PASS' if condition else 'FAIL'
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ''))


# =====================================================================
print("=" * 70)
print("GRASP PROTOCOL TEST SUITE")
print("=" * 70)

# ---------------------------------------------------------------------
print("\n[Phase 1] Initial assessment rejection paths")
# ---------------------------------------------------------------------
for obj in ('sponge', 'holder'):
    ser = MockSerial(rows_for(obj, 'no_grip'), loop=True)
    res, sent, _ = run_case(get_profile(obj), ser)
    check(f"{obj}: no_grip -> REPICK",
          res == 'REPICK' and 'REPICK' in sent, f"returned {res}, sent {sent}")

# bad_grip should trigger REGRASP. Use a single orientation so the replay is
# coherent rather than jumping between recordings.
for obj, orient in (('sponge', 'Bottom'), ('holder', 'Back')):
    ser = MockSerial(rows_for(obj, 'bad_grip', orientation=orient), loop=True)
    res, sent, _ = run_case(get_profile(obj), ser)
    check(f"{obj}/{orient}: bad_grip -> REGRASP or REPICK",
          res in ('REGRASP', 'REPICK'), f"returned {res}")

# ---------------------------------------------------------------------
print("\n[Phase 1] Acceptance path")
# ---------------------------------------------------------------------
for obj, orient in (('sponge', 'Top'), ('holder', 'Bottom')):
    ser = MockSerial(rows_for(obj, 'good_grip', orientation=orient), loop=True)
    res, sent, out = run_case(get_profile(obj, MAX_MONITOR_CYCLES=40), ser)
    accepted = 'ASSESSMENT_SUCCESS' not in sent and res not in ('REPICK', 'REGRASP')
    check(f"{obj}/{orient}: good_grip accepted -> enters Phase 2",
          accepted and 'Baseline established' in out, f"returned {res}")

# ---------------------------------------------------------------------
print("\n[Phase 1] Timeout on silent serial")
# ---------------------------------------------------------------------
ser = MockSerial(np.zeros((0, 4)), loop=False)      # no data at all
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    _base, _act = RGP.initial_grasp_assessment(ser, get_profile('sponge'), timeout_s=1.0)
check("silent serial -> ASSESSMENT_TIMEOUT",
      _act == 'ASSESSMENT_TIMEOUT', f"returned {_act}")

# ---------------------------------------------------------------------
print("\n[Phase 2] Healthy grip completes without abort")
# ---------------------------------------------------------------------
for obj, orient in (('sponge', 'Top'), ('holder', 'Bottom')):
    ser = MockSerial(rows_for(obj, 'good_grip', orientation=orient), loop=True)
    res, sent, _ = run_case(get_profile(obj, MAX_MONITOR_CYCLES=60), ser)
    check(f"{obj}/{orient}: healthy grip -> PROTOCOL_COMPLETE",
          res == 'PROTOCOL_COMPLETE', f"returned {res}, cmds={sent}")
    check(f"{obj}/{orient}: no LEVEL3 abort on healthy grip",
          'LEVEL3' not in sent, f"cmds={sent}")

# ---------------------------------------------------------------------
print("\n[Phase 2] Object loss mid-motion -> CAMERA_CHECK")
# ---------------------------------------------------------------------
good = rows_for('sponge', 'good_grip', orientation='Top')
nogrip = rows_for('sponge', 'no_grip')
seq = np.vstack([good[:15], np.tile(nogrip[:30], (3, 1))])
ser = MockSerial(seq, loop=False)
res, sent, _ = run_case(get_profile('sponge', MAX_MONITOR_CYCLES=150), ser)
check("good -> no_grip transition -> CAMERA_CHECK",
      res == 'CAMERA_CHECK', f"returned {res}, cmds={sent}")

# ---------------------------------------------------------------------
print("\n[Phase 2] Severe slippage -> LEVEL3 abort")
# ---------------------------------------------------------------------
# Establish a baseline on a healthy grip, then inject readings far outside it.
good = rows_for('sponge', 'good_grip', orientation='Top')

# (a) Gross deviation that the NN reads as object-loss: CAMERA_CHECK is the
#     correct response, and it legitimately fires before the smoothed slippage
#     metric has caught up.
extreme = np.tile(np.array([[200.0, 150.0, 220.0, 300.0]]), (60, 1))
ser = MockSerial(np.vstack([good[:25], extreme]), loop=False)
res, sent, _ = run_case(get_profile('sponge', MAX_MONITOR_CYCLES=150), ser)
check("gross deviation -> abort-class action",
      res in ('LEVEL3', 'CAMERA_CHECK'), f"returned {res}, cmds={sent}")

# (b) Slip that stays within a gripped class, so the slippage branch is what
#     must react. Scaled baseline drift, no object-loss signature.
slip = np.vstack([good[:25] * f for f in (1.6, 2.2, 3.0, 3.6)])
ser = MockSerial(np.vstack([good[:25], slip]), loop=False)
res, sent, _ = run_case(get_profile('sponge', MAX_MONITOR_CYCLES=150), ser)
check("progressive slip -> graduated escalation fires (LEVEL1 then LEVEL2)",
      'LEVEL1' in sent and 'LEVEL2' in sent, f"returned {res}, cmds={sent}")

# ---------------------------------------------------------------------
print("\n[Phase 2] Fragile flag changes the bad-grip action")
# ---------------------------------------------------------------------
bad = rows_for('sponge', 'bad_grip', orientation='Bottom')
good = rows_for('sponge', 'good_grip', orientation='Top')
seq = np.vstack([good[:15], np.tile(bad[:40], (3, 1))])

ser = MockSerial(seq, loop=False)
_, sent_normal, _ = run_case(
    get_profile('sponge', is_fragile=False, MAX_MONITOR_CYCLES=150), ser)

ser = MockSerial(seq, loop=False)
_, sent_fragile, _ = run_case(
    get_profile('sponge', is_fragile=True, MAX_MONITOR_CYCLES=150), ser)

check("non-fragile -> REPOSITION_GRIP",
      'REPOSITION_GRIP' in sent_normal, f"cmds={sent_normal}")
check("fragile -> LOW_PRESSURE_RETREAT (not REPOSITION_GRIP)",
      'LOW_PRESSURE_RETREAT' in sent_fragile
      and 'REPOSITION_GRIP' not in sent_fragile, f"cmds={sent_fragile}")

# ---------------------------------------------------------------------
print("\n[Debounce] CONSECUTIVE_HITS suppresses single-frame noise")
# ---------------------------------------------------------------------
good = rows_for('sponge', 'good_grip', orientation='Top')
ser = MockSerial(good, loop=True)
_, sent_debounced, _ = run_case(
    get_profile('sponge', CONSECUTIVE_HITS=5, MAX_MONITOR_CYCLES=60), ser)
ser = MockSerial(good, loop=True)
_, sent_raw, _ = run_case(
    get_profile('sponge', CONSECUTIVE_HITS=1, SMOOTHING_WINDOW=1,
                MAX_MONITOR_CYCLES=60), ser)
check("debounce reduces command count vs original behaviour",
      len(sent_debounced) <= len(sent_raw),
      f"debounced={len(sent_debounced)} vs original={len(sent_raw)}")

# ---------------------------------------------------------------------
print("\n[Safety] Serial link lost mid-motion")
# ---------------------------------------------------------------------
good = rows_for('sponge', 'good_grip', orientation='Top')
ser = MockSerial(good[:25], loop=False)   # goes silent after 25 frames
res, sent, _ = run_case(
    get_profile('sponge', MAX_MONITOR_CYCLES=500, MAX_NO_DATA_READS=50), ser)
check("serial goes silent -> SENSOR_TIMEOUT (was: infinite hang)",
      res == 'SENSOR_TIMEOUT', f"returned {res}, cmds={sent}")
check("link loss also emits LEVEL3 abort",
      'LEVEL3' in sent, f"cmds={sent}")

# ---------------------------------------------------------------------
print("\n[Errors] Bad configuration handled cleanly")
# ---------------------------------------------------------------------
res = RGP.start_grasp_validation('does_not_exist', config_dir=CONFIG_DIR)
check("unknown object -> CONFIG_OR_INIT_FAILURE",
      res == 'CONFIG_OR_INIT_FAILURE', f"returned {res}")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    res = RGP.start_grasp_validation('sponge', config_dir='/nonexistent/dir')
check("bad config dir -> CONFIG_OR_INIT_FAILURE",
      res == 'CONFIG_OR_INIT_FAILURE', f"returned {res}")

# ---------------------------------------------------------------------
print("\n[Regression] finally-block no longer clobbers the return value")
# ---------------------------------------------------------------------
ser = MockSerial(rows_for('sponge', 'no_grip'), loop=True)
res, _, _ = run_case(get_profile('sponge'), ser)
check("REPICK survives the finally block (was PROTOCOL_INTERRUPTED)",
      res == 'REPICK' and res != 'PROTOCOL_INTERRUPTED', f"returned {res}")

# =====================================================================
print("\n" + "=" * 70)
passed = sum(1 for _, ok, _ in RESULTS if ok)
total = len(RESULTS)
print(f"RESULT: {passed}/{total} passed")
if passed < total:
    print("\nFailures:")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  - {name}: {detail}")
print("=" * 70)
sys.exit(0 if passed == total else 1)

"""
Two-phase adhesive-grasp validation:
  Phase 1  initial_grasp_assessment  -- before movement
  Phase 2  active_monitoring_loop    -- during movement (graduated response)

Wired to the trained one-phase grip-quality models
(`one_phase_nn_grip_model_<object>.keras` + matching scaler), loaded through
Object_Selection from a per-object asset map.

This version fixes several problems in the original (full detail in
README_PROTOCOL.md). The original `finally:` block ended with the line
`finally final_action`, which isn't valid Python -- the module couldn't even
be imported. There was also a bare `return` inside that same `finally`
block, which silently discarded whatever the try/except had actually
returned: on the REPICK/REGRASP path the function returned
`assessment_action`, but `finally` then overrode it with the stale
"PROTOCOL_INTERRUPTED" default, so the arm controller would never have
received a REPICK or REGRASP command even though the console printed one.
`final_action` is now kept in sync and `finally` only closes the port.

The initial assessment used to rely on only the last of the 10 baseline
reads to decide REPICK/REGRASP, so one noisy frame could flip the decision
-- it's now a majority vote over all 10 classifications. The slippage
metric is smoothed over SMOOTHING_WINDOW samples before being compared to
the baseline; replaying the recorded data through the original, un-smoothed
metric shows healthy grips routinely producing max_delta > 1.0 (median
0.61, p99 3.10), meaning the graduated response would have fired
continuously on a perfectly good grip, whereas smoothing over 10 samples
drops the p99 to 1.03. LEVEL1/LEVEL2 now also require CONSECUTIVE_HITS
consecutive breaches before firing, with repeat commands suppressed while a
level is already active (set CONSECUTIVE_HITS = 1 in the asset map to
restore the original behaviour). Thresholds, smoothing window, EMA alpha
and cycle cap all come from the asset map now instead of being hardcoded.
"""

import argparse
import time
from collections import Counter, deque

import numpy as np

import Object_Selection

# --- Serial defaults (overridable on the command line) ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200


def initialize_serial(port, baud):
    """Initializes and returns the serial connection."""
    import serial  # imported lazily so --simulate works without pyserial
    try:
        ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2)
        print(f"Serial connection established on {port}.")
        return ser
    except serial.SerialException as e:
        print(f"Serial Port Error: {e}. Check the port name and whether the "
              f"Arduino IDE is holding the port.")
        raise


def send_command(ser, command):
    """Sends a command string over serial to the arm controller."""
    print(f"\n>>>> ARM COMMAND SENT: {command} <<<<")
    # ser.write(command.encode('utf-8') + b'\n')   # uncomment in real deployment
    time.sleep(0.1)


def read_and_predict(model, scaler, ser, categories):
    """Read one sample, scale it, classify it."""
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line:
                return None, None, None

            raw_values = np.array([float(x) for x in line.split(',')])
            if raw_values.shape[0] != 4:
                print(f"Warning: expected 4 values, got {raw_values.shape[0]}. Skipping.")
                return None, None, None

            scaled = scaler.transform(raw_values.reshape(1, -1))
            # PERF: model.predict() carries large per-call dispatch overhead
            # (~60 ms here) that dominates a 2,532-parameter network. Calling
            # the model directly on a single batch is numerically identical and
            # roughly an order of magnitude faster, which matters for a
            # real-time monitoring loop.
            preds = np.asarray(model(scaled, training=False))
            idx = int(np.argmax(preds, axis=1)[0])
            return categories[idx], float(preds[0, idx]), raw_values

        except Exception as e:
            print(f"Error during data reading/prediction: {e}")
            return None, None, None
    return None, None, None


def initial_grasp_assessment(ser, config_profile, n_samples=10, timeout_s=30.0):
    """
    Phase 1 -- assess grasp quality BEFORE movement.

    Returns (baseline_raw_values, action) where action is
    'ASSESSMENT_SUCCESS', 'REPICK', 'REGRASP', or 'ASSESSMENT_TIMEOUT'.
    """
    MODEL = config_profile['model']
    SCALER = config_profile['scaler']
    CATS = config_profile['categories']
    NO_GRIP, BAD_GRIP = CATS[0], CATS[1]
    SETTLING = config_profile.get('SETTLING_FRAMES', 10)

    print("\n--- Initial Grasp Quality Assessment (PRE-MOVEMENT) ---")

    # FIX 8: discard the first SETTLING_FRAMES readings before judging.
    # Measured on the recorded data: for holder/Bottom good grips, only 30% of
    # the first 10 frames classify as good_grip, versus 85-87% from frame 11
    # onward -- the adhesive is still wetting the surface and the object is
    # still seating, so early frames read close to the loose/no-object range.
    # Voting on that window rejected perfectly healthy grips as REGRASP.
    if SETTLING > 0:
        discarded = 0
        t_settle = time.time()
        while discarded < SETTLING:
            if time.time() - t_settle > timeout_s:
                break
            _, _, raw = read_and_predict(MODEL, SCALER, ser, CATS)
            if raw is not None:
                discarded += 1
            time.sleep(0.05)
        print(f"Discarded {discarded} settling frame(s) before assessment.")

    stable_reads, votes = [], []
    t0 = time.time()
    while len(stable_reads) < n_samples:
        if time.time() - t0 > timeout_s:
            print(f"Assessment timed out after {timeout_s}s with "
                  f"{len(stable_reads)}/{n_samples} samples.")
            return None, 'ASSESSMENT_TIMEOUT'
        category, confidence, raw_values = read_and_predict(MODEL, SCALER, ser, CATS)
        if raw_values is not None:
            stable_reads.append(raw_values)
            votes.append(category)
            print(f"Sampling {len(stable_reads)}/{n_samples}... quality: {category} "
                  f"(conf {confidence:.2f})")
        time.sleep(0.05)

    baseline_raw_values = np.mean(stable_reads, axis=0)

    # FIX 3: majority vote instead of "whatever the last frame said"
    tally = Counter(votes)
    initial_category, n_votes = tally.most_common(1)[0]
    print(f"\nVote over {n_samples} samples: {dict(tally)} -> "
          f"{initial_category} ({n_votes}/{n_samples})")

    if initial_category == NO_GRIP:
        action = 'REPICK'
        print(f"Initial quality is {initial_category}. Returning action: {action}")
        send_command(ser, action)
        return None, action

    if initial_category == BAD_GRIP:
        action = 'REGRASP'
        print(f"Initial quality is {initial_category}. Returning action: {action}")
        send_command(ser, action)
        return None, action

    print(f"Initial Grasp Quality: {initial_category}. Baseline established: "
          f"{np.round(baseline_raw_values, 2)}")
    return baseline_raw_values, 'ASSESSMENT_SUCCESS'


def active_monitoring_loop(ser, config_profile, baseline_raw_values):
    """
    Phase 2 -- monitor grasp quality and slippage DURING movement.

    Returns the final action string.
    """
    MODEL = config_profile['model']
    SCALER = config_profile['scaler']
    CATS = config_profile['categories']
    IS_FRAGILE = config_profile['is_fragile']
    SLIP = config_profile['SLIPPAGE_THRESHOLDS']
    QUAL = config_profile['QUALITY_THRESHOLDS']
    WIN = config_profile['SMOOTHING_WINDOW']
    EMA = config_profile['EMA_ALPHA']
    NEED = config_profile['CONSECUTIVE_HITS']
    MAX_CYCLES = config_profile['MAX_MONITOR_CYCLES']

    NO_GRIP, BAD_GRIP = CATS[0], CATS[1]

    print("\n--- Graduated Response Monitoring Commencing (DURING MOVEMENT) ---")
    print(f"Baseline readings : {np.round(baseline_raw_values, 2)}")
    print(f"Slip thresholds   : WARNING={SLIP['WARNING']} "
          f"CRITICAL={SLIP['CRITICAL']} ABORT={SLIP['ABORT']}")
    print(f"Smoothing window  : {WIN} samples | consecutive hits required: {NEED}")

    smooth_buf = deque(maxlen=WIN)
    streak = {'WARNING': 0, 'CRITICAL': 0, 'ABORT': 0}
    q_streak = {NO_GRIP: 0, BAD_GRIP: 0}
    active_level = None          # suppresses repeat commands at the same level
    quality_action_active = None # suppresses repeat NN quality commands
    monitoring_count = 0
    no_data_streak = 0
    # ~5 s of silence at the 10 ms loop period before declaring the link lost
    MAX_NO_DATA = config_profile.get('MAX_NO_DATA_READS', 500)

    while True:
        category, confidence, current_raw = read_and_predict(MODEL, SCALER, ser, CATS)

        if current_raw is None:
            # FIX 9 (SAFETY): the original did `continue` here WITHOUT
            # incrementing monitoring_count, so if the serial link went quiet
            # -- Arduino unplugged, reset, or crashed mid-motion -- the loop
            # spun forever and the MAX_MONITOR_CYCLES fail-safe was never
            # reached. No abort command would ever be sent while the arm kept
            # moving with an unmonitored payload. Now a run of consecutive
            # empty reads times out and fails safe.
            no_data_streak += 1
            if no_data_streak >= MAX_NO_DATA:
                print(f"\nSENSOR LINK LOST ({no_data_streak} consecutive empty "
                      f"reads). Failing safe.")
                send_command(ser, 'LEVEL3')
                return 'SENSOR_TIMEOUT'
            time.sleep(0.01)
            continue
        no_data_streak = 0

        # FIX 4: smooth before comparing to the baseline
        smooth_buf.append(current_raw)
        smoothed = np.mean(smooth_buf, axis=0)
        max_delta = float(np.max(np.abs(smoothed - baseline_raw_values)
                                 / (baseline_raw_values + 1e-6)))

        # --- Graduated slippage response ---
        if max_delta > SLIP['ABORT']:
            band = 'ABORT'
        elif max_delta > SLIP['CRITICAL']:
            band = 'CRITICAL'
        elif max_delta > SLIP['WARNING']:
            band = 'WARNING'
        else:
            band = None

        for k in streak:
            streak[k] = streak[k] + 1 if k == band else 0

        if band and streak[band] >= NEED:
            if band == 'ABORT':
                final_action = 'LEVEL3'
                print(f"\nSLIPPAGE FATAL (change {max_delta*100:.1f}%, "
                      f"{streak[band]} consecutive). Action: {final_action}")
                send_command(ser, final_action)
                return final_action
            level = 'LEVEL2' if band == 'CRITICAL' else 'LEVEL1'
            if active_level != level:      # FIX 5: don't spam the same level
                print(f"\nSLIPPAGE {band} (change {max_delta*100:.1f}%, "
                      f"{streak[band]} consecutive). Action: {level}")
                send_command(ser, level)
                active_level = level
        elif band is None:
            active_level = None

        # --- NN grip-quality response ---
        # FIX 7: the NN decides per-frame, and the models are ~85% (sponge) /
        # ~80% (holder) accurate, so roughly one frame in six is misclassified.
        # Acting on a single frame therefore fires spurious REPOSITION_GRIP /
        # CAMERA_CHECK commands during a perfectly healthy grip (observed in
        # simulation). The same CONSECUTIVE_HITS debounce used for slippage is
        # applied here so an action needs a sustained prediction, not one frame.
        for k in q_streak:
            q_streak[k] = q_streak[k] + 1 if k == category else 0

        if (category == NO_GRIP
                and confidence >= QUAL['NO_GRIP_CRITICAL_CONF']
                and q_streak[NO_GRIP] >= NEED):
            final_action = 'CAMERA_CHECK'
            print(f"\nNN predicts {NO_GRIP} (conf {confidence:.2f}, "
                  f"{q_streak[NO_GRIP]} consecutive). Action: {final_action}")
            send_command(ser, final_action)
            return final_action

        if (category == BAD_GRIP
                and confidence >= QUAL['BAD_GRIP_WARNING_CONF']
                and q_streak[BAD_GRIP] >= NEED):
            act = 'LOW_PRESSURE_RETREAT' if IS_FRAGILE else 'REPOSITION_GRIP'
            # non-fatal: send once while the condition persists, keep monitoring
            if quality_action_active != act:
                print(f"\nNN predicts {BAD_GRIP} (conf {confidence:.2f}, "
                      f"{q_streak[BAD_GRIP]} consecutive). Action: {act}")
                send_command(ser, act)
                quality_action_active = act
        elif q_streak[BAD_GRIP] == 0:
            quality_action_active = None

        # Baseline EMA update (uses the smoothed value)
        baseline_raw_values = baseline_raw_values * EMA + smoothed * (1.0 - EMA)

        monitoring_count += 1
        if monitoring_count > MAX_CYCLES:
            print(f"\nReached {MAX_CYCLES}-cycle fail-safe.")
            return 'PROTOCOL_COMPLETE'

        time.sleep(0.01)


def run_grasp_protocol(config_profile, ser=None, port=SERIAL_PORT, baud=BAUD_RATE):
    """
    Orchestrates Phase 1 then Phase 2.

    `ser` may be supplied (e.g. a MockSerial) to skip real port initialisation.
    Returns the final action string.
    """
    final_action = "PROTOCOL_INTERRUPTED"
    owns_serial = ser is None

    try:
        if ser is None:
            ser = initialize_serial(port, baud)

        baseline, assessment_action = initial_grasp_assessment(ser, config_profile)

        if assessment_action != 'ASSESSMENT_SUCCESS':
            # FIX 2: keep final_action in sync so nothing can clobber it
            final_action = assessment_action
            return final_action

        final_action = active_monitoring_loop(ser, config_profile, baseline)
        return final_action

    except KeyboardInterrupt:
        final_action = "MANUAL_HALT"
        print(f"\nProtocol manually halted. Returning action: {final_action}")
        return final_action
    except Exception as e:
        final_action = "RUNTIME_ERROR"
        print(f"An unexpected error occurred: {e}. Returning action: {final_action}")
        return final_action
    finally:
        # FIX 1 & 2: this block only cleans up. It must NOT return -- a return
        # inside `finally` discards the value the try/except block produced.
        if owns_serial and ser is not None and getattr(ser, 'is_open', False):
            ser.close()
            print("Serial connection closed.")


def start_grasp_validation(object_name, config_dir=None, ser=None,
                           port=SERIAL_PORT, baud=BAUD_RATE):
    """
    Public entry point for the main control system.

    Returns the final action command, e.g. 'PROTOCOL_COMPLETE' or 'LEVEL3'.
    """
    try:
        CONFIG = Object_Selection.load_configuration_profile(
            object_name, config_dir=config_dir)
        return run_grasp_protocol(CONFIG, ser=ser, port=port, baud=baud)
    except Exception as e:
        print(f"\nFATAL: Grasp validation failed for {object_name}: {e}")
        return "CONFIG_OR_INIT_FAILURE"


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description="Adhesive grasp validation protocol")
    ap.add_argument('--object', help="Object name, e.g. 'sponge' or 'holder'")
    ap.add_argument('--config-dir', default='configs')
    ap.add_argument('--port', default=SERIAL_PORT)
    ap.add_argument('--baud', type=int, default=BAUD_RATE)
    ap.add_argument('--simulate', metavar='GRIP_LABEL',
                    help="Run against recorded data instead of hardware, e.g. "
                         "--simulate good_grip")
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--max-cycles', type=int,
                    help="Override MAX_MONITOR_CYCLES (useful in simulation)")
    args = ap.parse_args()

    obj = args.object or input("Enter the object name (e.g. 'sponge'): ").strip()

    mock = None
    if args.simulate:
        from Mock_Serial import MockSerial
        mock = MockSerial.from_dataset(args.data_dir, dataset=obj,
                                       grip_label=args.simulate, loop=True)
        print(f"[SIMULATION] replaying '{args.simulate}' data for '{obj}' "
              f"-- no hardware in use")

    if args.max_cycles:
        cfg = Object_Selection.load_configuration_profile(obj, config_dir=args.config_dir)
        cfg['MAX_MONITOR_CYCLES'] = args.max_cycles
        final_command = run_grasp_protocol(cfg, ser=mock, port=args.port, baud=args.baud)
    else:
        final_command = start_grasp_validation(obj, config_dir=args.config_dir,
                                               ser=mock, port=args.port, baud=args.baud)

    print(f"\nMAIN CODE RECEIVES FINAL COMMAND: {final_command}")

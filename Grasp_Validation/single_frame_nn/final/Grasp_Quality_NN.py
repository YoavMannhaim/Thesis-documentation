"""
Standalone real-time grip-quality inference: reads photoresistor frames
from the Arduino over serial, classifies each one with the trained
one-phase model, and prints the result plus timing statistics.

This is the simple monitor. For the full two-phase protocol with the
graduated slippage response, use Run_Grasp_Protocol.py.

A few fixes went into this version (full detail in README_PROTOCOL.md): the
scaler used to load at import time from a hardcoded 'fitted_scaler.joblib',
which raised immediately if the file was missing and didn't match any
scaler training actually produced -- loading is lazy now and the path is a
CLI argument. MODEL_PATH was a placeholder ('path/to/your/saved/...') and
GRIP_CATEGORIES were placeholders too ('Category 0' etc); both are now the
real values. `ser` was being pulled out of the `finally` block via
`locals()`, which fails if the serial constructor itself raised, so it's
initialised to None up front instead. Also added --simulate so the pipeline
can be exercised without hardware.
"""

import argparse
import time

import numpy as np

DEFAULT_MODEL = 'one_phase_nn_grip_model_sponge.keras'
DEFAULT_SCALER = 'one_phase_nn_sponge_scaler.joblib'

SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

# Must match the training label order (Data_Loader.GRIP_LABEL_MAP)
GRIP_CATEGORIES = ['no_grip', 'bad_grip', 'semi_good_grip', 'good_grip']


def load_nn_model(path):
    """Loads the trained Keras model ONCE."""
    from tensorflow.keras.models import load_model
    try:
        model = load_model(path)
        print(f"Model loaded successfully: {path}")
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None


def load_scaler(path):
    """Loads the fitted scaler that was saved alongside the model."""
    from joblib import load
    try:
        scaler = load(path)
        print(f"Scaler loaded successfully: {path}")
        return scaler
    except Exception as e:
        print(f"Error loading scaler from {path}: {e}")
        return None


def preprocess_data(raw_data_array, scaler):
    """Converts, scales and reshapes raw sensor data to (1, 4)."""
    try:
        input_data = np.array(raw_data_array, dtype=np.float32).reshape(1, -1)
        if scaler is None:
            # No scaler: the model was trained on standardised inputs, so this
            # path will give poor predictions. Kept only as a last resort.
            print("WARNING: no scaler supplied -- predictions will be unreliable.")
            return input_data
        return scaler.transform(input_data)
    except Exception as e:
        print(f"Error processing data: {e}")
        return None


def run_realtime_inference(model_path=DEFAULT_MODEL, scaler_path=DEFAULT_SCALER,
                           port=SERIAL_PORT, baud=BAUD_RATE, ser=None,
                           max_predictions=None):
    """Main loop: read serial frames, classify, report."""
    start_total = time.time()
    prediction_count = 0
    prediction_times = []
    owns_serial = ser is None

    model = load_nn_model(model_path)
    if model is None:
        return
    scaler = load_scaler(scaler_path)

    try:
        if ser is None:
            import serial
            ser = serial.Serial(port, baud, timeout=0.1)
            time.sleep(2)
            print(f"Listening to Arduino on {port} at {baud} baud...")

        while True:
            if max_predictions and prediction_count >= max_predictions:
                break
            if ser.in_waiting > 0:
                start_pred = time.time()
                line = ser.readline().decode('utf-8').strip()
                if not line:
                    continue

                try:
                    raw_values = [float(x) for x in line.split(',')]
                except ValueError:
                    print(f"Skipping unparsable data: {line}")
                    continue

                if len(raw_values) != 4:
                    print(f"Skipping malformed data: {line}")
                    continue

                scaled = preprocess_data(raw_values, scaler)
                if scaled is None:
                    continue

                predictions = model.predict(scaled, verbose=0)
                idx = int(np.argmax(predictions, axis=1)[0])
                category = GRIP_CATEGORIES[idx]

                pred_time = time.time() - start_pred
                prediction_times.append(pred_time)
                prediction_count += 1
                avg_ms = np.mean(prediction_times) * 1000

                print(f"[{prediction_count:3d}] Photoresistors: {raw_values} -> "
                      f"**{category}** (conf {predictions[0, idx]:.2f}, "
                      f"{pred_time*1000:.1f}ms, avg {avg_ms:.1f}ms)")

    except KeyboardInterrupt:
        print("\n\nReal-time inference stopped.")
    except Exception as e:
        print(f"\nSerial/runtime error: {e}")
    finally:
        total_time = time.time() - start_total
        if prediction_count > 0:
            print("\n" + "=" * 60)
            print("PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Total runtime:        {total_time:.1f} seconds")
            print(f"Total predictions:    {prediction_count}")
            print(f"Avg prediction time:  {np.mean(prediction_times)*1000:.1f} ms")
            print(f"Fastest prediction:   {np.min(prediction_times)*1000:.1f} ms")
            print(f"Slowest prediction:   {np.max(prediction_times)*1000:.1f} ms")
            print(f"Predictions/sec:      {prediction_count/total_time:.1f}")
            print("=" * 60)
        else:
            print(f"\nTotal runtime: {total_time:.1f} seconds (no predictions made)")

        # `ser` is always bound at this point, so no locals() probe is needed.
        if owns_serial and ser is not None and getattr(ser, 'is_open', False):
            ser.close()
            print("Serial port closed.")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--scaler', default=DEFAULT_SCALER)
    ap.add_argument('--port', default=SERIAL_PORT)
    ap.add_argument('--baud', type=int, default=BAUD_RATE)
    ap.add_argument('--simulate', metavar='GRIP_LABEL',
                    help="Replay recorded data instead of using hardware")
    ap.add_argument('--dataset', default='sponge', choices=['sponge', 'holder'])
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--n', type=int, default=None,
                    help="Stop after N predictions")
    a = ap.parse_args()

    mock = None
    if a.simulate:
        from Mock_Serial import MockSerial
        mock = MockSerial.from_dataset(a.data_dir, dataset=a.dataset,
                                       grip_label=a.simulate, loop=True)
        print(f"[SIMULATION] replaying '{a.simulate}' ({a.dataset}) -- no hardware")

    run_realtime_inference(a.model, a.scaler, a.port, a.baud,
                           ser=mock, max_predictions=a.n)

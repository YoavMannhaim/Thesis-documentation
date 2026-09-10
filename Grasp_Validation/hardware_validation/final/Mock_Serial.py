"""
A drop-in stand-in for `serial.Serial` that replays recorded photoresistor
data from the training .xlsx logs instead of reading from an Arduino.

This exists so the grasp protocol can be exercised end-to-end -- including
the real Keras model and scaler -- without any hardware attached. It's a
test harness, not part of the deployed system.

Usage:
    from Mock_Serial import MockSerial
    ser = MockSerial.from_dataset('/path/to/xlsx', dataset='sponge',
                                  grip_label='good_grip')
    # then pass `ser` anywhere a serial.Serial is expected
"""

import numpy as np


class MockSerial:
    def __init__(self, rows, name='MOCK', loop=False):
        """
        rows : (N, 4) array of sensor readings to replay, one per readline().
        loop : if True, wrap around at the end instead of running dry.
        """
        self._rows = np.asarray(rows, dtype=float)
        self._i = 0
        self._loop = loop
        self.is_open = True
        self.name = name

    # --- serial.Serial API surface used by the protocol ---
    @property
    def in_waiting(self):
        if self._i < len(self._rows):
            return 1
        return 1 if self._loop and len(self._rows) else 0

    def readline(self):
        if self._i >= len(self._rows):
            if not self._loop or len(self._rows) == 0:
                return b''
            self._i = 0
        row = self._rows[self._i]
        self._i += 1
        return (",".join(f"{v:.1f}" for v in row) + "\n").encode('utf-8')

    def write(self, data):
        return len(data)

    def close(self):
        self.is_open = False

    # --- constructors ---
    @classmethod
    def from_dataset(cls, data_dir, dataset='sponge', grip_label='good_grip',
                     orientation=None, loop=True, limit=None):
        from Data_Loader import build_dataset, SENSOR_COLS
        df = build_dataset(data_dir, dataset=dataset, verbose=False)
        sub = df[df.grip_label == grip_label]
        if orientation:
            sub = sub[sub.orientation == orientation]
        if sub.empty:
            raise ValueError(f"no rows for {dataset}/{grip_label}/{orientation}")
        rows = sub[SENSOR_COLS].values
        if limit:
            rows = rows[:limit]
        return cls(rows, name=f'MOCK[{dataset}/{grip_label}]', loop=loop)

    @classmethod
    def from_sequence(cls, segments, loop=False):
        """
        Splice several phases together to simulate a grip that degrades, e.g.
            [(good_rows, 40), (bad_rows, 40), (nogrip_rows, 20)]
        """
        parts = [np.asarray(r, dtype=float)[:n] for r, n in segments]
        return cls(np.vstack(parts), name='MOCK[sequence]', loop=loop)

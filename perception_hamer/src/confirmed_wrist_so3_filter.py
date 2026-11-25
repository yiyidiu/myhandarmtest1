"""Recover a sustained wrist turn after the ordinary causal filter rejects a jump."""
import math
from dataclasses import replace

import numpy as np

from .causal_wrist_so3_filter import CausalWristSO3Filter, _so3_exp, _so3_log, project_to_so3


class ConfirmedWristSO3Filter(CausalWristSO3Filter):
    def reset(self):
        super().reset()
        self._recovery = None

    def update(self, timestamp_s, measurement, measurement_quality=1.0):
        previous_time = self._last_timestamp_s
        result = super().update(timestamp_s, measurement, measurement_quality)
        if result.status != "jump_rejected" or result.measurement_quality < .45:
            self._recovery = None
        else:
            timestamp = result.timestamp_s
            dt = timestamp - previous_time
            observed = project_to_so3(measurement)
            last = self._recovery
            # Consecutive raw samples must describe the same new region in SO(3).
            if (last is None or dt > .14 or
                    math.degrees(np.linalg.norm(_so3_log(last[2].T @ observed))) > 8. + 320.*dt):
                self._recovery = (timestamp, 1, observed)
            else:
                self._recovery = (last[0], last[1] + 1, observed)
            start, count, _ = self._recovery
            if count >= 3 and timestamp - start >= .10:
                gain = min(1., 210.*dt/result.innovation_deg)
                innovation = _so3_log(self._filtered.T @ observed)
                self._filtered = project_to_so3(self._filtered @ _so3_exp(gain * innovation))
                self._motion_observations = [(timestamp, observed.copy())]
                result = self._valid(timestamp, "reacquiring_confirmed_turn", result.measurement_quality,
                                     gain, result.innovation_deg)
        return replace(result, large_angle_mode="confirm")

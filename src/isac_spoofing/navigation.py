from __future__ import annotations

import numpy as np

from .config import NavigationConfig


def limit_norm(vectors: np.ndarray, maximum: float) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(norms, 1.0e-12))


class GPSPositionController:
    """Position feedback with PX4 default gains and ideal velocity tracking.

    The controller receives reference position/velocity and the available
    navigation position (GPS or causally propagated ISAC).
    The speed cap is a simulation setting, not PX4's full limit allocation.
    """

    def __init__(self, config: NavigationConfig, dt_s: float):
        config.validate()
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("navigation dt_s must be finite and positive")
        self.config = config
        self.dt_s = float(dt_s)
        self.position_gains_s_inv = np.array([
            config.position_gain_xy_s_inv,
            config.position_gain_xy_s_inv,
            config.position_gain_z_s_inv,
        ], dtype=np.float64)

    def command(
        self,
        reference_position_m: np.ndarray,
        reference_velocity_mps: np.ndarray,
        gps_observed_m: np.ndarray,
    ) -> np.ndarray:
        error = np.asarray(reference_position_m, dtype=np.float64) - np.asarray(
            gps_observed_m, dtype=np.float64
        )
        requested_velocity = (
            np.asarray(reference_velocity_mps, dtype=np.float64)
            + self.position_gains_s_inv * error
        )
        return limit_norm(requested_velocity, self.config.max_speed_mps)

    def advance(
        self,
        position_m: np.ndarray,
        command_velocity_mps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = limit_norm(command_velocity_mps, self.config.max_speed_mps)
        position = np.asarray(position_m, dtype=np.float64) + self.dt_s * velocity
        return position, velocity.copy()

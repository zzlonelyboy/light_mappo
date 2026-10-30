from __future__ import annotations

import numpy as np

from .config import GPSConfig, NavigationConfig
from .navigation import GPSPositionController
from .types import TrajectoryData


def generate_normal_gps_recording(
    reference: TrajectoryData,
    gps_config: GPSConfig,
    navigation_config: NavigationConfig,
    dt_s: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Record one completed, unattacked flight with an independent GPS stream.

    The caller supplies the shared reference and an RNG independent of the
    victim's GPS, attack and sensing streams. With navigation enabled, only
    normal GPS reports drive this recording's own plant. The reference velocity
    must already match the discrete reference path, as in the environment.
    No detector, ISAC observation, or victim state participates in this flight.
    """
    gps_config.validate()
    navigation_config.validate()
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("recording dt_s must be finite and positive")
    reference_position = np.asarray(reference.position_m, dtype=np.float64)
    reference_velocity = np.asarray(reference.velocity_mps, dtype=np.float64)
    if (
        reference_position.ndim != 3
        or reference_position.shape[0] < 1
        or reference_position.shape[1] < 1
        or reference_position.shape[2] != 3
        or reference_velocity.shape != reference_position.shape
        or not np.all(np.isfinite(reference_position))
        or not np.all(np.isfinite(reference_velocity))
    ):
        raise ValueError("recording reference positions and velocities must have shape (T,N,3) and be finite")

    noise = rng.normal(
        0.0, np.asarray(gps_config.noise_std_m), size=reference_position.shape,
    )
    if not navigation_config.enabled:
        recorded = (reference_position + noise).astype(np.float32)
    else:
        controller = GPSPositionController(navigation_config, dt_s)
        position = reference_position[0].copy()
        recorded = np.empty(reference_position.shape, dtype=np.float32)
        for time_index in range(reference_position.shape[0]):
            # Quantize before feedback exactly as in the victim environment.
            recorded[time_index] = (position + noise[time_index]).astype(np.float32)
            command = controller.command(
                reference_position[time_index], reference_velocity[time_index],
                recorded[time_index],
            )
            if time_index + 1 < reference_position.shape[0]:
                position, _ = controller.advance(position, command)

    if not np.all(np.isfinite(recorded)):
        raise FloatingPointError("non-finite normal GPS recording")
    recorded.setflags(write=False)
    return recorded

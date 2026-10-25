from __future__ import annotations

import numpy as np

from .config import ISACConfig
from .types import ISACMeasurement, InterUAVMeasurement


def unit(vectors: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(
        vectors,
        norms,
        out=np.zeros_like(vectors, dtype=np.float64),
        where=norms > eps,
    )


class ISACMeasurementModel:
    """ISAC range, radial-velocity and LoS simulator.

    The equations and tangent-plane direction perturbation follow the current
    Stage2 ISAC implementation, but this class has no Gym or RL dependency.
    """

    LIGHT_SPEED_MPS = 3.0e8

    def __init__(self, config: ISACConfig):
        self.config = config
        self.anchors_m = np.asarray(config.anchors_m, dtype=np.float64)

    def _snr_from_distance(self, distance_m: np.ndarray) -> np.ndarray:
        distance = np.asarray(distance_m, dtype=np.float64)
        if np.any(distance <= 0.0):
            raise ValueError("ISAC sensing distances must be positive")
        wavelength = 3.0e8 / self.config.carrier_hz
        # Converting both Pt and N0 from dBm to SI introduces the same 1e-3
        # factor in numerator and denominator, so it preserves the legacy SNR.
        tx_power_w = 10.0 ** ((self.config.tx_power_dbm - 30.0) / 10.0)
        noise_density_w_hz = 10.0 ** ((-174.0 - 30.0) / 10.0)
        noise_power_w = noise_density_w_hz * self.config.bandwidth_hz
        snr = (
            tx_power_w
            * self.config.radar_cross_section
            * wavelength**2
            / ((4.0 * np.pi) ** 3 * distance**4 * noise_power_w)
        )
        if np.any(~np.isfinite(snr)) or np.any(snr <= 0.0):
            raise FloatingPointError("invalid physical SNR")
        return snr

    def _snr(self, true_position_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(
            self.anchors_m[None, :, :] - true_position_m[:, None, :], axis=-1
        )
        snr = self._snr_from_distance(distance)
        return snr, 10.0 * np.log10(snr)

    def variances_from_range(
        self,
        range_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        distance = np.abs(np.asarray(range_m, dtype=np.float64))
        if np.any(distance <= 0.0):
            raise ValueError("ISAC ranges must be non-zero")
        snr = self._snr_from_distance(distance)
        wavelength = self.LIGHT_SPEED_MPS / self.config.carrier_hz
        range_variance = (
            (self.LIGHT_SPEED_MPS / self.config.bandwidth_hz) ** 2 / snr
        )
        radial_variance = (wavelength / self.config.coherent_time_s) ** 2 / snr
        direction_variance = (wavelength / self.config.effective_aperture_m) ** 2 / snr
        return range_variance, radial_variance, direction_variance, snr

    def measure_inter_uav(
        self,
        true_position_m: np.ndarray,
        rng: np.random.Generator,
    ) -> InterUAVMeasurement:
        """Simulate directed UAV-to-UAV radar range measurements.

        The same carrier, bandwidth, transmit power, and RCS configuration as
        the BS sensing links is used. The diagonal is invalid because a UAV
        does not range itself. Off-diagonal measurements are generated before
        neighbor selection so dynamic neighborhoods use measured—not true—range.
        """

        true_position = np.asarray(true_position_m, dtype=np.float64)
        if true_position.ndim != 2 or true_position.shape[1] != 3:
            raise ValueError("true_position_m must have shape (num_uavs, 3)")
        num_uavs = true_position.shape[0]
        displacement = (
            true_position[:, None, :] - true_position[None, :, :]
        )
        true_range = np.linalg.norm(displacement, axis=-1)
        valid = ~np.eye(num_uavs, dtype=bool)
        if np.any(true_range[valid] <= 0.0):
            raise ValueError("distinct UAVs must not occupy the same position")

        safe_range = np.where(valid, true_range, 1.0)
        snr = self._snr_from_distance(safe_range)
        range_variance = (
            (self.LIGHT_SPEED_MPS / self.config.bandwidth_hz) ** 2 / snr
        )
        range_measured = true_range + rng.normal(size=true_range.shape) * np.sqrt(
            range_variance
        )

        range_measured = np.where(valid, range_measured, 0.0)
        range_variance = np.where(valid, range_variance, 0.0)
        snr = np.where(valid, snr, 0.0)
        snr_db = np.zeros_like(snr)
        snr_db[valid] = 10.0 * np.log10(snr[valid])
        if np.any(~np.isfinite(range_measured[valid])):
            raise FloatingPointError("invalid inter-UAV range measurement")

        return InterUAVMeasurement(
            range_m=range_measured.astype(np.float32),
            valid_mask=valid,
            snr_linear=snr.astype(np.float32),
            snr_db=snr_db.astype(np.float32),
            range_variance=range_variance.astype(np.float32),
        )

    def measure(
        self,
        true_position_m: np.ndarray,
        velocity_mps: np.ndarray,
        rng: np.random.Generator,
    ) -> ISACMeasurement:
        true_position = np.asarray(true_position_m, dtype=np.float64)
        velocity = np.asarray(velocity_mps, dtype=np.float64)
        displacement = self.anchors_m[None, :, :] - true_position[:, None, :]
        true_range = np.linalg.norm(displacement, axis=-1)
        true_los = unit(displacement)
        true_radial = np.sum(velocity[:, None, :] * true_los, axis=-1)

        snr, snr_db = self._snr(true_position)
        wavelength = self.LIGHT_SPEED_MPS / self.config.carrier_hz
        range_variance = (self.LIGHT_SPEED_MPS / self.config.bandwidth_hz) ** 2 / snr
        radial_variance = (wavelength / self.config.coherent_time_s) ** 2 / snr
        direction_variance = (wavelength / self.config.effective_aperture_m) ** 2 / snr

        range_measured = true_range + rng.normal(size=true_range.shape) * np.sqrt(range_variance)
        radial_measured = true_radial + rng.normal(size=true_radial.shape) * np.sqrt(radial_variance)
        raw_noise = rng.normal(size=true_los.shape)
        tangent_noise = raw_noise - np.sum(
            raw_noise * true_los, axis=-1, keepdims=True
        ) * true_los
        los_measured = unit(
            true_los + tangent_noise * np.sqrt(direction_variance)[..., None]
        )

        # Four-BS assumption: every BS-UAV link is always available.  SNR only
        # controls measurement variance and never invalidates a measurement.
        valid = np.ones_like(snr, dtype=bool)
        return ISACMeasurement(
            range_m=range_measured.astype(np.float32),
            los_unit=los_measured.astype(np.float32),
            radial_velocity_mps=radial_measured.astype(np.float32),
            valid_mask=valid,
            snr_linear=snr.astype(np.float32),
            snr_db=snr_db.astype(np.float32),
            range_variance=range_variance.astype(np.float32),
            radial_velocity_variance=radial_variance.astype(np.float32),
            direction_variance=direction_variance.astype(np.float32),
        )

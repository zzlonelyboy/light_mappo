from __future__ import annotations

import numpy as np

from .config import ISACConfig
from .types import ISACMeasurement, InterUAVMeasurement


def unit(vectors: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > eps)


def dbm_to_watts(value: float | np.ndarray) -> np.ndarray:
    return np.power(10.0, (np.asarray(value, dtype=np.float64) - 30.0) / 10.0)


def los_from_angles(azimuth_rad: np.ndarray, elevation_rad: np.ndarray) -> np.ndarray:
    """World-coordinate unit direction from UAV to BS."""
    azimuth, elevation = np.broadcast_arrays(azimuth_rad, elevation_rad)
    return -np.stack((
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ), axis=-1)


def angles_from_los(los_unit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth/elevation of BS-to-UAV direction, opposite to supplied LoS."""
    direction = -unit(los_unit)
    return (
        np.arctan2(direction[..., 1], direction[..., 0]),
        np.arctan2(direction[..., 2], np.linalg.norm(direction[..., :2], axis=-1)),
    )


def delay_doppler_crb(
    received_power_w: np.ndarray,
    bandwidth_hz: float | np.ndarray,
    observation_time_s: float,
    noise_psd_w_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Unknown-gain white-noise bounds under uniform-energy/decoupling assumptions."""
    power = np.asarray(received_power_w, dtype=np.float64)
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
        raise ValueError("received power must be finite and positive")
    if np.any(~np.isfinite(bandwidth)) or np.any(bandwidth <= 0.0):
        raise ValueError("effective waveform bandwidth must be finite and positive")
    if not np.isfinite(observation_time_s) or observation_time_s <= 0.0:
        raise ValueError("observation_time_s must be finite and positive")
    if not np.isfinite(noise_psd_w_hz) or noise_psd_w_hz <= 0.0:
        raise ValueError("noise PSD must be finite and positive")
    numerator = 3.0 * noise_psd_w_hz / (2.0 * np.pi**2 * power)
    return (
        numerator / (observation_time_s * bandwidth**2),
        numerator / observation_time_s**3,
    )


def direction_covariance(
    azimuth_rad: np.ndarray,
    elevation_rad: np.ndarray,
    angle_covariance: np.ndarray,
) -> np.ndarray:
    """Propagate the two-angle covariance into a rank-two LoS covariance."""
    azimuth, elevation = np.broadcast_arrays(azimuth_rad, elevation_rad)
    jacobian = np.zeros(azimuth.shape + (3, 2), dtype=np.float64)
    jacobian[..., 0, 0] = np.cos(elevation) * np.sin(azimuth)
    jacobian[..., 1, 0] = -np.cos(elevation) * np.cos(azimuth)
    jacobian[..., 0, 1] = np.sin(elevation) * np.cos(azimuth)
    jacobian[..., 1, 1] = np.sin(elevation) * np.sin(azimuth)
    jacobian[..., 2, 1] = -np.cos(elevation)
    covariance = np.asarray(angle_covariance, dtype=np.float64)
    if covariance.shape != azimuth.shape + (2, 2):
        raise ValueError("angle covariance shape must equal angle shape + (2, 2)")
    return jacobian @ covariance @ np.swapaxes(jacobian, -1, -2)


class ISACMeasurementModel:
    """Received-power CRB measurement-level model, with separate BS/UAV resources.

    No raw waveform or concrete estimator is simulated. The environment generates
    measurements from true geometry. Processing covariances use reported power
    and noisy angles. Exact power reports are a declared reference assumption.
    """

    LIGHT_SPEED_MPS = 3.0e8
    direction_covariance = staticmethod(direction_covariance)

    def __init__(self, config: ISACConfig):
        config.validate()
        self.config = config
        self.anchors_m = np.asarray(config.anchors_m, dtype=np.float64)
        self.carrier_hz = np.asarray(config.carrier_frequencies_hz, dtype=np.float64)
        self.wavelength_m = self.LIGHT_SPEED_MPS / self.carrier_hz
        self.coverage_halfspace = np.asarray(config.coverage_sides, dtype=np.int8)
        self.noise_psd_w_hz = float(dbm_to_watts(config.thermal_noise_dbm_hz)) * 10.0**(config.noise_figure_db / 10.0)
        self.inter_uav_noise_psd_w_hz = float(dbm_to_watts(config.thermal_noise_dbm_hz)) * 10.0**(config.inter_uav_noise_figure_db / 10.0)
        if not np.isfinite(self.noise_psd_w_hz) or self.noise_psd_w_hz <= 0.0:
            raise ValueError("BS noise density is outside finite positive numerical range")

    def target_bandwidth_hz(self, num_uavs: int) -> float:
        if num_uavs < 1:
            raise ValueError("num_uavs must be positive")
        return self.config.bandwidth_hz / num_uavs

    def received_power_from_range(self, distance_m: np.ndarray, num_uavs: int) -> np.ndarray:
        """Environment-only monostatic link budget, including both array gains."""
        distance = np.asarray(distance_m, dtype=np.float64)
        if distance.shape[-1] != self.anchors_m.shape[0]:
            raise ValueError("distance final dimension must match anchor count")
        if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
            raise ValueError("BS distances must be finite and positive")
        target_power = float(dbm_to_watts(self.config.tx_power_dbm)) / num_uavs
        transmit_count = int(np.prod(self.config.tx_array_shape))
        receive_count = int(np.prod(self.config.rx_array_shape))
        reflection_power = (
            self.wavelength_m**2 * self.config.radar_cross_section
            / ((4.0 * np.pi)**3 * distance**4 * 10.0**(self.config.loss_db / 10.0))
        )
        power = (
            transmit_count * receive_count * target_power
            * reflection_power * self.config.transmit_match_power
        )
        if np.any(~np.isfinite(power)) or np.any(power <= 0.0):
            raise FloatingPointError("BS received power is not finite and positive")
        return power

    def _check_angles(self, azimuth: np.ndarray, elevation: np.ndarray) -> None:
        if np.any(~np.isfinite(azimuth)) or np.any(~np.isfinite(elevation)):
            raise ValueError("sensing angles must be finite")
        if np.any(np.abs(elevation) >= np.pi / 2.0):
            raise ValueError("elevation leaves the non-singular local angular domain")
        if np.any(np.abs(np.sin(elevation)) <= 1.0e-10):
            raise ValueError("horizontal-array elevation is singular at the array plane")
        if np.any(elevation * self.coverage_halfspace <= 0.0):
            raise ValueError(
                "geometry or angular measurement violates the configured known "
                "coverage halfspace; revise deployment, not the truth-derived branch"
            )

    def covariances_from_power(
        self,
        reported_power_w: np.ndarray,
        azimuth_rad: np.ndarray,
        elevation_rad: np.ndarray,
        num_uavs: int | None = None,
    ) -> np.ndarray:
        """Build [range, azimuth, elevation, radial velocity] covariance.

        Input power is a receiver quality report when called by processing.
        Neither true position nor a true reflection coefficient is required.
        """
        power, azimuth, elevation = np.broadcast_arrays(
            np.asarray(reported_power_w, dtype=np.float64),
            np.asarray(azimuth_rad, dtype=np.float64),
            np.asarray(elevation_rad, dtype=np.float64),
        )
        if power.shape[-1] != self.anchors_m.shape[0]:
            raise ValueError("quality arrays must end with the anchor dimension")
        if num_uavs is None:
            if power.ndim < 2:
                raise ValueError("num_uavs is required for quality arrays without a UAV axis")
            num_uavs = power.shape[-2]
        self._check_angles(azimuth, elevation)
        delay_var, doppler_var = delay_doppler_crb(
            power, self.target_bandwidth_hz(num_uavs),
            self.config.coherent_time_s, self.noise_psd_w_hz,
        )
        a = np.pi * np.cos(elevation) * np.sin(azimuth)
        b = np.pi * np.cos(elevation) * np.cos(azimuth)
        r = np.hypot(a, b)
        jacobian = np.empty(power.shape + (2, 2), dtype=np.float64)
        jacobian[..., 0, 0] = b / r**2
        jacobian[..., 0, 1] = -a / r**2
        jacobian[..., 1, 0] = -a / (np.pi * r * np.sin(elevation))
        jacobian[..., 1, 1] = -b / (np.pi * r * np.sin(elevation))
        dim_a, dim_b = self.config.rx_array_shape
        common = 6.0 * self.noise_psd_w_hz / (power * self.config.coherent_time_s)
        spatial = np.zeros(power.shape + (2, 2), dtype=np.float64)
        spatial[..., 0, 0] = common / (dim_a**2 - 1)
        spatial[..., 1, 1] = common / (dim_b**2 - 1)
        angle_covariance = jacobian @ spatial @ np.swapaxes(jacobian, -1, -2)
        covariance = np.zeros(power.shape + (4, 4), dtype=np.float64)
        covariance[..., 0, 0] = (self.LIGHT_SPEED_MPS / 2.0)**2 * delay_var
        covariance[..., 1:3, 1:3] = angle_covariance
        covariance[..., 3, 3] = (self.wavelength_m / 2.0)**2 * doppler_var
        if np.any(~np.isfinite(covariance)):
            raise FloatingPointError("non-finite CRB; sensing geometry is outside the model")
        return covariance

    def variances_from_range(self, range_m: np.ndarray):
        raise ValueError(
            "Measured range alone no longer determines the sensing covariance. "
            "Use covariances_from_power with a receiver power report and noisy angles."
        )

    def measure_inter_uav(
        self,
        true_position_m: np.ndarray,
        rng: np.random.Generator,
        velocity_mps: np.ndarray | None = None,
    ) -> InterUAVMeasurement:
        """One common waveform per transmitting UAV, separate monostatic parameters.

        Each directed range is the separated-target branch of that waveform.
        Gain already includes the coherent receive array. No power, bandwidth
        or observation time is divided again by the number of neighboring echoes.
        """
        position = np.asarray(true_position_m, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or not np.all(np.isfinite(position)):
            raise ValueError("true_position_m must be finite with shape (N, 3)")
        if velocity_mps is not None and (
            np.asarray(velocity_mps).shape != position.shape
            or not np.all(np.isfinite(velocity_mps))
        ):
            raise ValueError("velocity_mps must be finite with shape (N, 3)")
        count = position.shape[0]
        distance = np.linalg.norm(position[:, None] - position[None, :], axis=-1)
        valid = ~np.eye(count, dtype=bool)
        if np.any(distance[valid] <= 0.0):
            raise ValueError("distinct UAVs must not occupy the same position")
        safe_distance = np.where(valid, distance, 1.0)
        wavelength = self.LIGHT_SPEED_MPS / self.config.inter_uav_carrier_hz
        power = (
            float(dbm_to_watts(self.config.inter_uav_tx_power_dbm))
            * self.config.inter_uav_tx_gain * self.config.inter_uav_rx_gain
            * wavelength**2 * self.config.radar_cross_section
            / ((4.0 * np.pi)**3 * safe_distance**4 * 10.0**(self.config.inter_uav_loss_db / 10.0))
        )
        delay_var, _ = delay_doppler_crb(
            power, self.config.inter_uav_bandwidth_hz,
            self.config.inter_uav_observation_time_s, self.inter_uav_noise_psd_w_hz,
        )
        range_variance = (self.LIGHT_SPEED_MPS / 2.0)**2 * delay_var
        measured = distance + rng.normal(size=distance.shape) * np.sqrt(range_variance)
        snr = power / (self.inter_uav_noise_psd_w_hz * self.config.inter_uav_bandwidth_hz)
        snr_db = 10.0 * np.log10(snr)
        measured, range_variance, snr, snr_db, power = (
            np.where(valid, field, 0.0)
            for field in (measured, range_variance, snr, snr_db, power)
        )
        return InterUAVMeasurement(
            range_m=measured.astype(np.float32), valid_mask=valid,
            snr_linear=snr.astype(np.float32), snr_db=snr_db.astype(np.float32),
            range_variance=range_variance.astype(np.float32),
            received_power_w=power.copy(), reported_power_w=power.copy(),
        )

    def measure(
        self,
        true_position_m: np.ndarray,
        velocity_mps: np.ndarray,
        rng: np.random.Generator,
    ) -> ISACMeasurement:
        position = np.asarray(true_position_m, dtype=np.float64)
        velocity = np.asarray(velocity_mps, dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3 or velocity.shape != position.shape:
            raise ValueError("position and velocity must have shape (N, 3)")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("position and velocity must be finite")
        distance_vector = position[:, None, :] - self.anchors_m[None, :, :]
        distance = np.linalg.norm(distance_vector, axis=-1)
        azimuth = np.arctan2(distance_vector[..., 1], distance_vector[..., 0])
        elevation = np.arctan2(distance_vector[..., 2], np.linalg.norm(distance_vector[..., :2], axis=-1))
        self._check_angles(azimuth, elevation)
        true_los = los_from_angles(azimuth, elevation)
        radial = np.sum(velocity[:, None, :] * true_los, axis=-1)
        power = self.received_power_from_range(distance, position.shape[0])
        generated_covariance = self.covariances_from_power(
            power, azimuth, elevation, position.shape[0]
        )
        truth = np.stack((distance, azimuth, elevation, radial), axis=-1)
        noise = np.einsum(
            "...ij,...j->...i", np.linalg.cholesky(generated_covariance),
            rng.normal(size=truth.shape),
        )
        measured = truth + noise
        measured[..., 1] = (measured[..., 1] + np.pi) % (2.0 * np.pi) - np.pi
        # Exact quality report is explicit. It is not an implemented power estimator.
        reported_power = power.copy()
        reported_covariance = self.covariances_from_power(
            reported_power, measured[..., 1], measured[..., 2], position.shape[0]
        )
        los_cov = direction_covariance(
            measured[..., 1], measured[..., 2], reported_covariance[..., 1:3, 1:3]
        )
        snr = reported_power / (
            self.noise_psd_w_hz * self.target_bandwidth_hz(position.shape[0])
        )
        return ISACMeasurement(
            range_m=measured[..., 0].astype(np.float32),
            los_unit=los_from_angles(measured[..., 1], measured[..., 2]).astype(np.float32),
            radial_velocity_mps=measured[..., 3].astype(np.float32),
            valid_mask=np.ones(distance.shape, dtype=bool),
            snr_linear=snr.astype(np.float32), snr_db=(10.0 * np.log10(snr)).astype(np.float32),
            range_variance=reported_covariance[..., 0, 0].astype(np.float32),
            radial_velocity_variance=reported_covariance[..., 3, 3].astype(np.float32),
            direction_variance=(np.trace(los_cov, axis1=-2, axis2=-1) / 2.0).astype(np.float32),
            azimuth_rad=measured[..., 1].copy(), elevation_rad=measured[..., 2].copy(),
            measurement_covariance=generated_covariance,
            reported_covariance=reported_covariance,
            received_power_w=power, reported_power_w=reported_power,
            reported_los_covariance=los_cov,
        )

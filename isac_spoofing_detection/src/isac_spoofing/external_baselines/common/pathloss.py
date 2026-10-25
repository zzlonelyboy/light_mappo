"""Offline A2G path-loss observations for literature baselines.

The deterministic terms follow the 3GPP UMi-AV model used by Dang et al.
Actual observations include the height- and link-state-dependent shadow
fading specified for UMi-AV links. GPS-implied values are theoretical path
losses computed from the reported position and therefore do not contain an
independently invented measurement-error or correlation term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class A2GPathLossConfig:
    carrier_ghz: float = 3.0
    packet_samples: int = 150
    standard_model: str = "3gpp_tr_36_777_umi_av"
    reference_carrier_ghz: float = 2.0
    min_distance_m: float = 1.0
    min_altitude_m: float = 1.5

    def validate(self) -> None:
        if self.carrier_ghz <= 0.0:
            raise ValueError("carrier_ghz must be positive")
        if self.packet_samples < 4:
            raise ValueError("packet_samples must be at least four")
        if self.standard_model != "3gpp_tr_36_777_umi_av":
            raise ValueError("only the 3GPP TR 36.777 UMi-AV model is supported")
        if self.reference_carrier_ghz <= 0.0:
            raise ValueError("reference_carrier_ghz must be positive")
        if self.min_distance_m <= 0.0 or self.min_altitude_m <= 0.0:
            raise ValueError("minimum geometry values must be positive")


def _umi_aerial_terms_db(
    uav_position_m: np.ndarray,
    bs_position_m: np.ndarray,
    config: A2GPathLossConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return LoS probability, LoS/NLoS losses, and UAV height."""

    config.validate()
    uav = np.asarray(uav_position_m, dtype=np.float64)
    bs = np.asarray(bs_position_m, dtype=np.float64)
    if uav.shape[-1] != 3 or bs.shape[-1] != 3:
        raise ValueError("UAV and BS positions must end with three coordinates")

    displacement = uav - bs
    horizontal_distance = np.maximum(
        np.linalg.norm(displacement[..., :2], axis=-1),
        config.min_distance_m,
    )
    distance_3d = np.maximum(
        np.linalg.norm(displacement, axis=-1),
        config.min_distance_m,
    )
    altitude = np.maximum(uav[..., 2], config.min_altitude_m)
    log_altitude = np.log10(altitude)

    d1 = np.maximum(294.05 * log_altitude - 432.94, 18.0)
    q1 = np.maximum(233.98 * log_altitude - 0.95, 1.0e-6)
    los_probability = np.where(
        horizontal_distance <= d1,
        1.0,
        d1 / horizontal_distance
        + np.exp(-horizontal_distance / q1)
        * (1.0 - d1 / horizontal_distance),
    )
    los_probability = np.clip(los_probability, 0.0, 1.0)

    log_distance = np.log10(distance_3d)
    frequency_term = 20.0 * np.log10(config.carrier_ghz)
    reference_loss = (
        15.3
        + 37.6 * log_distance
        + 20.0
        * np.log10(config.carrier_ghz / config.reference_carrier_ghz)
    )
    los_loss = np.maximum(
        reference_loss,
        30.9
        + (22.25 - 0.5 * log_altitude) * log_distance
        + frequency_term,
    )
    nlos_loss = np.maximum(
        los_loss,
        32.4
        + (43.2 - 7.6 * log_altitude) * log_distance
        + frequency_term,
    )
    return los_probability, los_loss, nlos_loss, altitude


def umi_aerial_shadow_fading_std_db(
    altitude_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return 3GPP UMi-AV LoS and NLoS shadow-fading deviations in dB.

    TR 36.777 uses the terrestrial UMi values below or at 22.5 m. Above
    22.5 m, the aerial LoS deviation decreases with height while the NLoS
    deviation is 8 dB.
    """

    altitude = np.asarray(altitude_m, dtype=np.float64)
    if np.any(altitude <= 0.0):
        raise ValueError("altitude_m must be positive")
    low_altitude = altitude <= 22.5
    los_std = np.where(
        low_altitude,
        4.0,
        np.maximum(5.0 * np.exp(-0.01 * altitude), 2.0),
    )
    nlos_std = np.where(low_altitude, 7.82, 8.0)
    return los_std, nlos_std


def expected_a2g_pathloss_db(
    uav_position_m: np.ndarray,
    bs_position_m: np.ndarray,
    config: A2GPathLossConfig,
) -> np.ndarray:
    """Return the expected A2G path loss in dB.

    The implementation reproduces Eqs. (1)-(4) of the Deep Ensemble paper.
    Its reference term is adjusted by 20 log10(fc/2 GHz), which reduces to the
    published expression at the original 2 GHz carrier.
    """

    los_probability, los_loss, nlos_loss, _ = _umi_aerial_terms_db(
        uav_position_m,
        bs_position_m,
        config,
    )
    return los_probability * los_loss + (1.0 - los_probability) * nlos_loss


def sample_actual_and_gps_pathloss(
    true_position_m: np.ndarray,
    gps_position_m: np.ndarray,
    bs_positions_m: np.ndarray,
    rng: np.random.Generator,
    config: A2GPathLossConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate standard-based actual and theoretical path-loss sets.

    Each actual sample draws a LoS/NLoS state from the UMi-AV probability
    and applies the corresponding standard shadow-fading deviation. The
    GPS-implied set repeats the expected theoretical loss at the reported
    position. It contains no artificial measurement noise and has no
    user-selected correlation with the actual observation.
    """

    config.validate()
    true_position = np.asarray(true_position_m, dtype=np.float64)
    gps_position = np.asarray(gps_position_m, dtype=np.float64)
    bs_positions = np.asarray(bs_positions_m, dtype=np.float64)
    if true_position.shape != gps_position.shape:
        raise ValueError("true and GPS positions must have identical shapes")
    if true_position.ndim < 2 or true_position.shape[-1] != 3:
        raise ValueError("position arrays must have shape (...,N,3)")
    if bs_positions.ndim != 2 or bs_positions.shape[1] != 3:
        raise ValueError("bs_positions_m must have shape (M,3)")

    actual_los_probability, actual_los_loss, actual_nlos_loss, altitude = (
        _umi_aerial_terms_db(
            true_position[..., :, None, :],
            bs_positions,
            config,
        )
    )
    gps_mean = expected_a2g_pathloss_db(
        gps_position[..., :, None, :],
        bs_positions,
        config,
    )
    sample_shape = actual_los_probability.shape + (config.packet_samples,)
    los_state = rng.random(sample_shape) < actual_los_probability[..., None]
    los_std, nlos_std = umi_aerial_shadow_fading_std_db(altitude)
    shadow_std = np.where(
        los_state,
        los_std[..., None],
        nlos_std[..., None],
    )
    state_loss = np.where(
        los_state,
        actual_los_loss[..., None],
        actual_nlos_loss[..., None],
    )
    actual = state_loss + shadow_std * rng.standard_normal(sample_shape)
    gps_implied = np.broadcast_to(
        gps_mean[..., None],
        sample_shape,
    ).copy()
    return actual.astype(np.float32), gps_implied.astype(np.float32)

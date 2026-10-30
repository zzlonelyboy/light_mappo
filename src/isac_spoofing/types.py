from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrajectoryData:
    time_s: np.ndarray
    position_m: np.ndarray
    velocity_mps: np.ndarray
    heading: np.ndarray
    trajectory_type: str


@dataclass(frozen=True)
class AttackScenario:
    attack_type: str
    start_step: int
    end_step: int
    target_mask: np.ndarray
    parameters: dict[str, Any]
    nominal_duration_steps: int = 0
    actual_duration_steps: int = 0
    was_truncated: bool = False

    def metadata(self) -> dict[str, Any]:
        return {
            "attack_type": self.attack_type,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "nominal_duration_steps": self.nominal_duration_steps,
            "actual_duration_steps": self.actual_duration_steps,
            "was_truncated": self.was_truncated,
            "target_uavs": np.flatnonzero(self.target_mask).tolist(),
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class ISACMeasurement:
    range_m: np.ndarray
    los_unit: np.ndarray
    radial_velocity_mps: np.ndarray
    valid_mask: np.ndarray
    snr_linear: np.ndarray
    snr_db: np.ndarray
    range_variance: np.ndarray
    radial_velocity_variance: np.ndarray
    direction_variance: np.ndarray
    azimuth_rad: np.ndarray | None = None
    elevation_rad: np.ndarray | None = None
    measurement_covariance: np.ndarray | None = None
    reported_covariance: np.ndarray | None = None
    received_power_w: np.ndarray | None = None
    reported_power_w: np.ndarray | None = None
    reported_los_covariance: np.ndarray | None = None


@dataclass(frozen=True)
class InterUAVMeasurement:
    """Directed UAV-to-UAV range measurements for one time step."""

    range_m: np.ndarray
    valid_mask: np.ndarray
    snr_linear: np.ndarray
    snr_db: np.ndarray
    range_variance: np.ndarray
    received_power_w: np.ndarray | None = None
    reported_power_w: np.ndarray | None = None


@dataclass(frozen=True)
class StepRecord:
    time_index: int
    time_s: float
    true_position_m: np.ndarray
    velocity_mps: np.ndarray
    heading: np.ndarray
    gps_clean_m: np.ndarray
    gps_observed_m: np.ndarray
    isac_position_m: np.ndarray
    isac: ISACMeasurement
    inter_uav: InterUAVMeasurement
    features: np.ndarray
    attack_label: np.ndarray
    reference_position_m: np.ndarray | None = None
    reference_velocity_mps: np.ndarray | None = None
    command_velocity_mps: np.ndarray | None = None
    tracking_error_m: np.ndarray | None = None
    available_time_s: float | None = None
    navigation_mode: np.ndarray | None = None
    navigation_position_m: np.ndarray | None = None
    confirmed_alarm: np.ndarray | None = None
    detection_probability: np.ndarray | None = None


@dataclass(frozen=True)
class EpisodeData:
    time_s: np.ndarray
    true_position_m: np.ndarray
    velocity_mps: np.ndarray
    heading: np.ndarray
    gps_clean_m: np.ndarray
    gps_observed_m: np.ndarray
    isac_position_m: np.ndarray
    isac_range_m: np.ndarray
    isac_los_unit: np.ndarray
    isac_radial_velocity_mps: np.ndarray
    isac_valid_mask: np.ndarray
    isac_snr_db: np.ndarray
    isac_range_variance: np.ndarray
    isac_radial_velocity_variance: np.ndarray
    isac_direction_variance: np.ndarray
    inter_uav_range_m: np.ndarray
    inter_uav_valid_mask: np.ndarray
    inter_uav_snr_db: np.ndarray
    inter_uav_range_variance: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    has_attack: np.ndarray
    attack_start_step: np.ndarray
    attack_end_step_exclusive: np.ndarray
    attack_nominal_duration_steps: np.ndarray
    attack_actual_duration_steps: np.ndarray
    attack_was_truncated: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]
    reference_position_m: np.ndarray | None = None
    reference_velocity_mps: np.ndarray | None = None
    command_velocity_mps: np.ndarray | None = None
    tracking_error_m: np.ndarray | None = None
    available_time_s: np.ndarray | None = None
    navigation_mode: np.ndarray | None = None
    navigation_position_m: np.ndarray | None = None
    confirmed_alarm: np.ndarray | None = None
    detection_probability: np.ndarray | None = None
    isac_azimuth_rad: np.ndarray | None = None
    isac_elevation_rad: np.ndarray | None = None
    isac_received_power_w: np.ndarray | None = None
    isac_reported_power_w: np.ndarray | None = None
    isac_measurement_covariance: np.ndarray | None = None
    isac_reported_covariance: np.ndarray | None = None
    isac_reported_los_covariance: np.ndarray | None = None
    inter_uav_received_power_w: np.ndarray | None = None
    inter_uav_reported_power_w: np.ndarray | None = None
    # Audit-only normal recording. Zeros when metadata.replay_recording.present is false.
    replay_source_gps_m: np.ndarray | None = None

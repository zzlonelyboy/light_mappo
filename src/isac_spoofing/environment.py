from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .config import ProjectConfig
from .features import PaperFeatureExtractor
from .gps_attacks import GPSAttackInjector, generate_clean_gps
from .isac import ISACMeasurementModel
from .localization import RangeDirectionLocalizer
from .metrics import ProbabilityAlarmTracker
from .navigation import GPSPositionController
from .normal_recording import generate_normal_gps_recording
from .trajectories import (
    PredefinedTrajectoryGenerator,
    fixed_formation_neighbor_indices,
)
from .types import EpisodeData, StepRecord, TrajectoryData


class ISACSpoofingEnvironment:
    """Supervised detector simulator with optional GPS navigation feedback."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        self.config.validate()
        self.trajectory_generator = PredefinedTrajectoryGenerator(self.config.simulation)
        self.attack_injector = GPSAttackInjector(
            self.config.attack,
            self.config.simulation.dt_s,
        )
        self.isac_model = ISACMeasurementModel(self.config.isac)
        self.controller = GPSPositionController(
            self.config.navigation, self.config.simulation.dt_s
        )
        localization = self.config.localization
        self.feature_extractor = PaperFeatureExtractor(
            self.config.features,
            self.isac_model,
            localizer=RangeDirectionLocalizer(
                iterations=localization.iterations,
                damping=localization.damping,
                tolerance_m=localization.tolerance_m,
                step_clip_m=localization.step_clip_m,
                line_search_steps=localization.line_search_steps,
                relative_improvement_tolerance=(
                    localization.relative_improvement_tolerance
                ),
                min_anchors=localization.min_anchors,
            ),
            gps_noise_std_m=self.config.gps.noise_std_m,
        )
        self._initialized = False
        self._time_index = 0

    @property
    def done(self) -> bool:
        return self._initialized and self._time_index >= self.config.simulation.num_steps

    def reset(
        self,
        seed: int = 0,
        trajectory_type: str | None = None,
        attack_type: str | None = None,
    ) -> dict[str, object]:
        self._seed = int(seed)
        self._detector_attached = False
        self._rng = np.random.default_rng(self._seed)
        self._trajectory: TrajectoryData = self.trajectory_generator.generate(
            self._rng, trajectory_type=trajectory_type
        )
        # Reference feedforward must match the actual discrete reference path.
        reference_velocity = np.empty_like(self._trajectory.position_m)
        reference_velocity[:-1] = (
            np.diff(self._trajectory.position_m.astype(np.float64), axis=0)
            / self.config.simulation.dt_s
        )
        reference_velocity[-1] = reference_velocity[-2]
        reference_heading = self._command_heading(reference_velocity)
        if self.config.navigation.enabled:
            self._trajectory = replace(
                self._trajectory,
                velocity_mps=reference_velocity,
                heading=reference_heading,
            )
        self._reference = self._trajectory
        if self.config.navigation.enabled:
            # Exogenous white noise may be sampled in advance. No future plant
            # state or future GPS observation is generated here. Keep the
            # legacy draw order so open/closed-loop runs share event schedules.
            self._gps_noise_m = self._rng.normal(
                0.0, np.asarray(self.config.gps.noise_std_m),
                size=self._reference.position_m.shape,
            )
            self._position_m = self._reference.position_m[0].astype(np.float64)
            self._velocity_mps = self._reference.velocity_mps[0].astype(np.float64)
            self._heading = self._reference.heading[0].copy()
            self._gps_clean = np.full_like(self._reference.position_m, np.nan)
            self._gps_observed = np.full_like(self._reference.position_m, np.nan)
            self._labels = np.zeros(self._reference.position_m.shape[:2], dtype=np.uint8)
            scenario_rng = self._rng
        else:
            # Preserve the original random-number order for open-loop ablations.
            self._gps_clean = generate_clean_gps(
                self._trajectory.position_m, self.config.gps, self._rng
            )
            scenario_rng = self._rng
        self._scenario = self.attack_injector.sample(
            self.config.simulation.num_steps,
            self.config.simulation.num_uavs,
            scenario_rng,
            attack_type=attack_type,
        )
        self._prepare_replay_recording()
        if self.config.navigation.enabled:
            self._attack_rng = np.random.default_rng(0)
            self._attack_rng.bit_generator.state = self._rng.bit_generator.state
            if self._scenario.attack_type == "random":
                # Reserve the same random draws as legacy batch injection. The
                # separate injector stream consumes them causally during step.
                self._rng.normal(
                    0.0, float(self._scenario.parameters["sigma_m"]),
                    size=(
                        self._scenario.end_step - self._scenario.start_step,
                        int(np.count_nonzero(self._scenario.target_mask)), 3,
                    ),
                )
        else:
            self._gps_observed, self._labels = self.attack_injector.apply(
                self._gps_clean,
                self._scenario,
                self._rng,
                heading=self._trajectory.heading,
                replay_source_gps_m=(
                    self._replay_source_gps_m if self._replay_recording["present"] else None
                ),
            )
        fixed_neighbors = None
        if self.config.features.neighbor_selection_mode == "fixed_formation":
            fixed_neighbors = fixed_formation_neighbor_indices(
                self.config.simulation.num_uavs,
                self.config.features.num_neighbors,
                formation_type=self.config.simulation.formation_type,
                cube_edge_length_m=self.config.simulation.cube_edge_length_m,
            )
        self.feature_extractor.reset(
            fixed_neighbor_indices=fixed_neighbors,
        )
        self._has_attack = self._scenario.target_mask.astype(bool, copy=True)
        self._attack_start_step = np.full(
            self.config.simulation.num_uavs, -1, dtype=np.int32
        )
        self._attack_end_step_exclusive = np.full(
            self.config.simulation.num_uavs, -1, dtype=np.int32
        )
        self._attack_nominal_duration_steps = np.full(
            self.config.simulation.num_uavs, -1, dtype=np.int32
        )
        self._attack_actual_duration_steps = np.full(
            self.config.simulation.num_uavs, -1, dtype=np.int32
        )
        self._attack_was_truncated = np.zeros(
            self.config.simulation.num_uavs, dtype=bool
        )
        self._attack_start_step[self._has_attack] = self._scenario.start_step
        self._attack_end_step_exclusive[self._has_attack] = self._scenario.end_step
        self._attack_nominal_duration_steps[self._has_attack] = (
            self._scenario.nominal_duration_steps
        )
        self._attack_actual_duration_steps[self._has_attack] = (
            self._scenario.actual_duration_steps
        )
        self._attack_was_truncated[self._has_attack] = self._scenario.was_truncated
        self._navigation_mode = np.zeros(
            self.config.simulation.num_uavs, dtype=np.uint8
        )
        self._previous_isac_position = None
        self._previous_command_velocity = None
        self._last_probability_step = -1
        self._last_confirmed_alarm = np.zeros(
            self.config.simulation.num_uavs, dtype=bool
        )
        self._alarm_tracker = ProbabilityAlarmTracker(
            num_uavs=self.config.simulation.num_uavs,
            confirmation_steps=self.config.navigation.confirmation_steps,
            threshold=self.config.navigation.alarm_threshold,
            warmup_steps=self.config.features.warmup_steps,
        )
        self._time_index = 0
        self._initialized = True
        return self.metadata

    def _prepare_replay_recording(self) -> None:
        """Finish a private normal flight before advancing the victim plant.

        Source GPS uses the same reference and initial formation but a separate
        random stream. No source ISAC data or future victim state is generated.
        """
        self._replay_source_gps_m = np.zeros_like(self._reference.position_m)
        self._replay_source_gps_m.setflags(write=False)
        self._replay_recording: dict[str, object] = {
            "present": False,
            "storage_array": "replay_source_gps_m",
            "absent_array_value": 0.0,
        }
        if self._scenario.attack_type != "replay":
            return
        mode = self._scenario.parameters.setdefault("replay_source", self.config.attack.replay_source)
        if mode != "recorded_normal_episode":
            return
        # A namespaced SeedSequence does not consume or clone the victim RNG.
        recording_seed = int(np.random.SeedSequence(
            [self._seed, 0x5245504C]
        ).generate_state(1, dtype=np.uint64)[0])
        self._replay_source_gps_m = generate_normal_gps_recording(
            self._reference, self.config.gps, self.config.navigation,
            self.config.simulation.dt_s, np.random.default_rng(recording_seed),
        )
        offset = int(self._scenario.parameters["delay_steps"])
        source_id = f"normal_{self._reference.trajectory_type}_victim_{self._seed}_rng_{recording_seed}"
        self._replay_recording = {
            "present": True,
            "source_id": source_id,
            "recording_seed": recording_seed,
            "victim_seed": self._seed,
            "trajectory_type": self._reference.trajectory_type,
            "num_steps": self.config.simulation.num_steps,
            "dt_s": self.config.simulation.dt_s,
            "normal_gps_feedback": self.config.navigation.enabled,
            "source_attack_type": "clean",
            "source_mitigation_enabled": False,
            "shares_reference_and_initial_state": True,
            "recorded_before_victim_rollout": True,
            "gps_noise_reused_without_resampling": True,
            "uav_mapping": "same_formation_index",
            "source_start_step": self._scenario.start_step - offset,
            "source_end_step_exclusive": self._scenario.end_step - offset,
            "offset_semantics": "trajectory_phase_offset",
            "storage_array": "replay_source_gps_m",
            "reuse_policy": "private_to_victim_episode",
        }
        self._scenario.parameters["recording_source_id"] = source_id

    @property
    def metadata(self) -> dict[str, object]:
        if not self._initialized:
            raise RuntimeError("call reset() before requesting metadata")
        return {
            "schema_version": "3.0",
            "seed": self._seed,
            "trajectory_type": self._trajectory.trajectory_type,
            "attack": self._scenario.metadata(),
            "config": self.config.to_dict(),
            "replay_recording": dict(self._replay_recording),
            "measurement_timing": {
                "geometry_reference": "slot_start",
                "result_availability": "slot_end",
                "alarm_availability_offset_steps": 1,
                "peer_stage_duration_s": 0.01,
                "base_station_stage_duration_s": 0.01,
                "geometry_frozen_with_doppler": True,
                "guard_intervals_modelled": False,
            },
            "sensing": {
                "model": "signal_supported_received_power_crb",
                "measurement_level_simulation": True,
                "reported_power_assumption": "exact_receiver_quality_report",
                "receiver_covariance_source": "reported_power_and_measured_angles",
                "crb_is_covariance_reference": True,
                "association": "correct_and_separable_target_echoes",
            },
            "navigation": {
                "enabled": self.config.navigation.enabled,
                "model": (
                    "GPS/ISAC position feedback with PX4 reference XY/Z gains, "
                    "reference-velocity feedforward and ideal velocity tracking"
                    if self.config.navigation.enabled else "predefined open-loop trajectory"
                ),
                "step_order": (
                    "position[t] -> GPS[t] -> injection[t] -> command[t] "
                    "= velocity[t] -> ISAC/features[t] -> slot-end alarm[t] "
                    "-> navigation_mode[t+1], position[t+1]"
                ),
                "trusted_motion_source": (
                    "known issued command under ideal velocity tracking"
                    if self.config.navigation.enabled else
                    "trusted planned-flight/flight-controller motion"
                ),
                "tracking_error_m": "true_position_m - reference_position_m",
                "attack_recovery_policy": "GPS returns to current actual state without resetting the plant",
                "mitigation_enabled": self.config.navigation.mitigation_enabled,
                "confirmation_duration_s": self.config.navigation.confirmation_duration_s(
                    self.config.simulation.dt_s
                ),
                "confirmation_steps": self.config.navigation.confirmation_steps,
                "detector_controls_vehicle": (
                    self.config.navigation.enabled
                    and self.config.navigation.mitigation_enabled
                    and getattr(self, "_detector_attached", False)
                ),
                "collection_mode": (
                    "online_detector_driven_mitigation"
                    if self.config.navigation.enabled
                    and self.config.navigation.mitigation_enabled
                    and getattr(self, "_detector_attached", False)
                    else (
                        "online_detection_without_navigation_mitigation"
                        if getattr(self, "_detector_attached", False)
                        else "shared_gps_feedback_without_alarm_feedback"
                    )
                ),
                "parameters_calibrated_to_airframe": False,
                "position_gain_reference": "PX4 v1.14 position-loop reference gains",
                "velocity_tracking": "ideal, without lag or acceleration limits",
                "speed_limit_model": "simulation-specific 3D norm cap",
                "audit_only_arrays": [
                    "reference_position_m", "reference_velocity_mps",
                    "command_velocity_mps", "tracking_error_m",
                ],
            },
            "label_semantics": "labels[t, i] describes features[t, i] at the same time step",
            "feature_groups": {
                "isac_gps_residual": 6,
                "self": 2,
                "neighbors": len(self.feature_extractor.feature_names) - 8,
            },
            "neighbor_feature_schema": {
                "mode": self.config.features.neighbor_feature_mode,
                "selection_mode": self.config.features.neighbor_selection_mode,
                "feature_names": list(self.feature_extractor.feature_names[8:]),
                "uses_truth_derived_snr_or_variance": False,
                "dynamic_selection_source": (
                    "measured_inter_uav_range"
                    if self.config.features.neighbor_feature_mode == "inter_uav"
                    and self.config.features.neighbor_selection_mode == "dynamic_isac"
                    else None
                ),
            },
            "temporal_feature_policy": (
                "no handcrafted temporal features; temporal dependence is learned "
                "by the sequence detector"
            ),
            "event_semantics": {
                "has_attack": "per-UAV attack-event indicator",
                "attack_start_step": "inclusive; -1 for an unattacked UAV",
                "attack_end_step_exclusive": "exclusive; -1 for an unattacked UAV",
                "attack_nominal_duration_steps": (
                    "sampled complete active length; -1 for an unattacked UAV"
                ),
                "attack_actual_duration_steps": (
                    "observed active length; equal to nominal duration for every "
                    "attacked UAV and -1 for an unattacked UAV"
                ),
                "attack_was_truncated": (
                    "always false for generated attacks; retained for storage "
                    "compatibility with earlier datasets"
                ),
                "attack_onset_sampling": (
                    "sample duration first, then sample the onset uniformly over "
                    "feasible starts independent of detector warmup, so the complete "
                    "event fits; replay also requires a full sampled delay of history"
                ),
                "attack_boundary_policy": "complete_event_within_episode",
                "attack_strength_sampling": (
                    "sample one severity level uniformly, then sample one continuous "
                    "strength uniformly inside its attack-specific interval; the "
                    "sampled strength is fixed for the complete event"
                ),
                "collection_horizon": "the simulator always records the complete episode",
                "training_horizon": "the detector is trained on complete per-UAV sequences",
                "alarm_warmup_steps": self.config.features.warmup_steps,
                "deployment_decision": (
                    "one attack logit is converted by Sigmoid to p(attack); "
                    "p(attack) >= 0.5 produces a hard attack decision, and an "
                    "external K-step confirmation/recovery state machine creates "
                    "re-armable alarms"
                ),
            },
            "detector_information_assumptions": {
                "velocity_mps": (
                    "known issued velocity command, equal to executed velocity under "
                    "ideal tracking; no extra true-velocity sensor channel"
                    if self.config.navigation.enabled else
                    "trusted planned-flight/flight-controller velocity; the predefined "
                    "numerical trajectory follows it exactly"
                ),
                "heading": (
                    "unit direction of issued command; zero vector at zero speed"
                    if self.config.navigation.enabled else
                    "trusted planned-flight/flight-controller heading; the predefined "
                    "numerical trajectory follows it exactly"
                ),
                "true_position_m": "simulation and audit only; never passed to the detector",
                "neighbor_summary": (
                    "direct UAV-to-UAV range measurements are compared with GPS "
                    "pair ranges; only the resulting per-UAV feature summary is "
                    "passed to the detector"
                    if self.config.features.neighbor_feature_mode == "inter_uav"
                    else (
                        "trusted, lossless, delay-free exchange of GPS/ISAC "
                        "consistency summaries using "
                        f"{self.config.features.neighbor_feature_mode!r} features"
                    )
                ),
            },
        }

    @staticmethod
    def _command_heading(command_velocity_mps: np.ndarray) -> np.ndarray:
        velocity = np.asarray(command_velocity_mps, dtype=np.float64)
        speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
        return np.divide(
            velocity, speed, out=np.zeros_like(velocity), where=speed > 0.0
        ).astype(np.float32)

    def apply_detection_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Consume one completed slot's scores and affect only its next slot.

        This interface accepts detector probabilities only. It never consults
        attack labels, true positions, or tracking errors.
        """
        if not self._initialized or self._time_index == 0:
            raise RuntimeError("a completed measurement slot is required first")
        completed_step = self._time_index - 1
        if self._last_probability_step == completed_step:
            raise RuntimeError("probabilities for this slot were already submitted")
        if self._last_probability_step != completed_step - 1:
            raise RuntimeError("submit detector probabilities consecutively from slot zero")
        alarm = self._alarm_tracker.update(probabilities)
        self._last_probability_step = completed_step
        self._last_confirmed_alarm = alarm.copy()
        self._detector_attached = True
        if self.config.navigation.enabled and self.config.navigation.mitigation_enabled:
            self._navigation_mode |= alarm.astype(np.uint8)
        return alarm

    def step(self) -> StepRecord:
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise StopIteration("episode is complete")
        t = self._time_index
        navigation_mode = self._navigation_mode.copy()
        if self.config.navigation.enabled:
            position = self._position_m
            heading = self._command_heading(self._velocity_mps)
            # A stationary vehicle has no velocity direction. The attack
            # injector receives a fixed reference direction for this case.
            attack_heading = np.where(
                (np.linalg.norm(heading, axis=1) > 0.0)[:, None],
                heading,
                self._reference.heading[t],
            )
            stationary = np.linalg.norm(attack_heading, axis=1) == 0.0
            attack_heading[stationary] = [1.0, 0.0, 0.0]
            self._gps_clean[t] = (position + self._gps_noise_m[t]).astype(np.float32)
            # Cache only GPS observations already generated from the plant.
            self._gps_observed[t], self._labels[t] = self.attack_injector.apply_step(
                self._gps_clean[:t + 1], self._scenario, t, self._attack_rng,
                heading=attack_heading,
                replay_source_gps_m=(
                    self._replay_source_gps_m if self._replay_recording["present"] else None
                ),
            )
            navigation_position = self._gps_observed[t].astype(np.float64).copy()
            if np.any(navigation_mode):
                if self._previous_isac_position is None:
                    raise RuntimeError("ISAC navigation requires the previous slot's estimate")
                predicted_position = (
                    self._previous_isac_position
                    + self.config.simulation.dt_s * self._previous_command_velocity
                )
                selected = navigation_mode.astype(bool)
                navigation_position[selected] = predicted_position[selected]
            command_velocity = self.controller.command(
                self._reference.position_m[t], self._reference.velocity_mps[t],
                navigation_position,
            )
            # Ideal inner loop executes the command during interval [t, t+1).
            velocity = command_velocity
            heading = self._command_heading(command_velocity)
            self._heading = heading.copy()
        else:
            position = self._trajectory.position_m[t]
            velocity = self._trajectory.velocity_mps[t]
            heading = self._trajectory.heading[t]
            command_velocity = velocity
            navigation_position = self._gps_observed[t].copy()
        # Acquisition order follows the two stages. Both use the same slot
        # start geometry under the frozen-geometry reference approximation.
        inter_uav_measurement = self.isac_model.measure_inter_uav(
            position,
            self._rng,
        )
        measurement = self.isac_model.measure(position, velocity, self._rng)
        features, isac_position, _ = self.feature_extractor.compute(
            self._gps_observed[t],
            command_velocity,
            heading,
            measurement,
            inter_uav_measurement,
        )
        record = StepRecord(
            time_index=t,
            time_s=float(self._trajectory.time_s[t]),
            true_position_m=position.astype(np.float32).copy(),
            velocity_mps=velocity.astype(np.float32).copy(),
            heading=heading.astype(np.float32).copy(),
            gps_clean_m=self._gps_clean[t].copy(),
            gps_observed_m=self._gps_observed[t].copy(),
            isac_position_m=isac_position.copy(),
            isac=measurement,
            inter_uav=inter_uav_measurement,
            features=features.copy(),
            attack_label=self._labels[t].copy(),
            reference_position_m=self._reference.position_m[t].copy(),
            reference_velocity_mps=self._reference.velocity_mps[t].copy(),
            command_velocity_mps=command_velocity.astype(np.float32).copy(),
            tracking_error_m=(position - self._reference.position_m[t]).astype(np.float32),
            available_time_s=float((t + 1) * self.config.simulation.dt_s),
            navigation_mode=navigation_mode,
            navigation_position_m=navigation_position.astype(np.float32),
            confirmed_alarm=np.zeros(self.config.simulation.num_uavs, dtype=bool),
        )
        self._previous_isac_position = isac_position.astype(np.float64).copy()
        self._previous_command_velocity = command_velocity.astype(np.float64).copy()
        if self.config.navigation.enabled and t + 1 < self.config.simulation.num_steps:
            self._position_m, self._velocity_mps = self.controller.advance(
                position, command_velocity
            )
        self._time_index += 1
        return record

    def run_episode(
        self,
        seed: int = 0,
        trajectory_type: str | None = None,
        attack_type: str | None = None,
        detector: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> EpisodeData:
        self.reset(
            seed=seed,
            trajectory_type=trajectory_type,
            attack_type=attack_type,
        )
        self._detector_attached = detector is not None
        records = []
        for _ in range(self.config.simulation.num_steps):
            record = self.step()
            if detector is not None:
                probabilities = np.asarray(
                    detector(record.features.copy()), dtype=np.float64
                )
                alarm = self.apply_detection_probabilities(probabilities)
                record = replace(
                    record,
                    confirmed_alarm=alarm.copy(),
                    detection_probability=probabilities.astype(np.float32),
                )
            records.append(record)
        return EpisodeData(
            time_s=np.asarray([record.time_s for record in records], dtype=np.float32),
            true_position_m=np.stack([record.true_position_m for record in records]),
            velocity_mps=np.stack([record.velocity_mps for record in records]),
            heading=np.stack([record.heading for record in records]),
            gps_clean_m=np.stack([record.gps_clean_m for record in records]),
            gps_observed_m=np.stack([record.gps_observed_m for record in records]),
            isac_position_m=np.stack([record.isac_position_m for record in records]),
            isac_range_m=np.stack([record.isac.range_m for record in records]),
            isac_los_unit=np.stack([record.isac.los_unit for record in records]),
            isac_radial_velocity_mps=np.stack(
                [record.isac.radial_velocity_mps for record in records]
            ),
            isac_valid_mask=np.stack([record.isac.valid_mask for record in records]),
            isac_snr_db=np.stack([record.isac.snr_db for record in records]),
            isac_range_variance=np.stack(
                [record.isac.range_variance for record in records]
            ),
            isac_radial_velocity_variance=np.stack(
                [record.isac.radial_velocity_variance for record in records]
            ),
            isac_direction_variance=np.stack(
                [record.isac.direction_variance for record in records]
            ),
            inter_uav_range_m=np.stack(
                [record.inter_uav.range_m for record in records]
            ),
            inter_uav_valid_mask=np.stack(
                [record.inter_uav.valid_mask for record in records]
            ),
            inter_uav_snr_db=np.stack(
                [record.inter_uav.snr_db for record in records]
            ),
            inter_uav_range_variance=np.stack(
                [record.inter_uav.range_variance for record in records]
            ),
            features=np.stack([record.features for record in records]),
            labels=np.stack([record.attack_label for record in records]),
            has_attack=self._has_attack.copy(),
            attack_start_step=self._attack_start_step.copy(),
            attack_end_step_exclusive=self._attack_end_step_exclusive.copy(),
            attack_nominal_duration_steps=(
                self._attack_nominal_duration_steps.copy()
            ),
            attack_actual_duration_steps=self._attack_actual_duration_steps.copy(),
            attack_was_truncated=self._attack_was_truncated.copy(),
            feature_names=self.feature_extractor.feature_names,
            metadata=self.metadata,
            reference_position_m=np.stack([record.reference_position_m for record in records]),
            reference_velocity_mps=np.stack([record.reference_velocity_mps for record in records]),
            command_velocity_mps=np.stack([record.command_velocity_mps for record in records]),
            tracking_error_m=np.stack([record.tracking_error_m for record in records]),
            available_time_s=np.asarray([record.available_time_s for record in records]),
            navigation_mode=np.stack([record.navigation_mode for record in records]),
            navigation_position_m=np.stack([record.navigation_position_m for record in records]),
            confirmed_alarm=np.stack([record.confirmed_alarm for record in records]),
            detection_probability=(
                np.stack([record.detection_probability for record in records])
                if detector is not None else None
            ),
            isac_azimuth_rad=np.stack([record.isac.azimuth_rad for record in records]),
            isac_elevation_rad=np.stack([record.isac.elevation_rad for record in records]),
            isac_received_power_w=np.stack([record.isac.received_power_w for record in records]),
            isac_reported_power_w=np.stack([record.isac.reported_power_w for record in records]),
            isac_measurement_covariance=np.stack([record.isac.measurement_covariance for record in records]),
            isac_reported_covariance=np.stack([record.isac.reported_covariance for record in records]),
            isac_reported_los_covariance=np.stack([record.isac.reported_los_covariance for record in records]),
            inter_uav_received_power_w=np.stack([record.inter_uav.received_power_w for record in records]),
            inter_uav_reported_power_w=np.stack([record.inter_uav.reported_power_w for record in records]),
            replay_source_gps_m=self._replay_source_gps_m,
        )

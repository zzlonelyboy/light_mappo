from __future__ import annotations

import numpy as np

from .config import ProjectConfig
from .features import PaperFeatureExtractor
from .gps_attacks import GPSAttackInjector, generate_clean_gps
from .isac import ISACMeasurementModel
from .localization import RangeDirectionLocalizer
from .trajectories import (
    PredefinedTrajectoryGenerator,
    fixed_formation_neighbor_indices,
)
from .types import EpisodeData, StepRecord, TrajectoryData


class ISACSpoofingEnvironment:
    """Action-free full-episode simulator for supervised event detection."""

    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()
        self.config.validate()
        self.trajectory_generator = PredefinedTrajectoryGenerator(self.config.simulation)
        self.attack_injector = GPSAttackInjector(
            self.config.attack,
            self.config.simulation.dt_s,
        )
        self.isac_model = ISACMeasurementModel(self.config.isac)
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
        self._rng = np.random.default_rng(self._seed)
        self._trajectory: TrajectoryData = self.trajectory_generator.generate(
            self._rng, trajectory_type=trajectory_type
        )
        self._gps_clean = generate_clean_gps(
            self._trajectory.position_m, self.config.gps, self._rng
        )
        self._scenario = self.attack_injector.sample(
            self.config.simulation.num_steps,
            self.config.simulation.num_uavs,
            self._rng,
            attack_type=attack_type,
        )
        self._gps_observed, self._labels = self.attack_injector.apply(
            self._gps_clean,
            self._scenario,
            self._rng,
            heading=self._trajectory.heading,
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
        self._time_index = 0
        self._initialized = True
        return self.metadata

    @property
    def metadata(self) -> dict[str, object]:
        if not self._initialized:
            raise RuntimeError("call reset() before requesting metadata")
        return {
            "schema_version": "2.1",
            "seed": self._seed,
            "trajectory_type": self._trajectory.trajectory_type,
            "attack": self._scenario.metadata(),
            "config": self.config.to_dict(),
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
                    "[0, num_steps - nominal_duration_steps], independent of "
                    "detector warmup, so the complete event fits in the episode"
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
                    "external K-step (K in [1,5]) confirmation/recovery state machine creates "
                    "re-armable alarms"
                ),
            },
            "detector_information_assumptions": {
                "velocity_mps": (
                    "trusted planned-flight/flight-controller velocity; the predefined "
                    "numerical trajectory follows it exactly"
                ),
                "heading": (
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

    def step(self) -> StepRecord:
        if not self._initialized:
            raise RuntimeError("call reset() before step()")
        if self.done:
            raise StopIteration("episode is complete")
        t = self._time_index
        measurement = self.isac_model.measure(
            self._trajectory.position_m[t],
            self._trajectory.velocity_mps[t],
            self._rng,
        )
        inter_uav_measurement = self.isac_model.measure_inter_uav(
            self._trajectory.position_m[t],
            self._rng,
        )
        features, isac_position, _ = self.feature_extractor.compute(
            self._gps_observed[t],
            self._trajectory.velocity_mps[t],
            self._trajectory.heading[t],
            measurement,
            inter_uav_measurement,
        )
        record = StepRecord(
            time_index=t,
            time_s=float(self._trajectory.time_s[t]),
            true_position_m=self._trajectory.position_m[t].copy(),
            velocity_mps=self._trajectory.velocity_mps[t].copy(),
            heading=self._trajectory.heading[t].copy(),
            gps_clean_m=self._gps_clean[t].copy(),
            gps_observed_m=self._gps_observed[t].copy(),
            isac_position_m=isac_position.copy(),
            isac=measurement,
            inter_uav=inter_uav_measurement,
            features=features.copy(),
            attack_label=self._labels[t].copy(),
        )
        self._time_index += 1
        return record

    def run_episode(
        self,
        seed: int = 0,
        trajectory_type: str | None = None,
        attack_type: str | None = None,
    ) -> EpisodeData:
        self.reset(
            seed=seed,
            trajectory_type=trajectory_type,
            attack_type=attack_type,
        )
        records = [self.step() for _ in range(self.config.simulation.num_steps)]
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
        )

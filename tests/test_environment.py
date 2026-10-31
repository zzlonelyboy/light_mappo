from __future__ import annotations

from collections import Counter
from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isac_spoofing.config import (
    AttackConfig,
    FeatureConfig,
    GPSConfig,
    ISACConfig,
    NavigationConfig,
    ProjectConfig,
    SimulationConfig,
    load_config,
)
from isac_spoofing.cli.generate_dataset import (
    SCENARIO_TYPES,
    _experiment_assignment,
    _scenario_assignment,
    _split,
)
from isac_spoofing.data import (
    causal_windows,
    save_episode,
    save_episode_shard,
    uav_sequences,
)
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.features import (
    FEATURE_NAMES,
    FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES,
    INTER_UAV_FEATURE_NAMES,
    compute_formation_deformation_neighbor_features,
    compute_inter_uav_consistency_from_measurement,
    compute_neighbor_features_from_positions,
)
from isac_spoofing.gps_attacks import GPSAttackInjector
from isac_spoofing.isac import ISACMeasurementModel
from isac_spoofing.learning_data import (
    FeatureStandardizer,
    load_episode,
    load_episode_shard,
    load_split,
)
from isac_spoofing.localization import RangeDirectionLocalizer
from isac_spoofing.metrics import (
    AttackEvent,
    aggregate_event_results,
    binary_detection_metrics,
    event_detection_metrics,
    first_alarm_steps,
    k_step_alarm_events,
    match_alarm_events,
)
from isac_spoofing.types import AttackScenario
from isac_spoofing.trajectories import (
    fixed_formation_neighbor_indices,
    nominal_formation_offsets,
)


def small_config() -> ProjectConfig:
    return ProjectConfig(
        navigation=NavigationConfig(enabled=False),
        simulation=SimulationConfig(
            num_uavs=4,
            dt_s=0.1,
            duration_s=2.0,
            speed_mps=4.0,
            trajectory_types=("circle",),
            formation_type="compact_ring",
        ),
        gps=GPSConfig(noise_std_m=(0.4, 0.4, 0.6)),
        attack=AttackConfig(
            replay_source="delayed_clean_stream",
            clean_episode_probability=0.0,
            attack_types=("stealth",),
            duration_fraction_range=(0.4, 0.4),
            target_count_range=(2, 2),
            severity_levels=("fixed",),
            random_sigma_ranges_m=((1.0, 1.0),),
            replay_delay_ranges_s=((0.5, 0.5),),
            stealth_rate_ranges_mps=((0.5, 0.5),),
        ),
        isac=ISACConfig(),
        features=FeatureConfig(
            num_neighbors=2,
            neighbor_feature_mode="current",
        ),
    )


class EnvironmentTests(unittest.TestCase):
    def test_legacy_range_multilateration_initializer(self) -> None:
        anchors = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 30.0, 0.0],
                [0.0, 0.0, 40.0],
            ],
            dtype=np.float64,
        )
        true_position = np.asarray([[4.0, 7.0, 9.0]], dtype=np.float64)
        displacement = anchors[None, :, :] - true_position[:, None, :]
        ranges = np.linalg.norm(displacement, axis=2)
        los = displacement / ranges[:, :, None]
        valid = np.ones((1, 4), dtype=bool)
        localizer = RangeDirectionLocalizer(min_anchors=4)

        initial = localizer.initialize(anchors, ranges, los, valid)
        np.testing.assert_allclose(initial, true_position, atol=1.0e-3)

        unrelated_los = np.zeros_like(los)
        unrelated_los[..., 0] = 1.0
        initial_with_other_los = localizer.initialize(
            anchors, ranges, unrelated_los, valid
        )
        np.testing.assert_allclose(initial_with_other_los, initial, atol=0.0)

    def test_legacy_localizer_step_clip_and_descent(self) -> None:
        anchors = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [0.0, 30.0, 0.0],
                [0.0, 0.0, 40.0],
            ],
            dtype=np.float64,
        )
        true_position = np.asarray([[4.0, 7.0, 9.0]], dtype=np.float64)
        displacement = anchors[None, :, :] - true_position[:, None, :]
        ranges = np.linalg.norm(displacement, axis=2)
        los = displacement / ranges[:, :, None]
        valid = np.ones((1, 4), dtype=bool)
        initial = np.asarray([[30.0, 30.0, 30.0]], dtype=np.float64)
        localizer = RangeDirectionLocalizer(
            iterations=1,
            damping=1.0e-5,
            tolerance_m=1.0e-5,
            step_clip_m=10.0,
            line_search_steps=8,
            relative_improvement_tolerance=1.0e-3,
            min_anchors=4,
        )

        initial_cost = localizer._objective(initial[0], anchors, ranges[0], los[0])
        estimated = localizer.estimate(
            anchors,
            ranges,
            los,
            valid,
            initial_position_m=initial,
        )
        final_cost = localizer._objective(estimated[0], anchors, ranges[0], los[0])

        self.assertLessEqual(
            float(np.linalg.norm(estimated[0] - initial[0])),
            10.0 + 1.0e-5,
        )
        self.assertLessEqual(final_cost, initial_cost)

    def test_shapes_label_alignment_and_finite_features(self) -> None:
        config = small_config()
        episode = ISACSpoofingEnvironment(config).run_episode(seed=17)
        steps = config.simulation.num_steps
        self.assertEqual(episode.features.shape, (steps, 4, 11))
        self.assertEqual(
            FEATURE_NAMES,
            (
                "mean_range",
                "mean_radial_velocity",
                "mean_direction",
                "max_range",
                "max_radial_velocity",
                "max_direction",
                "gps_isac_position_error_norm",
                "gps_isac_position_error_heading_projection",
                "signed_self_neighbor_anomaly_contrast",
                "self_neighbor_common_projection",
                "mean_gps_isac_geometry_error",
            ),
        )
        self.assertEqual(episode.feature_names, FEATURE_NAMES)
        self.assertEqual(episode.labels.shape, (steps, 4))
        self.assertEqual(episode.has_attack.shape, (4,))
        self.assertEqual(episode.attack_start_step.shape, (4,))
        self.assertEqual(episode.attack_end_step_exclusive.shape, (4,))

        self.assertEqual(episode.attack_nominal_duration_steps.shape, (4,))
        self.assertEqual(episode.attack_actual_duration_steps.shape, (4,))
        self.assertEqual(episode.attack_was_truncated.shape, (4,))
        self.assertEqual(episode.isac_los_unit.shape, (steps, 4, 4, 3))
        self.assertEqual(episode.inter_uav_range_m.shape, (steps, 4, 4))
        self.assertEqual(episode.inter_uav_valid_mask.shape, (steps, 4, 4))
        self.assertEqual(episode.inter_uav_snr_db.shape, (steps, 4, 4))
        self.assertEqual(episode.inter_uav_range_variance.shape, (steps, 4, 4))
        self.assertFalse(
            np.any(
                np.diagonal(
                    episode.inter_uav_valid_mask,
                    axis1=1,
                    axis2=2,
                )
            )
        )
        self.assertTrue(np.all(episode.isac_valid_mask))
        self.assertTrue(np.all(np.isfinite(episode.features)))
        speed = np.linalg.norm(episode.velocity_mps, axis=2)
        np.testing.assert_allclose(speed, config.simulation.speed_mps, atol=1.0e-5)

        attack = episode.metadata["attack"]
        expected = np.zeros_like(episode.labels)
        expected[
            int(attack["start_step"]) : int(attack["end_step"]),
            np.asarray(attack["target_uavs"], dtype=int),
        ] = 1
        np.testing.assert_array_equal(episode.labels, expected)
        np.testing.assert_array_equal(episode.has_attack, np.any(expected, axis=0))
        np.testing.assert_array_equal(
            episode.attack_start_step[episode.has_attack],
            np.full(np.sum(episode.has_attack), int(attack["start_step"])),
        )
        np.testing.assert_array_equal(
            episode.attack_end_step_exclusive[episode.has_attack],
            np.full(np.sum(episode.has_attack), int(attack["end_step"])),
        )
        self.assertTrue(np.all(episode.attack_start_step[~episode.has_attack] == -1))
        self.assertTrue(
            np.all(episode.attack_end_step_exclusive[~episode.has_attack] == -1)
        )
        self.assertTrue(
            np.all(episode.attack_nominal_duration_steps[~episode.has_attack] == -1)
        )
        self.assertTrue(
            np.all(episode.attack_actual_duration_steps[~episode.has_attack] == -1)
        )
        self.assertFalse(np.any(episode.attack_was_truncated[~episode.has_attack]))
        self.assertTrue(
            np.all(
                episode.attack_actual_duration_steps[episode.has_attack]
                == episode.attack_end_step_exclusive[episode.has_attack]
                - episode.attack_start_step[episode.has_attack]
            )
        )
        changed = np.linalg.norm(episode.gps_observed_m - episode.gps_clean_m, axis=2) > 0.0
        self.assertFalse(np.any(changed[~episode.labels.astype(bool)]))
        self.assertTrue(np.any(changed[episode.labels.astype(bool)]))

    def test_target_centered_neighbor_features_distinguish_local_and_common_bias(
        self,
    ) -> None:
        isac = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ],
            dtype=np.float32,
        )
        common_gps = isac + np.asarray([2.0, 0.0, 0.0], dtype=np.float32)
        common = compute_neighbor_features_from_positions(
            common_gps,
            isac,
            position_scale_m=1.0,
            num_neighbors=3,
        )
        np.testing.assert_allclose(common[:, 0], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(common[:, 1], 2.0, atol=1.0e-6)
        np.testing.assert_allclose(common[:, 2], 0.0, atol=1.0e-6)

        local_gps = isac + np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        local_gps[0] = isac[0] + np.asarray([5.0, 0.0, 0.0], dtype=np.float32)
        local = compute_neighbor_features_from_positions(
            local_gps,
            isac,
            position_scale_m=1.0,
            num_neighbors=3,
        )
        self.assertAlmostEqual(float(local[0, 0]), 4.0, places=6)
        self.assertAlmostEqual(float(local[0, 1]), 5.0, places=6)

    def test_formation_deformation_features_have_physical_invariances(self) -> None:
        isac = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        common_gps = isac + np.asarray([2.0, 3.0, 0.0], dtype=np.float32)
        common = compute_formation_deformation_neighbor_features(
            common_gps,
            isac,
            position_scale_m=1.0,
            num_neighbors=3,
        )
        np.testing.assert_allclose(common, 0.0, atol=1.0e-6)

        local_gps = isac.copy()
        local_gps[0] += np.asarray([2.0, 3.0, 0.0], dtype=np.float32)
        local = compute_formation_deformation_neighbor_features(
            local_gps,
            isac,
            position_scale_m=1.0,
            num_neighbors=3,
        )
        self.assertEqual(
            local.shape,
            (4, len(FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES)),
        )
        self.assertAlmostEqual(float(local[0, 0]), np.sqrt(13.0), places=6)
        self.assertAlmostEqual(float(local[0, 1]), 0.1, places=6)
        self.assertAlmostEqual(float(local[0, 2]), 0.15, places=6)

    def test_environment_supports_formation_deformation_features(self) -> None:
        base = small_config()
        config = replace(
            base,
            features=replace(
                base.features,
                neighbor_feature_mode="formation_deformation",
            ),
        )
        config.validate()
        episode = ISACSpoofingEnvironment(config).run_episode(
            seed=31,
            attack_type="stealth",
        )

        self.assertEqual(
            episode.feature_names[8:],
            FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES,
        )
        self.assertTrue(np.all(np.isfinite(episode.features)))

    def test_environment_directly_generates_inter_uav_features(self) -> None:
        base = small_config()
        config = replace(
            base,
            features=replace(
                base.features,
                neighbor_feature_mode="inter_uav",
            ),
        )
        episode = ISACSpoofingEnvironment(config).run_episode(seed=32)

        self.assertEqual(episode.features.shape, (20, 4, 10))
        self.assertEqual(episode.feature_names[8:], INTER_UAV_FEATURE_NAMES)
        self.assertEqual(
            episode.metadata["neighbor_feature_schema"]["dynamic_selection_source"],
            "measured_inter_uav_range",
        )
        self.assertTrue(np.all(np.isfinite(episode.features)))

    def test_inter_uav_dynamic_neighbors_use_measured_range(self) -> None:
        gps = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        measured = np.asarray(
            [
                [0.0, 100.0, 1.0],
                [100.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        valid = ~np.eye(3, dtype=bool)
        features = compute_inter_uav_consistency_from_measurement(
            gps,
            measured,
            valid,
            num_neighbors=1,
        )

        # UAV 0 selects UAV 2 because its measured range is 1 m, even though
        # UAV 1 is closer in the GPS geometry. Its disparity is |1 - 20| = 19.
        np.testing.assert_allclose(features[0], [19.0, 19.0])

    def test_inter_uav_measurement_uses_configured_rcs(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 20.0, 0.0],
            ],
            dtype=np.float32,
        )
        low_rcs = ISACMeasurementModel(
            ISACConfig(radar_cross_section=3.0)
        ).measure_inter_uav(positions, np.random.default_rng(7))
        high_rcs = ISACMeasurementModel(
            ISACConfig(radar_cross_section=12.0)
        ).measure_inter_uav(positions, np.random.default_rng(7))
        valid = low_rcs.valid_mask

        # SNR scales linearly with RCS and range variance inversely with RCS.
        np.testing.assert_allclose(
            high_rcs.snr_linear[valid] / low_rcs.snr_linear[valid],
            4.0,
            rtol=1.0e-6,
        )
        np.testing.assert_allclose(
            low_rcs.range_variance[valid] / high_rcs.range_variance[valid],
            4.0,
            rtol=1.0e-6,
        )

    def test_fixed_formation_neighbor_indices_are_attack_independent(self) -> None:
        indices = fixed_formation_neighbor_indices(8, 3)

        np.testing.assert_array_equal(
            indices,
            np.asarray(
                [
                    [1, 2, 4],
                    [0, 3, 5],
                    [0, 3, 6],
                    [1, 2, 7],
                    [0, 5, 6],
                    [1, 4, 7],
                    [2, 4, 7],
                    [3, 5, 6],
                ],
                dtype=np.int64,
            ),
        )
        self.assertFalse(
            np.any(indices == np.arange(indices.shape[0])[:, None])
        )

    def test_nominal_cubic_formation_has_eight_meter_edges(self) -> None:
        offsets = nominal_formation_offsets(
            8,
            formation_type="cubic",
            cube_edge_length_m=8.0,
        )
        np.testing.assert_allclose(np.mean(offsets, axis=0), 0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            np.linalg.norm(offsets, axis=1),
            4.0 * np.sqrt(3.0),
            atol=1.0e-12,
        )
        pairwise = np.linalg.norm(
            offsets[:, None, :] - offsets[None, :, :],
            axis=-1,
        )
        nonzero = pairwise[pairwise > 0.0]
        self.assertAlmostEqual(float(np.min(nonzero)), 8.0, places=12)
        self.assertAlmostEqual(
            float(np.max(nonzero)), 8.0 * np.sqrt(3.0), places=12
        )

    def test_environment_supports_fixed_formation_neighbor_selection(self) -> None:
        base = small_config()
        trajectories = ("straight", "circle", "helix", "s_curve")
        config = replace(
            base,
            simulation=replace(
                base.simulation,
                trajectory_types=trajectories,
            ),
            features=replace(
                base.features,
                neighbor_selection_mode="fixed_formation",
            ),
        )
        environment = ISACSpoofingEnvironment(config)
        expected = fixed_formation_neighbor_indices(
            config.simulation.num_uavs,
            config.features.num_neighbors,
            formation_type=config.simulation.formation_type,
            cube_edge_length_m=config.simulation.cube_edge_length_m,
        )

        for trajectory_type in trajectories:
            metadata = environment.reset(
                seed=29,
                trajectory_type=trajectory_type,
                attack_type="stealth",
            )
            np.testing.assert_array_equal(
                environment.feature_extractor._fixed_neighbor_indices,
                expected,
            )
            self.assertEqual(
                metadata["neighbor_feature_schema"]["selection_mode"],
                "fixed_formation",
            )
            episode = environment.run_episode(
                seed=29,
                trajectory_type=trajectory_type,
                attack_type="stealth",
            )
            self.assertTrue(np.all(np.isfinite(episode.features)))

    def test_compact_feature_slices_match_declared_groups(self) -> None:
        environment = ISACSpoofingEnvironment(small_config())
        metadata = environment.reset(seed=23, attack_type="stealth")
        trajectory = environment._trajectory
        measurement = environment.isac_model.measure(
            trajectory.position_m[0],
            trajectory.velocity_mps[0],
            np.random.default_rng(101),
        )
        features, _, diagnostics = environment.feature_extractor.compute(
            environment._gps_observed[0],
            trajectory.velocity_mps[0],
            trajectory.heading[0],
            measurement,
        )

        self.assertEqual(metadata["schema_version"], "3.0")
        self.assertEqual(
            metadata["feature_groups"],
            {"isac_gps_residual": 6, "self": 2, "neighbors": 3},
        )
        np.testing.assert_allclose(features[:, 0:3], diagnostics["residual_mean"])
        np.testing.assert_allclose(features[:, 3:6], diagnostics["residual_max"])
        self.assertNotIn("residual_cusum", diagnostics)
        self.assertNotIn("residual_ewma_error", diagnostics)
        np.testing.assert_allclose(features[:, 6:8], diagnostics["self_features"])
        np.testing.assert_allclose(features[:, 8:11], diagnostics["neighbor_features"])

    def test_duration_first_uniform_feasible_onset_without_truncation(self) -> None:
        config = AttackConfig(
            clean_episode_probability=0.0,
            attack_types=("random",),
            duration_fraction_range=(0.4, 0.4),
            target_count_range=(1, 1),
        )
        injector = GPSAttackInjector(config, dt_s=0.1)
        scenarios = [
            injector.sample(
                num_steps=400,
                num_uavs=8,
                rng=np.random.default_rng(seed),
                attack_type="random",
            )
            for seed in range(64)
        ]

        latest_start = 400 - 161
        self.assertTrue(
            all(0 <= item.start_step <= latest_start for item in scenarios)
        )
        self.assertTrue(all(item.nominal_duration_steps == 161 for item in scenarios))
        self.assertTrue(all(item.actual_duration_steps == 161 for item in scenarios))
        self.assertTrue(any(item.start_step < 24 for item in scenarios))
        self.assertTrue(any(item.start_step > 215 for item in scenarios))
        self.assertTrue(
            all(
                item.end_step == item.start_step + item.nominal_duration_steps
                for item in scenarios
            )
        )
        self.assertTrue(all(item.end_step <= 400 for item in scenarios))
        self.assertTrue(all(not item.was_truncated for item in scenarios))

    def test_full_episode_attack_duration_is_clamped_and_complete(self) -> None:
        config = AttackConfig(
            clean_episode_probability=0.0,
            attack_types=("random",),
            duration_fraction_range=(1.0, 1.0),
            target_count_range=(1, 1),
        )
        scenario = GPSAttackInjector(config, dt_s=0.1).sample(
            num_steps=40,
            num_uavs=8,
            rng=np.random.default_rng(8),
            attack_type="random",
        )

        self.assertEqual(scenario.start_step, 0)
        self.assertEqual(scenario.end_step, 40)
        self.assertEqual(scenario.nominal_duration_steps, 40)
        self.assertEqual(scenario.actual_duration_steps, 40)
        self.assertFalse(scenario.was_truncated)

    def test_attack_strength_is_sampled_once_per_event(self) -> None:
        config = AttackConfig(clean_episode_probability=0.0, replay_source="delayed_clean_stream")
        injector = GPSAttackInjector(config, dt_s=0.1)
        definitions = {
            "random": (config.random_sigma_ranges_m, "sigma_m", "m"),
            "replay": (config.replay_delay_ranges_s, "delay_s", "s"),
            "stealth": (config.stealth_rate_ranges_mps, "rate_mps", "m/s"),
        }
        rng = np.random.default_rng(717)

        for attack_type, (ranges, parameter_name, unit) in definitions.items():
            seen_severities: set[str] = set()
            seen_values: set[float] = set()
            scenario = None
            for _ in range(96):
                scenario = injector.sample(
                    num_steps=400,
                    num_uavs=8,
                    rng=rng,
                    attack_type=attack_type,
                )
                severity = str(scenario.parameters["severity_level"])
                severity_index = config.severity_levels.index(severity)
                strength = float(scenario.parameters["strength_value"])
                low, high = ranges[severity_index]

                self.assertGreaterEqual(strength, low)
                self.assertLessEqual(strength, high)
                self.assertEqual(scenario.parameters["strength_unit"], unit)
                self.assertEqual(strength, scenario.parameters[parameter_name])
                seen_severities.add(severity)
                seen_values.add(round(strength, 8))

            self.assertEqual(seen_severities, set(config.severity_levels))
            self.assertGreater(len(seen_values), len(config.severity_levels))
            self.assertIsNotNone(scenario)

            strength_before_apply = float(scenario.parameters[parameter_name])
            nominal = np.zeros((400, 8, 3), dtype=np.float32)
            heading = np.zeros_like(nominal)
            heading[..., 0] = 1.0
            injector.apply(nominal, scenario, rng, heading=heading)
            self.assertEqual(
                float(scenario.parameters[parameter_name]),
                strength_before_apply,
            )

    def test_seed_reproduces_complete_episode(self) -> None:
        first = ISACSpoofingEnvironment(small_config()).run_episode(seed=91)
        second = ISACSpoofingEnvironment(small_config()).run_episode(seed=91)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.isac_range_m, second.isac_range_m)
        np.testing.assert_array_equal(
            first.inter_uav_range_m,
            second.inter_uav_range_m,
        )

    def test_default_s_curve_moves_from_start_toward_target_at_constant_speed(self) -> None:
        simulation = SimulationConfig()
        trajectory = ISACSpoofingEnvironment(
            ProjectConfig(simulation=simulation)
        ).trajectory_generator.generate(
            np.random.default_rng(7), trajectory_type="s_curve"
        )
        centroid = np.mean(trajectory.position_m, axis=1)
        start = np.asarray(simulation.s_curve_start_m)
        target = np.asarray(simulation.s_curve_target_m)
        displacement = target - start
        lateral = np.asarray(
            [-displacement[1], displacement[0], 0.0], dtype=np.float64
        )
        lateral /= np.linalg.norm(lateral)
        lateral_offset = (centroid - start) @ lateral

        np.testing.assert_allclose(centroid[0], start, atol=1.0e-6)
        distance_to_target = np.linalg.norm(centroid - target, axis=1)
        self.assertLess(float(np.min(distance_to_target)), 2.0e-2)
        self.assertGreater(float(np.max(lateral_offset)), 43.0)
        self.assertLess(float(np.min(lateral_offset)), -43.0)
        np.testing.assert_allclose(
            np.linalg.norm(trajectory.velocity_mps, axis=2),
            simulation.speed_mps,
            atol=1.0e-5,
        )
        relative_position = trajectory.position_m - centroid[:, None, :]
        np.testing.assert_allclose(
            relative_position,
            np.broadcast_to(relative_position[0:1], relative_position.shape),
            atol=2.0e-5,
        )

    def test_all_trajectories_share_the_same_compact_rigid_formation(self) -> None:
        trajectories = ("straight", "circle", "helix", "s_curve")
        simulation = replace(
            SimulationConfig(),
            duration_s=5.0,
            trajectory_types=trajectories,
        )
        generator = ISACSpoofingEnvironment(
            ProjectConfig(simulation=simulation)
        ).trajectory_generator
        reference_offsets = None

        for trajectory_type in trajectories:
            trajectory = generator.generate(
                np.random.default_rng(31415),
                trajectory_type=trajectory_type,
            )
            centroid = np.mean(trajectory.position_m, axis=1)
            relative_position = trajectory.position_m - centroid[:, None, :]

            np.testing.assert_allclose(
                relative_position,
                np.broadcast_to(relative_position[0:1], relative_position.shape),
                atol=2.0e-5,
            )
            np.testing.assert_allclose(
                trajectory.velocity_mps,
                np.broadcast_to(
                    trajectory.velocity_mps[:, 0:1, :],
                    trajectory.velocity_mps.shape,
                ),
                atol=1.0e-6,
            )
            if reference_offsets is None:
                reference_offsets = relative_position[0]
            else:
                np.testing.assert_allclose(
                    relative_position[0],
                    reference_offsets,
                    atol=2.0e-5,
                )

            pairwise_distance = np.linalg.norm(
                relative_position[0, :, None, :]
                - relative_position[0, None, :, :],
                axis=-1,
            )
            self.assertLess(float(np.max(pairwise_distance)), 16.0)

    def test_all_trajectory_and_attack_types(self) -> None:
        base = small_config()
        trajectories = ("straight", "circle", "helix", "s_curve")
        for trajectory in trajectories:
            simulation = replace(base.simulation, trajectory_types=trajectories)
            config = replace(base, simulation=simulation)
            episode = ISACSpoofingEnvironment(config).run_episode(
                seed=31, trajectory_type=trajectory
            )
            speed = np.linalg.norm(episode.velocity_mps, axis=2)
            np.testing.assert_allclose(speed, simulation.speed_mps, atol=1.0e-5)
            self.assertTrue(np.all(np.isfinite(episode.features)))

        for index, attack_type in enumerate(("random", "replay", "stealth")):
            attack = replace(base.attack, attack_types=(attack_type,))
            config = replace(base, attack=attack)
            episode = ISACSpoofingEnvironment(config).run_episode(seed=70 + index)
            changed = np.linalg.norm(
                episode.gps_observed_m - episode.gps_clean_m, axis=2
            ) > 0.0
            self.assertFalse(np.any(changed[~episode.labels.astype(bool)]))
            self.assertTrue(np.any(changed[episode.labels.astype(bool)]))

    def test_attacks_preserve_nominal_gps_noise_semantics(self) -> None:
        nominal = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        target_mask = np.asarray([True, False])
        injector = GPSAttackInjector(AttackConfig(replay_source="delayed_clean_stream"), dt_s=0.1)

        replay = AttackScenario(
            attack_type="replay",
            start_step=2,
            end_step=4,
            target_mask=target_mask,
            parameters={"delay_steps": 1},
        )
        replayed, labels = injector.apply(
            nominal,
            replay,
            np.random.default_rng(1),
        )
        np.testing.assert_array_equal(replayed[2, 0], nominal[1, 0])
        np.testing.assert_array_equal(replayed[3, 0], nominal[2, 0])
        np.testing.assert_array_equal(replayed[:, 1], nominal[:, 1])
        self.assertTrue(np.all(labels[2:4, 0] == 1))

        stealth = AttackScenario(
            attack_type="stealth",
            start_step=1,
            end_step=4,
            target_mask=target_mask,
            parameters={"rate_mps": 1.0},
        )
        heading = np.zeros((4, 2, 3), dtype=np.float32)
        heading[..., 0] = 1.0
        drifted, _ = injector.apply(
            nominal,
            stealth,
            np.random.default_rng(2),
            heading=heading,
        )
        np.testing.assert_array_equal(drifted[1, 0], nominal[1, 0])
        self.assertGreater(float(drifted[2, 0, 0]), float(nominal[2, 0, 0]))

    def test_episode_export_and_causal_windows(self) -> None:
        episode = ISACSpoofingEnvironment(small_config()).run_episode(seed=5)
        with tempfile.TemporaryDirectory() as directory:
            path = save_episode(episode, directory, "episode_00000")
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".json").exists())
            with np.load(path) as stored:
                self.assertEqual(stored["features"].shape[-1], 11)
                np.testing.assert_array_equal(stored["labels"], episode.labels)
                np.testing.assert_array_equal(stored["has_attack"], episode.has_attack)
                np.testing.assert_array_equal(
                    stored["attack_start_step"], episode.attack_start_step
                )
                np.testing.assert_array_equal(
                    stored["attack_end_step_exclusive"],
                    episode.attack_end_step_exclusive,
                )
                np.testing.assert_array_equal(
                    stored["attack_nominal_duration_steps"],
                    episode.attack_nominal_duration_steps,
                )
                np.testing.assert_array_equal(
                    stored["attack_actual_duration_steps"],
                    episode.attack_actual_duration_steps,
                )
                np.testing.assert_array_equal(
                    stored["attack_was_truncated"],
                    episode.attack_was_truncated,
                )
            loaded = load_episode(path)
            standardizer = FeatureStandardizer.fit((loaded,))
            normalized = standardizer.transform(loaded.features)
            self.assertEqual(normalized.shape, loaded.features.shape)
            deliberately_large = loaded.features + 100.0 * standardizer.std
            self.assertGreater(
                float(np.max(standardizer.transform(deliberately_large))), 50.0
            )
        windows = list(causal_windows(episode.features, episode.labels, window_steps=5))
        expected_count = episode.features.shape[1] * (episode.features.shape[0] - 4)
        self.assertEqual(len(windows), expected_count)
        self.assertEqual(windows[0][0].shape, (5, 11))
        sequences = list(uav_sequences(episode.features, episode.labels))
        self.assertEqual(len(sequences), episode.features.shape[1])
        self.assertEqual(sequences[0][0].shape, (episode.features.shape[0], 11))
        self.assertEqual(sequences[0][1].shape, (episode.features.shape[0],))

    def test_episode_shard_round_trip_and_split_loading(self) -> None:
        environment = ISACSpoofingEnvironment(small_config())
        episodes = []
        for index in range(2):
            episode_id = f"episode_{index:05d}"
            episode = environment.run_episode(seed=50 + index)
            episodes.append(
                replace(
                    episode,
                    metadata={**episode.metadata, "episode_id": episode_id},
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            split_directory = Path(directory) / "train"
            path = save_episode_shard(
                episodes,
                split_directory,
                "shard_00000",
            )
            with np.load(path, allow_pickle=False) as stored:
                self.assertEqual(stored["features"].shape, (2, 20, 4, 11))
                self.assertEqual(stored["labels"].shape, (2, 20, 4))
                self.assertEqual(stored["episode_ids"].tolist(), [
                    "episode_00000",
                    "episode_00001",
                ])

            loaded_shard = load_episode_shard(path)
            self.assertEqual(len(loaded_shard), 2)
            np.testing.assert_array_equal(
                loaded_shard[0].features,
                episodes[0].features,
            )
            np.testing.assert_array_equal(
                loaded_shard[1].labels,
                episodes[1].labels,
            )

            loaded_split = load_split(directory, "train")
            self.assertEqual(
                tuple(episode.episode_id for episode in loaded_split),
                ("episode_00000", "episode_00001"),
            )

    def test_default_json_loads(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "configs" / "default.json")
        self.assertEqual(config.simulation.num_uavs, 8)
        self.assertEqual(config.simulation.dt_s, 0.02)
        self.assertEqual(config.simulation.duration_s, 60.0)
        self.assertEqual(config.simulation.speed_mps, 4.0)
        self.assertEqual(config.simulation.trajectory_types, ("s_curve",))
        self.assertEqual(config.simulation.formation_type, "cubic")
        self.assertEqual(config.simulation.cube_edge_length_m, 8.0)
        self.assertEqual(config.simulation.formation_jitter_std_m, 0.5)
        self.assertEqual(config.simulation.s_curve_start_m, (0.0, 0.0, 20.0))
        self.assertEqual(
            config.simulation.s_curve_target_m, (100.0, 100.0, 60.0)
        )
        self.assertEqual(config.simulation.s_curve_lateral_amplitude_m, 44.0)
        self.assertEqual(config.gps.noise_std_m, (1.5, 1.5, 3.0))
        self.assertEqual(config.attack.clean_episode_probability, 0.25)
        self.assertEqual(config.attack.duration_fraction_range, (0.2, 0.4))
        self.assertEqual(config.attack.target_count_range, (1, 4))
        self.assertEqual(config.attack.severity_levels, ("weak", "medium", "strong"))
        self.assertEqual(
            config.attack.random_sigma_ranges_m,
            ((0.25, 0.75), (0.75, 1.5), (1.5, 3.0)),
        )
        self.assertEqual(
            config.attack.replay_delay_ranges_s,
            ((0.2, 0.5), (0.5, 1.5), (1.5, 3.0)),
        )
        self.assertEqual(
            config.attack.stealth_rate_ranges_mps,
            ((0.1, 0.3), (0.3, 0.8), (0.8, 1.2)),
        )
        self.assertEqual(config.to_dict(), ProjectConfig().to_dict())
        self.assertEqual(len(config.isac.anchors_m), 4)
        self.assertEqual(config.isac.carrier_hz, 3.0e9)
        self.assertEqual(config.isac.bandwidth_hz, 40.0e6)
        self.assertEqual(config.isac.tx_power_dbm, 27.0)
        self.assertEqual(config.isac.radar_cross_section, 0.05)
        self.assertEqual(ISACMeasurementModel.LIGHT_SPEED_MPS, 3.0e8)
        self.assertEqual(config.features.position_scale_m, 1.0)
        self.assertEqual(config.features.warmup_steps, 0)
        self.assertEqual(config.features.num_neighbors, 3)
        self.assertEqual(config.features.neighbor_feature_mode, "inter_uav")
        self.assertEqual(config.localization.iterations, 50)
        self.assertEqual(config.localization.damping, 1.0e-5)
        self.assertEqual(config.localization.tolerance_m, 1.0e-5)
        self.assertEqual(config.localization.step_clip_m, 10.0)
        self.assertEqual(config.localization.line_search_steps, 8)
        self.assertEqual(
            config.localization.relative_improvement_tolerance,
            1.0e-3,
        )
        self.assertEqual(config.localization.min_anchors, 4)

    def test_forced_four_scenarios_and_event_metadata(self) -> None:
        base = small_config()
        environment = ISACSpoofingEnvironment(
            replace(
                base,
                attack=replace(
                    base.attack,
                    attack_types=("random", "replay", "stealth"),
                ),
            )
        )
        for index, scenario_type in enumerate(SCENARIO_TYPES):
            episode = environment.run_episode(
                seed=200 + index,
                attack_type=scenario_type,
            )
            self.assertEqual(
                episode.metadata["attack"]["attack_type"], scenario_type
            )
            if scenario_type == "clean":
                self.assertFalse(np.any(episode.labels))
                self.assertFalse(np.any(episode.has_attack))
                self.assertTrue(np.all(episode.attack_start_step == -1))
                self.assertTrue(np.all(episode.attack_end_step_exclusive == -1))
            else:
                self.assertTrue(np.any(episode.labels))
                np.testing.assert_array_equal(
                    episode.has_attack, np.any(episode.labels.astype(bool), axis=0)
                )

    def test_balanced_scenario_assignment_by_split_and_total(self) -> None:
        total = 100
        assignment = _scenario_assignment(total, seed=20260716)
        self.assertEqual(set(assignment), set(range(total)))
        total_counts = Counter(assignment.values())
        self.assertEqual(set(total_counts), set(SCENARIO_TYPES))
        self.assertLessEqual(max(total_counts.values()) - min(total_counts.values()), 1)

        split_counts: dict[str, Counter[str]] = {
            "train": Counter(),
            "validation": Counter(),
            "test": Counter(),
        }
        for index, scenario_type in assignment.items():
            split = _split(index, total)
            split_counts[split][scenario_type] += 1
        self.assertEqual(
            {split: sum(counts.values()) for split, counts in split_counts.items()},
            {"train": 80, "validation": 10, "test": 10},
        )
        for counts in split_counts.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_balanced_scenario_trajectory_pairs_by_split(self) -> None:
        trajectories = ("straight", "circle", "helix", "s_curve")
        assignment = _experiment_assignment(160, seed=11, trajectory_types=trajectories)
        self.assertEqual(set(assignment), set(range(160)))
        split_pair_counts: dict[str, Counter[tuple[str, str]]] = {
            "train": Counter(),
            "validation": Counter(),
            "test": Counter(),
        }
        for index, pair in assignment.items():
            split = _split(index, 160)
            split_pair_counts[split][pair] += 1
        self.assertEqual(
            {
                split: sum(counts.values())
                for split, counts in split_pair_counts.items()
            },
            {"train": 128, "validation": 16, "test": 16},
        )
        for counts in split_pair_counts.values():
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_balanced_scenario_trajectory_pairs_in_test_only_mode(self) -> None:
        trajectories = ("straight", "circle", "helix", "s_curve")
        assignment = _experiment_assignment(
            160,
            seed=11,
            trajectory_types=trajectories,
            split_mode="test-only",
        )
        self.assertEqual(set(assignment), set(range(160)))
        self.assertTrue(
            all(
                _split(index, 160, split_mode="test-only") == "test"
                for index in assignment
            )
        )
        pair_counts = Counter(assignment.values())
        self.assertEqual(len(pair_counts), len(SCENARIO_TYPES) * len(trajectories))
        self.assertEqual(set(pair_counts.values()), {10})


class DetectionMetricTests(unittest.TestCase):
    def test_k_step_alarm_confirms_recovers_and_rearms(self) -> None:
        predictions = np.zeros((14, 1), dtype=np.uint8)
        predictions[0:2, 0] = 1  # ignored by warmup
        predictions[4:6, 0] = 1
        predictions[9:11, 0] = 1
        events = k_step_alarm_events(
            predictions,
            confirmation_steps=2,
            warmup_steps=2,
        )
        self.assertEqual(
            [(event.trigger_step, event.end_step_exclusive) for event in events[0]],
            [(5, 8), (10, 13)],
        )

    def test_long_false_alarm_covering_attack_is_fp_and_fn(self) -> None:
        predictions = np.zeros((10, 1), dtype=np.uint8)
        predictions[2:8, 0] = 1
        alarms = k_step_alarm_events(predictions, confirmation_steps=1)
        result = match_alarm_events(
            alarms,
            (AttackEvent(uav=0, start_step=4, end_step_exclusive=7, event_id="a"),),
            num_steps=10,
            dt_s=0.1,
        )
        self.assertEqual(result["true_positives"], 0)
        self.assertEqual(result["false_positives"], 1)
        self.assertEqual(result["false_negatives"], 1)

    def test_rearmed_alarm_matches_later_attack_and_aggregates(self) -> None:
        predictions = np.zeros((12, 1), dtype=np.uint8)
        predictions[1:3, 0] = 1
        predictions[6:8, 0] = 1
        alarms = k_step_alarm_events(predictions, confirmation_steps=2)
        first = match_alarm_events(
            alarms,
            (AttackEvent(uav=0, start_step=5, end_step_exclusive=9, event_id="a"),),
            num_steps=12,
            dt_s=0.1,
        )
        self.assertEqual(first["true_positives"], 1)
        self.assertEqual(first["false_positives"], 1)
        self.assertEqual(first["false_negatives"], 0)
        aggregate = aggregate_event_results(
            (first, first), bootstrap_samples=20, seed=9
        )
        self.assertEqual(aggregate["true_positives"], 2)
        self.assertEqual(aggregate["false_positives"], 2)
        self.assertAlmostEqual(float(aggregate["event_detection_rate"]), 1.0)
        self.assertAlmostEqual(float(aggregate["delay_mean_s"]), 0.2)
        self.assertIn("event_detection_rate", aggregate["confidence_intervals_95"])
        self.assertIn("delay_mean_s", aggregate["confidence_intervals_95"])

    def test_only_first_alarm_per_attack_event_is_counted(self) -> None:
        predictions = np.zeros((20, 1), dtype=np.uint8)
        predictions[2:4, 0] = 1
        predictions[6:8, 0] = 1
        predictions[13:15, 0] = 1
        alarms = k_step_alarm_events(predictions, confirmation_steps=1)
        result = match_alarm_events(
            alarms,
            (
                AttackEvent(uav=0, start_step=2, end_step_exclusive=10, event_id="a"),
                AttackEvent(uav=0, start_step=12, end_step_exclusive=18, event_id="b"),
            ),
            num_steps=20,
            dt_s=0.1,
        )

        self.assertEqual(result["true_positives"], 2)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertEqual(result["num_raw_alarm_events"], 3)
        self.assertEqual(result["num_evaluated_alarm_events"], 2)
        self.assertEqual(result["num_ignored_duplicate_alarms"], 1)
        self.assertEqual(
            [record["status"] for record in result["alarm_events"]],
            ["true_positive", "ignored_duplicate_alarm", "true_positive"],
        )

        aggregate = aggregate_event_results((result, result))
        self.assertEqual(aggregate["true_positives"], 4)
        self.assertEqual(aggregate["false_positives"], 0)
        self.assertEqual(aggregate["num_raw_alarm_events"], 6)
        self.assertEqual(aggregate["num_evaluated_alarm_events"], 4)
        self.assertEqual(aggregate["num_ignored_duplicate_alarms"], 2)

    def test_first_alarm_latches_once_and_respects_warmup(self) -> None:
        predictions = np.zeros((8, 2), dtype=np.uint8)
        predictions[1, 0] = 1
        predictions[4:7, 1] = 1
        np.testing.assert_array_equal(first_alarm_steps(predictions), [1, 4])
        np.testing.assert_array_equal(
            first_alarm_steps(predictions, warmup_steps=2), [-1, 4]
        )

    def test_event_matching_counts_early_late_and_silent_failures(self) -> None:
        predictions = np.zeros((10, 6), dtype=np.uint8)
        predictions[2, 0] = 1  # false alarm on an unattacked UAV
        predictions[5:8, 2] = 1  # one matched event, despite repeated positives
        predictions[1, 3] = 1  # premature false alarm
        predictions[8, 4] = 1  # late false alarm
        has_attack = np.asarray([False, False, True, True, True, True])
        starts = np.asarray([-1, -1, 3, 3, 3, 3])
        ends = np.asarray([-1, -1, 7, 7, 7, 7])

        result = event_detection_metrics(
            predictions,
            has_attack,
            starts,
            ends,
            dt_s=0.1,
        )
        self.assertEqual(result["num_detected_events"], 1)
        self.assertEqual(result["num_missed_events"], 3)
        self.assertEqual(result["num_false_alarms"], 3)
        self.assertEqual(result["num_predicted_alarms"], 4)
        self.assertAlmostEqual(float(result["event_precision"]), 0.25)
        self.assertAlmostEqual(float(result["event_recall"]), 0.25)
        self.assertAlmostEqual(float(result["event_f1"]), 0.25)
        self.assertAlmostEqual(float(result["delay_mean_s"]), 0.2)
        self.assertAlmostEqual(float(result["delay_median_s"]), 0.2)
        self.assertEqual(result["event_status"].tolist().count("detected"), 1)

    def test_step_metrics_require_hard_binary_predictions(self) -> None:
        result = binary_detection_metrics([0, 0, 1, 1], [0, 1, 1, 1])
        self.assertAlmostEqual(result["precision"], 2.0 / 3.0)
        self.assertEqual(result["recall"], 1.0)
        with self.assertRaises(ValueError):
            binary_detection_metrics([0, 1], [0.2, 0.8])


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional")
class ModelShapeTests(unittest.TestCase):
    def test_models_return_one_attack_logit(self) -> None:
        import torch

        torch.set_num_threads(1)

        from isac_spoofing.models.torch_models import (
            CausalGRUDetector,
            TimeStepMLP,
        )

        self.assertEqual(TimeStepMLP()(torch.zeros(3, 10)).shape, (3, 1))
        self.assertEqual(
            CausalGRUDetector()(torch.zeros(3, 5, 10)).shape,
            (3, 5, 1),
        )

if __name__ == "__main__":
    unittest.main()

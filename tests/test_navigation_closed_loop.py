from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np

from isac_spoofing.config import GPSConfig, NavigationConfig, ProjectConfig, SimulationConfig, load_config
from isac_spoofing.data import NAVIGATION_ARRAY_FIELDS, save_episode, save_episode_shard
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.gps_attacks import GPSAttackInjector
from isac_spoofing.learning_data import load_episode
from isac_spoofing.navigation import GPSPositionController
from isac_spoofing.raw_episode import load_raw_episode_file
from isac_spoofing.types import AttackScenario, TrajectoryData


def make_environment(scenario: AttackScenario) -> ISACSpoofingEnvironment:
    config = ProjectConfig(attack=replace(ProjectConfig().attack, replay_source="delayed_clean_stream"), gps=GPSConfig(noise_std_m=(0., 0., 0.)), simulation=SimulationConfig(
        num_uavs=4, dt_s=0.1, duration_s=16.0, speed_mps=2.0,
        trajectory_types=("straight",), formation_type="compact_ring",
    ))
    environment = ISACSpoofingEnvironment(config)
    time = np.arange(config.simulation.num_steps) * config.simulation.dt_s
    position = np.zeros((time.size, 4, 3), dtype=np.float32)
    position[..., 0] = 20.0 + 2.0 * time[:, None]
    position[..., 1] = [30.0, 38.0, 46.0, 54.0]
    position[..., 2] = 20.0
    velocity = np.zeros_like(position)
    velocity[..., 0] = 2.0
    heading = np.zeros_like(position)
    heading[..., 0] = 1.0
    reference = TrajectoryData(time.astype(np.float32), position, velocity, heading, "straight")
    environment.trajectory_generator.generate = Mock(return_value=reference)
    environment.attack_injector.sample = Mock(return_value=scenario)
    return environment


def lateral_scenario() -> AttackScenario:
    return AttackScenario(
        attack_type="stealth", start_step=20, end_step=80,
        target_mask=np.array([True, False, False, False]),
        parameters={"rate_mps": 1.0, "directions": [[0.0, 1.0, 0.0]] * 4},
        nominal_duration_steps=60, actual_duration_steps=60,
    )


class NavigationTests(unittest.TestCase):
    def test_px4_position_gains_and_ideal_velocity_tracking(self) -> None:
        config = NavigationConfig()
        self.assertEqual(config.position_gain_xy_s_inv, 0.95)
        self.assertEqual(config.position_gain_z_s_inv, 1.0)
        controller = GPSPositionController(config, 0.1)
        position = np.zeros((1, 3))
        command = controller.command(
            position, position, np.array([[-1.0, -2.0, -3.0]]),
        )
        np.testing.assert_allclose(command, [[0.95, 1.9, 3.0]], atol=1e-12)
        next_position, velocity = controller.advance(position, command)
        np.testing.assert_allclose(velocity, command, atol=1e-12)
        np.testing.assert_allclose(next_position, [[0.095, 0.19, 0.3]], atol=1e-12)

    def test_feedback_sign_and_speed_norm_limit(self) -> None:
        config = NavigationConfig()
        controller = GPSPositionController(config, 0.1)
        position = np.zeros((1, 3))
        reference_velocity = np.array([[2.0, 0.0, 0.0]])
        command = controller.command(position, reference_velocity, np.array([[0.0, 4.0, 0.0]]))
        np.testing.assert_allclose(command, [[2.0, -3.8, 0.0]], atol=1e-12)
        p, v = controller.advance(position, command)
        self.assertLess(p[0, 1], 0.0)
        np.testing.assert_allclose(v, command, atol=1e-12)
        for _ in range(10):
            command = controller.command(position, reference_velocity, np.array([[1e6, -1e6, 1e6]]))
            self.assertLessEqual(np.linalg.norm(command), config.max_speed_mps + 1e-12)
            old_position = position.copy()
            position, velocity = controller.advance(position, command)
            self.assertLessEqual(np.linalg.norm(velocity), config.max_speed_mps + 1e-12)
            np.testing.assert_allclose(position - old_position, 0.1 * command, atol=1e-12)

    def test_legacy_navigation_configuration_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_config.json"
            path.write_text(json.dumps({"navigation": {
                "position_gain_s_inv": 0.8,
                "velocity_time_constant_s": 0.5,
                "max_acceleration_mps2": 4.0,
            }}), encoding="utf-8")
            navigation = load_config(path).navigation
            self.assertEqual(navigation.position_gain_xy_s_inv, 0.8)
            self.assertEqual(navigation.position_gain_z_s_inv, 0.8)
            self.assertNotIn("velocity_time_constant_s", ProjectConfig(navigation=navigation).to_dict()["navigation"])
            path.write_text(json.dumps({"navigation": {
                "position_gain_s_inv": 0.8,
                "position_gain_z_s_inv": 1.0,
            }}), encoding="utf-8")
            navigation = load_config(path).navigation
            self.assertEqual(navigation.position_gain_xy_s_inv, 0.8)
            self.assertEqual(navigation.position_gain_z_s_inv, 1.0)

    def test_invalid_navigation_parameters_rejected(self) -> None:
        for name in ("position_gain_xy_s_inv", "position_gain_z_s_inv", "max_speed_mps"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(NavigationConfig(), **{name: value}).validate()
        with self.assertRaises(ValueError):
            replace(NavigationConfig(), enabled="false").validate()
        with self.assertRaises(ValueError):
            ProjectConfig(navigation=NavigationConfig(max_speed_mps=1.0)).validate()

    def test_config_round_trip_and_default_enable(self) -> None:
        config = ProjectConfig()
        self.assertTrue(config.navigation.enabled)
        self.assertFalse(config.navigation.mitigation_enabled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
            self.assertEqual(load_config(path), config)
            path.write_text(json.dumps({"navigation": {"enabled": False}}), encoding="utf-8")
            self.assertFalse(load_config(path).navigation.enabled)

    def test_confirmation_config_migrates_legacy_duration_without_changing_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            legacy = {"simulation": {"dt_s": 0.02}, "navigation": {"confirmation_duration_s": 0.5}}
            path.write_text(json.dumps(legacy), encoding="utf-8")
            migrated = load_config(path)
            self.assertEqual(migrated.navigation.confirmation_steps, 25)
            self.assertNotIn("confirmation_duration_s", migrated.to_dict()["navigation"])
            legacy["navigation"]["confirmation_steps"] = 5
            with self.assertRaisesRegex(ValueError, "conflicts"):
                path.write_text(json.dumps(legacy), encoding="utf-8")
                load_config(path)
            legacy["navigation"]["confirmation_steps"] = 25
            path.write_text(json.dumps(legacy), encoding="utf-8")
            self.assertEqual(load_config(path).navigation.confirmation_steps, 25)
            path.write_text(json.dumps({"simulation": {"dt_s": 0.1}, "navigation": {"confirmation_steps": 5}}), encoding="utf-8")
            current = load_config(path)
            self.assertEqual(current.navigation.confirmation_steps, 5)
            self.assertAlmostEqual(current.navigation.confirmation_duration_s(0.1), 0.5)
        for value in (True, 0, -1, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                replace(NavigationConfig(), confirmation_steps=value).validate()

    def test_lateral_feedback_causality_and_recovery(self) -> None:
        environment = make_environment(lateral_scenario())
        episode = environment.run_episode(seed=123)
        clean = make_environment(AttackScenario(
            "clean", 0, 0, np.zeros(4, dtype=bool), {},
        )).run_episode(seed=123)
        np.testing.assert_array_equal(episode.reference_position_m, clean.reference_position_m)
        np.testing.assert_array_equal(episode.true_position_m[:21], clean.true_position_m[:21])
        self.assertLess(episode.command_velocity_mps[21, 0, 1], 0.0)
        self.assertLess(episode.true_position_m[22, 0, 1], clean.true_position_m[22, 0, 1])
        self.assertGreater(abs(float(episode.tracking_error_m[79, 0, 1])), 1.0)
        self.assertGreater(abs(float(episode.tracking_error_m[80, 0, 1])), 1.0)
        self.assertLess(abs(float(episode.tracking_error_m[-1, 0, 1])), 0.05)
        np.testing.assert_array_equal(episode.gps_observed_m[80:], episode.gps_clean_m[80:])
        np.testing.assert_array_equal(episode.gps_clean_m, episode.true_position_m)
        np.testing.assert_array_equal(episode.true_position_m[:, 1:], clean.true_position_m[:, 1:])
        np.testing.assert_allclose(episode.tracking_error_m, episode.true_position_m - episode.reference_position_m, atol=4e-6)
        np.testing.assert_array_equal(episode.velocity_mps, episode.command_velocity_mps)
        expected_dx = 0.1 * episode.velocity_mps[:-1]
        np.testing.assert_allclose(np.diff(episode.true_position_m, axis=0), expected_dx, atol=4e-6)
        self.assertTrue(np.all(np.isfinite(episode.features)))
        self.assertEqual(episode.features.shape[-1], 10)
        self.assertTrue(np.all(episode.labels[20:80, 0] == 1))
        self.assertFalse(np.any(episode.labels[80:]))

    def test_isac_and_detector_use_current_actual_motion(self) -> None:
        environment = make_environment(lateral_scenario())
        environment.reset(seed=12)
        original_measure = environment.isac_model.measure
        original_inter = environment.isac_model.measure_inter_uav
        original_compute = environment.feature_extractor.compute
        environment.isac_model.measure = Mock(wraps=original_measure)
        environment.isac_model.measure_inter_uav = Mock(wraps=original_inter)
        environment.feature_extractor.compute = Mock(wraps=original_compute)
        for _ in range(45):
            record = environment.step()
        np.testing.assert_allclose(environment.isac_model.measure.call_args.args[0], record.true_position_m, atol=2e-6)
        np.testing.assert_allclose(environment.isac_model.measure.call_args.args[1], record.velocity_mps, atol=1e-6)
        np.testing.assert_allclose(environment.isac_model.measure_inter_uav.call_args.args[0], record.true_position_m, atol=2e-6)
        np.testing.assert_allclose(environment.feature_extractor.compute.call_args.args[1], record.velocity_mps, atol=1e-6)
        self.assertFalse(np.allclose(record.velocity_mps, record.reference_velocity_mps))

    def test_navigation_arrays_round_trip_for_episode_and_shard(self) -> None:
        environment = make_environment(lateral_scenario())
        episode = environment.run_episode(seed=3)
        episode = replace(episode, metadata={**episode.metadata, "episode_id": "navigation"})
        with tempfile.TemporaryDirectory() as directory:
            single = save_episode(episode, directory, "navigation")
            shard = save_episode_shard([episode], directory, "shard")
            for path in (single, shard):
                loaded = load_raw_episode_file(path, array_names=NAVIGATION_ARRAY_FIELDS)[0]
                for name in NAVIGATION_ARRAY_FIELDS:
                    np.testing.assert_array_equal(loaded.require(name), getattr(episode, name))
            self.assertEqual(load_episode(single).features.shape[-1], 10)

    def test_streaming_attacks_match_batch_and_replay_is_causal(self) -> None:
        nominal = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
        heading = np.zeros_like(nominal)
        heading[..., 0] = 1.0
        injector = GPSAttackInjector(replace(ProjectConfig().attack, replay_source="delayed_clean_stream"), 0.1)
        for kind, parameters in (("random", {"sigma_m": 1.0}), ("replay", {"delay_steps": 3}), ("stealth", {"rate_mps": 0.5})):
            start = 3 if kind == "replay" else 1
            scenario = AttackScenario(kind, start, 8, np.array([True, False, True]), dict(parameters))
            batch, batch_labels = injector.apply(nominal, scenario, np.random.default_rng(7), heading)
            scenario = replace(scenario, parameters=dict(parameters))
            rng = np.random.default_rng(7)
            stream, labels = zip(*(injector.apply_step(nominal[:t + 1], scenario, t, rng, heading[t]) for t in range(10)))
            np.testing.assert_allclose(stream, batch, atol=1e-6)
            np.testing.assert_array_equal(labels, batch_labels)
            if kind == "replay":
                np.testing.assert_array_equal(stream[3][[0, 2]], nominal[0, [0, 2]])
                np.testing.assert_array_equal(stream[6][[0, 2]], nominal[3, [0, 2]])
        with self.assertRaises(ValueError):
            injector.apply_step(nominal, scenario, 0, np.random.default_rng(7))

    def test_replay_environment_uses_observed_history_not_reference(self) -> None:
        scenario = AttackScenario(
            "replay", 20, 80, np.array([True, False, False, False]),
            {"delay_steps": 5}, nominal_duration_steps=60, actual_duration_steps=60,
        )
        episode = make_environment(scenario).run_episode(seed=5)
        np.testing.assert_array_equal(episode.gps_observed_m[20:80, 0], episode.gps_clean_m[15:75, 0])
        self.assertGreater(np.linalg.norm(episode.true_position_m[60, 0] - episode.reference_position_m[60, 0]), 0.1)

    def test_clean_all_reference_paths_reproducible_and_bounded(self) -> None:
        for kind in ("straight", "circle", "helix", "s_curve"):
            config = ProjectConfig(simulation=SimulationConfig(duration_s=3.0, trajectory_types=(kind,)))
            environment = ISACSpoofingEnvironment(config)
            first = environment.run_episode(seed=21, attack_type="clean")
            second = environment.run_episode(seed=21, attack_type="clean")
            np.testing.assert_array_equal(first.true_position_m, second.true_position_m)
            np.testing.assert_array_equal(first.features, second.features)
            self.assertTrue(np.all(np.isfinite(first.features)))
            self.assertLessEqual(float(np.linalg.norm(first.velocity_mps, axis=-1).max()), config.navigation.max_speed_mps + 1e-5)
            np.testing.assert_array_equal(first.velocity_mps, first.command_velocity_mps)
            # Discrete reference sampling can still produce finite tracking error.
            self.assertLess(float(np.linalg.norm(first.tracking_error_m, axis=-1).max()), 1.0)

    def test_open_closed_pair_keeps_scenario_and_random_bias_draws(self) -> None:
        config = ProjectConfig(simulation=SimulationConfig(duration_s=3.0))
        config = replace(config, attack=replace(
            config.attack, replay_delay_ranges_s=((.2, .3), (.3, .4), (.4, .5)),
        ))
        for kind in ("clean", "random", "replay", "stealth"):
            closed = ISACSpoofingEnvironment(config).run_episode(seed=42, attack_type=kind)
            opened = ISACSpoofingEnvironment(replace(
                config, navigation=replace(config.navigation, enabled=False)
            )).run_episode(seed=42, attack_type=kind)
            np.testing.assert_array_equal(closed.reference_position_m, opened.reference_position_m)
            np.testing.assert_array_equal(closed.labels, opened.labels)
            for key in ("start_step", "end_step", "target_uavs"):
                self.assertEqual(closed.metadata["attack"][key], opened.metadata["attack"][key])
            for key, value in opened.metadata["attack"]["parameters"].items():
                if key != "directions":
                    self.assertEqual(closed.metadata["attack"]["parameters"][key], value)
            if kind == "random":
                np.testing.assert_allclose(
                    closed.gps_observed_m - closed.gps_clean_m,
                    opened.gps_observed_m - opened.gps_clean_m, atol=8e-6,
                )


if __name__ == "__main__":
    unittest.main()

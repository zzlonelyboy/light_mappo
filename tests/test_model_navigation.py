from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from isac_spoofing.config import ProjectConfig
from isac_spoofing.data import save_episode, save_episode_shard
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.gps_attacks import GPSAttackInjector, generate_clean_gps
from isac_spoofing.raw_episode import load_raw_episode_file
from isac_spoofing.types import AttackScenario


def short_config():
    c = ProjectConfig()
    return replace(
        c,
        simulation=replace(c.simulation, duration_s=0.24, trajectory_types=("straight",)),
        navigation=replace(c.navigation, confirmation_steps=2),
    )


class ModelNavigationTests(unittest.TestCase):
    def test_false_alarm_switches_next_slot_without_oracle_labels(self):
        c = short_config()
        c = replace(c, navigation=replace(c.navigation, mitigation_enabled=True))
        calls = []
        def policy(features):
            self.assertIsInstance(features, np.ndarray)
            self.assertEqual(features.shape, (8, 10))
            calls.append(features.copy())
            probability = np.zeros(8)
            probability[0] = 0.99 if len(calls) <= 2 else 0.01
            return probability
        episode = ISACSpoofingEnvironment(c).run_episode(
            seed=73, attack_type="clean", detector=policy
        )
        self.assertFalse(np.any(episode.labels))
        self.assertFalse(np.any(episode.navigation_mode[:2]))
        self.assertTrue(episode.confirmed_alarm[1, 0])
        self.assertTrue(np.all(episode.navigation_mode[2:, 0] == 1))
        self.assertFalse(np.any(episode.navigation_mode[:, 1:]))
        np.testing.assert_allclose(
            episode.navigation_position_m[2:, 0],
            episode.isac_position_m[1:-1, 0]
            + c.simulation.dt_s * episode.command_velocity_mps[1:-1, 0],
            atol=1e-5,
        )
        np.testing.assert_array_equal(
            episode.navigation_position_m[:, 1:], episode.gps_observed_m[:, 1:]
        )
        np.testing.assert_allclose(
            np.diff(episode.true_position_m, axis=0),
            c.simulation.dt_s * episode.command_velocity_mps[:-1],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            episode.available_time_s,
            (np.arange(c.simulation.num_steps) + 1) * c.simulation.dt_s,
        )
        self.assertEqual(episode.metadata["measurement_timing"]["alarm_availability_offset_steps"], 1)

    def test_default_disabled_mitigation_keeps_gps_motion_despite_alarms(self):
        c = short_config()
        common = ISACSpoofingEnvironment(c).run_episode(seed=73, attack_type="clean")
        detected = ISACSpoofingEnvironment(c).run_episode(
            seed=73, attack_type="clean", detector=lambda features: np.ones(8),
        )
        self.assertTrue(np.any(detected.confirmed_alarm))
        self.assertTrue(np.all(detected.detection_probability == 1.0))
        self.assertFalse(np.any(detected.navigation_mode))
        np.testing.assert_array_equal(detected.navigation_position_m, detected.gps_observed_m)
        for name in ("true_position_m", "command_velocity_mps", "gps_observed_m", "features"):
            np.testing.assert_array_equal(getattr(detected, name), getattr(common, name))
        self.assertFalse(detected.metadata["navigation"]["detector_controls_vehicle"])
        self.assertEqual(
            detected.metadata["navigation"]["collection_mode"],
            "online_detection_without_navigation_mitigation",
        )

    def test_five_slot_confirmation_is_independent_of_slot_duration(self):
        for dt_s in (0.02, 0.04):
            with self.subTest(dt_s=dt_s):
                c = ProjectConfig()
                c = replace(c, simulation=replace(
                    c.simulation, dt_s=dt_s, duration_s=10 * dt_s,
                    trajectory_types=("straight",),
                ))
                episode = ISACSpoofingEnvironment(c).run_episode(
                    seed=73, attack_type="clean", detector=lambda features: np.full(8, 0.5),
                )
                for uav in range(c.simulation.num_uavs):
                    np.testing.assert_array_equal(
                        np.flatnonzero(episode.confirmed_alarm[:, uav]), [4],
                    )
                self.assertFalse(np.any(episode.navigation_mode))
                self.assertAlmostEqual(episode.available_time_s[4], 5 * dt_s)
                self.assertEqual(episode.metadata["navigation"]["confirmation_steps"], 5)
                self.assertAlmostEqual(
                    episode.metadata["navigation"]["confirmation_duration_s"], 5 * dt_s,
                )

    def test_offline_collection_has_feedback_but_no_invented_detector(self):
        episode = ISACSpoofingEnvironment(short_config()).run_episode(
            seed=2, attack_type="clean"
        )
        self.assertFalse(np.any(episode.navigation_mode))
        self.assertFalse(np.any(episode.confirmed_alarm))
        self.assertIsNone(episode.detection_probability)
        self.assertFalse(episode.metadata["navigation"]["detector_controls_vehicle"])
        self.assertGreater(np.max(np.abs(episode.gps_clean_m - episode.true_position_m)), 0)

    def test_scores_can_only_follow_completed_slots(self):
        env = ISACSpoofingEnvironment(short_config())
        env.reset(seed=1, attack_type="clean")
        with self.assertRaises(RuntimeError):
            env.apply_detection_probabilities(np.zeros(8))
        env.step()
        env.apply_detection_probabilities(np.zeros(8))
        with self.assertRaises(RuntimeError):
            env.apply_detection_probabilities(np.zeros(8))
        env.step()
        with self.assertRaises(ValueError):
            env.apply_detection_probabilities(np.full(8, np.nan))

    def test_new_quality_and_navigation_arrays_round_trip(self):
        episode = ISACSpoofingEnvironment(short_config()).run_episode(
            seed=2, attack_type="clean", detector=lambda x: np.zeros(8)
        )
        episode = replace(episode, metadata={**episode.metadata, "episode_id": "new_model"})
        fields = (
            "navigation_mode", "navigation_position_m", "confirmed_alarm",
            "available_time_s", "isac_reported_covariance", "isac_received_power_w",
            "isac_azimuth_rad", "detection_probability",
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                save_episode(episode, directory, "new_model"),
                save_episode_shard([episode], directory, "new_model_shard"),
            ]
            for path in paths:
                raw = load_raw_episode_file(path, array_names=fields)[0]
                for name in fields:
                    np.testing.assert_array_equal(raw.require(name), getattr(episode, name))

    def test_normal_gps_scale_and_replay_keep_same_historical_sample(self):
        c = ProjectConfig()
        truth = np.zeros((10000, 1, 3))
        nominal = generate_clean_gps(truth, c.gps, np.random.default_rng(2))
        np.testing.assert_allclose(
            np.std(nominal[:, 0], axis=0), c.gps.noise_std_m, rtol=0.04
        )
        scenario = AttackScenario(
            "replay", 10, 20, np.array([True]), {"delay_steps": 4}
        )
        injector = GPSAttackInjector(replace(c.attack, replay_source="delayed_clean_stream"), c.simulation.dt_s)
        reported, labels = injector.apply(
            nominal[:30], scenario, np.random.default_rng(7)
        )
        np.testing.assert_array_equal(reported[10:20], nominal[6:16])
        self.assertTrue(np.all(labels[10:20]))
        with self.assertRaisesRegex(ValueError, "history"):
            injector.apply(
                nominal[:30], replace(scenario, start_step=1),
                np.random.default_rng(0),
            )


if __name__ == "__main__":
    unittest.main()

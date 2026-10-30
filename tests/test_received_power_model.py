from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from isac_spoofing.config import GPSConfig, ISACConfig, ProjectConfig
from isac_spoofing.features import (
    PaperFeatureExtractor,
    compute_inter_uav_consistency_feature,
    compute_inter_uav_consistency_from_measurement,
)
from isac_spoofing.isac import ISACMeasurementModel, delay_doppler_crb
from isac_spoofing.localization import RangeDirectionLocalizer


class ReceivedPowerModelTests(unittest.TestCase):
    def test_peer_wrapper_reuses_current_measurement_model(self):
        positions = np.array([[0., 0., 12.], [8., 0., 12.], [0., 8., 12.]])
        gps = positions + np.array([[0.4, 0., 0.], [0., 0.1, 0.], [0., 0., 0.2]])
        model = ISACMeasurementModel(ISACConfig())
        measurement = model.measure_inter_uav(positions, np.random.default_rng(10))
        expected = compute_inter_uav_consistency_from_measurement(
            gps, measurement.range_m, measurement.valid_mask, num_neighbors=2,
        )
        actual = compute_inter_uav_consistency_feature(
            positions, gps, num_neighbors=2, rng=np.random.default_rng(10),
        )
        np.testing.assert_array_equal(actual, expected)

    def setUp(self):
        self.model = ISACMeasurementModel(ISACConfig())
        self.position = np.array([[40.0, 45.0, 30.0]])
        self.velocity = np.array([[2.0, 0.2, -0.1]])

    def test_resource_budget_and_monostatic_distance_scaling(self):
        config = self.model.config
        np.testing.assert_allclose(
            config.carrier_frequencies_hz, [2.94e9, 2.98e9, 3.02e9, 3.06e9]
        )
        self.assertEqual(config.total_bandwidth_hz, 160e6)
        self.assertEqual(self.model.target_bandwidth_hz(8), 5e6)
        distance = np.full((1, 4), 100.0)
        power = self.model.received_power_from_range(distance, 8)
        doubled_range = self.model.received_power_from_range(2 * distance, 8)
        np.testing.assert_allclose(doubled_range, power / 16)
        np.testing.assert_allclose(
            self.model.received_power_from_range(distance, 4), power * 2
        )

    def test_observation_time_and_bandwidth_have_different_effects(self):
        power = np.ones(4) * 1e-12
        tau, mu = delay_doppler_crb(power, 5e6, 0.01, 2e-20)
        tau_b, mu_b = delay_doppler_crb(power, 10e6, 0.01, 2e-20)
        tau_t, mu_t = delay_doppler_crb(power, 5e6, 0.02, 2e-20)
        np.testing.assert_allclose(tau_b, tau / 4)
        np.testing.assert_allclose(mu_b, mu)
        np.testing.assert_allclose(tau_t, tau / 2)
        np.testing.assert_allclose(mu_t, mu / 8)

    def test_more_targets_share_bs_resources_but_not_peer_waveform(self):
        d = np.full((1, 4), 80.0)
        theta = np.full((1, 4), 0.4)
        phi = np.full((1, 4), 0.6)
        r1 = self.model.covariances_from_power(
            self.model.received_power_from_range(d, 1), theta, phi, 1
        )
        r2 = self.model.covariances_from_power(
            self.model.received_power_from_range(d, 2), theta, phi, 2
        )
        np.testing.assert_allclose(r2[..., 0, 0], r1[..., 0, 0] * 8)
        np.testing.assert_allclose(r2[..., 1:3, 1:3], r1[..., 1:3, 1:3] * 2)
        pair = np.array([[0., 0., 20.], [8., 0., 20.]])
        p2 = self.model.measure_inter_uav(pair, np.random.default_rng(1))
        p3 = self.model.measure_inter_uav(
            np.vstack([pair, [0., 8., 20.]]), np.random.default_rng(1)
        )
        self.assertEqual(p2.received_power_w[0, 1], p3.received_power_w[0, 1])
        self.assertEqual(p2.range_variance[0, 1], p3.range_variance[0, 1])
        self.assertFalse(np.any(np.diag(p3.valid_mask)))

    def test_covariance_matches_repeated_observations(self):
        rng = np.random.default_rng(20261029)
        first = self.model.measure(self.position, self.velocity, rng)
        delta = self.position[:, None] - self.model.anchors_m[None]
        distance = np.linalg.norm(delta, axis=-1)
        theta = np.arctan2(delta[..., 1], delta[..., 0])
        phi = np.arctan2(delta[..., 2], np.linalg.norm(delta[..., :2], axis=-1))
        radial = np.sum(self.velocity[:, None] * (-delta / distance[..., None]), axis=-1)
        truth = np.stack([distance, theta, phi, radial], axis=-1)
        errors = []
        for _ in range(700):
            observation = self.model.measure(self.position, self.velocity, rng)
            values = np.stack([
                observation.range_m, observation.azimuth_rad,
                observation.elevation_rad, observation.radial_velocity_mps,
            ], axis=-1)
            errors.append(values - truth)
        empirical = np.var(errors, axis=0, ddof=1)
        expected = np.diagonal(first.measurement_covariance, axis1=-2, axis2=-1)
        np.testing.assert_allclose(empirical, expected, rtol=0.23, atol=1e-12)
        np.testing.assert_array_equal(first.received_power_w, first.reported_power_w)
        self.assertFalse(np.array_equal(first.measurement_covariance, first.reported_covariance))

    def test_receiver_features_do_not_read_generator_covariance(self):
        observation = self.model.measure(
            self.position, self.velocity, np.random.default_rng(5)
        )
        poisoned = replace(
            observation,
            measurement_covariance=np.full_like(observation.measurement_covariance, np.nan),
            received_power_w=np.full_like(observation.received_power_w, np.nan),
        )
        config = ProjectConfig()
        peer_measurement = self.model.measure_inter_uav(
            self.position, np.random.default_rng(3)
        )
        def features(m, velocity=self.velocity, heading=None):
            extractor = PaperFeatureExtractor(
                config.features, self.model,
                localizer=RangeDirectionLocalizer(min_anchors=4),
                gps_noise_std_m=GPSConfig().noise_std_m,
            )
            extractor.reset()
            return extractor.compute(
                self.position + [0.5, 0.2, 0.8], velocity,
                np.zeros_like(velocity) if heading is None else heading, m,
                peer_measurement,
            )
        with patch.object(self.model, "received_power_from_range", side_effect=AssertionError("truth access")):
            a = features(observation)
            b = features(poisoned, heading=np.ones_like(self.velocity) * 100)
        np.testing.assert_allclose(a[0], b[0])
        zero = features(observation, velocity=np.zeros_like(self.velocity))[0]
        self.assertEqual(zero[0, 7], 0.0)

    def test_deployment_halfspace_is_not_selected_from_truth(self):
        with self.assertRaisesRegex(ValueError, "halfspace"):
            self.model.measure(
                np.array([[40., 45., -20.]]), self.velocity, np.random.default_rng(0)
            )
        with self.assertRaisesRegex(ValueError, "singular"):
            self.model.measure(
                np.array([[40., 45., 0.]]), self.velocity, np.random.default_rng(0)
            )


if __name__ == "__main__":
    unittest.main()

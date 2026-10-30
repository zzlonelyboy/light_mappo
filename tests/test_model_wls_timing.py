import unittest

import numpy as np

from isac_spoofing.config import NavigationConfig
from isac_spoofing.isac import los_from_angles
from isac_spoofing.localization import RangeDirectionLocalizer
from isac_spoofing.metrics import (
    ProbabilityAlarmTracker, alarm_availability_offset_steps,
    k_step_alarm_events, match_alarm_events, AttackEvent,
)
from isac_spoofing.training import TrainingConfig


class NewWLSAndTimingTests(unittest.TestCase):
    def test_wls_downweights_an_uncertain_outlier(self):
        anchors = np.array([[0., 0., 0.], [20., 0., 0.], [0., 20., 0.], [20., 20., 0.]])
        truth = np.array([8., 7., 15.])
        q, _ = RangeDirectionLocalizer._range_angle_geometry(truth, anchors)
        q[0, 0] += 7
        directions = los_from_angles(q[:, 1], q[:, 2])[None]
        covariance = np.tile(np.diag([.01, .0001, .0001]), (1, 4, 1, 1))
        covariance[0, 0] *= 1e6
        localizer = RangeDirectionLocalizer(iterations=40, min_anchors=4)
        def estimate(cov):
            return localizer.estimate(
                anchors, q[None, :, 0], directions, np.ones((1, 4), bool),
                measurement_covariance=cov,
                azimuth_rad=q[None, :, 1], elevation_rad=q[None, :, 2],
            )[0]
        weighted = estimate(covariance)
        unweighted = estimate(np.tile(np.diag([.01, .0001, .0001]), (1, 4, 1, 1)))
        self.assertLess(np.linalg.norm(weighted - truth), .001)
        self.assertGreater(np.linalg.norm(unweighted - truth), .1)

    def test_angle_wrap_does_not_create_a_two_pi_outlier(self):
        anchors = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [10., 10., 0.]])
        truth = np.array([-10., -1e-4, 12.])
        q, _ = RangeDirectionLocalizer._range_angle_geometry(truth, anchors)
        q[0, 1] += 2 * np.pi
        covariance = np.tile(np.diag([.01, .0001, .0001]), (1, 4, 1, 1))
        result = RangeDirectionLocalizer(min_anchors=4).estimate(
            anchors, q[None, :, 0], los_from_angles(q[:, 1], q[:, 2])[None],
            np.ones((1, 4), bool), measurement_covariance=covariance,
            azimuth_rad=q[None, :, 1], elevation_rad=q[None, :, 2],
        )
        np.testing.assert_allclose(result[0], truth, atol=1e-4)

    def test_five_step_tracker_matches_batch_and_rearms(self):
        self.assertEqual(NavigationConfig().confirmation_steps, 5)
        TrainingConfig(confirmation_steps=5).validate()
        probabilities = np.r_[np.ones(4), np.zeros(1), np.ones(5), np.zeros(5), np.ones(5)][:, None]
        tracker = ProbabilityAlarmTracker(1, 5)
        triggers = np.stack([tracker.update(p) for p in probabilities])
        np.testing.assert_array_equal(np.flatnonzero(triggers[:, 0]), [9, 19])
        events = k_step_alarm_events(probabilities.astype(np.uint8), 5)
        self.assertEqual([event.trigger_step for event in events[0]], [9, 19])

    def test_new_slot_end_latency_and_legacy_metadata_are_distinct(self):
        predictions = np.zeros((5, 1), dtype=np.uint8)
        predictions[1, 0] = 1
        events = k_step_alarm_events(predictions, 1)
        attacks = [AttackEvent(uav=0, start_step=1, end_step_exclusive=4)]
        current = match_alarm_events(
            events, attacks, num_steps=5, dt_s=.02,
            alarm_availability_offset_steps=1,
        )
        self.assertEqual(current["detected_delays_s"], [.02])
        self.assertEqual(alarm_availability_offset_steps({}), 0)
        self.assertEqual(alarm_availability_offset_steps({
            "schema_version": "3.0",
            "measurement_timing": {"alarm_availability_offset_steps": 1},
        }), 1)
        with self.assertRaises(ValueError):
            alarm_availability_offset_steps({"schema_version": "3.0"})


if __name__ == "__main__":
    unittest.main()

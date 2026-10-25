from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.rebuild_inter_uav_neighbor_views import (
    _base_view_name,
    _rebuild_shard,
    _view_name,
)
from isac_spoofing.features import (
    INTER_UAV_FEATURE_NAMES,
    compute_inter_uav_consistency_from_measurement,
)


class RebuildInterUAVNeighborViewsTests(unittest.TestCase):
    def test_view_names_match_existing_dataset_convention(self) -> None:
        source = "d1_1000_multitrajectory_test_rcs6_seed20260802_interuav"
        self.assertEqual(_base_view_name(source), source)
        self.assertEqual(_view_name(source, 3), source)
        self.assertEqual(_view_name(source, 2), source + "_k2")
        self.assertEqual(_view_name(source + "_k7", 5), source + "_k5")

    def test_rebuild_shard_changes_only_neighbor_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory_text:
            directory = Path(directory_text)
            source = directory / "source" / "test" / "shard_00000.npz"
            output = directory / "view_k2" / "test" / "shard_00000.npz"
            source.parent.mkdir(parents=True)

            episodes, steps, num_uavs = 2, 3, 4
            feature_names = tuple(
                [f"base_{index}" for index in range(8)]
                + list(INTER_UAV_FEATURE_NAMES)
            )
            features = np.arange(
                episodes * steps * num_uavs * len(feature_names), dtype=np.float32
            ).reshape(episodes, steps, num_uavs, len(feature_names))
            gps = np.zeros((episodes, steps, num_uavs, 3), dtype=np.float32)
            gps[..., :, 0] = np.arange(num_uavs, dtype=np.float32) * 10.0
            gps_range = np.linalg.norm(
                gps[..., :, None, :] - gps[..., None, :, :], axis=-1
            )
            offsets = np.asarray(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [1.0, 0.0, 4.0, 5.0],
                    [2.0, 4.0, 0.0, 6.0],
                    [3.0, 5.0, 6.0, 0.0],
                ],
                dtype=np.float32,
            )
            measured = gps_range + offsets
            valid = np.broadcast_to(
                ~np.eye(num_uavs, dtype=bool), measured.shape
            ).copy()
            event_shape = (episodes, num_uavs)
            labels = np.zeros((episodes, steps, num_uavs), dtype=np.uint8)
            episode_ids = np.asarray(["episode_0", "episode_1"])
            np.savez_compressed(
                source,
                features=features,
                feature_names=np.asarray(feature_names),
                gps_observed_m=gps,
                inter_uav_range_m=measured,
                inter_uav_valid_mask=valid,
                labels=labels,
                has_attack=np.zeros(event_shape, dtype=bool),
                attack_start_step=np.full(event_shape, -1, dtype=np.int32),
                attack_end_step_exclusive=np.full(event_shape, -1, dtype=np.int32),
                attack_was_truncated=np.zeros(event_shape, dtype=bool),
                episode_ids=episode_ids,
            )
            source.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "storage_format": "npz_episode_shard",
                        "episode_count": episodes,
                        "episodes": [
                            {"episode_id": "episode_0"},
                            {"episode_id": "episode_1"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = _rebuild_shard(str(source), ((str(output), 2),))
            self.assertEqual(result["rebuilt"][0]["neighbor_size"], 2)
            with np.load(output, allow_pickle=False) as rebuilt:
                self.assertNotIn("gps_observed_m", rebuilt.files)
                np.testing.assert_array_equal(rebuilt["features"][..., :8], features[..., :8])
                expected = compute_inter_uav_consistency_from_measurement(
                    gps, measured, valid, num_neighbors=2
                )
                np.testing.assert_allclose(rebuilt["features"][..., 8:], expected)
                np.testing.assert_array_equal(rebuilt["labels"], labels)
                np.testing.assert_array_equal(rebuilt["episode_ids"], episode_ids)
            self.assertTrue(output.with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()

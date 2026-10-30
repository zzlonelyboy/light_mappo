from __future__ import annotations

import importlib.util
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


# Load the staged file explicitly, so this test cannot silently test an older
# editable installation while its source is awaiting synchronization.
_SOURCE = Path(__file__).resolve().parents[1] / "src/isac_spoofing/learning_data.py"
_SPEC = importlib.util.spec_from_file_location("memmap_learning_data_under_test", _SOURCE)
learning = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = learning
_SPEC.loader.exec_module(learning)


@contextmanager
def closing_maps(episodes):
    try:
        yield episodes
    finally:
        closed = set()
        for episode in episodes:
            for array in (episode.features, episode.labels):
                mapped = getattr(array, "_mmap", None)
                if mapped is not None and id(mapped) not in closed:
                    mapped.close()
                    closed.add(id(mapped))


class MemmapLearningDataTests(unittest.TestCase):
    def make_root(self, root):
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": "3.0", "storage": {"format": "npy_feature_memmap_v1"},
            "feature_names": ["distance", "direction", "peer"],
        }), encoding="utf-8")

    def make_split(self, root, split="train", identifiers=("episode_a", "episode_b")):
        directory = root / split
        directory.mkdir()
        count = len(identifiers)
        features = np.random.default_rng(8).normal(size=(count, 5, 2, 3)).astype(np.float32)
        labels = np.zeros((count, 5, 2), dtype=np.uint8)
        labels[:, 1:3, 0] = 1
        has_attack = np.zeros((count, 2), dtype=bool)
        has_attack[:, 0] = True
        starts = np.full((count, 2), -1, dtype=np.int32)
        ends = starts.copy()
        starts[:, 0], ends[:, 0] = 1, 3
        events = {
            "has_attack": has_attack, "attack_start_step": starts,
            "attack_end_step_exclusive": ends,
            "attack_was_truncated": np.zeros((count, 2), dtype=bool),
        }
        rows = [{"episode_id": name, "split": split,
                 "measurement_timing": {"alarm_availability_offset_steps": 1}}
                for name in identifiers]
        np.save(directory / "features.npy", features)
        np.save(directory / "labels.npy", labels)
        np.savez(directory / "events.npz", **events)
        (directory / "metadata.json").write_text(json.dumps({"episodes": rows}),
                                                 encoding="utf-8")
        return features, labels, events, rows

    def test_dispatch_preserves_read_only_shared_mmap_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            original, labels, _, _ = self.make_split(root)
            with closing_maps(learning.load_split(root, "train")) as episodes:
                self.assertEqual(len(episodes), 2)
                self.assertIsInstance(episodes[0].features, np.memmap)
                self.assertIsInstance(episodes[0].labels, np.memmap)
                self.assertFalse(episodes[0].features.flags.writeable)
                self.assertIs(episodes[0].features._mmap, episodes[1].features._mmap)
                np.testing.assert_array_equal(episodes[0].features, original[0])
                np.testing.assert_array_equal(episodes[1].labels, labels[1])
                with self.assertRaises(ValueError):
                    episodes[0].features[0, 0, 0] = 99

    def test_memmap_and_legacy_shard_load_identical_detector_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            features, labels, events, rows = self.make_split(root)
            with closing_maps(learning.load_split(root, "train")) as mapped:
                legacy = root / "legacy"
                (legacy / "train").mkdir(parents=True)
                file = legacy / "train/shard_00000.npz"
                np.savez_compressed(file, features=features, labels=labels, **events,
                                    feature_names=np.asarray(mapped[0].feature_names),
                                    episode_ids=np.asarray([row["episode_id"] for row in rows]))
                file.with_suffix(".json").write_text(json.dumps({
                    "storage_format": "npz_episode_shard", "episode_count": 2,
                    "episodes": rows,
                }), encoding="utf-8")
                loaded = learning.load_split(legacy, "train")
                for first, second in zip(mapped, loaded):
                    self.assertEqual(first.episode_id, second.episode_id)
                    self.assertEqual(first.metadata, second.metadata)
                    for name in ("features", "labels", "has_attack", "attack_start_step",
                                 "attack_end_step_exclusive", "attack_was_truncated"):
                        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))

    def test_split_leakage_is_rejected_using_metadata_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_root(root)
            self.make_split(root, identifiers=("shared_episode",))
            validation = root / "validation"
            validation.mkdir()
            # Held-out numerical arrays are deliberately absent. The train
            # loader checks membership without reading held-out measurements.
            (validation / "metadata.json").write_text(json.dumps({"episodes": [
                {"episode_id": "shared_episode", "split": "validation"}
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not cross splits"):
                learning.load_split(root, "train")

    def test_invalid_shapes_ids_and_values_fail_explicitly(self):
        cases = ("shape", "duplicate", "membership", "nonfinite", "nonbinary")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.make_root(root)
                features, labels, _, rows = self.make_split(root)
                if case == "shape":
                    np.save(root / "train/labels.npy", labels[:, :-1])
                    error = (ValueError, "labels must have shape")
                elif case == "duplicate":
                    rows[1]["episode_id"] = rows[0]["episode_id"]
                    error = (ValueError, "duplicate episode_id")
                elif case == "membership":
                    rows[0]["split"] = "test"
                    error = (ValueError, "metadata split")
                elif case == "nonfinite":
                    features[0, 0, 0, 0] = np.nan
                    np.save(root / "train/features.npy", features)
                    error = (FloatingPointError, "non-finite")
                else:
                    labels[0, 0, 0] = 3
                    np.save(root / "train/labels.npy", labels)
                    error = (ValueError, "only 0 and 1")
                (root / "train/metadata.json").write_text(json.dumps({"episodes": rows}),
                                                           encoding="utf-8")
                with self.assertRaisesRegex(*error):
                    learning.load_split(root, "train")


if __name__ == "__main__":
    unittest.main()

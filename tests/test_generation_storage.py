from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import isac_spoofing.config as config_module
from isac_spoofing.config import ProjectConfig
from isac_spoofing.data import ARRAY_FIELDS
from isac_spoofing.environment import ISACSpoofingEnvironment


CLI_PATH = Path(__file__).resolve().parents[1] / "src/isac_spoofing/cli/generate_dataset.py"
_spec = importlib.util.spec_from_file_location("_generation_storage_cli", CLI_PATH)
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def quick_config():
    base = ProjectConfig()
    return replace(
        base,
        simulation=replace(base.simulation, duration_s=0.2, trajectory_types=("straight",)),
        gps=replace(base.gps, noise_std_m=(1.5, 1.5, 3.0)),
        attack=replace(base.attack, replay_delay_ranges_s=((0.02, 0.04),) * 3),
    )


def invoke(config, output, *, audit_level="full", workers=1, episodes=8, replay_source=None):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(config_module.__file__).resolve().parents[1])
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    command = [
        sys.executable, str(CLI_PATH), "--config", str(config),
        "--output", str(output), "--episodes", str(episodes), "--seed", "20260714",
        "--episodes-per-shard", "2", "--workers", str(workers),
        "--audit-level", audit_level,
    ]
    if replay_source is not None:
        command.extend(["--replay-source", replay_source])
    return subprocess.run(
        command,
        capture_output=True, text=True, env=environment, timeout=120,
    )


def read_records(directory):
    records = {}
    for split in ("train", "validation", "test"):
        for p in sorted((directory / split).glob("*.npz")):
            metadata = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
            with np.load(p, allow_pickle=False) as arrays:
                for index, episode_id in enumerate(metadata["episode_ids"]):
                    records[episode_id] = {
                        "split": split,
                        "metadata": metadata["episodes"][index],
                        "arrays": {
                            name: arrays[name][index].copy()
                            for name in arrays.files
                            if name not in {"feature_names", "episode_ids"}
                        },
                    }
    return records


class GenerationStorageTests(unittest.TestCase):
    def test_cli_parallel_override_migrates_legacy_replay_and_saves_private_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw_config = quick_config().to_dict()
            # Old snapshots omit this field and must keep their historical meaning.
            raw_config["attack"].pop("replay_source")
            config_path = root / "legacy_config.json"
            config_path.write_text(json.dumps(raw_config), encoding="utf-8")
            variants = (
                ("legacy", None, "delayed_clean_stream"),
                ("recorded", "recorded_normal_episode", "recorded_normal_episode"),
            )
            for name, override, expected_mode in variants:
                output = root / name
                result = invoke(
                    config_path, output, audit_level="standard", workers=2,
                    episodes=8, replay_source=override,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["replay_source"], expected_mode)
                self.assertEqual(manifest["config"]["attack"]["replay_source"], expected_mode)
                records = read_records(output)
                self.assertEqual(len(records), 8)
                replay_count = 0
                source_ids = set()
                for record in records.values():
                    metadata = record["metadata"]
                    attack = metadata["attack"]
                    source = metadata["replay_recording"]
                    arrays = record["arrays"]
                    self.assertIn("replay_source_gps_m", arrays)
                    saved_source = arrays["replay_source_gps_m"]
                    self.assertEqual(saved_source.shape, arrays["gps_observed_m"].shape)
                    expected_present = (
                        attack["attack_type"] == "replay"
                        and expected_mode == "recorded_normal_episode"
                    )
                    self.assertEqual(source["present"], expected_present)
                    if attack["attack_type"] == "replay":
                        replay_count += 1
                        self.assertEqual(attack["parameters"]["replay_source"], expected_mode)
                    if expected_present:
                        self.assertEqual(source["storage_array"], "replay_source_gps_m")
                        self.assertNotIn(source["source_id"], source_ids)
                        source_ids.add(source["source_id"])
                        start, end = attack["start_step"], attack["end_step"]
                        delay = attack["parameters"]["delay_steps"]
                        targets = attack["target_uavs"]
                        np.testing.assert_array_equal(
                            arrays["gps_observed_m"][start:end, targets],
                            saved_source[start-delay:end-delay, targets],
                        )
                    else:
                        np.testing.assert_array_equal(saved_source, np.zeros_like(saved_source))
                self.assertGreater(replay_count, 0)
                self.assertEqual(manifest["replay_recording_count"], len(source_ids))
                listed_sources = {
                    source_id
                    for ids in manifest["replay_recording_sources_by_split"].values()
                    for source_id in ids
                }
                self.assertEqual(listed_sources, source_ids)

    def test_standard_policy_preserves_required_arrays_and_features(self):
        episode = ISACSpoofingEnvironment(quick_config()).run_episode(
            seed=14, attack_type="clean", trajectory_type="straight",
        )
        standard = cli._apply_audit_level(episode, "standard")
        self.assertIs(cli._apply_audit_level(episode, "full"), episode)
        for field in cli.OPTIONAL_COVARIANCE_AUDIT_FIELDS:
            self.assertIsNotNone(getattr(episode, field))
            self.assertIsNone(getattr(standard, field))
        for field in ARRAY_FIELDS:
            self.assertIs(getattr(standard, field), getattr(episode, field))
        self.assertIs(standard.navigation_position_m, episode.navigation_position_m)
        self.assertIs(standard.isac_reported_power_w, episode.isac_reported_power_w)
        with self.assertRaisesRegex(ValueError, "audit level"):
            cli._apply_audit_level(episode, "unknown")

    def test_batches_bound_pending_results_without_changing_schedule(self):
        assignment = cli._experiment_assignment(23, 20260714, ("s_curve",))
        tasks = [(index, 20260714 + index, *assignment[index]) for index in range(23)]
        for workers in (1, 2, 4):
            batches = list(cli._bounded_episode_batches(tasks, workers))
            self.assertTrue(all(0 < len(batch) <= 2 * workers for batch in batches))
            self.assertEqual([task for batch in batches for task in batch], tasks)

    def test_cli_serial_parallel_and_audit_levels_have_identical_training_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(quick_config().to_dict()), encoding="utf-8")
            variants = {"full": ("full", 1), "standard": ("standard", 1),
                        "parallel": ("standard", 2)}
            data = {}
            manifests = {}
            for name, (audit, workers) in variants.items():
                output = root / name
                result = invoke(config_path, output, audit_level=audit, workers=workers)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                manifests[name] = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
                self.assertEqual(progress["completed_episodes"], 8)
                self.assertEqual(progress["saved_episodes"], 8)
                self.assertEqual(progress["total_episodes"], 8)
                self.assertTrue(progress["complete"])
                self.assertEqual(progress["status"], "complete")
                self.assertGreater(progress["elapsed_seconds"], 0)
                self.assertEqual(manifests[name]["storage"]["audit_level"], audit)
                expected_omissions = list(cli.OPTIONAL_COVARIANCE_AUDIT_FIELDS) if audit == "standard" else []
                self.assertEqual(manifests[name]["storage"]["omitted_optional_array_fields"], expected_omissions)
                data[name] = read_records(output)
                self.assertEqual(len(data[name]), 8)
            full = data["full"]
            for name in ("standard", "parallel"):
                self.assertEqual(data[name].keys(), full.keys())
                self.assertEqual(manifests[name]["scenario_counts_by_split"], manifests["full"]["scenario_counts_by_split"])
                for episode_id, expected in full.items():
                    actual = data[name][episode_id]
                    self.assertEqual(actual["split"], expected["split"])
                    self.assertEqual(actual["metadata"], expected["metadata"])
                    self.assertEqual(set(actual["arrays"]),
                                     set(expected["arrays"]) - set(cli.OPTIONAL_COVARIANCE_AUDIT_FIELDS))
                    for field, values in actual["arrays"].items():
                        np.testing.assert_array_equal(values, expected["arrays"][field])
            self.assertEqual(manifests["parallel"]["runtime"]["max_tasks_per_parallel_batch"], 4)

    def test_cli_refuses_to_overwrite_nonempty_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(quick_config().to_dict()), encoding="utf-8")
            output = root / "existing"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("original", encoding="utf-8")
            result = invoke(config_path, output, episodes=1)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must be empty", result.stderr)
            self.assertEqual(marker.read_text(encoding="utf-8"), "original")
            self.assertFalse((output / "progress.json").exists())

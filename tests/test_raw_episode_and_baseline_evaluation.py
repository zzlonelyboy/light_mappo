from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isac_spoofing.baseline_evaluation import (
    BaselineEvaluationConfig,
    evaluate_baseline_predictions,
)
from isac_spoofing.raw_episode import load_raw_split


def _episode_arrays(num_steps: int = 8, num_uavs: int = 2) -> dict[str, np.ndarray]:
    labels = np.zeros((num_steps, num_uavs), dtype=np.uint8)
    labels[2:, 0] = 1
    return {
        "gps_observed_m": np.zeros((num_steps, num_uavs, 3), dtype=np.float64),
        "inter_uav_range_m": np.ones((num_steps, num_uavs, num_uavs), dtype=np.float64),
        "inter_uav_valid_mask": ~np.eye(num_uavs, dtype=bool)[None].repeat(num_steps, axis=0),
        "inter_uav_range_variance": np.ones((num_steps, num_uavs, num_uavs), dtype=np.float64),
        "labels": labels,
        "has_attack": np.array([True, False]),
        "attack_start_step": np.array([2, -1]),
        "attack_end_step_exclusive": np.array([num_steps, -1]),
        "attack_was_truncated": np.array([False, False]),
    }


def test_raw_loader_restores_sharded_episode_boundary(tmp_path: Path) -> None:
    split = tmp_path / "validation"
    split.mkdir()
    arrays = _episode_arrays()
    np.savez_compressed(
        split / "shard_00000.npz",
        **{name: np.stack([value, value], axis=0) for name, value in arrays.items()},
        feature_names=np.asarray(["unused"]),
        episode_ids=np.asarray(["episode_a", "episode_b"]),
    )
    metadata = {
        "storage_format": "npz_episode_shard",
        "episode_count": 2,
        "episodes": [
            {"episode_id": "episode_a", "attack": {"attack_type": "random"}},
            {"episode_id": "episode_b", "attack": {"attack_type": "clean"}},
        ],
    }
    (split / "shard_00000.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    episodes = load_raw_split(
        tmp_path,
        "validation",
        array_names=tuple(arrays),
    )

    assert [episode.episode_id for episode in episodes] == ["episode_a", "episode_b"]
    assert episodes[0].labels.shape == (8, 2)
    assert episodes[1].require("gps_observed_m").shape == (8, 2, 3)


def test_evaluation_adapter_applies_confirmation_once(tmp_path: Path) -> None:
    split = tmp_path / "validation"
    split.mkdir()
    arrays = _episode_arrays()
    np.savez_compressed(
        split / "episode_a.npz",
        **arrays,
        feature_names=np.asarray(["unused"]),
    )
    (split / "episode_a.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_a",
                "attack": {"attack_type": "random"},
            }
        ),
        encoding="utf-8",
    )
    episode = load_raw_split(
        tmp_path,
        "validation",
        array_names=tuple(arrays),
    )[0]
    predictions = np.zeros_like(episode.labels)
    predictions[2:, 0] = 1

    result = evaluate_baseline_predictions(
        [episode],
        [predictions],
        BaselineEvaluationConfig(dt_s=0.1, confirmation_steps=3),
    )

    assert result["aggregate"]["true_positives"] == 1
    assert np.isclose(result["aggregate"]["delay_mean_s"], 0.2)

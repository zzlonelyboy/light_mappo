from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .types import EpisodeData


ARRAY_FIELDS = (
    "time_s",
    "true_position_m",
    "velocity_mps",
    "heading",
    "gps_clean_m",
    "gps_observed_m",
    "isac_position_m",
    "isac_range_m",
    "isac_los_unit",
    "isac_radial_velocity_mps",
    "isac_valid_mask",
    "isac_snr_db",
    "isac_range_variance",
    "isac_radial_velocity_variance",
    "isac_direction_variance",
    "inter_uav_range_m",
    "inter_uav_valid_mask",
    "inter_uav_snr_db",
    "inter_uav_range_variance",
    "features",
    "labels",
    "has_attack",
    "attack_start_step",
    "attack_end_step_exclusive",
    "attack_nominal_duration_steps",
    "attack_actual_duration_steps",
    "attack_was_truncated",
)


NAVIGATION_ARRAY_FIELDS = (
    "reference_position_m",
    "reference_velocity_mps",
    "command_velocity_mps",
    "tracking_error_m",
)

MODEL_ARRAY_FIELDS = (
    "available_time_s",
    "navigation_mode",
    "navigation_position_m",
    "confirmed_alarm",
    "detection_probability",
    "isac_azimuth_rad",
    "isac_elevation_rad",
    "isac_received_power_w",
    "isac_reported_power_w",
    "isac_measurement_covariance",
    "isac_reported_covariance",
    "isac_reported_los_covariance",
    "inter_uav_received_power_w",
    "inter_uav_reported_power_w",
    "replay_source_gps_m",
)

OPTIONAL_ARRAY_FIELDS = (*NAVIGATION_ARRAY_FIELDS, *MODEL_ARRAY_FIELDS)


def save_episode(
    episode: EpisodeData,
    output_directory: str | Path,
    episode_id: str,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{episode_id}.npz"
    arrays = {name: getattr(episode, name) for name in ARRAY_FIELDS}
    arrays.update({
        name: getattr(episode, name) for name in OPTIONAL_ARRAY_FIELDS
        if getattr(episode, name) is not None
    })
    arrays["feature_names"] = np.asarray(episode.feature_names)
    np.savez_compressed(npz_path, **arrays)
    with (output / f"{episode_id}.json").open("w", encoding="utf-8") as handle:
        json.dump(episode.metadata, handle, ensure_ascii=False, indent=2)
    return npz_path


def save_episode_shard(
    episodes: Sequence[EpisodeData],
    output_directory: str | Path,
    shard_id: str,
) -> Path:
    """Store multiple equal-shape episodes in one compressed NPZ shard.

    Every array in ``ARRAY_FIELDS`` receives a leading episode axis. Episode
    metadata remain separate inside the adjacent shard JSON, so the original
    episode boundary and split-level statistical unit are preserved.
    """
    episode_list = tuple(episodes)
    if not episode_list:
        raise ValueError("an episode shard must contain at least one episode")
    feature_names = episode_list[0].feature_names
    if any(episode.feature_names != feature_names for episode in episode_list[1:]):
        raise ValueError("all episodes in a shard must use the same feature schema")

    episode_ids = tuple(
        str(episode.metadata.get("episode_id", "")) for episode in episode_list
    )
    if any(not episode_id for episode_id in episode_ids):
        raise ValueError("every sharded episode must have an episode_id")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode_ids must be unique inside a shard")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / f"{shard_id}.npz"
    json_path = output / f"{shard_id}.json"
    temporary_npz = output / f".{shard_id}.tmp.npz"
    temporary_json = output / f".{shard_id}.tmp.json"

    arrays = {
        name: np.stack([getattr(episode, name) for episode in episode_list], axis=0)
        for name in ARRAY_FIELDS
    }
    for name in OPTIONAL_ARRAY_FIELDS:
        values = [getattr(episode, name) for episode in episode_list]
        if all(value is not None for value in values):
            arrays[name] = np.stack(values, axis=0)
        elif any(value is not None for value in values):
            raise ValueError(f"inconsistent optional navigation array in shard: {name}")
    arrays["feature_names"] = np.asarray(feature_names)
    arrays["episode_ids"] = np.asarray(episode_ids)
    metadata = {
        "schema_version": "1.0",
        "storage_format": "npz_episode_shard",
        "episode_axis": 0,
        "episode_count": len(episode_list),
        "episode_ids": list(episode_ids),
        "episodes": [episode.metadata for episode in episode_list],
    }

    np.savez_compressed(temporary_npz, **arrays)
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    temporary_npz.replace(npz_path)
    temporary_json.replace(json_path)
    return npz_path


def causal_windows(
    features: np.ndarray,
    labels: np.ndarray,
    window_steps: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    """Yield causal windows without mixing UAVs or future observations."""
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("expected features (T,N,F) and labels (T,N)")
    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    num_steps, num_uavs, _ = features.shape
    for uav in range(num_uavs):
        for end in range(window_steps - 1, num_steps):
            start = end - window_steps + 1
            yield features[start : end + 1, uav], labels[end, uav], (end, uav)


def uav_sequences(
    features: np.ndarray,
    labels: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    """Yield complete per-UAV sequences for the main causal detector.

    The complete episode remains the split/cluster unit, while each yielded
    sequence is one model sample with shapes ``(T,F)`` and ``(T,)``.
    """
    feature_array = np.asarray(features)
    label_array = np.asarray(labels)
    if feature_array.ndim != 3 or label_array.shape != feature_array.shape[:2]:
        raise ValueError("expected features (T,N,F) and labels (T,N)")
    for uav in range(feature_array.shape[1]):
        yield feature_array[:, uav], label_array[:, uav], uav

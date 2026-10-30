"""Resumable feature derivation and loading for the Deep Ensemble baseline."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from ..common.episode_data import (
    RawEpisode,
    discover_shards,
    iter_raw_episodes,
    validate_raw_shard,
)
from ..common.pathloss import A2GPathLossConfig, sample_actual_and_gps_pathloss
from ..common.seeding import stable_seed
from .features import FEATURE_NAMES, deep_ensemble_features


SIDECAR_FIELDS = (
    "features",
    "labels",
    "time_s",
    "has_attack",
    "attack_start_step",
    "attack_end_step_exclusive",
    "attack_was_truncated",
    "episode_ids",
)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def derive_episode_features(
    episode: RawEpisode,
    bs_positions_m: np.ndarray,
    config: A2GPathLossConfig,
    base_seed: int,
    time_chunk_steps: int = 100,
) -> np.ndarray:
    """Derive features with bounded memory and episode-level reproducibility."""

    if time_chunk_steps < 1:
        raise ValueError("time_chunk_steps must be positive")
    num_steps, num_uavs, _ = episode.true_position_m.shape
    num_bs = int(np.asarray(bs_positions_m).shape[0])
    features = np.empty(
        (num_steps, num_uavs, num_bs, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    rng = np.random.default_rng(
        stable_seed(base_seed, episode.episode_id, "deep_ensemble_pathloss")
    )
    for start in range(0, num_steps, time_chunk_steps):
        end = min(start + time_chunk_steps, num_steps)
        actual, gps_implied = sample_actual_and_gps_pathloss(
            true_position_m=episode.true_position_m[start:end],
            gps_position_m=episode.gps_observed_m[start:end],
            bs_positions_m=bs_positions_m,
            rng=rng,
            config=config,
        )
        features[start:end] = deep_ensemble_features(actual, gps_implied)
    return features


def _expected_marker(
    source_shard: Path,
    episode_count: int,
    config: A2GPathLossConfig,
    base_seed: int,
    bs_positions_m: np.ndarray,
) -> dict[str, object]:
    return {
        "source_shard": str(source_shard.resolve()),
        "source_size_bytes": int(source_shard.stat().st_size),
        "episode_count": int(episode_count),
        "base_seed": int(base_seed),
        "pathloss_config": asdict(config),
        "bs_positions_m": np.asarray(bs_positions_m, dtype=float).tolist(),
        "feature_names": list(FEATURE_NAMES),
        "alarm_timing_schema_version": 1,
    }


def generate_feature_shards(
    dataset_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...],
    bs_positions_m: np.ndarray,
    config: A2GPathLossConfig,
    base_seed: int,
    time_chunk_steps: int = 100,
    max_episodes_per_split: int | None = None,
) -> dict[str, object]:
    """Generate one resumable feature sidecar for every selected source shard."""

    config.validate()
    if max_episodes_per_split is not None and max_episodes_per_split < 1:
        raise ValueError("max_episodes_per_split must be positive when provided")
    summary: dict[str, object] = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "splits": {},
    }
    for split in splits:
        processed = 0
        generated = 0
        skipped = 0
        split_records: list[dict[str, object]] = []
        for source_shard in discover_shards(dataset_dir, split):
            available = validate_raw_shard(source_shard)
            remaining = (
                available
                if max_episodes_per_split is None
                else max_episodes_per_split - processed
            )
            if remaining <= 0:
                break
            episode_count = min(available, remaining)
            target = Path(output_dir) / split / source_shard.name
            marker_path = target.with_suffix(".complete.json")
            expected = _expected_marker(
                source_shard,
                episode_count,
                config,
                base_seed,
                bs_positions_m,
            )
            if target.exists() and marker_path.exists():
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                if marker == expected:
                    processed += episode_count
                    skipped += episode_count
                    split_records.append(
                        {"shard": target.name, "status": "reused", "episodes": episode_count}
                    )
                    continue

            episode_features: list[np.ndarray] = []
            episodes: list[RawEpisode] = []
            for episode in iter_raw_episodes(source_shard):
                if len(episodes) >= episode_count:
                    break
                episodes.append(episode)
                episode_features.append(
                    derive_episode_features(
                        episode,
                        bs_positions_m=bs_positions_m,
                        config=config,
                        base_seed=base_seed,
                        time_chunk_steps=time_chunk_steps,
                    )
                )
            if len(episodes) != episode_count:
                raise RuntimeError(f"failed to read {episode_count} episodes from {source_shard}")
            arrays = {
                "alarm_availability_offset_steps": np.asarray(
                    [episode.alarm_availability_offset_steps for episode in episodes],
                    dtype=np.int32,
                ),
                "features": np.stack(episode_features),
                "labels": np.stack([episode.labels for episode in episodes]),
                "time_s": np.stack([episode.time_s for episode in episodes]),
                "has_attack": np.stack([episode.has_attack for episode in episodes]),
                "attack_start_step": np.stack(
                    [episode.attack_start_step for episode in episodes]
                ),
                "attack_end_step_exclusive": np.stack(
                    [episode.attack_end_step_exclusive for episode in episodes]
                ),
                "attack_was_truncated": np.stack(
                    [episode.attack_was_truncated for episode in episodes]
                ),
                "episode_ids": np.asarray(
                    [episode.episode_id for episode in episodes],
                    dtype="U128",
                ),
            }
            _atomic_save_npz(target, arrays)
            _atomic_write_json(marker_path, expected)
            processed += episode_count
            generated += episode_count
            split_records.append(
                {"shard": target.name, "status": "generated", "episodes": episode_count}
            )
        summary["splits"][split] = {
            "processed_episodes": processed,
            "generated_episodes": generated,
            "reused_episodes": skipped,
            "shards": split_records,
        }
    _atomic_write_json(Path(output_dir) / "generation_summary.json", summary)
    return summary


def validate_sidecar(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in SIDECAR_FIELDS if field not in data.files]
        if missing:
            raise KeyError(f"{path} is missing sidecar fields {missing}")
        count = int(data["episode_ids"].shape[0])
        if data["features"].ndim != 5 or data["features"].shape[-1] != len(FEATURE_NAMES):
            raise ValueError(f"{path} has an invalid feature tensor")
    return count


def discover_sidecars(feature_dir: Path, split: str) -> tuple[Path, ...]:
    paths = tuple(sorted((Path(feature_dir) / split).glob("shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no feature sidecars found for split {split}")
    return paths


def iter_sidecar_episodes(
    feature_dir: Path,
    split: str,
    predicate: Callable[[str], bool] | None = None,
) -> Iterator[dict[str, np.ndarray | str]]:
    for path in discover_sidecars(feature_dir, split):
        validate_sidecar(path)
        with np.load(path, allow_pickle=False) as data:
            for index in range(data["episode_ids"].shape[0]):
                episode_id = str(data["episode_ids"][index])
                if predicate is not None and not predicate(episode_id):
                    continue
                yield {
                    "alarm_availability_offset_steps": (
                        int(data["alarm_availability_offset_steps"][index])
                        if "alarm_availability_offset_steps" in data.files else 0
                    ),
                    "episode_id": episode_id,
                    "features": np.asarray(data["features"][index]),
                    "labels": np.asarray(data["labels"][index]),
                    "time_s": np.asarray(data["time_s"][index]),
                    "has_attack": np.asarray(data["has_attack"][index]),
                    "attack_start_step": np.asarray(data["attack_start_step"][index]),
                    "attack_end_step_exclusive": np.asarray(
                        data["attack_end_step_exclusive"][index]
                    ),
                    "attack_was_truncated": np.asarray(
                        data["attack_was_truncated"][index]
                    ),
                }


def collect_slot_samples(
    feature_dir: Path,
    split: str,
    max_samples: int,
    seed: int,
    predicate: Callable[[str], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect a deterministic, approximately uniform sample across sidecars."""

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    paths = discover_sidecars(feature_dir, split)
    rng = np.random.default_rng(seed)
    per_shard = max(1, int(np.ceil(max_samples / len(paths))))
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            identifiers = np.asarray(data["episode_ids"]).astype(str)
            episode_mask = np.ones(identifiers.shape[0], dtype=bool)
            if predicate is not None:
                episode_mask = np.asarray(
                    [predicate(identifier) for identifier in identifiers],
                    dtype=bool,
                )
            selected_features = np.asarray(data["features"])[episode_mask]
            selected_labels = np.asarray(data["labels"])[episode_mask]
            if selected_features.size == 0:
                continue
            flat_features = selected_features.reshape(
                -1,
                selected_features.shape[-2],
                selected_features.shape[-1],
            )
            flat_labels = selected_labels.reshape(-1)
            count = min(per_shard, flat_labels.size)
            indices = rng.choice(flat_labels.size, size=count, replace=False)
            feature_batches.append(flat_features[indices])
            label_batches.append(flat_labels[indices])
    if not feature_batches:
        raise ValueError("the episode predicate selected no slot samples")
    features = np.concatenate(feature_batches, axis=0)
    labels = np.concatenate(label_batches, axis=0).astype(np.float32)
    if labels.size > max_samples:
        indices = rng.choice(labels.size, size=max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    return features.astype(np.float32), labels

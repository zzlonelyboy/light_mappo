from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LoadedEpisode:
    """Detector-visible arrays and event metadata loaded from one episode."""

    episode_id: str
    path: Path
    features: np.ndarray
    labels: np.ndarray
    has_attack: np.ndarray
    attack_start_step: np.ndarray
    attack_end_step_exclusive: np.ndarray
    attack_was_truncated: np.ndarray
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    @property
    def num_steps(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_uavs(self) -> int:
        return int(self.features.shape[1])


def select_episode_features(
    episode: LoadedEpisode,
    feature_names: Sequence[str],
) -> LoadedEpisode:
    """Return an episode view containing the requested named features in order."""

    requested = tuple(str(name) for name in feature_names)
    if not requested:
        raise ValueError("at least one feature must be selected")
    if len(set(requested)) != len(requested):
        raise ValueError("selected feature names must be unique")
    if requested == episode.feature_names:
        return episode
    index_by_name = {name: index for index, name in enumerate(episode.feature_names)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
        raise ValueError(
            f"episode {episode.episode_id} does not contain features: {missing}"
        )
    indices = [index_by_name[name] for name in requested]
    return LoadedEpisode(
        episode_id=episode.episode_id,
        path=episode.path,
        features=episode.features[..., indices],
        labels=episode.labels,
        has_attack=episode.has_attack,
        attack_start_step=episode.attack_start_step,
        attack_end_step_exclusive=episode.attack_end_step_exclusive,
        attack_was_truncated=episode.attack_was_truncated,
        feature_names=requested,
        metadata=episode.metadata,
    )


def discover_episode_files(data_root: str | Path, split: str) -> tuple[Path, ...]:
    directory = Path(data_root) / str(split)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset split directory does not exist: {directory}")
    files = tuple(sorted(directory.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no episode NPZ files found in {directory}")
    return files


def _validate_binary(array: np.ndarray, name: str) -> None:
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")


def load_episode(path: str | Path) -> LoadedEpisode:
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"episode metadata is missing: {json_path}")

    with np.load(npz_path, allow_pickle=False) as stored:
        required = {
            "features",
            "labels",
            "has_attack",
            "attack_start_step",
            "attack_end_step_exclusive",
            "attack_was_truncated",
            "feature_names",
        }
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {sorted(missing)}")
        features = np.asarray(stored["features"], dtype=np.float32)
        labels = np.asarray(stored["labels"], dtype=np.uint8)
        has_attack = np.asarray(stored["has_attack"], dtype=bool)
        attack_start = np.asarray(stored["attack_start_step"], dtype=np.int32)
        attack_end = np.asarray(
            stored["attack_end_step_exclusive"], dtype=np.int32
        )
        attack_was_truncated = np.asarray(
            stored["attack_was_truncated"], dtype=bool
        )
        feature_names = tuple(str(value) for value in stored["feature_names"].tolist())

    if features.ndim != 3:
        raise ValueError(f"{npz_path}: features must have shape (T,N,F)")
    if labels.shape != features.shape[:2]:
        raise ValueError(f"{npz_path}: labels must have shape (T,N)")
    if len(feature_names) != features.shape[2]:
        raise ValueError(f"{npz_path}: feature name count does not match feature dim")
    event_shape = (features.shape[1],)
    for name, array in (
        ("has_attack", has_attack),
        ("attack_start_step", attack_start),
        ("attack_end_step_exclusive", attack_end),
        ("attack_was_truncated", attack_was_truncated),
    ):
        if array.shape != event_shape:
            raise ValueError(f"{npz_path}: {name} must have shape {event_shape}")
    _validate_binary(labels, "labels")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(f"{npz_path}: features contain non-finite values")

    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    episode_id = str(metadata.get("episode_id", npz_path.stem))
    return LoadedEpisode(
        episode_id=episode_id,
        path=npz_path,
        features=features,
        labels=labels,
        has_attack=has_attack,
        attack_start_step=attack_start,
        attack_end_step_exclusive=attack_end,
        attack_was_truncated=attack_was_truncated,
        feature_names=feature_names,
        metadata=metadata,
    )


def load_episode_shard(path: str | Path) -> tuple[LoadedEpisode, ...]:
    """Load a compressed shard and restore its individual episode records."""
    npz_path = Path(path)
    json_path = npz_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"shard metadata is missing: {json_path}")

    with json_path.open("r", encoding="utf-8") as handle:
        shard_metadata = json.load(handle)
    if shard_metadata.get("storage_format") != "npz_episode_shard":
        raise ValueError(f"{json_path}: unsupported shard storage format")
    episode_metadata = shard_metadata.get("episodes")
    if not isinstance(episode_metadata, list) or not episode_metadata:
        raise ValueError(f"{json_path}: shard must contain episode metadata")

    with np.load(npz_path, allow_pickle=False) as stored:
        required = {
            "features",
            "labels",
            "has_attack",
            "attack_start_step",
            "attack_end_step_exclusive",
            "attack_was_truncated",
            "feature_names",
            "episode_ids",
        }
        missing = required - set(stored.files)
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {sorted(missing)}")
        features = np.asarray(stored["features"], dtype=np.float32)
        labels = np.asarray(stored["labels"], dtype=np.uint8)
        has_attack = np.asarray(stored["has_attack"], dtype=bool)
        attack_start = np.asarray(stored["attack_start_step"], dtype=np.int32)
        attack_end = np.asarray(
            stored["attack_end_step_exclusive"], dtype=np.int32
        )
        attack_was_truncated = np.asarray(
            stored["attack_was_truncated"], dtype=bool
        )
        feature_names = tuple(str(value) for value in stored["feature_names"].tolist())
        episode_ids = tuple(str(value) for value in stored["episode_ids"].tolist())

    if features.ndim != 4:
        raise ValueError(f"{npz_path}: sharded features must have shape (E,T,N,F)")
    episode_count, _, num_uavs, feature_dim = features.shape
    if labels.shape != features.shape[:3]:
        raise ValueError(f"{npz_path}: sharded labels must have shape (E,T,N)")
    if len(feature_names) != feature_dim:
        raise ValueError(f"{npz_path}: feature name count does not match feature dim")
    if len(episode_ids) != episode_count or len(episode_metadata) != episode_count:
        raise ValueError(f"{npz_path}: shard episode counts are inconsistent")
    if int(shard_metadata.get("episode_count", -1)) != episode_count:
        raise ValueError(f"{json_path}: episode_count is inconsistent")
    event_shape = (episode_count, num_uavs)
    for name, array in (
        ("has_attack", has_attack),
        ("attack_start_step", attack_start),
        ("attack_end_step_exclusive", attack_end),
        ("attack_was_truncated", attack_was_truncated),
    ):
        if array.shape != event_shape:
            raise ValueError(f"{npz_path}: {name} must have shape {event_shape}")
    _validate_binary(labels, "labels")
    if not np.all(np.isfinite(features)):
        raise FloatingPointError(f"{npz_path}: features contain non-finite values")

    episodes: list[LoadedEpisode] = []
    for index, metadata in enumerate(episode_metadata):
        if not isinstance(metadata, dict):
            raise ValueError(f"{json_path}: episode metadata must be objects")
        metadata_id = str(metadata.get("episode_id", ""))
        if metadata_id != episode_ids[index]:
            raise ValueError(f"{json_path}: episode_id order does not match the NPZ")
        episodes.append(
            LoadedEpisode(
                episode_id=metadata_id,
                path=npz_path,
                features=features[index],
                labels=labels[index],
                has_attack=has_attack[index],
                attack_start_step=attack_start[index],
                attack_end_step_exclusive=attack_end[index],
                attack_was_truncated=attack_was_truncated[index],
                feature_names=feature_names,
                metadata=metadata,
            )
        )
    return tuple(episodes)


def load_split(data_root: str | Path, split: str) -> tuple[LoadedEpisode, ...]:
    loaded: list[LoadedEpisode] = []
    for path in discover_episode_files(data_root, split):
        if path.name.startswith("shard_"):
            loaded.extend(load_episode_shard(path))
        else:
            loaded.append(load_episode(path))
    episodes = tuple(loaded)
    reference_names = episodes[0].feature_names
    for episode in episodes[1:]:
        if episode.feature_names != reference_names:
            raise ValueError(
                f"feature schema mismatch in {episode.path}: "
                f"expected {reference_names}, got {episode.feature_names}"
            )
    return episodes


@dataclass(frozen=True)
class FeatureStandardizer:
    """Per-feature training statistics; transformation deliberately has no clip."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    epsilon: float = 1.0e-8

    @classmethod
    def fit(
        cls,
        episodes: Iterable[LoadedEpisode],
        epsilon: float = 1.0e-8,
    ) -> FeatureStandardizer:
        episode_list = tuple(episodes)
        if not episode_list:
            raise ValueError("at least one training episode is required")
        feature_names = episode_list[0].feature_names
        dimension = len(feature_names)
        total = 0
        total_sum = np.zeros(dimension, dtype=np.float64)
        total_square = np.zeros(dimension, dtype=np.float64)
        for episode in episode_list:
            if episode.feature_names != feature_names:
                raise ValueError("all training episodes must use the same features")
            flattened = episode.features.reshape(-1, dimension).astype(
                np.float64, copy=False
            )
            total += flattened.shape[0]
            total_sum += np.sum(flattened, axis=0)
            total_square += np.sum(np.square(flattened), axis=0)
        if total < 1:
            raise ValueError("training episodes contain no feature observations")
        mean = total_sum / total
        variance = np.maximum(total_square / total - np.square(mean), 0.0)
        std = np.sqrt(variance)
        std = np.where(std < float(epsilon), 1.0, std)
        return cls(
            feature_names=feature_names,
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            epsilon=float(epsilon),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=np.float32)
        if array.shape[-1] != len(self.feature_names):
            raise ValueError(
                f"expected final feature dimension {len(self.feature_names)}, "
                f"got {array.shape[-1]}"
            )
        transformed = (array - self.mean) / self.std
        if not np.all(np.isfinite(transformed)):
            raise FloatingPointError("standardized features contain non-finite values")
        return transformed.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon,
            "clip": None,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> FeatureStandardizer:
        return cls(
            feature_names=tuple(str(value) for value in values["feature_names"]),
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
            epsilon=float(values.get("epsilon", 1.0e-8)),
        )


class UAVSequenceDataset:
    """PyTorch-compatible complete per-UAV sequence dataset."""

    def __init__(
        self,
        episodes: Sequence[LoadedEpisode],
        standardizer: FeatureStandardizer,
    ):
        if not episodes:
            raise ValueError("dataset requires at least one episode")
        self.episodes = tuple(episodes)
        self.standardizer = standardizer
        sequence_lengths = {episode.num_steps for episode in self.episodes}
        if len(sequence_lengths) != 1:
            raise ValueError("D1 batches require equal episode sequence lengths")
        self._indices = tuple(
            (episode_index, uav)
            for episode_index, episode in enumerate(self.episodes)
            for uav in range(episode.num_uavs)
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install the learning extra with: pip install -e '.[learning]'"
            ) from exc
        episode_index, uav = self._indices[index]
        episode = self.episodes[episode_index]
        features = self.standardizer.transform(episode.features[:, uav])
        labels = episode.labels[:, uav].astype(np.int64, copy=False)
        return torch.from_numpy(features), torch.from_numpy(labels)


def training_label_counts(episodes: Sequence[LoadedEpisode]) -> tuple[int, int]:
    negative = 0
    positive = 0
    for episode in episodes:
        positive += int(np.sum(episode.labels == 1))
        negative += int(np.sum(episode.labels == 0))
    return negative, positive

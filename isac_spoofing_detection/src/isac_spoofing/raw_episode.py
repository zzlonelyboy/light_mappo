from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from .data import ARRAY_FIELDS


@dataclass(frozen=True)
class RawEpisode:
    """One episode with caller-selected raw simulator arrays.

    The interface preserves episode boundaries for both individual NPZ files
    and episode shards. Baselines can request only the arrays they need instead
    of loading the detector-visible feature tensor alone.
    """

    episode_id: str
    path: Path
    arrays: Mapping[str, np.ndarray]
    feature_names: tuple[str, ...]
    metadata: dict[str, object]

    def require(self, name: str) -> np.ndarray:
        try:
            return self.arrays[str(name)]
        except KeyError as error:
            raise KeyError(
                f"episode {self.episode_id!r} was loaded without array {name!r}"
            ) from error

    @property
    def labels(self) -> np.ndarray:
        return self.require("labels")

    @property
    def num_steps(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_uavs(self) -> int:
        return int(self.labels.shape[1])


def discover_raw_episode_files(
    data_root: str | Path,
    split: str,
) -> tuple[Path, ...]:
    directory = Path(data_root) / str(split)
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset split directory does not exist: {directory}")
    files = tuple(sorted(directory.glob("*.npz")))
    if not files:
        raise FileNotFoundError(f"no episode NPZ files found in {directory}")
    return files


def _requested_array_names(array_names: Sequence[str] | None) -> tuple[str, ...]:
    requested = tuple(ARRAY_FIELDS if array_names is None else map(str, array_names))
    if not requested:
        raise ValueError("at least one raw array must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError("requested raw array names must be unique")
    unknown = sorted(set(requested) - set(ARRAY_FIELDS))
    if unknown:
        raise ValueError(f"unknown raw episode arrays: {unknown}")
    if "labels" not in requested:
        requested = (*requested, "labels")
    return requested


def _load_metadata(path: Path) -> dict[str, object]:
    json_path = path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"episode metadata is missing: {json_path}")
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"{json_path}: metadata must be a JSON object")
    return metadata


def _validate_episode_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    labels = np.asarray(arrays["labels"])
    if labels.ndim != 2:
        raise ValueError(f"{path}: labels must have shape (T,N)")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(f"{path}: labels must contain only 0 and 1")
    num_steps, num_uavs = labels.shape
    for name, array in arrays.items():
        value = np.asarray(array)
        if name in {
            "has_attack",
            "attack_start_step",
            "attack_end_step_exclusive",
            "attack_nominal_duration_steps",
            "attack_actual_duration_steps",
            "attack_was_truncated",
        }:
            if value.shape != (num_uavs,):
                raise ValueError(f"{path}: {name} must have shape ({num_uavs},)")
        elif name == "time_s":
            if value.shape != (num_steps,):
                raise ValueError(f"{path}: time_s must have shape ({num_steps},)")
        elif value.ndim >= 2 and value.shape[0] != num_steps:
            raise ValueError(f"{path}: {name} has an inconsistent time dimension")


def load_raw_episode_file(
    path: str | Path,
    array_names: Sequence[str] | None = None,
) -> tuple[RawEpisode, ...]:
    """Load an individual episode file or restore episodes from one shard."""

    npz_path = Path(path)
    metadata = _load_metadata(npz_path)
    requested = _requested_array_names(array_names)
    is_shard = metadata.get("storage_format") == "npz_episode_shard"

    with np.load(npz_path, allow_pickle=False) as stored:
        missing = set(requested) - set(stored.files)
        if missing:
            raise ValueError(f"{npz_path} is missing arrays: {sorted(missing)}")
        feature_names = (
            tuple(str(value) for value in stored["feature_names"].tolist())
            if "feature_names" in stored.files
            else ()
        )
        if not is_shard:
            arrays = {name: np.asarray(stored[name]) for name in requested}
            _validate_episode_arrays(npz_path, arrays)
            episode_id = str(metadata.get("episode_id", npz_path.stem))
            return (
                RawEpisode(
                    episode_id=episode_id,
                    path=npz_path,
                    arrays=arrays,
                    feature_names=feature_names,
                    metadata=metadata,
                ),
            )

        episode_metadata = metadata.get("episodes")
        if not isinstance(episode_metadata, list) or not episode_metadata:
            raise ValueError(f"{npz_path}: shard must contain episode metadata")
        if "episode_ids" not in stored.files:
            raise ValueError(f"{npz_path}: shard is missing episode_ids")
        episode_ids = tuple(str(value) for value in stored["episode_ids"].tolist())
        if len(episode_ids) != len(episode_metadata):
            raise ValueError(f"{npz_path}: shard episode counts are inconsistent")
        stacked = {name: np.asarray(stored[name]) for name in requested}
        if any(value.shape[0] != len(episode_ids) for value in stacked.values()):
            raise ValueError(f"{npz_path}: raw arrays have inconsistent episode axes")

    episodes: list[RawEpisode] = []
    for index, item_metadata in enumerate(episode_metadata):
        if not isinstance(item_metadata, dict):
            raise ValueError(f"{npz_path}: episode metadata must be objects")
        episode_id = str(item_metadata.get("episode_id", ""))
        if episode_id != episode_ids[index]:
            raise ValueError(f"{npz_path}: episode_id order does not match the NPZ")
        arrays = {name: value[index] for name, value in stacked.items()}
        _validate_episode_arrays(npz_path, arrays)
        episodes.append(
            RawEpisode(
                episode_id=episode_id,
                path=npz_path,
                arrays=arrays,
                feature_names=feature_names,
                metadata=item_metadata,
            )
        )
    return tuple(episodes)


def iter_raw_episodes(
    data_root: str | Path,
    split: str,
    array_names: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> Iterator[RawEpisode]:
    """Stream raw episodes from a split without crossing episode boundaries."""

    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive when supplied")
    emitted = 0
    for path in discover_raw_episode_files(data_root, split):
        for episode in load_raw_episode_file(path, array_names=array_names):
            yield episode
            emitted += 1
            if max_episodes is not None and emitted >= max_episodes:
                return


def load_raw_split(
    data_root: str | Path,
    split: str,
    array_names: Sequence[str] | None = None,
    max_episodes: int | None = None,
) -> tuple[RawEpisode, ...]:
    return tuple(
        iter_raw_episodes(
            data_root,
            split,
            array_names=array_names,
            max_episodes=max_episodes,
        )
    )

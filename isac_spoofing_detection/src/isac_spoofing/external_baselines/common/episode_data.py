"""Streaming access to raw episode fields and derived baseline shards."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


REQUIRED_RAW_FIELDS = (
    "time_s",
    "true_position_m",
    "gps_observed_m",
    "labels",
    "has_attack",
    "attack_start_step",
    "attack_end_step_exclusive",
    "attack_was_truncated",
    "episode_ids",
)


@dataclass(frozen=True)
class RawEpisode:
    episode_id: str
    time_s: np.ndarray
    true_position_m: np.ndarray
    gps_observed_m: np.ndarray
    labels: np.ndarray
    has_attack: np.ndarray
    attack_start_step: np.ndarray
    attack_end_step_exclusive: np.ndarray
    attack_was_truncated: np.ndarray

    @property
    def dt_s(self) -> float:
        if self.time_s.size < 2:
            raise ValueError("an episode requires at least two time samples")
        return float(np.median(np.diff(self.time_s)))


def discover_shards(dataset_dir: Path, split: str) -> tuple[Path, ...]:
    split_dir = Path(dataset_dir) / split
    paths = tuple(sorted(split_dir.glob("shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no episode shards found under {split_dir}")
    return paths


def validate_raw_shard(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in REQUIRED_RAW_FIELDS if field not in data.files]
        if missing:
            raise KeyError(f"{path} is missing raw fields {missing}")
        count = int(data["episode_ids"].shape[0])
        for field in REQUIRED_RAW_FIELDS:
            if data[field].shape[0] != count:
                raise ValueError(f"{path} has inconsistent episode axis for {field}")
    return count


def iter_raw_episodes(path: Path) -> Iterator[RawEpisode]:
    with np.load(path, allow_pickle=False) as data:
        count = int(data["episode_ids"].shape[0])
        for index in range(count):
            identifier = data["episode_ids"][index]
            if isinstance(identifier, bytes):
                episode_id = identifier.decode("utf-8")
            else:
                episode_id = str(identifier)
            yield RawEpisode(
                episode_id=episode_id,
                time_s=np.asarray(data["time_s"][index]),
                true_position_m=np.asarray(data["true_position_m"][index]),
                gps_observed_m=np.asarray(data["gps_observed_m"][index]),
                labels=np.asarray(data["labels"][index]),
                has_attack=np.asarray(data["has_attack"][index]),
                attack_start_step=np.asarray(data["attack_start_step"][index]),
                attack_end_step_exclusive=np.asarray(
                    data["attack_end_step_exclusive"][index]
                ),
                attack_was_truncated=np.asarray(
                    data["attack_was_truncated"][index]
                ),
            )

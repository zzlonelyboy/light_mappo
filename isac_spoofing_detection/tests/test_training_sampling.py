from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from isac_spoofing.learning_data import (
    FeatureStandardizer,
    LoadedEpisode,
    UAVSequenceDataset,
    training_label_counts,
)
from isac_spoofing.training import TrainingConfig


NUM_STEPS = 6
NUM_UAVS = 4
FEATURE_NAMES = ("feature_0", "feature_1")


def _episode(
    episode_index: int,
    attacked_uavs: tuple[int, ...] = (),
) -> LoadedEpisode:
    """Build a small episode whose UAV identity remains visible in its features."""

    features = np.empty(
        (NUM_STEPS, NUM_UAVS, len(FEATURE_NAMES)), dtype=np.float32
    )
    for step in range(NUM_STEPS):
        for uav in range(NUM_UAVS):
            features[step, uav, 0] = 100.0 * episode_index + 10.0 * uav + step
            features[step, uav, 1] = -20.0 * episode_index + 2.0 * uav - step

    labels = np.zeros((NUM_STEPS, NUM_UAVS), dtype=np.uint8)
    has_attack = np.zeros(NUM_UAVS, dtype=bool)
    attack_start = np.full(NUM_UAVS, -1, dtype=np.int32)
    attack_end = np.full(NUM_UAVS, -1, dtype=np.int32)
    for target_index, uav in enumerate(attacked_uavs):
        # Different target lengths ensure label_counts follows the selected UAV,
        # rather than accidentally using an all-UAV aggregate.
        start = 1
        end = min(NUM_STEPS, 3 + target_index)
        labels[start:end, uav] = 1
        has_attack[uav] = True
        attack_start[uav] = start
        attack_end[uav] = end

    episode_id = f"episode_{episode_index:05d}"
    return LoadedEpisode(
        episode_id=episode_id,
        path=Path(f"{episode_id}.npz"),
        features=features,
        labels=labels,
        has_attack=has_attack,
        attack_start_step=attack_start,
        attack_end_step_exclusive=attack_end,
        attack_was_truncated=np.zeros(NUM_UAVS, dtype=bool),
        feature_names=FEATURE_NAMES,
        metadata={
            "episode_id": episode_id,
            "attack": {
                "attack_type": "random" if attacked_uavs else "clean",
                "target_uavs": list(attacked_uavs),
            },
        },
    )


@pytest.fixture
def episodes() -> tuple[LoadedEpisode, ...]:
    return (_episode(0), _episode(1, attacked_uavs=(1, 3)))


@pytest.fixture
def standardizer(
    episodes: tuple[LoadedEpisode, ...],
) -> FeatureStandardizer:
    return FeatureStandardizer.fit(episodes)


def test_all_uavs_preserves_original_episode_uav_expansion(
    episodes: tuple[LoadedEpisode, ...],
    standardizer: FeatureStandardizer,
) -> None:
    dataset = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="all_uavs",
        seed=17,
    )
    expected = tuple(
        (episode_index, uav)
        for episode_index in range(len(episodes))
        for uav in range(NUM_UAVS)
    )

    assert len(dataset) == len(episodes) * NUM_UAVS
    assert dataset.sequence_indices() == expected

    dataset.set_epoch(9)
    assert dataset.sequence_indices() == expected
    with pytest.raises(ValueError, match="one_target_per_episode"):
        dataset.selected_uav_indices()


def test_one_target_has_one_sequence_per_episode_and_attack_uses_target(
    episodes: tuple[LoadedEpisode, ...],
    standardizer: FeatureStandardizer,
) -> None:
    dataset = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="one_target_per_episode",
        seed=23,
    )

    assert len(dataset) == len(episodes)
    for epoch in range(12):
        dataset.set_epoch(epoch)
        indices = dataset.sequence_indices()
        assert tuple(episode_index for episode_index, _ in indices) == (0, 1)
        assert episodes[1].has_attack[indices[1][1]]


def test_one_target_rotation_covers_clean_and_attack_candidates(
    episodes: tuple[LoadedEpisode, ...],
    standardizer: FeatureStandardizer,
) -> None:
    dataset = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="one_target_per_episode",
        seed=31,
    )

    clean_choices: list[int] = []
    attack_choices: list[int] = []
    for epoch in range(NUM_UAVS):
        dataset.set_epoch(epoch)
        clean_uav, attack_uav = dataset.selected_uav_indices()
        clean_choices.append(clean_uav)
        attack_choices.append(attack_uav)

    assert set(clean_choices) == set(range(NUM_UAVS))
    assert set(attack_choices[:2]) == {1, 3}
    assert attack_choices[2:] == attack_choices[:2]


def test_one_target_selection_is_reproducible_and_access_order_independent(
    episodes: tuple[LoadedEpisode, ...],
    standardizer: FeatureStandardizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # __getitem__ only needs torch.from_numpy.  A tiny stub keeps this unit test
    # independent of a CUDA/PyTorch installation and still exercises access order.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(from_numpy=lambda value: value),
    )
    first = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="one_target_per_episode",
        seed=101,
    )
    second = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="one_target_per_episode",
        seed=101,
    )
    first.set_epoch(7)
    second.set_epoch(7)
    expected = first.sequence_indices()

    # Reverse item access must not consume RNG state or alter the selection.
    first[1]
    first[0]
    second[0]
    second[1]
    assert first.sequence_indices() == expected
    assert second.sequence_indices() == expected

    first.set_epoch(7)
    second.set_epoch(7)
    assert first.sequence_indices() == second.sequence_indices() == expected


def test_label_counts_follow_current_sampled_sequences(
    episodes: tuple[LoadedEpisode, ...],
    standardizer: FeatureStandardizer,
) -> None:
    all_uavs = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="all_uavs",
    )
    assert all_uavs.label_counts() == training_label_counts(episodes)

    one_target = UAVSequenceDataset(
        episodes,
        standardizer,
        sampling_strategy="one_target_per_episode",
        seed=47,
    )
    for epoch in range(4):
        one_target.set_epoch(epoch)
        expected_positive = sum(
            int(np.sum(episodes[episode_index].labels[:, uav] == 1))
            for episode_index, uav in one_target.sequence_indices()
        )
        expected_total = len(episodes) * NUM_STEPS
        assert one_target.label_counts() == (
            expected_total - expected_positive,
            expected_positive,
        )


def test_standardization_is_identical_for_both_sampling_modes(
    episodes: tuple[LoadedEpisode, ...],
) -> None:
    all_uav_standardizer = FeatureStandardizer.fit(episodes)
    one_target_standardizer = FeatureStandardizer.fit(episodes)
    flattened = np.concatenate(
        [episode.features.reshape(-1, len(FEATURE_NAMES)) for episode in episodes],
        axis=0,
    )

    np.testing.assert_allclose(
        all_uav_standardizer.mean,
        np.mean(flattened, axis=0),
        rtol=1.0e-6,
    )
    np.testing.assert_array_equal(
        all_uav_standardizer.mean, one_target_standardizer.mean
    )
    np.testing.assert_array_equal(
        all_uav_standardizer.std, one_target_standardizer.std
    )

    all_uavs = UAVSequenceDataset(
        episodes,
        all_uav_standardizer,
        sampling_strategy="all_uavs",
    )
    one_target = UAVSequenceDataset(
        episodes,
        one_target_standardizer,
        sampling_strategy="one_target_per_episode",
    )
    np.testing.assert_array_equal(
        all_uavs.standardizer.mean, one_target.standardizer.mean
    )
    np.testing.assert_array_equal(
        all_uavs.standardizer.std, one_target.standardizer.std
    )


@pytest.mark.parametrize("invalid_strategy", ["one_uav", "", None])
def test_training_config_rejects_invalid_sampling_strategy(
    invalid_strategy: object,
) -> None:
    with pytest.raises(ValueError, match="training_sampling"):
        TrainingConfig(training_sampling=invalid_strategy).validate()  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_interval", [0, -1])
def test_training_config_rejects_nonpositive_validation_interval(
    invalid_interval: int,
) -> None:
    with pytest.raises(ValueError, match="validation_interval_epochs"):
        TrainingConfig(validation_interval_epochs=invalid_interval).validate()


def test_training_config_accepts_comparison_settings() -> None:
    TrainingConfig(
        training_sampling="all_uavs",
        validation_interval_epochs=1,
    ).validate()
    TrainingConfig(
        training_sampling="one_target_per_episode",
        validation_interval_epochs=8,
    ).validate()

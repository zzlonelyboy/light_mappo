from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from isac_spoofing.metrics import (
    AttackEvent,
    ProbabilityAlarmTracker,
    alarm_availability_offset_steps,
    k_step_alarm_events,
    match_alarm_events,
)
from isac_spoofing.training import TrainingConfig


def test_confirmation_25_steps_and_rearming_match_offline_state_machine() -> None:
    rng = np.random.default_rng(170)
    probability = rng.uniform(0.0, 1.0, size=(160, 3))
    probability[10:40, 0] = 0.9
    probability[40:70, 0] = 0.1
    probability[70:100, 0] = 0.9
    probability[:30, 1] = 0.9
    tracker = ProbabilityAlarmTracker(3, 25, warmup_steps=5)
    triggers = np.stack([tracker.update(row) for row in probability])
    expected = k_step_alarm_events(probability >= 0.5, 25, warmup_steps=5)
    for uav in range(3):
        assert np.flatnonzero(triggers[:, uav]).tolist() == [
            event.trigger_step for event in expected[uav]
        ]
    assert np.flatnonzero(triggers[:, 0]).tolist() == [34, 94]


def test_slot_end_delay_is_one_sample_later_than_legacy_timestamp() -> None:
    hard = np.ones((30, 1), dtype=np.uint8)
    alarms = k_step_alarm_events(hard, 25)
    attacks = (AttackEvent(0, 0, 30, "attack"),)
    legacy = match_alarm_events(alarms, attacks, 30, 0.02)
    current = match_alarm_events(
        alarms, attacks, 30, 0.02, alarm_availability_offset_steps=1
    )
    assert legacy["detected_delays_s"] == pytest.approx([0.48])
    assert current["detected_delays_s"] == pytest.approx([0.50])


def test_metadata_prevents_silent_schema3_timestamp_fallback() -> None:
    assert alarm_availability_offset_steps({}) == 0
    metadata = {
        "schema_version": "3.0",
        "measurement_timing": {"alarm_availability_offset_steps": 1},
    }
    assert alarm_availability_offset_steps(metadata) == 1
    with pytest.raises(ValueError, match="require"):
        alarm_availability_offset_steps({"schema_version": "3.0"})
    with pytest.raises(ValueError, match="non-negative integer"):
        alarm_availability_offset_steps({
            "measurement_timing": {"alarm_availability_offset_steps": 0.5}
        })


def test_training_defaults_to_five_confirmation_slots() -> None:
    config = TrainingConfig()
    config.validate()
    assert config.confirmation_steps == 5
    assert config.confirmation_steps * config.time_step_s == pytest.approx(0.1)
    replace(config, confirmation_steps=6).validate()
    replace(config, confirmation_steps=25).validate()


def test_soft_early_delay_uses_slot_end_offset_without_changing_other_terms() -> None:
    torch = pytest.importorskip("torch")
    from isac_spoofing.training import binary_detection_objective

    logits = torch.full((1, 4, 1), 1.3, dtype=torch.float64)
    labels = torch.ones((1, 4), dtype=torch.long)
    config = TrainingConfig(
        loss_objective="event_likelihood", confirmation_steps=2,
        time_step_s=0.02, early_detection_deadline_s=1.0,
        alarm_availability_offset_steps=0,
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    old = binary_detection_objective(logits, labels, criterion, config)
    new = binary_detection_objective(
        logits, labels, criterion, replace(config, alarm_availability_offset_steps=1)
    )
    assert float(new.early - old.early) == pytest.approx(0.02)
    assert float(new.detection) == pytest.approx(float(old.detection))
    assert float(new.false_alarm) == pytest.approx(float(old.false_alarm))


def test_deep_ensemble_sidecar_preserves_alarm_availability(tmp_path, monkeypatch) -> None:
    from isac_spoofing.external_baselines.common.pathloss import A2GPathLossConfig
    from isac_spoofing.external_baselines.deep_ensemble import dataset, evaluation
    from isac_spoofing.external_baselines.deep_ensemble.features import FEATURE_NAMES

    raw = tmp_path / "source" / "test"
    raw.mkdir(parents=True)
    source = raw / "shard_00000.npz"
    steps = 5
    position = np.full((1, steps, 1, 3), 30.0, dtype=np.float32)
    np.savez_compressed(
        source,
        time_s=np.arange(steps, dtype=np.float64)[None] * 0.02,
        true_position_m=position, gps_observed_m=position.copy(),
        labels=np.ones((1, steps, 1), dtype=np.uint8),
        has_attack=np.ones((1, 1), dtype=bool),
        attack_start_step=np.zeros((1, 1), dtype=np.int32),
        attack_end_step_exclusive=np.full((1, 1), steps, dtype=np.int32),
        attack_was_truncated=np.zeros((1, 1), dtype=bool),
        episode_ids=np.asarray(["sample"]),
    )
    metadata = {
        "schema_version": "3.0",
        "measurement_timing": {"alarm_availability_offset_steps": 1},
    }
    source.with_suffix(".json").write_text(
        json.dumps({"episodes": [metadata]}), encoding="utf-8"
    )
    monkeypatch.setattr(
        dataset, "derive_episode_features",
        lambda episode, **kwargs: np.zeros((steps, 1, 1, len(FEATURE_NAMES)), np.float32),
    )
    output = tmp_path / "sidecars"
    dataset.generate_feature_shards(
        tmp_path / "source", output, ("test",),
        np.asarray([[0.0, 0.0, 0.0]]), A2GPathLossConfig(packet_samples=8), 17,
    )
    restored = tuple(dataset.iter_sidecar_episodes(output, "test"))[0]
    assert restored["alarm_availability_offset_steps"] == 1

    class PerBS:
        def predict_proba(self, features):
            return np.full(features.shape[:-1], 0.9)

    monkeypatch.setattr(
        evaluation, "predict_positive_probability",
        lambda model, data: np.full(data.shape[0], 0.9),
    )
    probability = evaluation.predict_episodes(output, "test", PerBS(), object())
    result = evaluation.evaluate_episode_probabilities(probability, 0.5, 2)
    assert result["delay_mean_s"] == pytest.approx(0.04)
    assert probability[0].alarm_availability_offset_steps == 1

    # Old sidecars remain explicitly on the old timestamp convention.
    stored = output / "test" / "shard_00000.npz"
    with np.load(stored, allow_pickle=False) as arrays:
        legacy = {key: arrays[key] for key in arrays.files
                  if key != "alarm_availability_offset_steps"}
    np.savez_compressed(stored, **legacy)
    assert tuple(dataset.iter_sidecar_episodes(output, "test"))[0]["alarm_availability_offset_steps"] == 0

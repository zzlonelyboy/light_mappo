from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "baseline_xgboost.py"
SPEC = importlib.util.spec_from_file_location("baseline_xgboost", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
baseline_xgboost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline_xgboost)


def _episode(
    episode_id: str,
    probabilities: np.ndarray,
    attack_start: int | None,
) -> SimpleNamespace:
    num_steps, num_uavs = probabilities.shape
    has_attack = np.zeros(num_uavs, dtype=bool)
    starts = np.full(num_uavs, -1, dtype=np.int64)
    ends = np.full(num_uavs, -1, dtype=np.int64)
    labels = np.zeros((num_steps, num_uavs), dtype=np.uint8)
    if attack_start is not None:
        has_attack[0] = True
        starts[0] = attack_start
        ends[0] = num_steps
        labels[attack_start:, 0] = 1
    return SimpleNamespace(
        episode_id=episode_id,
        num_steps=num_steps,
        labels=labels,
        has_attack=has_attack,
        attack_start_step=starts,
        attack_end_step_exclusive=ends,
        metadata={"attack": {"attack_type": "synthetic"}},
    )


def test_evaluation_applies_confirmation_steps_and_repository_matching() -> None:
    probabilities = np.zeros((8, 1), dtype=np.float64)
    probabilities[2:, 0] = 0.9
    episode = _episode("attacked", probabilities, attack_start=2)

    aggregate_k1, _ = baseline_xgboost._evaluate_probabilities(
        [episode], [probabilities], 0.5, 0.1, 1, 0
    )
    aggregate_k3, results_k3 = baseline_xgboost._evaluate_probabilities(
        [episode], [probabilities], 0.5, 0.1, 3, 0
    )

    assert aggregate_k1["delay_mean_s"] == 0.0
    assert np.isclose(aggregate_k3["delay_mean_s"], 0.2)
    assert results_k3[0]["true_positives"] == 1
    assert results_k3[0]["false_positives"] == 0


def test_threshold_selection_uses_validation_far_constraint() -> None:
    rows = [
        {
            "threshold": 0.3,
            "event_f1": 0.9,
            "event_recall": 1.0,
            "false_alarms_per_uav_hour": 10.0,
        },
        {
            "threshold": 0.7,
            "event_f1": 0.8,
            "event_recall": 0.8,
            "false_alarms_per_uav_hour": 1.0,
        },
    ]

    selected, satisfied = baseline_xgboost._select_threshold(rows, 2.0)

    assert satisfied is True
    assert selected["threshold"] == 0.7


def test_threshold_selection_reports_unsatisfied_constraint() -> None:
    rows = [
        {
            "threshold": 0.3,
            "event_f1": 0.9,
            "event_recall": 1.0,
            "false_alarms_per_uav_hour": 10.0,
        }
    ]

    selected, satisfied = baseline_xgboost._select_threshold(rows, 2.0)

    assert selected["threshold"] == 0.3
    assert satisfied is False

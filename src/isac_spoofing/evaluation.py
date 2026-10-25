from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .learning_data import load_split
from .metrics import (
    aggregate_event_results,
    attack_events_from_arrays,
    binary_detection_metrics,
    k_step_alarm_events,
    match_alarm_events,
)
from .training import (
    ATTACK_PROBABILITY_THRESHOLD,
    load_trained_detector,
    predict_episode,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_dataset_manifest(data_root: str | Path) -> dict[str, Any]:
    path = Path(data_root) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluation_parameters(
    manifest: dict[str, Any],
    dt_s: float | None,
    warmup_steps: int | None,
    checkpoint_warmup_steps: int | None = None,
) -> tuple[float, int]:
    config = manifest.get("config", {})
    simulation = config.get("simulation", {})
    features = config.get("features", {})
    resolved_dt = float(simulation.get("dt_s", 0.0) if dt_s is None else dt_s)
    if warmup_steps is not None:
        resolved_warmup = int(warmup_steps)
    elif checkpoint_warmup_steps is not None:
        resolved_warmup = int(checkpoint_warmup_steps)
    else:
        # Backward compatibility: checkpoints created before alarm warmup was
        # decoupled from history length used history_steps implicitly.
        resolved_warmup = int(features.get("history_steps", 0))
    if resolved_dt <= 0.0:
        raise ValueError("dt_s must be supplied or present in the dataset manifest")
    if resolved_warmup < 0:
        raise ValueError("warmup_steps must be non-negative")
    return resolved_dt, resolved_warmup


def evaluate_split(
    checkpoint_path: str | Path,
    data_root: str | Path,
    split: str,
    confirmation_steps: int,
    output_directory: str | Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 20260716,
    device: str = "auto",
    dt_s: float | None = None,
    warmup_steps: int | None = None,
    save_predictions: bool = True,
) -> dict[str, object]:
    """Run frozen inference, K-step alarms, first-alarm event matching, and aggregation."""

    if not 1 <= confirmation_steps <= 5:
        raise ValueError("confirmation_steps must lie in [1, 5]")
    manifest = load_dataset_manifest(data_root)
    bundle = load_trained_detector(checkpoint_path, requested_device=device)
    checkpoint_warmup = bundle.checkpoint.get("monitoring_warmup_steps")
    resolved_dt, resolved_warmup = _evaluation_parameters(
        manifest,
        dt_s,
        warmup_steps,
        checkpoint_warmup_steps=(
            None if checkpoint_warmup is None else int(checkpoint_warmup)
        ),
    )
    episodes = load_split(data_root, split)
    episode_results: list[dict[str, object]] = []
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    output = Path(output_directory) if output_directory is not None else None
    prediction_directory = output / "predictions" / split if output else None
    if save_predictions and prediction_directory is not None:
        prediction_directory.mkdir(parents=True, exist_ok=True)

    for episode in episodes:
        logits, attack_probability, predictions = predict_episode(bundle, episode)
        alarm_events = k_step_alarm_events(
            predictions,
            confirmation_steps=confirmation_steps,
            warmup_steps=resolved_warmup,
        )
        attacks = attack_events_from_arrays(
            episode.has_attack,
            episode.attack_start_step,
            episode.attack_end_step_exclusive,
        )
        event_result = match_alarm_events(
            alarm_events,
            attacks,
            num_steps=episode.num_steps,
            dt_s=resolved_dt,
            warmup_steps=resolved_warmup,
        )
        event_result.update(
            {
                "episode_id": episode.episode_id,
                "scenario": str(episode.metadata.get("attack", {}).get("attack_type", "unknown")),
                "trajectory_type": str(
                    episode.metadata.get("trajectory_type", "unknown")
                ),
                "num_truncated_attacked_uavs": int(
                    np.sum(episode.attack_was_truncated & episode.has_attack)
                ),
            }
        )
        episode_results.append(event_result)
        all_labels.append(episode.labels.reshape(-1))
        all_predictions.append(predictions.reshape(-1))
        if save_predictions and prediction_directory is not None:
            np.savez_compressed(
                prediction_directory / f"{episode.episode_id}.npz",
                attack_logits=logits,
                attack_probability=attack_probability,
                predictions=predictions,
                labels=episode.labels,
            )

    aggregate = aggregate_event_results(
        episode_results,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    time_step_metrics = binary_detection_metrics(
        np.concatenate(all_labels), np.concatenate(all_predictions)
    )
    scenario_results: dict[str, object] = {}
    for scenario in sorted({str(result["scenario"]) for result in episode_results}):
        selected = [
            result for result in episode_results if str(result["scenario"]) == scenario
        ]
        scenario_results[scenario] = aggregate_event_results(
            selected,
            bootstrap_samples=0,
            seed=seed,
        )

    result: dict[str, object] = {
        "schema_version": "1.3",
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "confirmation_steps": int(confirmation_steps),
        "warmup_steps": resolved_warmup,
        "dt_s": resolved_dt,
        "output_mode": str(
            bundle.checkpoint["model_config"].get(
                "output_mode", "legacy_two_class_logits"
            )
        ),
        "decision_rule": (
            "sigmoid_attack_probability_gte_threshold"
            if int(bundle.checkpoint["model_config"].get("output_dim", 2)) == 1
            else "legacy_softmax_attack_probability_gte_threshold"
        ),
        "argmax_decision": False,
        "probability_threshold": ATTACK_PROBABILITY_THRESHOLD,
        "event_matching_policy": "first_alarm_per_attack_event",
        "duplicate_alarm_policy": "ignored",
        "aggregate": aggregate,
        "time_step_metrics": time_step_metrics,
        "by_scenario": scenario_results,
        "episodes": episode_results,
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"evaluation_{split}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(_json_ready(result), handle, ensure_ascii=False, indent=2)
    return result


def json_ready(value: Any) -> Any:
    """Public JSON conversion for experiment entry points."""

    return _json_ready(value)

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import (
    aggregate_event_results,
    attack_events_from_arrays,
    binary_detection_metrics,
    k_step_alarm_events,
    match_alarm_events,
)
from .raw_episode import RawEpisode


@dataclass(frozen=True)
class BaselineEvaluationConfig:
    dt_s: float
    confirmation_steps: int
    warmup_steps: int = 0
    bootstrap_samples: int = 0
    seed: int = 20260716

    def __post_init__(self) -> None:
        if self.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.confirmation_steps < 1:
            raise ValueError("confirmation_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative")


def evaluate_episode_predictions(
    episode: RawEpisode,
    predictions: np.ndarray,
    config: BaselineEvaluationConfig,
) -> dict[str, object]:
    hard = np.asarray(predictions, dtype=np.uint8)
    if hard.shape != episode.labels.shape:
        raise ValueError(
            f"episode {episode.episode_id}: predictions must have shape "
            f"{episode.labels.shape}"
        )
    if not np.all((hard == 0) | (hard == 1)):
        raise ValueError("predictions must contain only 0 and 1")

    alarms = k_step_alarm_events(
        hard,
        confirmation_steps=config.confirmation_steps,
        warmup_steps=config.warmup_steps,
    )
    attacks = attack_events_from_arrays(
        episode.require("has_attack"),
        episode.require("attack_start_step"),
        episode.require("attack_end_step_exclusive"),
    )
    result = match_alarm_events(
        alarms,
        attacks,
        num_steps=episode.num_steps,
        dt_s=config.dt_s,
        warmup_steps=config.warmup_steps,
    )
    attack_metadata = episode.metadata.get("attack", {})
    scenario = (
        str(attack_metadata.get("attack_type", "unknown"))
        if isinstance(attack_metadata, dict)
        else "unknown"
    )
    result.update(
        {
            "episode_id": episode.episode_id,
            "scenario": scenario,
            "trajectory_type": str(
                episode.metadata.get("trajectory_type", "unknown")
            ),
        }
    )
    return result


def evaluate_baseline_predictions(
    episodes: Sequence[RawEpisode],
    predictions: Sequence[np.ndarray],
    config: BaselineEvaluationConfig,
) -> dict[str, object]:
    """Apply the repository event protocol to arbitrary hard baseline outputs."""

    episode_list = tuple(episodes)
    prediction_list = tuple(predictions)
    if not episode_list:
        raise ValueError("at least one episode is required")
    if len(episode_list) != len(prediction_list):
        raise ValueError("episodes and predictions must have the same length")

    episode_results = [
        evaluate_episode_predictions(episode, hard, config)
        for episode, hard in zip(episode_list, prediction_list)
    ]
    aggregate = aggregate_event_results(
        episode_results,
        bootstrap_samples=config.bootstrap_samples,
        seed=config.seed,
    )
    time_step_metrics = binary_detection_metrics(
        np.concatenate([episode.labels.reshape(-1) for episode in episode_list]),
        np.concatenate([hard.reshape(-1) for hard in prediction_list]),
    )
    by_scenario: dict[str, object] = {}
    for scenario in sorted({str(item["scenario"]) for item in episode_results}):
        selected = [
            item for item in episode_results if str(item["scenario"]) == scenario
        ]
        by_scenario[scenario] = aggregate_event_results(
            selected,
            bootstrap_samples=0,
            seed=config.seed,
        )
    return {
        "aggregate": aggregate,
        "time_step_metrics": time_step_metrics,
        "by_scenario": by_scenario,
        "episodes": episode_results,
        "confirmation_steps": config.confirmation_steps,
        "warmup_steps": config.warmup_steps,
        "dt_s": config.dt_s,
        "event_matching_policy": "first_alarm_per_attack_event",
    }

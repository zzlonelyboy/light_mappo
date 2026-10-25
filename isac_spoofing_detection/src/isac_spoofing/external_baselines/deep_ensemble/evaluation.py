"""Unified event-level evaluation for the Deep Ensemble baseline."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from ...metrics import (
    aggregate_event_results,
    attack_events_from_arrays,
    k_step_alarm_events,
    match_alarm_events,
)
from .dataset import iter_sidecar_episodes
from .ensemble import load_ensemble, predict_positive_probability
from .model import PerBSModelSet
from .streaming_meta_mlp import load_streaming_meta_mlp


@dataclass(frozen=True)
class EpisodeProbability:
    episode_id: str
    probability: np.ndarray
    time_s: np.ndarray
    has_attack: np.ndarray
    attack_start_step: np.ndarray
    attack_end_step_exclusive: np.ndarray


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def predict_episodes(
    feature_dir: Path,
    split: str,
    per_bs_models: PerBSModelSet,
    ensemble_model,
    predicate: Callable[[str], bool] | None = None,
) -> list[EpisodeProbability]:
    predictions: list[EpisodeProbability] = []
    for episode in iter_sidecar_episodes(feature_dir, split, predicate):
        features = np.asarray(episode["features"], dtype=np.float32)
        bs_probabilities = per_bs_models.predict_proba(features)
        shape = bs_probabilities.shape[:-1]
        probability = predict_positive_probability(
            ensemble_model,
            bs_probabilities.reshape(-1, bs_probabilities.shape[-1]),
        ).reshape(shape)
        predictions.append(
            EpisodeProbability(
                episode_id=str(episode["episode_id"]),
                probability=probability,
                time_s=np.asarray(episode["time_s"]),
                has_attack=np.asarray(episode["has_attack"]),
                attack_start_step=np.asarray(episode["attack_start_step"]),
                attack_end_step_exclusive=np.asarray(
                    episode["attack_end_step_exclusive"]
                ),
            )
        )
    if not predictions:
        raise ValueError("no episodes were selected for evaluation")
    return predictions


def evaluate_episode_probabilities(
    episodes: Iterable[EpisodeProbability],
    threshold: float,
    confirmation_steps: int,
    warmup_steps: int = 0,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must lie in (0,1)")
    episode_results: list[dict[str, object]] = []
    for episode in episodes:
        hard = episode.probability >= threshold
        alarms = k_step_alarm_events(
            hard,
            confirmation_steps=confirmation_steps,
            warmup_steps=warmup_steps,
        )
        attacks = attack_events_from_arrays(
            episode.has_attack,
            episode.attack_start_step,
            episode.attack_end_step_exclusive,
        )
        if episode.time_s.size < 2:
            raise ValueError("episode time axis must contain at least two values")
        dt_s = float(np.median(np.diff(episode.time_s)))
        episode_results.append(
            match_alarm_events(
                alarms,
                attacks,
                num_steps=episode.probability.shape[0],
                dt_s=dt_s,
                warmup_steps=warmup_steps,
            )
        )
    return aggregate_event_results(
        episode_results,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )


def select_event_threshold(
    episodes: Iterable[EpisodeProbability],
    thresholds: Iterable[float],
    confirmation_steps: int,
) -> dict[str, object]:
    """Select an event-level operating threshold on validation episodes."""

    episode_list = tuple(episodes)
    candidate_values = tuple(float(value) for value in thresholds)
    if not candidate_values:
        raise ValueError("at least one threshold candidate is required")
    if len(set(candidate_values)) != len(candidate_values):
        raise ValueError("threshold candidates must be unique")

    rows: list[dict[str, float]] = []
    for threshold in candidate_values:
        metrics = evaluate_episode_probabilities(
            episode_list,
            threshold=threshold,
            confirmation_steps=confirmation_steps,
            bootstrap_samples=0,
        )
        delay = metrics["delay_mean_s"]
        mean_delay_s = float(delay) if delay is not None else math.inf
        row = {
            "threshold": threshold,
            "event_precision": float(metrics["event_precision"]),
            "event_recall": float(metrics["event_recall"]),
            "event_f1": float(metrics["event_f1"]),
            "false_alarms_per_uav_hour": float(
                metrics["false_alarms_per_uav_hour"]
            ),
            "mean_detection_delay_s": mean_delay_s,
        }
        rows.append(row)
        print(
            f"[Threshold search] threshold={threshold:.1f} "
            f"Event-F1={row['event_f1']:.6f} "
            f"FAR={row['false_alarms_per_uav_hour']:.6f} "
            f"delay={row['mean_detection_delay_s']:.6f}s",
            flush=True,
        )

    selected = min(
        rows,
        key=lambda row: (
            -row["event_f1"],
            row["false_alarms_per_uav_hour"],
            row["mean_detection_delay_s"],
        ),
    )
    return {
        "selection_split": "validation",
        "selection_metric": "event_f1",
        "tie_breaking": [
            "lower_false_alarms_per_uav_hour",
            "shorter_mean_detection_delay",
        ],
        "confirmation_steps": int(confirmation_steps),
        "selected": selected,
        "all_results": rows,
    }


def select_ensemble_by_bce(
    bs_probabilities: np.ndarray,
    labels: np.ndarray,
    ensemble_dir: Path,
    output_dir: Path,
    candidate_names: tuple[str, ...],
    decision_threshold: float = 0.5,
) -> dict[str, object]:
    """Select the Algorithm 3 meta-classifier by validation BCE."""

    values = np.asarray(bs_probabilities, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("ensemble inputs require shape (samples,BS) and labels")
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must lie in (0,1)")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilon = np.finfo(np.float64).eps
    rows: list[dict[str, object]] = []
    best_key = None
    best_row = None
    for name in candidate_names:
        model = load_ensemble(Path(ensemble_dir) / f"{name}.joblib")
        probabilities = predict_positive_probability(model, values)
        clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
        average_bce = float(
            -np.mean(
                targets * np.log(clipped)
                + (1 - targets) * np.log(1.0 - clipped)
            )
        )
        predictions = probabilities >= decision_threshold
        accuracy = float(np.mean(predictions == targets))
        row = {
            "candidate": name,
            "average_bce": average_bce,
            "step_accuracy_at_fixed_threshold": accuracy,
        }
        rows.append(row)
        key = (average_bce, -accuracy)
        if best_key is None or key < best_key:
            best_key = key
            best_row = row
        print(
            f"[Ensemble selection][{name}] validation_bce={average_bce:.6f} "
            f"step_accuracy@{decision_threshold:.2f}={accuracy:.6f}",
            flush=True,
        )
    if best_row is None:
        raise RuntimeError("no ensemble candidate was evaluated")
    shutil.copy2(
        Path(ensemble_dir) / f"{best_row['candidate']}.joblib",
        output_dir / "selected_ensemble.joblib",
    )
    payload = {
        "selection_metric": "average_binary_cross_entropy",
        "decision_threshold": float(decision_threshold),
        "selected": {
            **best_row,
            "threshold": float(decision_threshold),
        },
        "all_results": rows,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(_json_ready(payload), indent=2),
        encoding="utf-8",
    )
    return payload


def evaluate_selected_model(
    feature_dir: Path,
    model_dir: Path,
    selection_dir: Path,
    split: str,
    confirmation_steps: int,
    output_path: Path,
    device: str = "auto",
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
    decision_threshold: float | None = None,
    threshold_candidates: Iterable[float] | None = None,
    threshold_search_output_path: Path | None = None,
) -> dict[str, object]:
    selection = json.loads(
        (Path(selection_dir) / "selection.json").read_text(encoding="utf-8")
    )
    threshold = (
        float(selection["selected"]["threshold"])
        if decision_threshold is None
        else float(decision_threshold)
    )
    per_bs_models = PerBSModelSet.load(model_dir, device=device)
    if selection.get("model_format") == "torch_streaming_meta_mlp":
        ensemble = load_streaming_meta_mlp(
            Path(selection_dir) / str(selection["model_file"]),
            device=device,
        )
    else:
        ensemble = load_ensemble(Path(selection_dir) / "selected_ensemble.joblib")
    episode_predictions = predict_episodes(
        feature_dir,
        split,
        per_bs_models,
        ensemble,
    )
    threshold_search = None
    if threshold_candidates is not None:
        if split != "validation":
            raise ValueError("threshold search is only permitted on validation")
        threshold_search = select_event_threshold(
            episode_predictions,
            thresholds=threshold_candidates,
            confirmation_steps=confirmation_steps,
        )
        threshold = float(threshold_search["selected"]["threshold"])
        if threshold_search_output_path is not None:
            Path(threshold_search_output_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            Path(threshold_search_output_path).write_text(
                json.dumps(_json_ready(threshold_search), indent=2),
                encoding="utf-8",
            )
    metrics = evaluate_episode_probabilities(
        episode_predictions,
        threshold=threshold,
        confirmation_steps=confirmation_steps,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    payload = {
        "split": split,
        "candidate": selection["selected"]["candidate"],
        "threshold": threshold,
        "confirmation_steps": confirmation_steps,
        "metrics": metrics,
    }
    if threshold_search is not None:
        payload["threshold_selection"] = {
            "split": "validation",
            "metric": "event_f1",
        }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(_json_ready(payload), indent=2),
        encoding="utf-8",
    )
    return payload

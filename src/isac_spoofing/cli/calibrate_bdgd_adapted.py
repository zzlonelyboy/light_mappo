from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..baseline_evaluation import (
    BaselineEvaluationConfig,
    evaluate_baseline_predictions,
)
from ..baselines.bdgd_adapted import (
    BDGDAdaptedConfig,
    BDGDAdaptedDetector,
    build_local_evidence_episode,
    estimate_connection_probability,
    estimate_missed_attack_probability,
    gaussian_false_accusation_probability,
    paper_adaptive_threshold_scalar,
)
from ..evaluation import json_ready, load_dataset_manifest
from ..metrics import binary_detection_metrics
from ..raw_episode import RawEpisode, load_raw_split


RAW_ARRAYS = (
    "gps_observed_m",
    "inter_uav_range_m",
    "inter_uav_valid_mask",
    "inter_uav_range_variance",
    "labels",
    "has_attack",
    "attack_start_step",
    "attack_end_step_exclusive",
    "attack_was_truncated",
)


def _float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    if any(item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("all tolerance multipliers must be positive")
    return result


def _dataset_settings(
    manifest: dict[str, object],
    num_neighbors_override: int | None,
) -> tuple[float, int, int]:
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("dataset manifest config must be an object")
    simulation = config.get("simulation", {})
    features = config.get("features", {})
    if not isinstance(simulation, dict) or not isinstance(features, dict):
        raise ValueError("dataset manifest is missing simulation or feature settings")
    dt_s = float(simulation.get("dt_s", 0.0))
    num_neighbors = (
        int(num_neighbors_override)
        if num_neighbors_override is not None
        else int(features.get("num_neighbors", 3))
    )
    warmup_steps = int(features.get("warmup_steps", 0))
    if dt_s <= 0.0:
        raise ValueError("dataset manifest must define a positive simulation dt_s")
    if num_neighbors < 1:
        raise ValueError("num_neighbors must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    return dt_s, num_neighbors, warmup_steps


def _scenario(episode: RawEpisode) -> str:
    attack = episode.metadata.get("attack", {})
    if isinstance(attack, dict):
        return str(attack.get("attack_type", "unknown"))
    return "unknown"


def _finite_or_infinity(value: object) -> float:
    converted = float(value)
    return converted if np.isfinite(converted) else float("inf")


def _select_candidate(
    candidates: Sequence[dict[str, object]],
    f1_tie_tolerance: float,
) -> dict[str, object]:
    best_f1 = max(float(item["event_f1"]) for item in candidates)
    tied = [
        item
        for item in candidates
        if best_f1 - float(item["event_f1"]) <= f1_tie_tolerance
    ]
    return min(
        tied,
        key=lambda item: (
            _finite_or_infinity(item["false_alarms_per_uav_hour"]),
            -float(item["event_precision"]),
            _finite_or_infinity(item["delay_mean_s"]),
            float(item["error_tolerance_multiplier"]),
        ),
    )


def calibrate(
    data_root: str | Path,
    output_path: str | Path,
    error_tolerance_multipliers: Sequence[float],
    confirmation_steps: int,
    num_neighbors_override: int | None = None,
    max_episodes: int | None = None,
    variance_floor: float = 1.0e-8,
    f1_tie_tolerance: float = 1.0e-4,
) -> dict[str, object]:
    """Calibrate P1, P2 and n only on the validation split."""

    if confirmation_steps < 1:
        raise ValueError("confirmation_steps must be positive")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    if f1_tie_tolerance < 0.0:
        raise ValueError("f1_tie_tolerance must be non-negative")
    multipliers = tuple(sorted(set(map(float, error_tolerance_multipliers))))
    if not multipliers or any(value <= 0.0 for value in multipliers):
        raise ValueError("positive error-tolerance multipliers are required")

    manifest = load_dataset_manifest(data_root)
    dt_s, num_neighbors, warmup_steps = _dataset_settings(
        manifest,
        num_neighbors_override,
    )
    episodes = load_raw_split(
        data_root,
        "validation",
        array_names=RAW_ARRAYS,
        max_episodes=max_episodes,
    )
    if not episodes:
        raise ValueError("validation split contains no episodes")
    num_uavs = episodes[0].num_uavs
    if any(episode.num_uavs != num_uavs for episode in episodes):
        raise ValueError("all calibration episodes must have the same UAV count")
    if not 1 <= num_neighbors <= num_uavs - 1:
        raise ValueError("num_neighbors must lie in [1,num_uavs-1]")

    print(
        "Building the validation neighbor graphs used to estimate the "
        "environment-specific BDGD probabilities."
    )
    evidence = tuple(
        build_local_evidence_episode(
            episode,
            num_neighbors=num_neighbors,
            variance_floor=variance_floor,
            symmetrize_neighbors=True,
        )
        for episode in episodes
    )
    connection_probability, selected_link_count, eligible_pair_count = (
        estimate_connection_probability(
            evidence,
            [episode.require("inter_uav_valid_mask") for episode in episodes],
            warmup_steps=warmup_steps,
        )
    )
    evaluation_config = BaselineEvaluationConfig(
        dt_s=dt_s,
        confirmation_steps=confirmation_steps,
        warmup_steps=warmup_steps,
        bootstrap_samples=0,
    )

    candidates: list[dict[str, object]] = []
    for multiplier in multipliers:
        print(
            "Evaluating validation event performance for BDGD tolerance "
            f"multiplier n={multiplier:g}."
        )
        missed_probability, missed_count, attack_link_count = (
            estimate_missed_attack_probability(
                evidence,
                [episode.labels for episode in episodes],
                error_tolerance_multiplier=multiplier,
                warmup_steps=warmup_steps,
            )
        )
        false_probability = gaussian_false_accusation_probability(multiplier)
        detector = BDGDAdaptedDetector(
            BDGDAdaptedConfig(
                error_tolerance_multiplier=multiplier,
                num_neighbors=num_neighbors,
                connection_probability=connection_probability,
                missed_attack_probability=missed_probability,
                variance_floor=variance_floor,
                symmetrize_neighbors=True,
            )
        )
        detector_results = [
            detector.run_episode(
                episode,
                malicious_count=int(np.sum(episode.require("has_attack"))),
            )
            for episode in episodes
        ]
        evaluation = evaluate_baseline_predictions(
            episodes,
            [item.predictions for item in detector_results],
            evaluation_config,
        )
        aggregate = evaluation["aggregate"]
        step = binary_detection_metrics(
            np.concatenate(
                [episode.labels[warmup_steps:].reshape(-1) for episode in episodes]
            ),
            np.concatenate(
                [
                    item.predictions[warmup_steps:].reshape(-1)
                    for item in detector_results
                ]
            ),
        )
        labels_flat = np.concatenate(
            [episode.labels[warmup_steps:].reshape(-1) for episode in episodes]
        )
        scores_flat = np.concatenate(
            [
                item.reputation_score[warmup_steps:].reshape(-1)
                for item in detector_results
            ]
        )
        attacked_scores = scores_flat[labels_flat == 1]
        benign_scores = scores_flat[labels_flat == 0]
        if attacked_scores.size == 0 or benign_scores.size == 0:
            raise ValueError(
                "validation calibration requires attacked and benign time slots"
            )
        threshold_by_m = {
            str(malicious_count): paper_adaptive_threshold_scalar(
                num_uavs=num_uavs,
                malicious_count=malicious_count,
                connection_probability=connection_probability,
                missed_attack_probability=missed_probability,
                false_accusation_probability=false_probability,
            )
            for malicious_count in range(num_uavs)
        }
        candidates.append(
            {
                "error_tolerance_multiplier": multiplier,
                "connection_probability": connection_probability,
                "missed_attack_probability": missed_probability,
                "false_accusation_probability": false_probability,
                "threshold_by_m": threshold_by_m,
                "selected_neighbor_link_count": selected_link_count,
                "eligible_directed_pair_count": eligible_pair_count,
                "missed_attack_link_count": missed_count,
                "attacked_neighbor_link_count": attack_link_count,
                "mean_reputation_attack_slots": float(np.mean(attacked_scores)),
                "mean_reputation_benign_slots": float(np.mean(benign_scores)),
                "reputation_ordering_valid": bool(
                    np.mean(attacked_scores) > np.mean(benign_scores)
                ),
                "step_precision": step["precision"],
                "step_recall": step["recall"],
                "step_f1": step["f1"],
                "step_false_positive_rate": step["false_alarm_rate"],
                "event_precision": aggregate["event_precision"],
                "event_recall": aggregate["event_recall"],
                "event_f1": aggregate["event_f1"],
                "false_alarms_per_uav_hour": aggregate[
                    "false_alarms_per_uav_hour"
                ],
                "delay_mean_s": aggregate["delay_mean_s"],
            }
        )

    if not all(bool(item["reputation_ordering_valid"]) for item in candidates):
        print(
            "WARNING: at least one candidate does not assign a higher mean "
            "reputation score to attacked UAV/time samples."
        )
    selected = _select_candidate(candidates, f1_tie_tolerance)
    result: dict[str, object] = {
        "schema_version": "3.0",
        "method": "BDGD-Adapted",
        "status": "validation_calibrated",
        "source_method_doi": "10.1016/j.ress.2025.112101",
        "data_root": str(Path(data_root).resolve()),
        "calibration_split": "validation",
        "calibration_episode_count": len(episodes),
        "episode_ids": [episode.episode_id for episode in episodes],
        "scenario_counts": {
            scenario: sum(_scenario(episode) == scenario for episode in episodes)
            for scenario in sorted({_scenario(episode) for episode in episodes})
        },
        "num_uavs": num_uavs,
        "num_neighbors": num_neighbors,
        "confirmation_steps": confirmation_steps,
        "warmup_steps": warmup_steps,
        "variance_floor": variance_floor,
        "symmetric_neighbor_graph": True,
        "selected_n": selected["error_tolerance_multiplier"],
        "connection_probability": selected["connection_probability"],
        "missed_attack_probability": selected["missed_attack_probability"],
        "false_accusation_probability": selected[
            "false_accusation_probability"
        ],
        "threshold_by_m": selected["threshold_by_m"],
        "threshold_scope": "episode-level constant",
        "adaptive_threshold_attack_count_source": (
            "episode ground-truth scenario count; paper-aligned oracle prior"
        ),
        "p1_definition": (
            "selected directed neighbor links divided by eligible directed pairs "
            "on the validation split"
        ),
        "p2_definition": (
            "P(z_ij <= n | target UAV attacked at the time step, selected valid "
            "neighbor link) on the validation split"
        ),
        "p3_definition": "erfc(n / sqrt(2))",
        "selection_policy": (
            "maximum validation Event F1; within the configured F1 tolerance, "
            "lower FAR, higher Event Precision, lower mean delay, then smaller n"
        ),
        "f1_tie_tolerance": f1_tie_tolerance,
        "candidate_results": candidates,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the environment-adapted BDGD probabilities and select n "
            "using validation event performance"
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--error-tolerance-multipliers",
        type=_float_list,
        default=(1.0, 2.0, 3.0, 4.0, 5.0),
    )
    parser.add_argument("--confirmation-steps", type=int, default=5)
    parser.add_argument("--num-neighbors", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--variance-floor", type=float, default=1.0e-8)
    parser.add_argument("--f1-tie-tolerance", type=float, default=1.0e-4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = calibrate(
        data_root=args.data_root,
        output_path=args.output,
        error_tolerance_multipliers=args.error_tolerance_multipliers,
        confirmation_steps=args.confirmation_steps,
        num_neighbors_override=args.num_neighbors,
        max_episodes=args.max_episodes,
        variance_floor=args.variance_floor,
        f1_tie_tolerance=args.f1_tie_tolerance,
    )
    print(
        json.dumps(
            {
                "selected_n": result["selected_n"],
                "connection_probability": result["connection_probability"],
                "missed_attack_probability": result[
                    "missed_attack_probability"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..baseline_evaluation import (
    BaselineEvaluationConfig,
    evaluate_baseline_predictions,
)
from ..baselines.bdgd_adapted import (
    BDGDAdaptedConfig,
    BDGDAdaptedDetector,
    gaussian_false_accusation_probability,
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


def _load_calibration(path: str | Path) -> dict[str, object]:
    calibration_path = Path(path)
    with calibration_path.open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    if not isinstance(calibration, dict):
        raise ValueError("calibration artifact must be a JSON object")
    if calibration.get("method") != "BDGD-Adapted":
        raise ValueError("calibration artifact is not for BDGD-Adapted")
    if calibration.get("status") != "validation_calibrated":
        raise ValueError("BDGD calibration must come from the validation split")
    if calibration.get("calibration_split") != "validation":
        raise ValueError("test evaluation refuses a non-validation calibration")
    required = {
        "selected_n",
        "connection_probability",
        "missed_attack_probability",
        "false_accusation_probability",
        "num_uavs",
        "num_neighbors",
        "confirmation_steps",
        "warmup_steps",
        "variance_floor",
    }
    missing = required - set(calibration)
    if missing:
        raise ValueError(f"calibration artifact is missing fields: {sorted(missing)}")
    return calibration


def _dataset_dt_s(manifest: dict[str, object]) -> float:
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("dataset manifest config must be an object")
    simulation = config.get("simulation", {})
    if not isinstance(simulation, dict):
        raise ValueError("dataset manifest is missing simulation settings")
    dt_s = float(simulation.get("dt_s", 0.0))
    if dt_s <= 0.0:
        raise ValueError("dataset manifest must define a positive simulation dt_s")
    return dt_s


def _scenario(episode: RawEpisode) -> str:
    attack = episode.metadata.get("attack", {})
    if isinstance(attack, dict):
        return str(attack.get("attack_type", "unknown"))
    return "unknown"


def evaluate(
    data_root: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    max_episodes: int | None = None,
) -> dict[str, object]:
    """Evaluate fixed validation-calibrated BDGD parameters on one split."""

    calibration = _load_calibration(calibration_path)
    manifest = load_dataset_manifest(data_root)
    dt_s = _dataset_dt_s(manifest)
    episodes = load_raw_split(
        data_root,
        split,
        array_names=RAW_ARRAYS,
        max_episodes=max_episodes,
    )
    if not episodes:
        raise ValueError(f"{split} split contains no episodes")
    expected_num_uavs = int(calibration["num_uavs"])
    if any(episode.num_uavs != expected_num_uavs for episode in episodes):
        raise ValueError("evaluation UAV count does not match the calibration")

    selected_n = float(calibration["selected_n"])
    connection_probability = float(calibration["connection_probability"])
    missed_attack_probability = float(calibration["missed_attack_probability"])
    num_neighbors = int(calibration["num_neighbors"])
    confirmation_steps = int(calibration["confirmation_steps"])
    warmup_steps = int(calibration["warmup_steps"])
    variance_floor = float(calibration["variance_floor"])
    expected_p3 = gaussian_false_accusation_probability(selected_n)
    if not np.isclose(
        float(calibration["false_accusation_probability"]),
        expected_p3,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("calibration P3 is inconsistent with its selected n")
    print(
        "Evaluating fixed validation-calibrated BDGD parameters with one "
        "episode-level reputation threshold determined by the scenario M."
    )
    detector = BDGDAdaptedDetector(
        BDGDAdaptedConfig(
            error_tolerance_multiplier=selected_n,
            num_neighbors=num_neighbors,
            connection_probability=connection_probability,
            missed_attack_probability=missed_attack_probability,
            variance_floor=variance_floor,
            symmetrize_neighbors=True,
        )
    )
    malicious_counts = [
        int(np.sum(episode.require("has_attack"))) for episode in episodes
    ]
    detector_results = [
        detector.run_episode(episode, malicious_count=malicious_count)
        for episode, malicious_count in zip(episodes, malicious_counts)
    ]
    evaluation = evaluate_baseline_predictions(
        episodes,
        [item.predictions for item in detector_results],
        BaselineEvaluationConfig(
            dt_s=dt_s,
            confirmation_steps=confirmation_steps,
            warmup_steps=warmup_steps,
            bootstrap_samples=0,
        ),
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
    episode_diagnostics = [
        {
            "episode_id": episode.episode_id,
            "scenario": _scenario(episode),
            "malicious_uav_count_oracle": malicious_count,
            "adaptive_threshold": detector_result.adaptive_threshold,
            "threshold_is_constant_within_episode": True,
            "mean_reputation": float(np.mean(detector_result.reputation_score)),
            "positive_prediction_count": int(
                np.sum(detector_result.predictions)
            ),
        }
        for episode, malicious_count, detector_result in zip(
            episodes,
            malicious_counts,
            detector_results,
        )
    ]
    result: dict[str, object] = {
        "schema_version": "3.0",
        "method": "BDGD-Adapted",
        "status": "fixed_calibration_evaluation",
        "data_root": str(Path(data_root).resolve()),
        "split": split,
        "episode_count": len(episodes),
        "calibration_artifact": str(Path(calibration_path).resolve()),
        "calibration_split": "validation",
        "selected_n": selected_n,
        "connection_probability": connection_probability,
        "missed_attack_probability": missed_attack_probability,
        "false_accusation_probability": float(
            calibration["false_accusation_probability"]
        ),
        "num_uavs": expected_num_uavs,
        "num_neighbors": num_neighbors,
        "confirmation_steps": confirmation_steps,
        "warmup_steps": warmup_steps,
        "threshold_scope": "episode-level constant",
        "adaptive_threshold_attack_count_source": (
            "episode ground-truth scenario count; paper-aligned oracle prior"
        ),
        "test_time_parameter_policy": (
            "n, P1, P2, and P3 are loaded unchanged from validation; only Rc(M) "
            "is evaluated once for each episode"
        ),
        "step_metrics_are_diagnostic_only": True,
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
        "by_scenario": evaluation["by_scenario"],
        "episodes": episode_diagnostics,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(result), handle, ensure_ascii=False, indent=2)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BDGD-Adapted using fixed validation-calibrated parameters "
            "and the common event-level protocol"
        )
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-episodes", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(
        data_root=args.data_root,
        calibration_path=args.calibration,
        output_path=args.output,
        split=args.split,
        max_episodes=args.max_episodes,
    )
    print(
        json.dumps(
            {
                "step_f1": result["step_f1"],
                "event_f1": result["event_f1"],
                "false_alarms_per_uav_hour": result[
                    "false_alarms_per_uav_hour"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

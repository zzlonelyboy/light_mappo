"""Evaluate a frozen checkpoint with and without alarm-driven ISAC navigation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from isac_spoofing.config import load_config
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.evaluation import StreamingDetector, json_ready
from isac_spoofing.metrics import (
    aggregate_event_results,
    alarm_availability_offset_steps,
    attack_events_from_arrays,
    k_step_alarm_events,
    match_alarm_events,
)
from isac_spoofing.training import load_trained_detector


def summarize_rollout(episode, probabilities: np.ndarray, config) -> dict[str, object]:
    """Use labels and true trajectories only after the rollout for evaluation."""
    confirmation_steps = config.navigation.confirmation_steps
    hard = (probabilities >= config.navigation.alarm_threshold).astype(np.uint8)
    events = k_step_alarm_events(
        hard, confirmation_steps=confirmation_steps,
        warmup_steps=config.features.warmup_steps,
    )
    attacks = attack_events_from_arrays(
        episode.has_attack, episode.attack_start_step,
        episode.attack_end_step_exclusive,
    )
    event_metrics = match_alarm_events(
        events, attacks, num_steps=hard.shape[0], dt_s=config.simulation.dt_s,
        warmup_steps=config.features.warmup_steps,
        alarm_availability_offset_steps=alarm_availability_offset_steps(episode.metadata),
    )
    error = episode.true_position_m - episode.reference_position_m
    norm = np.linalg.norm(error, axis=-1)
    actual_pair = np.linalg.norm(
        episode.true_position_m[:, :, None, :] - episode.true_position_m[:, None, :, :],
        axis=-1,
    )
    reference_pair = np.linalg.norm(
        episode.reference_position_m[:, :, None, :] - episode.reference_position_m[:, None, :, :],
        axis=-1,
    )
    upper = np.triu_indices(hard.shape[1], k=1)
    pair_error = (actual_pair - reference_pair)[:, upper[0], upper[1]]
    return {
        "events": event_metrics,
        "tracking_rmse_m": float(np.sqrt(np.mean(norm**2))),
        "tracking_max_m": float(norm.max()),
        "final_tracking_mean_m": float(norm[-1].mean()),
        "formation_range_rmse_m": float(np.sqrt(np.mean(pair_error**2))) if pair_error.size else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "default.json",
    )
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trajectory-type", choices=("straight", "circle", "helix", "s_curve"))
    parser.add_argument("--attack-type", choices=("clean", "random", "replay", "stealth"))
    parser.add_argument(
        "--allow-domain-shift", action="store_true",
        help="Explicitly permit legacy or mismatched physics, labelled unvalidated in output.",
    )
    parser.add_argument("--output", type=Path, default=Path("tmp/navigation_mitigation_evaluation.json"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = load_config(args.config)
    if not config.navigation.enabled or not config.navigation.mitigation_enabled:
        parser.error("both navigation.enabled and navigation.mitigation_enabled must be true")
    bundle = load_trained_detector(args.checkpoint, requested_device=args.device)
    results: list[dict[str, object]] = []
    domain_status = None
    for index in range(args.episodes):
        seed = args.seed + index
        common_environment = ISACSpoofingEnvironment(config)
        common_policy = StreamingDetector(
            bundle, common_environment.feature_extractor.feature_names,
            config.to_dict(), allow_domain_shift=args.allow_domain_shift,
        )
        # Collect the same GPS-feedback trajectory for detection comparison.
        # No detector controls this collection, and no true label switches navigation.
        common = common_environment.run_episode(
            seed=seed, trajectory_type=args.trajectory_type,
            attack_type=args.attack_type,
        )
        common_probability = np.stack([common_policy(row) for row in common.features])
        online_environment = ISACSpoofingEnvironment(config)
        online_policy = StreamingDetector(
            bundle, online_environment.feature_extractor.feature_names,
            config.to_dict(), allow_domain_shift=args.allow_domain_shift,
        )
        online = online_environment.run_episode(
            seed=seed, trajectory_type=args.trajectory_type,
            attack_type=args.attack_type, detector=online_policy,
        )
        online_probability = np.stack(online_policy.probabilities)
        domain_status = online_policy.domain_status
        results.append({
            "seed": seed,
            "scenario": common.metadata.get("attack"),
            "shared_gps_feedback": summarize_rollout(common, common_probability, config),
            "online_alarm_mitigation": summarize_rollout(online, online_probability, config),
        })
        print(f"Completed paired episode {index + 1}/{args.episodes}, seed={seed}", flush=True)
    output = {
        "schema_version": "1.0",
        "checkpoint": str(args.checkpoint.resolve()),
        "config": config.to_dict(),
        "checkpoint_domain": domain_status,
        "comparison_policy": "paired external seeds; online paths may diverge after each detector's alarms",
        "shared_detection": aggregate_event_results(
            [r["shared_gps_feedback"]["events"] for r in results], bootstrap_samples=0,
        ),
        "online_detection": aggregate_event_results(
            [r["online_alarm_mitigation"]["events"] for r in results], bootstrap_samples=0,
        ),
        "episodes": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_ready(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

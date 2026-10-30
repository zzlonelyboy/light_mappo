"""Audit GPS feedback and detector-driven next-slot ISAC navigation.

Controlled scores test implementation causality, not trained detection quality.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isac_spoofing.config import ProjectConfig, load_config
from isac_spoofing.environment import ISACSpoofingEnvironment

TRAJECTORIES = ("straight", "circle", "helix", "s_curve")
SCENARIOS = ("clean", "random", "replay", "stealth")


def audit_dynamics(episode, config: ProjectConfig) -> dict[str, object]:
    dt = config.simulation.dt_s
    command = episode.command_velocity_mps
    navigation = episode.navigation_position_m
    assert command is not None and navigation is not None
    gain = np.array([config.navigation.position_gain_xy_s_inv] * 2
                    + [config.navigation.position_gain_z_s_inv])
    requested = (episode.reference_velocity_mps.astype(np.float64)
                 + gain * (episode.reference_position_m.astype(np.float64)
                           - navigation.astype(np.float64)))
    norm = np.linalg.norm(requested, axis=-1, keepdims=True)
    factor = np.ones_like(norm)
    nonzero = norm > 0
    factor[nonzero] = np.minimum(1.0, config.navigation.max_speed_mps / norm[nonzero])
    np.testing.assert_allclose(command, requested * factor, atol=4e-5, rtol=1e-6)
    np.testing.assert_array_equal(episode.velocity_mps, command)
    np.testing.assert_allclose(np.diff(episode.true_position_m, axis=0),
                               dt * command[:-1], atol=2e-5, rtol=1e-5)
    np.testing.assert_allclose(episode.available_time_s,
                               (np.arange(len(command)) + 1) * dt, atol=1e-10)
    assert np.isfinite(episode.features).all()
    assert episode.features.shape[-1] == 10
    maximum_speed = float(np.linalg.norm(command, axis=-1).max())
    assert maximum_speed <= config.navigation.max_speed_mps + 1e-5
    tracking = np.linalg.norm(episode.tracking_error_m, axis=-1)
    return {
        "command_law_verified": True, "ideal_velocity_tracking_verified": True,
        "position_recursion_verified": True, "slot_end_availability_verified": True,
        "features_finite": True, "feature_dimension": 10,
        "max_speed_mps": maximum_speed,
        "tracking_rmse_m": float(np.sqrt(np.mean(tracking ** 2))),
        "max_tracking_error_m": float(tracking.max()),
    }


def audit_false_alarm_switch(config: ProjectConfig, seed: int) -> dict[str, object]:
    dt = config.simulation.dt_s
    count = config.navigation.confirmation_steps
    short = replace(config, simulation=replace(
        config.simulation, duration_s=max(0.8, (count + 8) * dt),
        trajectory_types=("straight",)))
    calls = 0

    def controlled_scores(features: np.ndarray) -> np.ndarray:
        nonlocal calls
        assert features.shape == (config.simulation.num_uavs, 10)
        assert np.isfinite(features).all()
        calls += 1
        scores = np.zeros(config.simulation.num_uavs)
        scores[0] = 0.99 if calls <= count else 0.01
        return scores

    switched = ISACSpoofingEnvironment(short).run_episode(
        seed=seed, trajectory_type="straight", attack_type="clean",
        detector=controlled_scores)
    gps_only = ISACSpoofingEnvironment(short).run_episode(
        seed=seed, trajectory_type="straight", attack_type="clean")
    assert not np.any(switched.labels)
    assert not np.any(switched.navigation_mode[:count])
    assert switched.confirmed_alarm[count - 1, 0]
    assert np.all(switched.navigation_mode[count:, 0] == 1)
    assert not np.any(switched.navigation_mode[:, 1:])
    assert not np.any(gps_only.navigation_mode)
    assert not np.any(gps_only.confirmed_alarm)
    np.testing.assert_array_equal(switched.true_position_m[:count + 1],
                                  gps_only.true_position_m[:count + 1])
    np.testing.assert_array_equal(switched.command_velocity_mps[:count],
                                  gps_only.command_velocity_mps[:count])
    np.testing.assert_allclose(
        switched.navigation_position_m[count:, 0],
        switched.isac_position_m[count - 1:-1, 0]
        + dt * switched.command_velocity_mps[count - 1:-1, 0],
        atol=1e-5, rtol=1e-6)
    np.testing.assert_array_equal(switched.navigation_position_m[:, 1:],
                                  switched.gps_observed_m[:, 1:])
    np.testing.assert_array_equal(switched.gps_observed_m, switched.gps_clean_m)
    audit_dynamics(switched, short)
    return {
        "scenario": "clean_with_controlled_false_alarm",
        "confirmation_steps": count,
        "first_confirmed_alarm_step": count - 1,
        "first_isac_navigation_step": count,
        "alarm_available_time_s": float(switched.available_time_s[count - 1]),
        "false_alarm_switches_without_attack_label": True,
        "next_slot_only": True, "navigation_mode_latched": True,
        "previous_isac_plus_command_propagation_verified": True,
        "gps_report_not_overwritten": True,
        "unattached_detector_does_not_invent_alarms": True,
        "detector_callback_inputs": "features only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--duration-s", type=float, default=6.0)
    parser.add_argument("--trajectory-types", nargs="+", choices=TRAJECTORIES,
                        default=list(TRAJECTORIES))
    parser.add_argument("--output", type=Path,
                        default=Path("tmp/navigation_model_validation.json"))
    args = parser.parse_args()
    base = load_config(args.config) if args.config else ProjectConfig()
    config = replace(base, simulation=replace(
        base.simulation, duration_s=args.duration_s,
        trajectory_types=tuple(args.trajectory_types)))
    config.validate()
    if not config.navigation.enabled or not config.navigation.mitigation_enabled:
        parser.error("navigation.enabled and navigation.mitigation_enabled must be true")
    delay = max(1, int(round(max(high for _, high in config.attack.replay_delay_ranges_s)
                            / config.simulation.dt_s)))
    active = min(config.simulation.num_steps,
                 int(max(config.attack.duration_fraction_range)
                     * config.simulation.num_steps) + 1)
    if delay + active > config.simulation.num_steps:
        parser.error("duration is too short for complete replay history; increase "
                     "--duration-s or shorten the configured replay delays")
    results = []
    for trajectory in args.trajectory_types:
        print(f"Checking GPS-feedback dynamics on {trajectory}.", flush=True)
        for scenario in SCENARIOS:
            episode = ISACSpoofingEnvironment(config).run_episode(
                seed=args.seed, trajectory_type=trajectory, attack_type=scenario)
            record = audit_dynamics(episode, config)
            assert not np.any(episode.navigation_mode)
            assert not np.any(episode.confirmed_alarm)
            np.testing.assert_array_equal(episode.navigation_position_m,
                                          episode.gps_observed_m)
            if scenario == "replay":
                attack = episode.metadata["attack"]
                lag = int(attack["parameters"]["delay_steps"])
                start, end = int(attack["start_step"]), int(attack["end_step"])
                targets = np.asarray(attack["target_uavs"], dtype=int)
                assert start >= lag
                np.testing.assert_array_equal(episode.gps_observed_m[start:end, targets],
                                              episode.gps_clean_m[start - lag:end - lag, targets])
                record["replay_uses_available_noisy_history"] = True
            record.update({"trajectory": trajectory, "scenario": scenario,
                           "attack": episode.metadata["attack"],
                           "collection_mode": "GPS_feedback_without_alarm_feedback"})
            results.append(record)
            print(f"  {scenario}: command, integration and timing passed; "
                  f"max tracking {record['max_tracking_error_m']:.3f} m.", flush=True)
    switching = audit_false_alarm_switch(config, args.seed)
    print(f"False-alarm switching passed at step {switching['first_isac_navigation_step']}.",
          flush=True)
    payload = {
        "schema_version": "3.0", "seed": args.seed, "config": config.to_dict(),
        "results": results, "false_alarm_switch": switching,
        "scope": "implementation checks only; no trained detector or performance estimates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    print(f"Saved {len(results)} dynamics cases and switching audit to "
          f"{args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()

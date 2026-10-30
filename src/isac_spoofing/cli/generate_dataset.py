from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

import numpy as np

from isac_spoofing.config import load_config
from isac_spoofing.data import save_episode_shard
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.features import feature_names_for_neighbor_mode
from isac_spoofing.types import EpisodeData


SCENARIO_TYPES: tuple[str, ...] = ("clean", "random", "replay", "stealth")
TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
VALIDATION_END_FRACTION = TRAIN_FRACTION + VALIDATION_FRACTION
SPLIT_POLICY = "episode-level 80/10/10"
TEST_ONLY_SPLIT_POLICY = "test-only"
SPLIT_MODES = ("standard", "test-only")
STRENGTH_UNITS: dict[str, str] = {
    "random": "m",
    "replay": "s",
    "stealth": "m/s",
}


_WORKER_ENVIRONMENT: ISACSpoofingEnvironment | None = None
_WORKER_AUDIT_LEVEL = "full"
AUDIT_LEVELS = ("standard", "full")
OPTIONAL_COVARIANCE_AUDIT_FIELDS = (
    "isac_measurement_covariance",
    "isac_reported_covariance",
    "isac_reported_los_covariance",
)


def _apply_audit_level(episode: EpisodeData, audit_level: str) -> EpisodeData:
    """Select stored audit matrices after all measurements/features are computed."""
    if audit_level == "full":
        return episode
    if audit_level != "standard":
        raise ValueError(f"unsupported audit level: {audit_level!r}")
    return replace(episode, **{
        name: None for name in OPTIONAL_COVARIANCE_AUDIT_FIELDS
    })


def _bounded_episode_batches(tasks: list, workers: int):
    """Bound ordered Pool.imap's pending full-episode results to 2*workers."""
    batch_size = max(1, 2 * workers)
    for first in range(0, len(tasks), batch_size):
        yield tasks[first:first + batch_size]


def _initialize_episode_worker(
    config_path: str | None, audit_level: str = "full", replay_source: str | None = None
) -> None:
    """Create one reusable simulator inside each worker process."""
    global _WORKER_ENVIRONMENT, _WORKER_AUDIT_LEVEL
    path = None if config_path is None else Path(config_path)
    config = load_config(path)
    if replay_source is not None:
        config = replace(config, attack=replace(config.attack, replay_source=replay_source))
    _WORKER_ENVIRONMENT = ISACSpoofingEnvironment(config)
    _WORKER_AUDIT_LEVEL = audit_level


def _run_episode_worker(
    task: tuple[int, int, str, str],
) -> tuple[int, EpisodeData]:
    """Generate one deterministic episode for ordered multiprocessing."""
    if _WORKER_ENVIRONMENT is None:
        raise RuntimeError("episode worker was not initialized")
    index, seed, trajectory_type, attack_type = task
    episode = _WORKER_ENVIRONMENT.run_episode(
        seed=seed,
        trajectory_type=trajectory_type,
        attack_type=attack_type,
    )
    return index, _apply_audit_level(episode, _WORKER_AUDIT_LEVEL)


def _split(index: int, total: int, split_mode: str = "standard") -> str:
    if split_mode == "test-only":
        return "test"
    if split_mode != "standard":
        raise ValueError(f"unsupported split mode: {split_mode!r}")
    fraction = (index + 0.5) / max(total, 1)
    if fraction < TRAIN_FRACTION:
        return "train"
    if fraction < VALIDATION_END_FRACTION:
        return "validation"
    return "test"


def _balanced_scenario_schedule(
    count: int,
    rng: np.random.Generator,
    remainder_offset: int = 0,
) -> tuple[list[str], int]:
    """Return a shuffled four-scenario schedule with counts differing by at most one.

    The remainder offset is carried across data splits, so the complete dataset
    is also balanced when individual split sizes are not divisible by four.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    quotient, remainder = divmod(count, len(SCENARIO_TYPES))
    schedule = list(SCENARIO_TYPES) * quotient
    schedule.extend(
        SCENARIO_TYPES[(remainder_offset + index) % len(SCENARIO_TYPES)]
        for index in range(remainder)
    )
    rng.shuffle(schedule)
    next_offset = (remainder_offset + remainder) % len(SCENARIO_TYPES)
    return schedule, next_offset


def _scenario_assignment(total: int, seed: int) -> dict[int, str]:
    """Assign balanced scenarios within train/validation/test episode groups."""
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for index in range(total):
        split_indices[_split(index, total)].append(index)

    rng = np.random.default_rng(int(seed))
    assignment: dict[int, str] = {}
    remainder_offset = 0
    for split in ("train", "validation", "test"):
        indices = split_indices[split]
        schedule, remainder_offset = _balanced_scenario_schedule(
            len(indices), rng, remainder_offset
        )
        assignment.update(zip(indices, schedule))
    return assignment


def _experiment_assignment(
    total: int,
    seed: int,
    trajectory_types: tuple[str, ...],
    split_mode: str = "standard",
) -> dict[int, tuple[str, str]]:
    """Balance scenario/trajectory pairs independently inside every split."""
    if not trajectory_types:
        raise ValueError("trajectory_types must not be empty")
    ordered_pairs = [
        (scenario, trajectory_types[(scenario_index + shift) % len(trajectory_types)])
        for shift in range(len(trajectory_types))
        for scenario_index, scenario in enumerate(SCENARIO_TYPES)
    ]
    split_indices: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for index in range(total):
        split_indices[_split(index, total, split_mode=split_mode)].append(index)

    rng = np.random.default_rng(int(seed))
    assignment: dict[int, tuple[str, str]] = {}
    remainder_offset = 0
    for split in ("train", "validation", "test"):
        indices = split_indices[split]
        quotient, remainder = divmod(len(indices), len(ordered_pairs))
        schedule = ordered_pairs * quotient
        schedule.extend(
            ordered_pairs[(remainder_offset + index) % len(ordered_pairs)]
            for index in range(remainder)
        )
        remainder_offset = (remainder_offset + remainder) % len(ordered_pairs)
        rng.shuffle(schedule)
        assignment.update(zip(indices, schedule))
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate episode-level ISAC/GPS spoofing data"
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--episodes-per-shard", type=int, default=100)
    parser.add_argument(
        "--replay-source",
        choices=("recorded_normal_episode", "delayed_clean_stream"),
        default=None,
        help="Override the replay source in the configuration, including in worker processes.",
    )
    parser.add_argument(
        "--audit-level", choices=AUDIT_LEVELS, default="full",
        help=(
            "'standard' omits only the three optional full covariance matrices "
            "after feature construction. Measurements, powers, navigation and "
            "detector features remain unchanged. 'full' preserves all audit arrays."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of episode-generation processes. Results are consumed in "
            "episode-index order, so seeds, split assignment, and shard order "
            "remain deterministic."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=SPLIT_MODES,
        default="standard",
        help=(
            "Use 'standard' for the original episode-level 80/10/10 split, or "
            "'test-only' to write every episode to the test split."
        ),
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.episodes_per_shard < 1:
        parser.error("--episodes-per-shard must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        if not args.output.is_dir():
            parser.error("--output must be a directory")
        if any(args.output.iterdir()):
            parser.error("--output must be empty; existing datasets are never overwritten")

    config = load_config(args.config)
    if args.replay_source is not None:
        config = replace(config, attack=replace(config.attack, replay_source=args.replay_source))
    feature_names = feature_names_for_neighbor_mode(
        config.features.neighbor_feature_mode
    )
    if set(config.attack.attack_types) != {"random", "replay", "stealth"}:
        parser.error(
            "balanced collection requires attack_types to contain random, replay, and stealth"
        )
    experiment_assignment = _experiment_assignment(
        args.episodes,
        args.seed,
        config.simulation.trajectory_types,
        split_mode=args.split_mode,
    )
    split_policy = (
        TEST_ONLY_SPLIT_POLICY if args.split_mode == "test-only" else SPLIT_POLICY
    )
    sampling_policy = (
        "balanced scenario/trajectory pairs in test-only dataset"
        if args.split_mode == "test-only"
        else "balanced scenario/trajectory pairs within each split"
    )
    episode_sampling_policy = (
        "balanced_scenario_trajectory_pairs_test_only"
        if args.split_mode == "test-only"
        else "balanced_scenario_trajectory_pairs_by_split"
    )
    attack_counts: Counter[str] = Counter()
    trajectory_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    truncation_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    severity_by_attack_counts: dict[str, Counter[str]] = {
        attack_type: Counter() for attack_type in STRENGTH_UNITS
    }
    sampled_strengths: dict[str, list[float]] = {
        attack_type: [] for attack_type in STRENGTH_UNITS
    }
    nominal_duration_steps: list[int] = []
    actual_duration_steps: list[int] = []
    split_attack_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    split_severity_counts: dict[str, dict[str, Counter[str]]] = {
        split: {
            attack_type: Counter() for attack_type in STRENGTH_UNITS
        }
        for split in ("train", "validation", "test")
    }
    episode_buffers: dict[str, list[EpisodeData]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    shard_episode_counts: dict[str, list[int]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    replay_sources_by_split: dict[str, list[str]] = {
        "train": [], "validation": [], "test": [],
    }
    seen_replay_sources: set[str] = set()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed_count = 0

    def write_progress(complete: bool = False, error: str | None = None) -> None:
        payload = {
            "completed_episodes": completed_count,
            "saved_episodes": sum(sum(counts) for counts in shard_episode_counts.values()),
            "total_episodes": args.episodes,
            "elapsed_seconds": time.perf_counter() - started,
            "complete": complete,
            "status": "failed" if error else ("complete" if complete else "running"),
            "audit_level": args.audit_level,
            "error": error,
        }
        temporary = args.output / ".progress.tmp.json"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(args.output / "progress.json")

    write_progress()

    def flush_shard(split: str) -> None:
        buffer = episode_buffers[split]
        if not buffer:
            return
        shard_index = len(shard_episode_counts[split])
        shard_id = f"shard_{shard_index:05d}"
        save_episode_shard(buffer, args.output / split, shard_id)
        shard_episode_counts[split].append(len(buffer))
        buffer.clear()

    tasks = [
        (
            index,
            args.seed + index,
            experiment_assignment[index][1],
            experiment_assignment[index][0],
        )
        for index in range(args.episodes)
    ]

    def consume_episode(index: int, episode: EpisodeData) -> None:
        nonlocal completed_count
        seed = args.seed + index
        split = _split(index, args.episodes, split_mode=args.split_mode)
        requested_scenario, requested_trajectory = experiment_assignment[index]
        episode_id = f"episode_{index:05d}_seed_{seed}"
        recording = dict(episode.metadata.get("replay_recording", {}))
        expects_recording = (
            requested_scenario == "replay"
            and config.attack.replay_source == "recorded_normal_episode"
        )
        if bool(recording.get("present", False)) != expects_recording:
            raise RuntimeError(f"replay recording presence does not match configuration: {episode_id}")
        if expects_recording:
            source_id = str(recording["source_id"])
            if source_id in seen_replay_sources:
                raise RuntimeError(f"normal replay source reused by multiple episodes: {source_id}")
            seen_replay_sources.add(source_id)
            replay_sources_by_split[split].append(source_id)
            recording.update({"split": split, "victim_episode_id": episode_id})
        episode = replace(
            episode,
            metadata={
                **episode.metadata,
                "episode_id": episode_id,
                "split": split,
                "replay_recording": recording,
                "sampling_policy": episode_sampling_policy,
                "attack_strength_sampling_policy": (
                    "uniform_severity_then_uniform_within_interval_event_fixed"
                ),
                "attack_boundary_policy": "complete_event_within_episode",
            },
        )
        episode_buffers[split].append(episode)
        if len(episode_buffers[split]) >= args.episodes_per_shard:
            flush_shard(split)
        attack_type = str(episode.metadata["attack"]["attack_type"])
        trajectory_type = str(episode.metadata["trajectory_type"])
        if attack_type != requested_scenario or trajectory_type != requested_trajectory:
            raise RuntimeError(
                "generated episode does not match its scheduled assignment: "
                f"episode={episode_id}, expected=({requested_scenario}, "
                f"{requested_trajectory}), actual=({attack_type}, {trajectory_type})"
            )
        attack_counts[attack_type] += 1
        trajectory_counts[trajectory_type] += 1
        pair_counts[f"{attack_type}:{trajectory_type}"] += 1
        split_attack_counts[split][attack_type] += 1
        if attack_type != "clean":
            attack_metadata = episode.metadata["attack"]
            attack_parameters = attack_metadata["parameters"]
            start_step = int(attack_metadata["start_step"])
            end_step = int(attack_metadata["end_step"])
            nominal_duration = int(attack_metadata["nominal_duration_steps"])
            actual_duration = int(attack_metadata["actual_duration_steps"])
            if (
                bool(attack_metadata["was_truncated"])
                or actual_duration != nominal_duration
                or end_step - start_step != nominal_duration
                or end_step > config.simulation.num_steps
            ):
                raise RuntimeError(
                    "complete-event generation invariant failed for "
                    f"{episode_id}: {attack_metadata!r}"
                )
            severity_level = str(attack_parameters["severity_level"])
            strength_value = float(attack_parameters["strength_value"])
            severity_counts[severity_level] += 1
            severity_by_attack_counts[attack_type][severity_level] += 1
            split_severity_counts[split][attack_type][severity_level] += 1
            sampled_strengths[attack_type].append(strength_value)
            truncation_counts[
                "truncated" if bool(attack_metadata["was_truncated"]) else "complete"
            ] += 1
            nominal_duration_steps.append(nominal_duration)
            actual_duration_steps.append(actual_duration)

        completed_count = index + 1
        if completed_count == args.episodes or completed_count % 10 == 0:
            write_progress()
            print(
                f"generated {completed_count}/{args.episodes} episodes "
                f"with {args.workers} worker(s), "
                f"elapsed {time.perf_counter() - started:.1f}s",
                flush=True,
            )

    try:
        if args.workers == 1:
            environment = ISACSpoofingEnvironment(config)
            for index, seed, trajectory_type, attack_type in tasks:
                episode = environment.run_episode(
                    seed=seed,
                    trajectory_type=trajectory_type,
                    attack_type=attack_type,
                )
                consume_episode(index, _apply_audit_level(episode, args.audit_level))
        else:
            config_path = None if args.config is None else str(args.config.resolve())
            context = mp.get_context("spawn")
            pool = context.Pool(
                processes=args.workers,
                initializer=_initialize_episode_worker,
                initargs=(config_path, args.audit_level, config.attack.replay_source),
            )
            try:
                for batch in _bounded_episode_batches(tasks, args.workers):
                    for expected, result in zip(
                        batch, pool.imap(_run_episode_worker, batch, chunksize=1)
                    ):
                        index, episode = result
                        if index != expected[0]:
                            raise RuntimeError(
                                "parallel generator returned episodes out of order: "
                                f"expected {expected[0]}, got {index}"
                            )
                        consume_episode(index, episode)
            except BaseException:
                pool.terminate()
                pool.join()
                raise
            else:
                pool.close()
                pool.join()
        for split in ("train", "validation", "test"):
            flush_shard(split)
    except BaseException as exc:
        write_progress(error=f"{type(exc).__name__}: {exc}")
        raise
    args.output.mkdir(parents=True, exist_ok=True)
    strength_summary = {
        attack_type: {
            "unit": STRENGTH_UNITS[attack_type],
            "count": len(values),
            "observed_min": min(values, default=None),
            "observed_max": max(values, default=None),
            "observed_mean": float(np.mean(values)) if values else None,
        }
        for attack_type, values in sampled_strengths.items()
    }
    manifest = {
        "schema_version": "3.0",
        "measurement_timing": {
            "geometry_reference": "slot_start",
            "result_availability": "slot_end",
            "alarm_availability_offset_steps": 1,
        },
        "sensing_model": "signal_supported_received_power_crb",
        "collection_mode": "shared_gps_feedback_without_alarm_feedback",
        "replay_source": config.attack.replay_source,
        "replay_offset_semantics": (
            "trajectory_phase_offset"
            if config.attack.replay_source == "recorded_normal_episode"
            else "measurement_delay"
        ),
        "replay_recording_policy": (
            "one_private_normal_recording_per_replay_episode"
            if config.attack.replay_source == "recorded_normal_episode" else "not_applicable"
        ),
        "replay_recording_count": len(seen_replay_sources),
        "replay_recording_sources_by_split": replay_sources_by_split,
        "confirmation_duration_s": config.navigation.confirmation_duration_s(config.simulation.dt_s),
        "confirmation_steps": config.navigation.confirmation_steps,
        "episodes": args.episodes,
        "base_seed": args.seed,
        "feature_dim": len(feature_names),
        "feature_names": list(feature_names),
        "storage": {
            "format": "npz_episode_shards",
            "episode_axis": 0,
            "episodes_per_shard": args.episodes_per_shard,
            "audit_level": args.audit_level,
            "omitted_optional_array_fields": (
                list(OPTIONAL_COVARIANCE_AUDIT_FIELDS)
                if args.audit_level == "standard" else []
            ),
            "audit_policy": (
                "omit only optional full covariance matrices after feature construction; "
                "retain all required arrays, navigation, GPS, angles, powers and features"
                if args.audit_level == "standard" else "retain all available audit arrays"
            ),
            "total_shards": sum(
                len(counts) for counts in shard_episode_counts.values()
            ),
            "shard_counts_by_split": {
                split: len(counts)
                for split, counts in shard_episode_counts.items()
            },
            "episodes_in_shards_by_split": shard_episode_counts,
        },
        "split_policy": split_policy,
        "sampling_policy": sampling_policy,
        "scenario_counts": dict(attack_counts),
        "scenario_counts_by_split": {
            split: {
                scenario: counts[scenario]
                for scenario in SCENARIO_TYPES
            }
            for split, counts in split_attack_counts.items()
        },
        "trajectory_counts": dict(trajectory_counts),
        "scenario_trajectory_counts": dict(pair_counts),
        "attack_strength_sampling_policy": (
            "uniform severity level, then uniform continuous strength within "
            "the matching interval; one fixed strength per event"
        ),
        "attack_boundary_policy": (
            "sample duration first, then sample onset uniformly from the feasible "
            "range so every attack event is complete within its episode"
        ),
        "attack_severity_counts": {
            level: severity_counts[level]
            for level in config.attack.severity_levels
        },
        "attack_severity_counts_by_type": {
            attack_type: {
                level: counts[level]
                for level in config.attack.severity_levels
            }
            for attack_type, counts in severity_by_attack_counts.items()
        },
        "attack_severity_counts_by_split": {
            split: {
                attack_type: {
                    level: counts[level]
                    for level in config.attack.severity_levels
                }
                for attack_type, counts in attack_counts_by_type.items()
            }
            for split, attack_counts_by_type in split_severity_counts.items()
        },
        "sampled_attack_strength_summary": strength_summary,
        "attack_truncation_counts": dict(truncation_counts),
        "attack_duration_steps": {
            "nominal_min": min(nominal_duration_steps, default=None),
            "nominal_max": max(nominal_duration_steps, default=None),
            "actual_min": min(actual_duration_steps, default=None),
            "actual_max": max(actual_duration_steps, default=None),
        },
        "config": config.to_dict(),
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "generation_workers": args.workers,
            "parallel_ordering": "ordered_by_episode_index",
            "max_tasks_per_parallel_batch": (
                min(2 * args.workers, args.episodes) if args.workers > 1 else 1
            ),
        },
    }
    with (args.output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_progress(complete=True)
    print(f"generated {args.episodes} episodes in {args.output.resolve()}")
    print(f"scenario distribution: {dict(attack_counts)}")
    print(f"trajectory distribution: {dict(trajectory_counts)}")
    print(f"attack severity distribution: {dict(severity_counts)}")
    print(
        "shard distribution: "
        f"{ {split: len(counts) for split, counts in shard_episode_counts.items()} }"
    )
    print(
        "scenario distribution by split: "
        f"{ {split: dict(counts) for split, counts in split_attack_counts.items()} }"
    )


if __name__ == "__main__":
    main()

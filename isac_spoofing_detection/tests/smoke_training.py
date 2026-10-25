from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile

from isac_spoofing.config import (
    AttackConfig,
    FeatureConfig,
    GPSConfig,
    ISACConfig,
    ProjectConfig,
    SimulationConfig,
)
from isac_spoofing.cli.generate_dataset import _split
from isac_spoofing.data import save_episode_shard
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.evaluation import evaluate_split
from isac_spoofing.training import TrainingConfig, train_detector


def smoke_config() -> ProjectConfig:
    return ProjectConfig(
        simulation=SimulationConfig(
            num_uavs=4,
            dt_s=0.1,
            duration_s=2.0,
            speed_mps=4.0,
            trajectory_types=("s_curve",),
            formation_type="compact_ring",
        ),
        gps=GPSConfig(noise_std_m=(0.4, 0.4, 0.6)),
        attack=AttackConfig(
            clean_episode_probability=0.25,
            attack_types=("random", "replay", "stealth"),
            duration_fraction_range=(0.4, 0.4),
            target_count_range=(2, 2),
            severity_levels=("fixed",),
            random_sigma_ranges_m=((1.0, 1.0),),
            replay_delay_ranges_s=((0.5, 0.5),),
            stealth_rate_ranges_mps=((0.5, 0.5),),
        ),
        isac=ISACConfig(),
        features=FeatureConfig(num_neighbors=2),
    )


def main() -> None:
    config = smoke_config()
    scenarios = ("clean", "random", "replay", "stealth")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        data_root = root / "data"
        split_episodes = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for index in range(16):
            split = _split(index, 16)
            episode = ISACSpoofingEnvironment(config).run_episode(
                seed=700 + index,
                attack_type=scenarios[index % len(scenarios)],
            )
            episode_id = f"episode_{index:05d}"
            episode = replace(
                episode,
                metadata={
                    **episode.metadata,
                    "episode_id": episode_id,
                    "split": split,
                },
            )
            split_episodes[split].append(episode)
        for split, episodes in split_episodes.items():
            save_episode_shard(
                episodes,
                data_root / split,
                "shard_00000",
            )
        with (data_root / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump({"config": config.to_dict()}, handle)

        print(
            "This smoke run verifies that actual simulated episodes can complete "
            "training, frozen 0.5-probability inference, K-step alarms, and event "
            "aggregation."
        )
        for objective in ("bce", "event_likelihood"):
            run_root = root / objective
            training = train_detector(
                data_root,
                run_root / "training",
                TrainingConfig(
                    model_type="mlp",
                    loss_objective=objective,
                    confirmation_steps=1,
                    confirmation_temperature=1.0,
                    time_step_s=0.1,
                    detection_loss_weight=0.1,
                    delay_loss_weight=0.1,
                    false_alarm_loss_weight=0.1,
                    early_detection_deadline_s=1.0,
                    event_ramp_epochs=1,
                    hidden_dim=8,
                    mlp_hidden_dims=(8, 4),
                    batch_size=4,
                    max_epochs=2,
                    patience=2,
                    seed=3,
                    device="cpu",
                    num_threads=1,
                ),
            )
            evaluation = evaluate_split(
                training["checkpoint"],
                data_root,
                "test",
                confirmation_steps=1,
                output_directory=run_root,
                bootstrap_samples=10,
                seed=4,
                device="cpu",
                save_predictions=False,
            )
            if float(training["final_validation_time_step_metrics"]["accuracy"]) <= 0.0:
                raise RuntimeError("smoke training produced 0 validation accuracy")
            if int(evaluation["aggregate"]["num_episodes"]) != 2:
                raise RuntimeError("smoke evaluation did not process the test split")
            print(objective)
            print(json.dumps(evaluation["aggregate"], ensure_ascii=False, indent=2))
        print("training-smoke-ok")


if __name__ == "__main__":
    main()

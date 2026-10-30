from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from isac_spoofing.config import GPSConfig, NavigationConfig, ProjectConfig, load_config
from isac_spoofing.data import save_episode_shard
from isac_spoofing.environment import ISACSpoofingEnvironment
from isac_spoofing.gps_attacks import GPSAttackInjector
from isac_spoofing.learning_data import load_episode_shard
from isac_spoofing.navigation import GPSPositionController
from isac_spoofing.normal_recording import generate_normal_gps_recording
from isac_spoofing.raw_episode import load_raw_episode_file
from isac_spoofing.types import AttackScenario, TrajectoryData


def short_recorded_config():
    config = ProjectConfig()
    return replace(
        config,
        simulation=replace(
            config.simulation, duration_s=0.6, trajectory_types=("straight",),
        ),
        attack=replace(
            config.attack, replay_source="recorded_normal_episode",
            replay_delay_ranges_s=((0.02, 0.04),) * 3,
            duration_fraction_range=(0.5, 0.5), target_count_range=(2, 2),
        ),
    )


def stationary_reference(steps=20, dt_s=0.02):
    position = np.broadcast_to(
        np.array([[40., 45., 20.], [48., 45., 20.]], dtype=np.float32),
        (steps, 2, 3),
    ).copy()
    return TrajectoryData(
        time_s=(np.arange(steps) * dt_s).astype(np.float32),
        position_m=position,
        velocity_mps=np.zeros_like(position),
        heading=np.zeros_like(position), trajectory_type="stationary_test",
    )


def replay_scenario(source="recorded_normal_episode"):
    return AttackScenario(
        attack_type="replay", start_step=2, end_step=8,
        target_mask=np.array([True, False, True]),
        parameters={"delay_steps": 2, "replay_source": source},
        nominal_duration_steps=6, actual_duration_steps=6,
    )


def test_replay_source_defaults_migrate_old_snapshots_without_changing_their_meaning(tmp_path):
    assert ProjectConfig().attack.replay_source == "recorded_normal_episode"
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default.json"
    assert load_config(default_path).attack.replay_source == "recorded_normal_episode"
    raw = short_recorded_config().to_dict()
    raw["attack"].pop("replay_source")
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_config(path).attack.replay_source == "delayed_clean_stream"
    raw["attack"]["replay_source"] = "unknown_recording_mode"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="replay_source"):
        load_config(path)


def test_normal_recording_contains_closed_loop_motion_and_frozen_gps_noise():
    reference = stationary_reference()
    gps = GPSConfig(noise_std_m=(0.5, 0.5, 0.8))
    navigation = NavigationConfig()
    seed = 731
    noise = np.random.default_rng(seed).normal(
        0.0, gps.noise_std_m, size=reference.position_m.shape,
    )
    recording = generate_normal_gps_recording(
        reference, gps, navigation, 0.02, np.random.default_rng(seed),
    )
    assert recording.dtype == np.float32
    assert not recording.flags.writeable
    # Subtracting the sampled sensor noise recovers the independently moving
    # normal plant. Its displacement must obey the normal GPS feedback command.
    plant = recording.astype(np.float64) - noise
    controller = GPSPositionController(navigation, 0.02)
    commands = controller.command(
        reference.position_m, reference.velocity_mps, recording,
    )
    np.testing.assert_allclose(plant[0], reference.position_m[0], atol=4e-6)
    np.testing.assert_allclose(np.diff(plant, axis=0), 0.02 * commands[:-1], atol=8e-6)
    assert np.max(np.linalg.norm(plant - reference.position_m, axis=-1)) > 0.01


def test_normal_recording_respects_explicit_open_loop_ablation():
    reference = stationary_reference()
    gps = GPSConfig(noise_std_m=(0.5, 0.5, 0.8))
    navigation = NavigationConfig(enabled=False)
    noise = np.random.default_rng(12).normal(
        0.0, gps.noise_std_m, size=reference.position_m.shape,
    )
    recording = generate_normal_gps_recording(
        reference, gps, navigation, 0.02, np.random.default_rng(12),
    )
    np.testing.assert_array_equal(
        recording, (reference.position_m.astype(np.float64) + noise).astype(np.float32),
    )


def test_recorded_replay_batch_and_stream_copy_the_original_noisy_samples():
    nominal = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
    recording = np.random.default_rng(71).normal(20., 0.8, nominal.shape).astype(np.float32)
    original = recording.copy()
    injector = GPSAttackInjector(short_recorded_config().attack, 0.02)
    scenario = replay_scenario()
    batch, labels = injector.apply(
        nominal, scenario, np.random.default_rng(1), replay_source_gps_m=recording,
    )
    streamed, streamed_labels = zip(*(
        injector.apply_step(
            nominal[:t + 1], scenario, t, np.random.default_rng(999),
            replay_source_gps_m=recording,
        ) for t in range(nominal.shape[0])
    ))
    np.testing.assert_array_equal(streamed, batch)
    np.testing.assert_array_equal(streamed_labels, labels)
    np.testing.assert_array_equal(batch[2:8, [0, 2]], recording[:6, [0, 2]])
    np.testing.assert_array_equal(batch[:, 1], nominal[:, 1])
    np.testing.assert_array_equal(batch[[0, 1, 8, 9]], nominal[[0, 1, 8, 9]])
    np.testing.assert_array_equal(recording, original)


def test_recorded_source_does_not_follow_the_victim_history():
    nominal = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
    recording = -nominal - 1
    injector = GPSAttackInjector(short_recorded_config().attack, 0.02)
    scenario = replay_scenario()
    a, _ = injector.apply(
        nominal, scenario, np.random.default_rng(1), replay_source_gps_m=recording,
    )
    b, _ = injector.apply(
        nominal + 5000, scenario, np.random.default_rng(2),
        replay_source_gps_m=recording,
    )
    np.testing.assert_array_equal(a[2:8, [0, 2]], b[2:8, [0, 2]])
    # These source indices lie after the attack onset in the victim clock.
    # They must still come from the completed normal recording.
    np.testing.assert_array_equal(b[4:8, [0, 2]], recording[2:6, [0, 2]])


@pytest.mark.parametrize("source_problem", ["missing", "wrong_uavs", "wrong_coordinates", "too_short", "nonfinite"])
def test_recorded_replay_requires_a_valid_available_recording(source_problem):
    nominal = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
    recording = np.ones_like(nominal)
    if source_problem == "missing":
        recording = None
    elif source_problem == "wrong_uavs":
        recording = np.zeros((10, 4, 3), dtype=np.float32)
    elif source_problem == "wrong_coordinates":
        recording = np.zeros((10, 3, 2), dtype=np.float32)
    elif source_problem == "too_short":
        recording = recording[:3]
    else:
        recording[3, 0, 0] = np.nan
    injector = GPSAttackInjector(short_recorded_config().attack, 0.02)
    with pytest.raises((ValueError, FloatingPointError)):
        injector.apply_step(
            nominal[:6], replay_scenario(), 5, np.random.default_rng(1),
            replay_source_gps_m=recording,
        )


def test_delayed_clean_stream_preserves_the_legacy_replay_behavior():
    nominal = np.arange(90, dtype=np.float32).reshape(10, 3, 3)
    config = replace(short_recorded_config().attack, replay_source="delayed_clean_stream")
    injector = GPSAttackInjector(config, 0.02)
    scenario = replace(replay_scenario(), parameters={"delay_steps": 2})
    batch, labels = injector.apply(nominal, scenario, np.random.default_rng(1))
    streamed, streamed_labels = zip(*(
        injector.apply_step(nominal[:t + 1], scenario, t, np.random.default_rng(9))
        for t in range(nominal.shape[0])
    ))
    np.testing.assert_array_equal(batch[2:8, [0, 2]], nominal[:6, [0, 2]])
    np.testing.assert_array_equal(batch, streamed)
    np.testing.assert_array_equal(labels, streamed_labels)


def test_environment_replays_private_recording_and_senses_the_actual_victim():
    config = short_recorded_config()
    environment = ISACSpoofingEnvironment(config)
    with patch.object(environment.isac_model, "measure", wraps=environment.isac_model.measure) as bs_spy:
        with patch.object(environment.isac_model, "measure_inter_uav", wraps=environment.isac_model.measure_inter_uav) as peer_spy:
            episode = environment.run_episode(seed=73, attack_type="replay")
    info = episode.metadata["replay_recording"]
    assert info["present"]
    assert info["normal_gps_feedback"]
    assert info["shares_reference_and_initial_state"]
    assert info["storage_array"] == "replay_source_gps_m"
    attack = episode.metadata["attack"]
    start, end = attack["start_step"], attack["end_step"]
    delay = attack["parameters"]["delay_steps"]
    targets = attack["target_uavs"]
    assert end - start > delay
    np.testing.assert_array_equal(
        episode.gps_observed_m[start:end, targets],
        episode.replay_source_gps_m[start-delay:end-delay, targets],
    )
    np.testing.assert_array_equal(
        episode.gps_observed_m[~episode.labels.astype(bool)],
        episode.gps_clean_m[~episode.labels.astype(bool)],
    )
    assert info["source_start_step"] == start - delay
    assert info["source_end_step_exclusive"] == end - delay
    assert not np.array_equal(episode.replay_source_gps_m[:start], episode.gps_clean_m[:start])
    for index, call in enumerate(bs_spy.call_args_list):
        np.testing.assert_allclose(call.args[0], episode.true_position_m[index], atol=1e-5)
        np.testing.assert_allclose(call.args[1], episode.command_velocity_mps[index], atol=1e-6)
    for index, call in enumerate(peer_spy.call_args_list):
        np.testing.assert_allclose(call.args[0], episode.true_position_m[index], atol=1e-5)


def test_recording_is_reproducible_private_and_independent_of_replay_phase():
    config = short_recorded_config()
    environment = ISACSpoofingEnvironment(config)
    first = environment.run_episode(seed=73, attack_type="replay")
    repeated = environment.run_episode(seed=73, attack_type="replay")
    other_seed = environment.run_episode(seed=74, attack_type="replay")
    np.testing.assert_array_equal(first.replay_source_gps_m, repeated.replay_source_gps_m)
    np.testing.assert_array_equal(first.true_position_m, repeated.true_position_m)
    np.testing.assert_array_equal(first.features, repeated.features)
    assert first.metadata["replay_recording"]["source_id"] != other_seed.metadata["replay_recording"]["source_id"]
    assert first.metadata["replay_recording"]["recording_seed"] != other_seed.metadata["replay_recording"]["recording_seed"]
    different_phase_config = replace(
        config, attack=replace(config.attack, replay_delay_ranges_s=((0.12, 0.12),) * 3),
    )
    different_phase = ISACSpoofingEnvironment(different_phase_config).run_episode(seed=73, attack_type="replay")
    np.testing.assert_array_equal(first.reference_position_m, different_phase.reference_position_m)
    np.testing.assert_array_equal(first.replay_source_gps_m, different_phase.replay_source_gps_m)
    assert not np.array_equal(first.true_position_m, different_phase.true_position_m)
    reference = TrajectoryData(
        time_s=first.time_s, position_m=first.reference_position_m,
        velocity_mps=first.reference_velocity_mps,
        heading=np.zeros_like(first.reference_velocity_mps), trajectory_type="straight",
    )
    regenerated = generate_normal_gps_recording(
        reference, config.gps, config.navigation, config.simulation.dt_s,
        np.random.default_rng(first.metadata["replay_recording"]["recording_seed"]),
    )
    np.testing.assert_array_equal(first.replay_source_gps_m, regenerated)


def test_mixed_shard_preserves_recording_as_audit_only(tmp_path):
    config = short_recorded_config()
    environment = ISACSpoofingEnvironment(config)
    replay = environment.run_episode(seed=73, attack_type="replay")
    clean = environment.run_episode(seed=74, attack_type="clean")
    assert not clean.metadata["replay_recording"]["present"]
    np.testing.assert_array_equal(clean.replay_source_gps_m, np.zeros_like(clean.gps_clean_m))
    episodes = [
        replace(episode, metadata={**episode.metadata, "episode_id": name, "split": "train"})
        for name, episode in (("replay", replay), ("clean", clean))
    ]
    path = save_episode_shard(episodes, tmp_path, "mixed")
    raw = load_raw_episode_file(path, array_names=("replay_source_gps_m", "gps_observed_m", "features"))
    learned = load_episode_shard(path)
    for original, audit, visible in zip(episodes, raw, learned):
        np.testing.assert_array_equal(audit.require("replay_source_gps_m"), original.replay_source_gps_m)
        np.testing.assert_array_equal(visible.features, original.features)
        assert visible.features.shape[-1] == 10
        assert not hasattr(visible, "replay_source_gps_m")
        assert "replay_source_gps_m" not in visible.feature_names
        assert audit.metadata["replay_recording"] == original.metadata["replay_recording"]

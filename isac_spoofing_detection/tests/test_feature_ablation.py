from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

from isac_spoofing.feature_sets import resolve_feature_set


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_feature_ablation.py"
SPEC = importlib.util.spec_from_file_location("feature_ablation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model="gru",
        class_weighting="none",
        loss_objective="event_likelihood",
        confirmation_temperature=1.0,
        time_step_s=0.1,
        detection_loss_weight=0.1,
        delay_loss_weight=0.03,
        false_alarm_loss_weight=0.1,
        early_detection_deadline_s=3.0,
        event_ramp_epochs=20,
        initial_checkpoint_root=None,
        dry_run=False,
        hidden_dim=64,
        num_layers=1,
        input_encoder_dim=16,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=0.0001,
        epochs=500,
        patience=100,
        device="cuda",
        num_threads=1,
        k_values="1,2,3,4,5",
        bootstrap_samples=1000,
        fixed_k=5,
        include_test=False,
    )


def _command_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_temporal_free_ablation_feature_dimensions() -> None:
    feature_sets = ABLATION.FEATURE_SETS
    assert len(feature_sets["full_f11"]) == 11
    assert len(feature_sets["no_handcrafted_temporal"]) == 11
    assert len(feature_sets["no_temporal_no_bs_residual"]) == 5
    assert len(feature_sets["no_temporal_no_gps_isac_position"]) == 9
    assert len(feature_sets["no_temporal_no_neighbor"]) == 8
    assert feature_sets["self_consensus_only"][-1] == "self_consensus_deviation"
    assert len(feature_sets["self_consensus_only"]) == 9
    assert (
        feature_sets["signed_anomaly_contrast_only"][-1]
        == "signed_self_neighbor_anomaly_contrast"
    )
    assert len(feature_sets["signed_anomaly_contrast_only"]) == 9

    for name in (
        "no_handcrafted_temporal",
        "no_temporal_no_bs_residual",
        "no_temporal_no_gps_isac_position",
        "no_temporal_no_neighbor",
    ):
        assert "cusum_range" not in feature_sets[name]
        assert "cusum_radial_velocity" not in feature_sets[name]
        assert "cusum_direction" not in feature_sets[name]
        assert "mean_error_vector_disagreement" not in feature_sets[name]


def test_self_consensus_only_resolves_by_name_for_robust_schema() -> None:
    available = (
        "mean_range",
        "mean_radial_velocity",
        "mean_direction",
        "max_range",
        "max_radial_velocity",
        "max_direction",
        "gps_isac_position_error_norm",
        "gps_isac_position_error_heading_projection",
        "self_consensus_deviation",
        "neighbor_consensus_bias",
        "neighbor_consensus_dispersion",
    )

    selected = resolve_feature_set("self_consensus_only", available)

    assert selected == available[:9]


def test_self_consensus_only_rejects_incompatible_neighbor_schema() -> None:
    available = tuple(ABLATION.FEATURE_SETS["full_f11"])

    with pytest.raises(ValueError, match="requires a dataset"):
        resolve_feature_set("self_consensus_only", available)


def test_signed_anomaly_contrast_only_resolves_current_schema() -> None:
    available = tuple(ABLATION.FEATURE_SETS["full_f11"])

    selected = resolve_feature_set("signed_anomaly_contrast_only", available)

    assert selected == available[:9]


def test_signed_anomaly_contrast_only_rejects_robust_schema() -> None:
    available = (
        "mean_range",
        "mean_radial_velocity",
        "mean_direction",
        "max_range",
        "max_radial_velocity",
        "max_direction",
        "gps_isac_position_error_norm",
        "gps_isac_position_error_heading_projection",
        "self_consensus_deviation",
        "neighbor_consensus_bias",
        "neighbor_consensus_dispersion",
    )

    with pytest.raises(ValueError, match="requires a dataset"):
        resolve_feature_set("signed_anomaly_contrast_only", available)


def test_formation_deformation_ablation_sets_resolve_by_feature_name() -> None:
    available = (
        "mean_range",
        "mean_radial_velocity",
        "mean_direction",
        "max_range",
        "max_radial_velocity",
        "max_direction",
        "gps_isac_position_error_norm",
        "gps_isac_position_error_heading_projection",
        "signed_self_neighbor_anomaly_contrast",
        "median_longitudinal_formation_strain",
        "median_transverse_formation_strain",
    )

    assert resolve_feature_set("no_neighbor", available) == available[:8]
    assert resolve_feature_set(
        "signed_anomaly_contrast_only",
        available,
    ) == available[:9]
    assert resolve_feature_set(
        "signed_plus_transverse_deformation",
        available,
    ) == available[:9] + (available[10],)
    assert resolve_feature_set(
        "formation_deformation_full",
        available,
    ) == available


def test_dynamic_and_fixed_signed_topology_ablation_resolves_by_name() -> None:
    available = (
        "mean_range",
        "mean_radial_velocity",
        "mean_direction",
        "max_range",
        "max_radial_velocity",
        "max_direction",
        "gps_isac_position_error_norm",
        "gps_isac_position_error_heading_projection",
        "signed_self_neighbor_anomaly_contrast",
        "fixed_formation_signed_self_neighbor_anomaly_contrast",
    )

    assert resolve_feature_set("no_neighbor", available) == available[:8]
    assert resolve_feature_set(
        "dynamic_signed_anomaly_contrast_only", available
    ) == available[:9]
    assert resolve_feature_set(
        "fixed_signed_anomaly_contrast_only", available
    ) == available[:8] + (available[9],)


def test_event_likelihood_settings_are_forwarded_to_every_training_run() -> None:
    command = ABLATION._build_command(
        _args(),
        Path("dataset"),
        Path("run"),
        "no_neighbor",
        20260718,
    )

    assert _command_value(command, "--loss_objective") == "event_likelihood"
    assert _command_value(command, "--confirmation_temperature") == "1.0"
    assert _command_value(command, "--time_step_s") == "0.1"
    assert _command_value(command, "--detection_loss_weight") == "0.1"
    assert _command_value(command, "--delay_loss_weight") == "0.03"
    assert _command_value(command, "--false_alarm_loss_weight") == "0.1"
    assert _command_value(command, "--early_detection_deadline_s") == "3.0"
    assert _command_value(command, "--event_ramp_epochs") == "20"
    assert _command_value(command, "--fixed_k") == "5"
    assert _command_value(command, "--feature_set") == "no_neighbor"
    assert _command_value(command, "--input_encoder_dim") == "16"
    assert "--skip_test" in command


def test_early_detection_metrics_are_aggregated_by_feature_set() -> None:
    record: dict[str, object] = {
        "feature_set": "full_f11",
        "feature_dim": 11,
        "loss_objective": "event_likelihood",
        "reported_split": "validation",
        "selected_k": 5,
    }
    for metric in ABLATION.SUMMARY_METRICS:
        record[metric] = None if metric.startswith("delay_") else 0.5

    summary = ABLATION._aggregate([record])

    assert len(summary) == 1
    assert summary[0]["loss_objective"] == "event_likelihood"
    assert summary[0]["detection_rate_within_1p0s_mean"] == 0.5
    assert summary[0]["delay_mean_s_mean"] is None
    assert summary[0]["delay_median_s_mean"] is None
    assert ABLATION._format_metric(summary[0]["delay_mean_s_mean"]) == "undefined"

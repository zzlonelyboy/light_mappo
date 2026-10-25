from __future__ import annotations

import numpy as np

from isac_spoofing.external_baselines.common.pathloss import (
    A2GPathLossConfig,
    expected_a2g_pathloss_db,
    sample_actual_and_gps_pathloss,
    umi_aerial_shadow_fading_std_db,
)
from isac_spoofing.external_baselines.deep_ensemble.features import (
    FEATURE_NAMES,
    deep_ensemble_features,
)
from isac_spoofing.external_baselines.deep_ensemble.ensemble import (
    DEFAULT_CANDIDATES,
    build_ensemble_candidate,
)
from isac_spoofing.external_baselines.deep_ensemble import evaluation


def test_expected_pathloss_increases_with_distance() -> None:
    config = A2GPathLossConfig(packet_samples=16)
    bs = np.asarray([0.0, 0.0, 5.0])
    positions = np.asarray(
        [
            [20.0, 0.0, 30.0],
            [200.0, 0.0, 30.0],
        ]
    )
    pathloss = expected_a2g_pathloss_db(positions, bs, config)
    assert pathloss.shape == (2,)
    assert pathloss[1] > pathloss[0]


def test_actual_pathloss_is_independent_of_gps_report() -> None:
    config = A2GPathLossConfig(packet_samples=16)
    true_position = np.asarray([[[20.0, 20.0, 30.0]]])
    gps_clean = true_position.copy()
    gps_spoofed = gps_clean + np.asarray([[[50.0, -20.0, 10.0]]])
    base_stations = np.asarray([[0.0, 0.0, 5.0], [100.0, 0.0, 10.0]])
    actual_clean, gps_clean_pathloss = sample_actual_and_gps_pathloss(
        true_position,
        gps_clean,
        base_stations,
        np.random.default_rng(123),
        config,
    )
    actual_spoofed, gps_spoofed_pathloss = sample_actual_and_gps_pathloss(
        true_position,
        gps_spoofed,
        base_stations,
        np.random.default_rng(123),
        config,
    )
    np.testing.assert_allclose(actual_clean, actual_spoofed)
    assert not np.allclose(gps_clean_pathloss, gps_spoofed_pathloss)


def test_umi_aerial_shadow_fading_matches_standard_values() -> None:
    altitude = np.asarray([10.0, 30.0, 200.0])
    los_std, nlos_std = umi_aerial_shadow_fading_std_db(altitude)
    np.testing.assert_allclose(
        los_std,
        [4.0, 5.0 * np.exp(-0.3), 2.0],
    )
    np.testing.assert_allclose(nlos_std, [7.82, 8.0, 8.0])


def test_gps_implied_set_is_theoretical_and_actual_set_is_stochastic() -> None:
    config = A2GPathLossConfig(packet_samples=512)
    position = np.asarray([[[80.0, 40.0, 30.0]]])
    base_stations = np.asarray([[0.0, 0.0, 5.0]])
    actual, gps_implied = sample_actual_and_gps_pathloss(
        position,
        position.copy(),
        base_stations,
        np.random.default_rng(321),
        config,
    )
    assert np.std(actual) > 1.0
    np.testing.assert_allclose(
        np.std(gps_implied, axis=-1),
        0.0,
        atol=1.0e-5,
    )


def test_statistical_feature_shape_and_identity() -> None:
    rng = np.random.default_rng(7)
    actual = rng.normal(size=(2, 3, 4, 32))
    features = deep_ensemble_features(actual, actual.copy())
    assert features.shape == (2, 3, 4, len(FEATURE_NAMES))
    np.testing.assert_allclose(features, 0.0, atol=1.0e-7)


def test_mean_shift_changes_mean_and_wasserstein_features() -> None:
    rng = np.random.default_rng(9)
    actual = rng.normal(size=(5, 64))
    gps_implied = actual + 3.0
    features = deep_ensemble_features(actual, gps_implied)
    np.testing.assert_allclose(features[:, 0], 3.0, atol=1.0e-6)
    np.testing.assert_allclose(features[:, -1], 3.0, atol=1.0e-6)


def test_paper_meta_mlp_uses_one_hidden_layer_with_num_bs_neurons() -> None:
    model = build_ensemble_candidate("mlp", seed=17, input_dim=4)
    classifier = model.named_steps["mlpclassifier"]
    assert classifier.hidden_layer_sizes == (4,)
    assert classifier.learning_rate_init == 0.005
    assert classifier.max_iter == 500
    assert classifier.n_iter_no_change == 100


def test_algorithm_3_includes_all_six_meta_classifiers() -> None:
    assert set(DEFAULT_CANDIDATES) == {
        "mlp",
        "decision_tree",
        "logistic_regression",
        "svm_linear",
        "svm_rbf",
        "gaussian_nb",
    }


def test_paper_meta_classifier_hyperparameters() -> None:
    input_dim = 4
    decision_tree = build_ensemble_candidate(
        "decision_tree", seed=17, input_dim=input_dim
    )
    logistic = build_ensemble_candidate(
        "logistic_regression", seed=17, input_dim=input_dim
    ).named_steps["logisticregression"]
    svm_linear = build_ensemble_candidate(
        "svm_linear", seed=17, input_dim=input_dim
    ).named_steps["svc"]
    svm_rbf = build_ensemble_candidate(
        "svm_rbf", seed=17, input_dim=input_dim
    ).named_steps["svc"]
    assert decision_tree.max_depth == input_dim
    assert logistic.C == 0.01
    assert svm_linear.kernel == "linear"
    assert svm_linear.C == 0.025
    assert svm_rbf.kernel == "rbf"
    assert svm_rbf.C == 1.0
    assert svm_rbf.gamma == 2.0


def test_event_threshold_selection_uses_f1_then_far_then_delay(
    monkeypatch,
) -> None:
    results = {
        0.1: (0.95, 0.20, 1.0),
        0.2: (0.95, 0.10, 2.0),
        0.3: (0.95, 0.10, 1.5),
        0.4: (0.94, 0.00, 1.0),
    }

    def fake_evaluate(
        episodes,
        threshold,
        confirmation_steps,
        bootstrap_samples,
    ):
        del episodes, confirmation_steps, bootstrap_samples
        event_f1, far, delay = results[threshold]
        return {
            "event_precision": 0.9,
            "event_recall": 0.9,
            "event_f1": event_f1,
            "false_alarms_per_uav_hour": far,
            "delay_mean_s": delay,
        }

    monkeypatch.setattr(
        evaluation,
        "evaluate_episode_probabilities",
        fake_evaluate,
    )
    selection = evaluation.select_event_threshold(
        episodes=(),
        thresholds=(0.1, 0.2, 0.3, 0.4),
        confirmation_steps=5,
    )
    assert selection["selected"]["threshold"] == 0.3

from __future__ import annotations

import numpy as np

from isac_spoofing.baselines.bdgd_adapted import (
    BDGDAdaptedConfig,
    BDGDAdaptedDetector,
    _paper_adaptive_threshold,
    build_local_evidence_arrays,
    estimate_connection_probability,
    estimate_missed_attack_probability,
)


def _synthetic_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_steps = 6
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    physical = np.repeat(positions[None], num_steps, axis=0)
    difference = physical[:, :, None, :] - physical[:, None, :, :]
    measured = np.linalg.norm(difference, axis=-1)
    gps = physical.copy()
    gps[2:, 0, 0] += 5.0
    gps[2:, 0, 1] += 4.0
    valid = ~np.eye(4, dtype=bool)[None].repeat(num_steps, axis=0)
    variance = np.full_like(measured, 0.01)
    return gps, measured, valid, variance


def _detector(multiplier: float) -> BDGDAdaptedDetector:
    return BDGDAdaptedDetector(
        BDGDAdaptedConfig(
            error_tolerance_multiplier=multiplier,
            num_neighbors=3,
            connection_probability=1.0,
            missed_attack_probability=0.0,
        )
    )


def test_accusation_score_is_higher_for_spoofed_uav() -> None:
    result = _detector(5.0).run_arrays(*_synthetic_arrays(), malicious_count=1)

    assert np.all(result.predictions[:2] == 0)
    assert np.all(
        result.reputation_score[2:, 0]
        > result.reputation_score[2:, 1:].max(axis=1)
    )
    assert np.all(result.predictions[2:, 0] == 1)
    assert np.allclose(
        result.reputation_score,
        result.accusation_count / result.neighbor_count,
    )


def test_threshold_is_one_scalar_for_the_whole_episode() -> None:
    result = _detector(5.0).run_arrays(*_synthetic_arrays(), malicious_count=1)

    assert isinstance(result.adaptive_threshold, float)
    assert 0.0 <= result.adaptive_threshold <= 1.0


def test_higher_tolerance_produces_no_more_local_accusations() -> None:
    arrays = _synthetic_arrays()
    low = _detector(2.0).run_arrays(*arrays, malicious_count=1)
    high = _detector(100.0).run_arrays(*arrays, malicious_count=1)

    assert np.sum(high.local_accusations) <= np.sum(low.local_accusations)


def test_symmetrization_does_not_restore_invalid_reverse_links() -> None:
    gps, measured, valid, variance = _synthetic_arrays()
    valid[:, 0, 1] = False
    valid[:, 1, 0] = True
    evidence = build_local_evidence_arrays(
        gps,
        measured,
        valid,
        variance,
        num_neighbors=3,
        symmetrize_neighbors=True,
    )

    assert np.all(~evidence.neighbor_mask[:, 0, 1])
    assert np.all(evidence.neighbor_mask <= valid)


def test_validation_probability_estimators_use_selected_attacked_links() -> None:
    arrays = _synthetic_arrays()
    evidence = build_local_evidence_arrays(
        *arrays,
        num_neighbors=3,
        symmetrize_neighbors=True,
    )
    labels = np.zeros((6, 4), dtype=np.uint8)
    labels[2:, 0] = 1
    p1, selected_count, eligible_count = estimate_connection_probability(
        [evidence],
        [arrays[2]],
    )
    p2, missed_count, attack_link_count = estimate_missed_attack_probability(
        [evidence],
        [labels],
        error_tolerance_multiplier=5.0,
    )

    assert np.isclose(p1, 1.0)
    assert selected_count == eligible_count
    assert np.isclose(p2, 0.0)
    assert missed_count == 0
    assert attack_link_count > 0


def test_paper_threshold_implements_equations_11_to_13() -> None:
    threshold = _paper_adaptive_threshold(
        num_uavs=4,
        malicious_count=1,
        connection_probability=np.array([1.0]),
        missed_attack_probability=np.array([0.0]),
        false_accusation_probability=0.0,
    )

    expected_normal = 1.0 / 5.0
    expected_malicious = 4.0 / 5.0
    assert np.allclose(threshold, (expected_normal + expected_malicious) / 2.0)

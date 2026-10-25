from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Sequence

import numpy as np

from ..raw_episode import RawEpisode


@dataclass(frozen=True)
class BDGDAdaptedConfig:
    """Validation-calibrated adaptation of the BDGD detector.

    The paper's RSSI range is replaced by direct inter-UAV ISAC ranging.
    ``connection_probability`` and ``missed_attack_probability`` must be
    estimated once on the validation split and remain fixed at test time.
    The episode-level malicious-node count is still required by the original
    analytical threshold and is therefore an explicitly documented oracle
    scenario prior.
    """

    error_tolerance_multiplier: float = 3.0
    num_neighbors: int = 3
    connection_probability: float | None = None
    missed_attack_probability: float | None = None
    variance_floor: float = 1.0e-8
    symmetrize_neighbors: bool = True

    def __post_init__(self) -> None:
        if self.error_tolerance_multiplier <= 0.0:
            raise ValueError("error_tolerance_multiplier must be positive")
        if self.num_neighbors < 1:
            raise ValueError("num_neighbors must be positive")
        if self.variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        for name, value in (
            ("connection_probability", self.connection_probability),
            ("missed_attack_probability", self.missed_attack_probability),
        ):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")


@dataclass(frozen=True)
class BDGDLocalEvidence:
    normalized_residual: np.ndarray
    neighbor_mask: np.ndarray


@dataclass(frozen=True)
class BDGDResult:
    normalized_residual: np.ndarray
    neighbor_mask: np.ndarray
    local_accusations: np.ndarray
    accusation_count: np.ndarray
    neighbor_count: np.ndarray
    reputation_score: np.ndarray
    adaptive_threshold: float
    connection_probability: float
    missed_attack_probability: float
    false_accusation_probability: float
    predictions: np.ndarray


def _pairwise_gps_ranges(gps_position_m: np.ndarray) -> np.ndarray:
    gps = np.asarray(gps_position_m, dtype=np.float64)
    if gps.ndim != 3 or gps.shape[2] != 3:
        raise ValueError("gps_position_m must have shape (T,N,3)")
    difference = gps[:, :, None, :] - gps[:, None, :, :]
    return np.linalg.norm(difference, axis=-1)


def _nearest_neighbor_mask(
    measured_range_m: np.ndarray,
    valid_mask: np.ndarray,
    num_neighbors: int,
    symmetrize: bool,
) -> np.ndarray:
    measured = np.asarray(measured_range_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if measured.ndim != 3 or measured.shape != valid.shape:
        raise ValueError("inter-UAV range and valid mask must have shape (T,N,N)")
    if measured.shape[1] != measured.shape[2]:
        raise ValueError("inter-UAV range matrices must be square")

    _, num_uavs, _ = measured.shape
    if not 1 <= int(num_neighbors) <= num_uavs - 1:
        raise ValueError("num_neighbors must lie in [1,num_uavs-1]")

    selected = np.zeros_like(valid)
    for step in range(measured.shape[0]):
        for observer in range(num_uavs):
            candidates = np.flatnonzero(valid[step, observer])
            candidates = candidates[candidates != observer]
            if candidates.size == 0:
                continue
            count = min(int(num_neighbors), int(candidates.size))
            order = np.argsort(
                measured[step, observer, candidates],
                kind="stable",
            )[:count]
            selected[step, observer, candidates[order]] = True
    if symmetrize:
        selected |= np.swapaxes(selected, 1, 2)
        selected &= valid
    return selected


def build_local_evidence_arrays(
    gps_observed_m: np.ndarray,
    inter_uav_range_m: np.ndarray,
    inter_uav_valid_mask: np.ndarray,
    inter_uav_range_variance: np.ndarray,
    num_neighbors: int,
    variance_floor: float = 1.0e-8,
    symmetrize_neighbors: bool = True,
) -> BDGDLocalEvidence:
    """Build the project-adapted BDGD neighbor graph and residual statistic."""

    measured = np.asarray(inter_uav_range_m, dtype=np.float64)
    valid = np.asarray(inter_uav_valid_mask, dtype=bool)
    variance = np.asarray(inter_uav_range_variance, dtype=np.float64)
    if measured.shape != valid.shape or measured.shape != variance.shape:
        raise ValueError("inter-UAV range arrays must have identical shapes")
    if measured.ndim != 3:
        raise ValueError("inter-UAV range arrays must have shape (T,N,N)")
    gps_ranges = _pairwise_gps_ranges(gps_observed_m)
    if gps_ranges.shape != measured.shape:
        raise ValueError("GPS and inter-UAV range arrays describe different shapes")
    if np.any(~np.isfinite(measured[valid])):
        raise FloatingPointError("valid inter-UAV ranges must be finite")
    if np.any(~np.isfinite(variance[valid])) or np.any(variance[valid] < 0.0):
        raise FloatingPointError(
            "valid inter-UAV range variances must be finite and non-negative"
        )

    neighbor_mask = _nearest_neighbor_mask(
        measured,
        valid,
        num_neighbors,
        symmetrize_neighbors,
    )
    sigma = np.sqrt(np.maximum(variance, float(variance_floor)))
    normalized_residual = np.zeros_like(measured, dtype=np.float64)
    normalized_residual[neighbor_mask] = (
        np.abs(gps_ranges - measured)[neighbor_mask] / sigma[neighbor_mask]
    )
    if np.any(~np.isfinite(normalized_residual[neighbor_mask])):
        raise FloatingPointError("non-finite normalized BDGD residual generated")
    return BDGDLocalEvidence(
        normalized_residual=normalized_residual,
        neighbor_mask=neighbor_mask,
    )


def build_local_evidence_episode(
    episode: RawEpisode,
    num_neighbors: int,
    variance_floor: float = 1.0e-8,
    symmetrize_neighbors: bool = True,
) -> BDGDLocalEvidence:
    return build_local_evidence_arrays(
        gps_observed_m=episode.require("gps_observed_m"),
        inter_uav_range_m=episode.require("inter_uav_range_m"),
        inter_uav_valid_mask=episode.require("inter_uav_valid_mask"),
        inter_uav_range_variance=episode.require("inter_uav_range_variance"),
        num_neighbors=num_neighbors,
        variance_floor=variance_floor,
        symmetrize_neighbors=symmetrize_neighbors,
    )


def estimate_connection_probability(
    evidence: Sequence[BDGDLocalEvidence],
    valid_masks: Sequence[np.ndarray],
    warmup_steps: int = 0,
) -> tuple[float, int, int]:
    """Estimate P1 from selected links over all eligible validation pairs."""

    if len(evidence) == 0 or len(evidence) != len(valid_masks):
        raise ValueError("evidence and valid_masks must be non-empty and aligned")
    selected_count = 0
    eligible_count = 0
    for item, raw_valid in zip(evidence, valid_masks):
        valid = np.asarray(raw_valid, dtype=bool)
        if valid.shape != item.neighbor_mask.shape:
            raise ValueError("validation masks and neighbor graphs must align")
        if not 0 <= int(warmup_steps) < valid.shape[0]:
            raise ValueError("warmup_steps must leave at least one evaluated step")
        valid = valid[int(warmup_steps) :]
        selected = item.neighbor_mask[int(warmup_steps) :]
        num_uavs = valid.shape[1]
        off_diagonal = ~np.eye(num_uavs, dtype=bool)[None, :, :]
        eligible = valid & off_diagonal
        if np.any(selected & ~eligible):
            raise AssertionError("neighbor graph contains an ineligible link")
        selected_count += int(np.sum(selected))
        eligible_count += int(np.sum(eligible))
    if eligible_count <= 0:
        raise ValueError("validation set contains no eligible inter-UAV pair")
    probability = selected_count / float(eligible_count)
    if not 0.0 <= probability <= 1.0:
        raise AssertionError("estimated connection probability is invalid")
    return probability, selected_count, eligible_count


def estimate_missed_attack_probability(
    evidence: Sequence[BDGDLocalEvidence],
    labels: Sequence[np.ndarray],
    error_tolerance_multiplier: float,
    warmup_steps: int = 0,
) -> tuple[float, int, int]:
    """Estimate P2=P(z<=n | attacked target, selected validation link)."""

    if len(evidence) == 0 or len(evidence) != len(labels):
        raise ValueError("evidence and labels must be non-empty and aligned")
    missed_count = 0
    attack_link_count = 0
    threshold = float(error_tolerance_multiplier)
    for item, raw_labels in zip(evidence, labels):
        target_labels = np.asarray(raw_labels, dtype=np.uint8)
        expected_shape = item.neighbor_mask.shape[:2]
        if target_labels.shape != expected_shape:
            raise ValueError("labels must have shape (T,N)")
        if not 0 <= int(warmup_steps) < target_labels.shape[0]:
            raise ValueError("warmup_steps must leave at least one evaluated step")
        selected = item.neighbor_mask[int(warmup_steps) :]
        residual = item.normalized_residual[int(warmup_steps) :]
        evaluated_labels = target_labels[int(warmup_steps) :]
        attacked_target_links = selected & (
            evaluated_labels[:, None, :] == 1
        )
        attack_link_count += int(np.sum(attacked_target_links))
        missed_count += int(
            np.sum(
                attacked_target_links
                & (residual <= threshold)
            )
        )
    if attack_link_count <= 0:
        raise ValueError(
            "validation set contains no attacked selected neighbor-link observation"
        )
    probability = missed_count / float(attack_link_count)
    if not 0.0 <= probability <= 1.0:
        raise AssertionError("estimated missed-attack probability is invalid")
    return probability, missed_count, attack_link_count


def gaussian_false_accusation_probability(
    error_tolerance_multiplier: float,
) -> float:
    probability = float(erfc(float(error_tolerance_multiplier) / sqrt(2.0)))
    if not 0.0 <= probability <= 1.0:
        raise AssertionError("Gaussian false-accusation probability is invalid")
    return probability


def _paper_adaptive_threshold(
    num_uavs: int,
    malicious_count: int,
    connection_probability: float | np.ndarray,
    missed_attack_probability: float | np.ndarray,
    false_accusation_probability: float,
) -> np.ndarray:
    """Equations (11)-(13) from BDGD on the paper's 0-to-1 scale."""

    if not 0 <= malicious_count < num_uavs:
        raise ValueError("malicious_count must lie in [0,num_uavs)")
    p1 = np.asarray(connection_probability, dtype=np.float64)
    p2 = np.asarray(missed_attack_probability, dtype=np.float64)
    if np.any((p1 < 0.0) | (p1 > 1.0)):
        raise ValueError("connection_probability must lie in [0,1]")
    if np.any((p2 < 0.0) | (p2 > 1.0)):
        raise ValueError("missed_attack_probability must lie in [0,1]")
    if not 0.0 <= float(false_accusation_probability) <= 1.0:
        raise ValueError("false_accusation_probability must lie in [0,1]")
    denominator = p1 * float(num_uavs) + 1.0
    expected_normal = (
        (1.0 - p2) * p1 * float(malicious_count)
        + float(false_accusation_probability)
        * p1
        * float(num_uavs - malicious_count)
    ) / denominator
    expected_malicious = (
        (1.0 - p2) * p1 * float(num_uavs)
    ) / denominator
    threshold = 0.5 * (expected_normal + expected_malicious)
    if np.any((threshold < 0.0) | (threshold > 1.0)):
        raise AssertionError("paper adaptive threshold lies outside [0,1]")
    return threshold


def paper_adaptive_threshold_scalar(
    num_uavs: int,
    malicious_count: int,
    connection_probability: float,
    missed_attack_probability: float,
    false_accusation_probability: float,
) -> float:
    return float(
        _paper_adaptive_threshold(
            num_uavs=num_uavs,
            malicious_count=malicious_count,
            connection_probability=connection_probability,
            missed_attack_probability=missed_attack_probability,
            false_accusation_probability=false_accusation_probability,
        )
    )


class BDGDAdaptedDetector:
    def __init__(self, config: BDGDAdaptedConfig) -> None:
        self.config = config

    def run_arrays(
        self,
        gps_observed_m: np.ndarray,
        inter_uav_range_m: np.ndarray,
        inter_uav_valid_mask: np.ndarray,
        inter_uav_range_variance: np.ndarray,
        malicious_count: int,
    ) -> BDGDResult:
        if self.config.connection_probability is None:
            raise ValueError("connection_probability must come from calibration")
        if self.config.missed_attack_probability is None:
            raise ValueError("missed_attack_probability must come from calibration")

        evidence = build_local_evidence_arrays(
            gps_observed_m=gps_observed_m,
            inter_uav_range_m=inter_uav_range_m,
            inter_uav_valid_mask=inter_uav_valid_mask,
            inter_uav_range_variance=inter_uav_range_variance,
            num_neighbors=self.config.num_neighbors,
            variance_floor=self.config.variance_floor,
            symmetrize_neighbors=self.config.symmetrize_neighbors,
        )
        local_accusations = evidence.neighbor_mask & (
            evidence.normalized_residual
            > self.config.error_tolerance_multiplier
        )
        accusation_count = np.sum(local_accusations, axis=1, dtype=np.int32)
        neighbor_count = np.sum(evidence.neighbor_mask, axis=1, dtype=np.int32)
        reputation_score = np.divide(
            accusation_count,
            neighbor_count,
            out=np.zeros_like(accusation_count, dtype=np.float64),
            where=neighbor_count > 0,
        )
        if np.any((reputation_score < 0.0) | (reputation_score > 1.0)):
            raise AssertionError("reputation score lies outside [0,1]")

        num_uavs = int(evidence.neighbor_mask.shape[1])
        false_accusation_probability = gaussian_false_accusation_probability(
            self.config.error_tolerance_multiplier
        )
        adaptive_threshold = paper_adaptive_threshold_scalar(
            num_uavs=num_uavs,
            malicious_count=int(malicious_count),
            connection_probability=float(self.config.connection_probability),
            missed_attack_probability=float(self.config.missed_attack_probability),
            false_accusation_probability=false_accusation_probability,
        )

        # Algorithm 2 prints R < Rc, but Equation (3), Equations (11)-(13),
        # and Figure 3 imply that malicious nodes have a higher accusation
        # ratio. The mathematically consistent direction is used here.
        predictions = (reputation_score >= adaptive_threshold).astype(np.uint8)
        return BDGDResult(
            normalized_residual=evidence.normalized_residual,
            neighbor_mask=evidence.neighbor_mask,
            local_accusations=local_accusations,
            accusation_count=accusation_count,
            neighbor_count=neighbor_count,
            reputation_score=reputation_score,
            adaptive_threshold=adaptive_threshold,
            connection_probability=float(self.config.connection_probability),
            missed_attack_probability=float(self.config.missed_attack_probability),
            false_accusation_probability=false_accusation_probability,
            predictions=predictions,
        )

    def run_episode(
        self,
        episode: RawEpisode,
        malicious_count: int,
    ) -> BDGDResult:
        return self.run_arrays(
            gps_observed_m=episode.require("gps_observed_m"),
            inter_uav_range_m=episode.require("inter_uav_range_m"),
            inter_uav_valid_mask=episode.require("inter_uav_valid_mask"),
            inter_uav_range_variance=episode.require(
                "inter_uav_range_variance"
            ),
            malicious_count=malicious_count,
        )

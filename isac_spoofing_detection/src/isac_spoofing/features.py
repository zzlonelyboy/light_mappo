from __future__ import annotations

from typing import Any

import numpy as np

from .config import FeatureConfig
from .isac import ISACMeasurementModel, unit
from .localization import RangeDirectionLocalizer
from .types import ISACMeasurement, InterUAVMeasurement


RESIDUAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"{stat}_{modality}"
    for stat in ("mean", "max")
    for modality in ("range", "radial_velocity", "direction")
)
SELF_FEATURE_NAMES: tuple[str, ...] = (
    "gps_isac_position_error_norm",
    "gps_isac_position_error_heading_projection",
)
INTER_UAV_MIN_FEATURE_NAME = "inter_uav_range_min"
INTER_UAV_MAX_FEATURE_NAME = "inter_uav_range_max"
INTER_UAV_FEATURE_NAMES: tuple[str, str] = (
    INTER_UAV_MIN_FEATURE_NAME,
    INTER_UAV_MAX_FEATURE_NAME,
)
CURRENT_NEIGHBOR_FEATURE_NAMES: tuple[str, ...] = (
    "signed_self_neighbor_anomaly_contrast",
    "self_neighbor_common_projection",
    "mean_gps_isac_geometry_error",
)
FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES: tuple[str, ...] = (
    "signed_self_neighbor_anomaly_contrast",
    "median_longitudinal_formation_strain",
    "median_transverse_formation_strain",
)
FIXED_FORMATION_SIGNED_ANOMALY_CONTRAST_FEATURE_NAME = (
    "fixed_formation_signed_self_neighbor_anomaly_contrast"
)
UNCERTAINTY_ROBUST_NEIGHBOR_FEATURE_NAMES: tuple[str, ...] = (
    "self_consensus_deviation",
    "neighbor_consensus_bias",
    "neighbor_consensus_dispersion",
)
COORDINATED_NEIGHBOR_FEATURE_NAMES: tuple[str, ...] = (
    "coordinated_deviation_median_distance",
    "coordinated_mean_geometry_error",
    "coordinated_max_geometry_error",
)
NEIGHBOR_FEATURE_NAMES_BY_MODE: dict[str, tuple[str, ...]] = {
    "current": CURRENT_NEIGHBOR_FEATURE_NAMES,
    "formation_deformation": FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES,
    "uncertainty_robust": UNCERTAINTY_ROBUST_NEIGHBOR_FEATURE_NAMES,
    "coordinated": COORDINATED_NEIGHBOR_FEATURE_NAMES,
    "inter_uav": INTER_UAV_FEATURE_NAMES,
}
# Compatibility aliases: existing datasets and commands continue to use the
# current target-centred feature schema unless explicitly configured otherwise.
NEIGHBOR_FEATURE_NAMES: tuple[str, ...] = CURRENT_NEIGHBOR_FEATURE_NAMES
FEATURE_NAMES: tuple[str, ...] = (
    RESIDUAL_FEATURE_NAMES + SELF_FEATURE_NAMES + NEIGHBOR_FEATURE_NAMES
)


def neighbor_feature_names(mode: str) -> tuple[str, ...]:
    try:
        return NEIGHBOR_FEATURE_NAMES_BY_MODE[str(mode)]
    except KeyError as exc:
        available = ", ".join(NEIGHBOR_FEATURE_NAMES_BY_MODE)
        raise ValueError(
            f"unknown neighbor feature mode {mode!r}; choose from: {available}"
        ) from exc


def feature_names_for_neighbor_mode(mode: str) -> tuple[str, ...]:
    return RESIDUAL_FEATURE_NAMES + SELF_FEATURE_NAMES + neighbor_feature_names(mode)


def _resolve_neighbor_indices(
    isac_position_m: np.ndarray,
    num_neighbors: int,
    fixed_neighbor_indices: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    isac = np.asarray(isac_position_m, dtype=np.float32)
    num_uavs = isac.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    pair_distance = np.linalg.norm(
        isac[..., :, None, :] - isac[..., None, :, :], axis=-1
    )
    diagonal = np.arange(num_uavs)
    pair_distance[..., diagonal, diagonal] = np.inf
    if k == 0:
        empty = np.empty(isac.shape[:-2] + (num_uavs, 0), dtype=np.int64)
        return empty, pair_distance
    if fixed_neighbor_indices is None:
        indices = np.argpartition(
            pair_distance, kth=k - 1, axis=-1
        )[..., :k]
        return indices, pair_distance

    fixed = np.asarray(fixed_neighbor_indices, dtype=np.int64)
    if fixed.shape != (num_uavs, k):
        raise ValueError(
            "fixed_neighbor_indices must have shape "
            f"({num_uavs}, {k}), got {fixed.shape}"
        )
    if np.any(fixed < 0) or np.any(fixed >= num_uavs):
        raise ValueError("fixed neighbor indices are out of range")
    if np.any(fixed == np.arange(num_uavs)[:, None]):
        raise ValueError("a UAV cannot be its own fixed neighbor")
    if any(len(set(row.tolist())) != k for row in fixed):
        raise ValueError("fixed neighbors must be unique for every UAV")
    indices = np.broadcast_to(
        fixed,
        isac.shape[:-2] + fixed.shape,
    )
    return indices, pair_distance


def compute_neighbor_features_from_positions(
    gps_position_m: np.ndarray,
    isac_position_m: np.ndarray,
    *,
    position_scale_m: float,
    num_neighbors: int,
    fixed_neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the three target-centered instantaneous neighbor features.

    Inputs may have any leading dimensions followed by ``(num_uavs, 3)``.
    The result has the corresponding leading dimensions followed by
    ``(num_uavs, 3)``. No feature is clipped.
    """

    gps = np.asarray(gps_position_m, dtype=np.float32)
    isac = np.asarray(isac_position_m, dtype=np.float32)
    if gps.shape != isac.shape or gps.ndim < 2 or gps.shape[-1] != 3:
        raise ValueError(
            "GPS and ISAC positions must have the same shape (..., num_uavs, 3)"
        )
    if position_scale_m <= 0.0:
        raise ValueError("position_scale_m must be positive")
    num_uavs = gps.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    if k == 0:
        return np.zeros(gps.shape[:-1] + (len(NEIGHBOR_FEATURE_NAMES),), dtype=np.float32)

    position_error = (gps - isac) / float(position_scale_m)
    anomaly_norm = np.linalg.norm(position_error, axis=-1)
    neighbor_indices, isac_pair_distance = _resolve_neighbor_indices(
        isac,
        k,
        fixed_neighbor_indices,
    )

    target_shape = position_error.shape[:-2] + (num_uavs, num_uavs, 3)
    error_source = np.broadcast_to(position_error[..., None, :, :], target_shape)
    neighbor_error = np.take_along_axis(
        error_source, neighbor_indices[..., None], axis=-2
    )
    anomaly_source = np.broadcast_to(
        anomaly_norm[..., None, :], anomaly_norm.shape[:-1] + (num_uavs, num_uavs)
    )
    neighbor_anomaly = np.take_along_axis(
        anomaly_source, neighbor_indices, axis=-1
    )

    neighbor_anomaly_median = np.median(neighbor_anomaly, axis=-1)
    signed_contrast = anomaly_norm - neighbor_anomaly_median
    neighbor_error_mean = np.mean(neighbor_error, axis=-2)
    neighbor_error_mean_norm = np.linalg.norm(neighbor_error_mean, axis=-1)
    common_projection_numerator = np.sum(
        position_error * neighbor_error_mean, axis=-1
    )
    common_projection = np.divide(
        common_projection_numerator,
        neighbor_error_mean_norm,
        out=np.zeros_like(neighbor_error_mean_norm, dtype=np.float32),
        where=neighbor_error_mean_norm > np.finfo(np.float32).eps,
    )

    gps_source = np.broadcast_to(gps[..., None, :, :], target_shape)
    neighbor_gps = np.take_along_axis(
        gps_source, neighbor_indices[..., None], axis=-2
    )
    selected_gps_distance = np.linalg.norm(gps[..., :, None, :] - neighbor_gps, axis=-1)
    selected_isac_distance = np.take_along_axis(
        isac_pair_distance, neighbor_indices, axis=-1
    )
    geometry_error_mean = np.mean(
        np.abs(selected_gps_distance - selected_isac_distance)
        / float(position_scale_m),
        axis=-1,
    )

    features = np.stack(
        (signed_contrast, common_projection, geometry_error_mean), axis=-1
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("non-finite neighbor feature generated")
    return features


def compute_coordinated_neighbor_features(
    gps_position_m: np.ndarray,
    isac_position_m: np.ndarray,
    *,
    position_scale_m: float,
    num_neighbors: int,
) -> np.ndarray:
    """Compute three coordinated neighbourhood features."""
    gps = np.asarray(gps_position_m, dtype=np.float32)
    isac = np.asarray(isac_position_m, dtype=np.float32)
    if gps.shape != isac.shape or gps.ndim < 2 or gps.shape[-1] != 3:
        raise ValueError("GPS and ISAC positions must have same shape")
    if position_scale_m <= 0.0:
        raise ValueError("position_scale_m must be positive")
    num_uavs = gps.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    if k == 0:
        return np.zeros(gps.shape[:-1] + (len(COORDINATED_NEIGHBOR_FEATURE_NAMES),), dtype=np.float32)
    delta_p = gps - isac
    neighbor_indices, isac_pair_distance = _resolve_neighbor_indices(isac, k, fixed_neighbor_indices=None)
    target_shape = delta_p.shape[:-2] + (num_uavs, num_uavs, 3)
    delta_source = np.broadcast_to(delta_p[..., None, :, :], target_shape)
    neighbor_delta = np.take_along_axis(delta_source, neighbor_indices[..., None], axis=-2)
    neighbor_median = np.median(neighbor_delta, axis=-2)
    e_rel = np.linalg.norm(delta_p - neighbor_median, axis=-1) / float(position_scale_m)
    gps_source = np.broadcast_to(gps[..., None, :, :], target_shape)
    neighbor_gps = np.take_along_axis(gps_source, neighbor_indices[..., None], axis=-2)
    isac_source = np.broadcast_to(isac[..., None, :, :], target_shape)
    neighbor_isac = np.take_along_axis(isac_source, neighbor_indices[..., None], axis=-2)
    gps_pair_dist = np.linalg.norm(gps[..., :, None, :] - neighbor_gps, axis=-1)
    isac_pair_dist = np.take_along_axis(isac_pair_distance, neighbor_indices, axis=-1)
    e_geo = np.abs(gps_pair_dist - isac_pair_dist) / float(position_scale_m)
    e_geo_mean = np.mean(e_geo, axis=-1)
    e_geo_max = np.max(e_geo, axis=-1)
    features = np.stack((e_rel, e_geo_mean, e_geo_max), axis=-1).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("non-finite coordinated neighbour feature")
    return features


def compute_formation_deformation_neighbor_features(
    gps_position_m: np.ndarray,
    isac_position_m: np.ndarray,
    *,
    position_scale_m: float,
    num_neighbors: int,
    fixed_neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute target contrast and robust relative-formation strains.

    For each target-neighbor pair, the GPS-versus-ISAC relative-position
    deformation is decomposed into components parallel and transverse to the
    ISAC-derived inter-UAV link.  Both components are normalized by the link
    length and aggregated with the neighborhood median.  The returned feature
    block is instantaneous, has three dimensions for any positive neighborhood
    size, and uses no clipping or simulator-truth quantities.
    """

    gps = np.asarray(gps_position_m, dtype=np.float64)
    isac = np.asarray(isac_position_m, dtype=np.float64)
    if gps.shape != isac.shape or gps.ndim < 2 or gps.shape[-1] != 3:
        raise ValueError(
            "GPS and ISAC positions must have the same shape (..., num_uavs, 3)"
        )
    if position_scale_m <= 0.0:
        raise ValueError("position_scale_m must be positive")
    num_uavs = gps.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    output_shape = gps.shape[:-1] + (
        len(FORMATION_DEFORMATION_NEIGHBOR_FEATURE_NAMES),
    )
    if k == 0:
        return np.zeros(output_shape, dtype=np.float32)

    neighbor_indices, _ = _resolve_neighbor_indices(
        isac,
        k,
        fixed_neighbor_indices,
    )
    position_error = (gps - isac) / float(position_scale_m)
    anomaly_norm = np.linalg.norm(position_error, axis=-1)
    anomaly_source = np.broadcast_to(
        anomaly_norm[..., None, :],
        anomaly_norm.shape[:-1] + (num_uavs, num_uavs),
    )
    neighbor_anomaly = np.take_along_axis(
        anomaly_source,
        neighbor_indices,
        axis=-1,
    )
    signed_contrast = anomaly_norm - np.median(neighbor_anomaly, axis=-1)

    target_shape = gps.shape[:-2] + (num_uavs, num_uavs, 3)
    gps_source = np.broadcast_to(gps[..., None, :, :], target_shape)
    isac_source = np.broadcast_to(isac[..., None, :, :], target_shape)
    neighbor_gps = np.take_along_axis(
        gps_source,
        neighbor_indices[..., None],
        axis=-2,
    )
    neighbor_isac = np.take_along_axis(
        isac_source,
        neighbor_indices[..., None],
        axis=-2,
    )

    relative_gps = gps[..., :, None, :] - neighbor_gps
    relative_isac = isac[..., :, None, :] - neighbor_isac
    link_length = np.linalg.norm(relative_isac, axis=-1)
    numerical_epsilon = np.finfo(np.float64).eps
    link_unit = relative_isac / (link_length[..., None] + numerical_epsilon)
    relative_deformation = relative_gps - relative_isac
    longitudinal_deformation = np.sum(
        relative_deformation * link_unit,
        axis=-1,
    )
    transverse_deformation = (
        relative_deformation
        - longitudinal_deformation[..., None] * link_unit
    )
    inverse_link_scale = 1.0 / (link_length + numerical_epsilon)
    longitudinal_strain = (
        np.abs(longitudinal_deformation) * inverse_link_scale
    )
    transverse_strain = (
        np.linalg.norm(transverse_deformation, axis=-1) * inverse_link_scale
    )
    median_longitudinal_strain = np.median(longitudinal_strain, axis=-1)
    median_transverse_strain = np.median(transverse_strain, axis=-1)

    features = np.stack(
        (
            signed_contrast,
            median_longitudinal_strain,
            median_transverse_strain,
        ),
        axis=-1,
    ).astype(np.float32)
    if features.shape != output_shape or not np.all(np.isfinite(features)):
        raise FloatingPointError(
            "invalid relative-formation deformation features generated"
        )
    return features


def estimate_gps_isac_error_precision(
    measured_range_m: np.ndarray,
    measured_los_unit: np.ndarray,
    *,
    isac_model: ISACMeasurementModel,
    gps_noise_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Estimate per-UAV GPS--ISAC error precision from detector-visible data.

    The inputs have shape ``(..., num_uavs, num_bs)`` and
    ``(..., num_uavs, num_bs, 3)``.  SNR and measurement variances are
    reconstructed from measured ranges; simulator truth-derived SNR/variance
    fields are deliberately not accepted by this interface.
    """

    ranges = np.abs(np.asarray(measured_range_m, dtype=np.float64))
    directions = np.asarray(measured_los_unit, dtype=np.float64)
    if ranges.ndim < 2:
        raise ValueError("measured_range_m must end in (num_uavs, num_bs)")
    if directions.shape != ranges.shape + (3,):
        raise ValueError(
            "measured_los_unit must have shape measured_range_m.shape + (3,)"
        )
    direction_norm = np.linalg.norm(directions, axis=-1, keepdims=True)
    if np.any(direction_norm <= 0.0) or np.any(~np.isfinite(direction_norm)):
        raise ValueError("measured LoS directions must be finite and non-zero")
    directions = directions / direction_norm

    range_variance, _, direction_variance, _ = (
        isac_model.variances_from_range(ranges)
    )
    identity = np.eye(3, dtype=np.float64)
    precision = np.zeros(ranges.shape[:-1] + (3, 3), dtype=np.float64)
    for anchor_index in range(ranges.shape[-1]):
        los = directions[..., anchor_index, :]
        longitudinal = los[..., :, None] * los[..., None, :]
        transverse = identity - longitudinal
        precision += longitudinal / range_variance[
            ..., anchor_index, None, None
        ]
        precision += transverse / (
            ranges[..., anchor_index, None, None] ** 2
            * direction_variance[..., anchor_index, None, None]
        )

    gps_std = np.asarray(gps_noise_std_m, dtype=np.float64)
    if gps_std.shape != (3,) or np.any(gps_std < 0.0):
        raise ValueError("gps_noise_std_m must contain three non-negative values")
    if np.any(gps_std > 0.0):
        isac_covariance = np.linalg.inv(precision)
        gps_covariance = np.diag(gps_std * gps_std)
        precision = np.linalg.inv(isac_covariance + gps_covariance)
    if not np.all(np.isfinite(precision)):
        raise FloatingPointError("non-finite GPS--ISAC error precision generated")
    return precision


def compute_uncertainty_robust_neighbor_features(
    gps_position_m: np.ndarray,
    isac_position_m: np.ndarray,
    measured_range_m: np.ndarray,
    measured_los_unit: np.ndarray,
    *,
    isac_model: ISACMeasurementModel,
    gps_noise_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    num_neighbors: int,
    consensus_iterations: int = 8,
    fixed_neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Return three uncertainty-aware robust consensus features per UAV.

    The robust centre is an uncertainty-weighted geometric median computed by
    iteratively reweighted least squares.  No feature clipping is applied.
    """

    gps = np.asarray(gps_position_m, dtype=np.float64)
    isac = np.asarray(isac_position_m, dtype=np.float64)
    ranges = np.asarray(measured_range_m, dtype=np.float64)
    directions = np.asarray(measured_los_unit, dtype=np.float64)
    if gps.shape != isac.shape or gps.ndim < 2 or gps.shape[-1] != 3:
        raise ValueError(
            "GPS and ISAC positions must have the same shape (..., num_uavs, 3)"
        )
    expected_range_shape = gps.shape[:-1] + (directions.shape[-2],)
    if ranges.shape != expected_range_shape or directions.shape != ranges.shape + (3,):
        raise ValueError("ISAC range/LoS arrays are inconsistent with UAV positions")
    if consensus_iterations < 1:
        raise ValueError("consensus_iterations must be positive")

    num_uavs = gps.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    output_shape = gps.shape[:-1] + (
        len(UNCERTAINTY_ROBUST_NEIGHBOR_FEATURE_NAMES),
    )
    if k == 0:
        return np.zeros(output_shape, dtype=np.float32)

    error = gps - isac
    precision = estimate_gps_isac_error_precision(
        ranges,
        directions,
        isac_model=isac_model,
        gps_noise_std_m=gps_noise_std_m,
    )
    neighbor_indices, _ = _resolve_neighbor_indices(
        isac,
        k,
        fixed_neighbor_indices,
    )

    target_shape = error.shape[:-2] + (num_uavs, num_uavs, 3)
    error_source = np.broadcast_to(error[..., None, :, :], target_shape)
    neighbor_error = np.take_along_axis(
        error_source, neighbor_indices[..., None], axis=-2
    )
    precision_target_shape = error.shape[:-2] + (
        num_uavs,
        num_uavs,
        3,
        3,
    )
    precision_source = np.broadcast_to(
        precision[..., None, :, :, :], precision_target_shape
    )
    neighbor_precision = np.take_along_axis(
        precision_source,
        neighbor_indices[..., None, None],
        axis=-3,
    )

    consensus = np.median(neighbor_error, axis=-2)
    numerical_epsilon = np.finfo(np.float64).eps
    for _ in range(consensus_iterations):
        difference = neighbor_error - consensus[..., None, :]
        squared_distance = np.einsum(
            "...ki,...kij,...kj->...k",
            difference,
            neighbor_precision,
            difference,
            optimize=True,
        )
        inverse_distance = 1.0 / np.maximum(
            np.sqrt(np.maximum(squared_distance, 0.0)),
            numerical_epsilon,
        )
        weighted_precision = (
            neighbor_precision * inverse_distance[..., None, None]
        )
        information = np.sum(weighted_precision, axis=-3)
        precision_error = np.einsum(
            "...kij,...kj->...ki",
            neighbor_precision,
            neighbor_error,
            optimize=True,
        )
        information_vector = np.sum(
            precision_error * inverse_distance[..., None], axis=-2
        )
        consensus = np.linalg.solve(
            information, information_vector[..., None]
        )[..., 0]

    consensus_precision = np.sum(neighbor_precision, axis=-3)
    parallel_solve = np.linalg.solve(
        precision + consensus_precision, precision
    )
    combined_precision = precision - np.matmul(precision, parallel_solve)
    combined_precision = 0.5 * (
        combined_precision + np.swapaxes(combined_precision, -1, -2)
    )
    self_difference = error - consensus
    self_consensus_deviation = np.sqrt(
        np.maximum(
            np.einsum(
                "...i,...ij,...j->...",
                self_difference,
                combined_precision,
                self_difference,
                optimize=True,
            ),
            0.0,
        )
    )
    consensus_bias = np.sqrt(
        np.maximum(
            np.einsum(
                "...i,...ij,...j->...",
                consensus,
                consensus_precision,
                consensus,
                optimize=True,
            ),
            0.0,
        )
    )
    neighbor_difference = neighbor_error - consensus[..., None, :]
    neighbor_distance = np.sqrt(
        np.maximum(
            np.einsum(
                "...ki,...kij,...kj->...k",
                neighbor_difference,
                neighbor_precision,
                neighbor_difference,
                optimize=True,
            ),
            0.0,
        )
    )
    consensus_dispersion = np.median(neighbor_distance, axis=-1)
    features = np.stack(
        (self_consensus_deviation, consensus_bias, consensus_dispersion),
        axis=-1,
    ).astype(np.float32)
    if features.shape != output_shape or not np.all(np.isfinite(features)):
        raise FloatingPointError(
            "invalid uncertainty-aware robust neighbor features generated"
        )
    return features


class PaperFeatureExtractor:
    """Construct 6 + 2 + 3 instantaneous detector features.

    Temporal dependence is intentionally left to the sequence model.  The
    extractor therefore contains no EWMA, CUSUM, history window, or other
    hand-crafted temporal state.
    """

    def __init__(
        self,
        config: FeatureConfig,
        isac_model: ISACMeasurementModel,
        localizer: RangeDirectionLocalizer | None = None,
        gps_noise_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.config = config
        self.isac_model = isac_model
        self.localizer = localizer or RangeDirectionLocalizer()
        self.gps_noise_std_m = tuple(float(value) for value in gps_noise_std_m)
        self.feature_names = feature_names_for_neighbor_mode(
            self.config.neighbor_feature_mode
        )
        self.reset()

    def reset(
        self,
        fixed_neighbor_indices: np.ndarray | None = None,
    ) -> None:
        self._previous_isac_position: np.ndarray | None = None
        self._fixed_neighbor_indices = (
            None
            if fixed_neighbor_indices is None
            else np.asarray(fixed_neighbor_indices, dtype=np.int64).copy()
        )

    @staticmethod
    def _mean_max(link_residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(link_residual, axis=1)
        maximum = np.max(link_residual, axis=1)
        return mean.astype(np.float32), maximum.astype(np.float32)

    def _residual_features(
        self, current: np.ndarray, maximum: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        features = np.concatenate([current, maximum], axis=1)
        diagnostics = {
            "residual_mean": current.copy(),
            "residual_max": maximum.copy(),
        }
        return features.astype(np.float32), diagnostics

    def _neighbor_features(
        self,
        gps_position_m: np.ndarray,
        isac_position_m: np.ndarray,
        normalized_position_error: np.ndarray,
        measurement: ISACMeasurement,
        inter_uav_measurement: InterUAVMeasurement | None,
    ) -> np.ndarray:
        expected_error = (
            np.asarray(gps_position_m, dtype=np.float32)
            - np.asarray(isac_position_m, dtype=np.float32)
        ) / self.config.position_scale_m
        if not np.allclose(
            expected_error,
            np.asarray(normalized_position_error, dtype=np.float32),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("normalized_position_error is inconsistent with positions")
        if self.config.neighbor_feature_mode == "current":
            return compute_neighbor_features_from_positions(
                gps_position_m,
                isac_position_m,
                position_scale_m=self.config.position_scale_m,
                num_neighbors=self.config.num_neighbors,
                fixed_neighbor_indices=self._fixed_neighbor_indices,
            )
        if self.config.neighbor_feature_mode == "formation_deformation":
            return compute_formation_deformation_neighbor_features(
                gps_position_m,
                isac_position_m,
                position_scale_m=self.config.position_scale_m,
                num_neighbors=self.config.num_neighbors,
                fixed_neighbor_indices=self._fixed_neighbor_indices,
            )
        if self.config.neighbor_feature_mode == "uncertainty_robust":
            return compute_uncertainty_robust_neighbor_features(
                gps_position_m,
                isac_position_m,
                measurement.range_m,
                measurement.los_unit,
                isac_model=self.isac_model,
                gps_noise_std_m=self.gps_noise_std_m,
                num_neighbors=self.config.num_neighbors,
                fixed_neighbor_indices=self._fixed_neighbor_indices,
            )
        if self.config.neighbor_feature_mode == "inter_uav":
            if inter_uav_measurement is None:
                raise ValueError(
                    "inter_uav neighbor features require an inter-UAV measurement"
                )
            return compute_inter_uav_consistency_from_measurement(
                gps_position_m,
                inter_uav_measurement.range_m,
                inter_uav_measurement.valid_mask,
                num_neighbors=self.config.num_neighbors,
                fixed_neighbor_indices=self._fixed_neighbor_indices,
            )
        raise RuntimeError(
            f"unsupported neighbor feature mode: {self.config.neighbor_feature_mode}"
        )

    def compute(
        self,
        gps_position_m: np.ndarray,
        trusted_velocity_mps: np.ndarray,
        trusted_heading: np.ndarray,
        measurement: ISACMeasurement,
        inter_uav_measurement: InterUAVMeasurement | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Build detector-visible features from GPS, ISAC, and trusted motion.

        ``trusted_velocity_mps`` and ``trusted_heading`` represent planned or
        flight-controller motion information.  Simulator ground-truth position
        is deliberately not an input to this method.
        """
        gps = np.asarray(gps_position_m, dtype=np.float32)
        velocity = np.asarray(trusted_velocity_mps, dtype=np.float32)
        heading = np.asarray(trusted_heading, dtype=np.float32)
        anchors = self.isac_model.anchors_m.astype(np.float32)

        gps_displacement = anchors[None, :, :] - gps[:, None, :]
        gps_range = np.linalg.norm(gps_displacement, axis=-1)
        gps_los = unit(gps_displacement)
        gps_radial = np.sum(velocity[:, None, :] * gps_los, axis=-1)
        range_var, radial_var, direction_var, estimated_snr = (
            self.isac_model.variances_from_range(measurement.range_m)
        )
        sigma_range = np.sqrt(range_var)
        sigma_radial = np.sqrt(radial_var)
        sigma_direction = np.sqrt(direction_var)

        range_residual = np.abs(measurement.range_m - gps_range) / sigma_range
        radial_residual = (
            np.abs(measurement.radial_velocity_mps - gps_radial) / sigma_radial
        )
        cosine = np.sum(measurement.los_unit * gps_los, axis=-1)
        direction_residual = np.abs(1.0 - cosine) / sigma_direction
        link_residual = np.stack(
            [range_residual, radial_residual, direction_residual], axis=2
        ).astype(np.float32)
        residual_mean, residual_max = self._mean_max(link_residual)
        residual_features, diagnostics = self._residual_features(
            residual_mean, residual_max
        )

        if self._previous_isac_position is None:
            initial = self.localizer.initialize(
                anchors,
                measurement.range_m,
                measurement.los_unit,
                measurement.valid_mask,
            )
        else:
            initial = self._previous_isac_position
        isac_position = self.localizer.estimate(
            anchors,
            measurement.range_m,
            measurement.los_unit,
            measurement.valid_mask,
            initial_position_m=initial,
        )
        self._previous_isac_position = isac_position.copy()

        normalized_error = (gps - isac_position) / self.config.position_scale_m
        self_error_norm = np.linalg.norm(normalized_error, axis=1, keepdims=True)
        self_error_projection = np.sum(
            normalized_error * heading, axis=1, keepdims=True
        )
        self_features = np.concatenate(
            [self_error_norm, self_error_projection], axis=1
        ).astype(np.float32)
        neighbor_features = self._neighbor_features(
            gps,
            isac_position,
            normalized_error,
            measurement,
            inter_uav_measurement,
        )
        features = np.concatenate(
            [residual_features, self_features, neighbor_features], axis=1
        ).astype(np.float32)
        if not np.all(np.isfinite(features)):
            raise FloatingPointError("non-finite detector feature generated")
        if features.shape != (gps.shape[0], len(self.feature_names)):
            raise RuntimeError(
                f"expected feature shape ({gps.shape[0]}, {len(self.feature_names)}), "
                f"got {features.shape}"
            )
        diagnostics.update(
            {
                "link_residual": link_residual,
                "estimated_snr": estimated_snr.astype(np.float32),
                "self_features": self_features,
                "neighbor_features": neighbor_features,
                "neighbor_feature_mode": self.config.neighbor_feature_mode,
            }
        )
        return features, isac_position, diagnostics


# ── UAV 互测距一致性特征 ─────────────────────────────────────────

def compute_inter_uav_consistency_from_measurement(
    gps_observed_m: np.ndarray,
    measured_range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    num_neighbors: int = 3,
    fixed_neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate measured inter-UAV range/GPS disparities for each UAV.

    Dynamic neighborhoods are selected from the measured range matrix. A
    fixed attack-independent topology can be supplied for formation studies.
    The result has shape ``(..., num_uavs, 2)`` and contains the minimum and
    maximum absolute link disparities over the selected neighbors.
    """

    gps = np.asarray(gps_observed_m, dtype=np.float64)
    measured = np.asarray(measured_range_m, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if gps.ndim < 2 or gps.shape[-1] != 3:
        raise ValueError("gps_observed_m must have shape (..., num_uavs, 3)")
    num_uavs = gps.shape[-2]
    expected_range_shape = gps.shape[:-1] + (num_uavs,)
    if measured.shape != expected_range_shape or valid.shape != expected_range_shape:
        raise ValueError(
            "inter-UAV range and mask must have shape (..., num_uavs, num_uavs)"
        )
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    if k == 0:
        return np.zeros(gps.shape[:-1] + (2,), dtype=np.float32)

    diagonal = np.arange(num_uavs)
    selection_range = np.where(
        valid & np.isfinite(measured) & (measured > 0.0),
        measured,
        np.inf,
    )
    selection_range[..., diagonal, diagonal] = np.inf
    if np.any(np.sum(np.isfinite(selection_range), axis=-1) < k):
        raise ValueError("fewer valid inter-UAV links than requested neighbors")

    if fixed_neighbor_indices is None:
        neighbor_idx = np.argpartition(
            selection_range, kth=k - 1, axis=-1
        )[..., :k]
    else:
        fixed = np.asarray(fixed_neighbor_indices, dtype=np.int64)
        if fixed.shape != (num_uavs, k):
            raise ValueError(
                "fixed_neighbor_indices must have shape "
                f"({num_uavs}, {k}), got {fixed.shape}"
            )
        if np.any(fixed < 0) or np.any(fixed >= num_uavs):
            raise ValueError("fixed neighbor indices are out of range")
        if np.any(fixed == diagonal[:, None]):
            raise ValueError("a UAV cannot be its own fixed neighbor")
        if any(len(set(row.tolist())) != k for row in fixed):
            raise ValueError("fixed neighbors must be unique for every UAV")
        neighbor_idx = np.broadcast_to(fixed, gps.shape[:-2] + fixed.shape)

    gps_pair_range = np.linalg.norm(
        gps[..., :, None, :] - gps[..., None, :, :], axis=-1
    )
    selected_gps = np.take_along_axis(gps_pair_range, neighbor_idx, axis=-1)
    selected_isac = np.take_along_axis(measured, neighbor_idx, axis=-1)
    selected_valid = np.take_along_axis(valid, neighbor_idx, axis=-1)
    if not np.all(selected_valid):
        raise ValueError("fixed topology contains an invalid inter-UAV link")

    disparity = np.abs(selected_isac - selected_gps)
    features = np.stack(
        [np.min(disparity, axis=-1), np.max(disparity, axis=-1)], axis=-1
    ).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise FloatingPointError("non-finite inter-UAV consistency")
    return features


def compute_inter_uav_consistency_feature(
    true_position_m: np.ndarray,
    gps_observed_m: np.ndarray,
    *,
    carrier_hz: float = 3.0e9,
    bandwidth_hz: float = 30.0e6,
    tx_power_dbm: float = 27.0,
    radar_cross_section: float = 6.0,
    num_neighbors: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate and aggregate inter-UAV ranging consistency features.

    Simulates direct UAV-to-UAV ISAC ranging and compares ISAC-measured
    inter-UAV distances against GPS-derived distances.  The ranging noise
    follows the identical radar equation as the base-station ISAC model.

    Returns ``(..., num_uavs, 2)`` float32 array:
      F0 = min disparity over K neighbours (best pair)
      F1 = max disparity over K neighbours (worst pair)
    """
    true_pos = np.asarray(true_position_m, dtype=np.float64)
    gps_pos = np.asarray(gps_observed_m, dtype=np.float64)
    if true_pos.shape != gps_pos.shape or true_pos.ndim < 2 or true_pos.shape[-1] != 3:
        raise ValueError("positions must have shape (..., num_uavs, 3)")
    num_uavs = true_pos.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k <= 0:
        return np.zeros(true_pos.shape[:-1] + (2,), dtype=np.float32)

    flat_true = true_pos.reshape(-1, num_uavs, 3)
    flat_gps = gps_pos.reshape(-1, num_uavs, 3)
    pair_dists = np.linalg.norm(
        flat_true[:, :, None, :] - flat_true[:, None, :, :], axis=-1)
    r_true = pair_dists.copy()

    wavelength = 3.0e8 / carrier_hz
    tx_power_w = 10.0 ** ((tx_power_dbm - 30.0) / 10.0)
    noise_density_w_hz = 10.0 ** ((-174.0 - 30.0) / 10.0)
    noise_power_w = noise_density_w_hz * bandwidth_hz

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (tx_power_w * radar_cross_section * wavelength**2
               / ((4.0 * np.pi) ** 3 * r_true**4 * noise_power_w))
    snr = np.where(r_true > 0.0, snr, np.inf)
    range_variance = (3.0e8 / bandwidth_hz) ** 2 / snr
    sigma_range = np.sqrt(np.maximum(range_variance, 0.0))

    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, r_true.shape).astype(np.float64)
    r_isac = r_true + noise * sigma_range
    diagonal = np.arange(num_uavs)
    r_isac[:, diagonal, diagonal] = 0.0
    valid = np.broadcast_to(~np.eye(num_uavs, dtype=bool), r_isac.shape)
    features = compute_inter_uav_consistency_from_measurement(
        flat_gps,
        r_isac,
        valid,
        num_neighbors=k,
    )
    return features.reshape(true_pos.shape[:-1] + (2,))

from __future__ import annotations

from typing import Any

import numpy as np

from .config import FeatureConfig, ISACConfig
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
    reported_covariance: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate per-UAV GPS--ISAC error precision from detector-visible data.

    The inputs have shape ``(..., num_uavs, num_bs)`` and
    ``(..., num_uavs, num_bs, 3)``.  SNR and measurement variances are
    obtained from the front-end quality report, not reconstructed from range.
    Simulator generation covariances are not accepted as a replacement report.
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

    if reported_covariance is None:
        raise ValueError("error precision requires reported distance/angle covariance")
    covariance = np.asarray(reported_covariance, dtype=np.float64)
    if covariance.shape == ranges.shape + (4, 4):
        covariance = covariance[..., :3, :3]
    if covariance.shape != ranges.shape + (3, 3):
        raise ValueError("reported covariance shape must match range/angle observations")
    delta = -ranges[..., None] * directions
    x, y, z = np.moveaxis(delta, -1, 0)
    horizontal_squared = x * x + y * y
    horizontal = np.sqrt(horizontal_squared)
    if np.any(ranges <= 0.0) or np.any(horizontal <= 0.0):
        raise ValueError("reported geometry is singular at zero range or zenith")
    jacobian = np.stack(
        [
            -directions,
            np.stack([-y / horizontal_squared, x / horizontal_squared,
                      np.zeros_like(x)], axis=-1),
            np.stack([-x * z / (ranges * ranges * horizontal),
                      -y * z / (ranges * ranges * horizontal),
                      horizontal / (ranges * ranges)], axis=-1),
        ], axis=-2,
    )
    try:
        inverse_cholesky = np.linalg.inv(np.linalg.cholesky(covariance))
    except np.linalg.LinAlgError as exc:
        raise ValueError("reported covariance must be positive definite") from exc
    whitened = inverse_cholesky @ jacobian
    precision = np.sum(
        np.swapaxes(whitened, -1, -2) @ whitened, axis=-3,
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
    reported_covariance: np.ndarray | None = None,
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
        reported_covariance=reported_covariance,
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
    """Construct link, position and neighborhood instantaneous features.

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
        gps_std = np.asarray(self.gps_noise_std_m, dtype=np.float64)
        if gps_std.shape != (3,) or np.any(~np.isfinite(gps_std)) or np.any(gps_std < 0.0):
            raise ValueError("gps_noise_std_m must contain three finite nonnegative values")
        self._gps_covariance = np.diag(gps_std * gps_std)
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
                reported_covariance=measurement.reported_covariance,
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
        """Build features from reports and the known executed velocity command.

        trusted_velocity_mps keeps its historical argument name for caller
        compatibility. It means the issued command under ideal velocity
        tracking. trusted_heading is retained only for signature compatibility.
        The projection direction is derived from the command, with the
        projection set to zero at zero commanded speed.
        """
        gps = np.asarray(gps_position_m, dtype=np.float64)
        velocity = np.asarray(trusted_velocity_mps, dtype=np.float64)
        if gps.ndim != 2 or gps.shape[-1] != 3 or velocity.shape != gps.shape:
            raise ValueError("GPS and command velocity must have shape (N, 3)")
        if not np.all(np.isfinite(gps)) or not np.all(np.isfinite(velocity)):
            raise ValueError("GPS and command velocity must be finite")
        heading = unit(velocity)
        anchors = self.isac_model.anchors_m.astype(np.float64)

        gps_displacement = anchors[None, :, :] - gps[:, None, :]
        gps_range = np.linalg.norm(gps_displacement, axis=-1)
        gps_los = unit(gps_displacement)
        gps_radial = np.sum(velocity[:, None, :] * gps_los, axis=-1)
        sigma_range, sigma_radial, sigma_direction, direction_difference_covariance = (
            self._residual_scales(measurement, velocity)
        )

        range_residual = np.abs(measurement.range_m - gps_range) / sigma_range
        radial_residual = (
            np.abs(measurement.radial_velocity_mps - gps_radial) / sigma_radial
        )
        # For unit vectors, 1 - u_I.T u_g == ||u_I-u_g||^2 / 2.
        # The latter avoids cancellation for very precise angle observations.
        measured_los = unit(np.asarray(measurement.los_unit, dtype=np.float64))
        direction_raw = 0.5 * np.sum((measured_los - gps_los) ** 2, axis=-1)
        direction_residual = direction_raw / sigma_direction
        link_residual = np.stack(
            [range_residual, radial_residual, direction_residual], axis=2
        ).astype(np.float32)
        residual_mean, residual_max = self._mean_max(link_residual)
        residual_features, diagnostics = self._residual_features(
            residual_mean, residual_max
        )

        reported_covariance = getattr(measurement, "reported_covariance", None)
        if self._previous_isac_position is None and reported_covariance is None:
            initial = self.localizer.initialize(
                anchors,
                measurement.range_m,
                measurement.los_unit,
                measurement.valid_mask,
            )
        else:
            initial = self._previous_isac_position
        quality_arguments = {}
        if reported_covariance is not None:
            quality_arguments = {
                "measurement_covariance": reported_covariance,
                "azimuth_rad": measurement.azimuth_rad,
                "elevation_rad": measurement.elevation_rad,
            }
        isac_position = self.localizer.estimate(
            anchors, measurement.range_m, measurement.los_unit,
            measurement.valid_mask, initial_position_m=initial,
            **quality_arguments,
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
                "estimated_snr": np.asarray(measurement.snr_linear, dtype=np.float32),
                "quality_source": (
                    "reported_power_crb" if reported_covariance is not None
                    else "legacy_reported_variances"
                ),
                "residual_scale_range_m": sigma_range,
                "residual_scale_radial_velocity_mps": sigma_radial,
                "residual_scale_direction": sigma_direction,
                "direction_difference_covariance": direction_difference_covariance,
                "self_features": self_features,
                "neighbor_features": neighbor_features,
                "neighbor_feature_mode": self.config.neighbor_feature_mode,
            }
        )
        return features, isac_position, diagnostics

    def _residual_scales(
        self, measurement: ISACMeasurement, command_velocity_mps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Nominal GPS--ISAC scales from receiver-visible geometry and quality."""
        ranges = np.asarray(measurement.range_m, dtype=np.float64)
        los = unit(np.asarray(measurement.los_unit, dtype=np.float64))
        if np.any(ranges <= 0.0) or not np.all(np.isfinite(ranges)):
            raise ValueError("positive measured ranges are required for local uncertainty propagation")
        reported = getattr(measurement, "reported_covariance", None)
        if reported is None:
            # Older observations may already carry receiver variance reports.
            # Do not infer their quality again from measured range.
            range_variance = np.asarray(measurement.range_variance, dtype=np.float64)
            radial_variance = np.asarray(measurement.radial_velocity_variance, dtype=np.float64)
            projector = np.eye(3) - los[..., :, None] * los[..., None, :]
            los_covariance = (
                np.asarray(measurement.direction_variance)[..., None, None] * projector
            )
        else:
            covariance = np.asarray(reported, dtype=np.float64)
            if covariance.shape != ranges.shape + (4, 4):
                raise ValueError("reported covariance must have shape (N, M, 4, 4)")
            range_variance = covariance[..., 0, 0]
            radial_variance = covariance[..., 3, 3]
            los_covariance = getattr(measurement, "reported_los_covariance", None)
            if los_covariance is None:
                outward = -los
                theta = np.arctan2(outward[..., 1], outward[..., 0])
                phi = np.arctan2(outward[..., 2], np.linalg.norm(outward[..., :2], axis=-1))
                ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
                angle_jacobian = np.stack(
                    [
                        np.stack([cp * st, -cp * ct, np.zeros_like(cp)], axis=-1),
                        np.stack([sp * ct, sp * st, -cp], axis=-1),
                    ], axis=-1,
                )
                los_covariance = (
                    angle_jacobian @ covariance[..., 1:3, 1:3]
                    @ np.swapaxes(angle_jacobian, -1, -2)
                )
            else:
                los_covariance = np.asarray(los_covariance, dtype=np.float64)
        projector = np.eye(3) - los[..., :, None] * los[..., None, :]
        gps_direction_jacobian = -projector / ranges[..., None, None]
        gps_direction_covariance = (
            gps_direction_jacobian @ self._gps_covariance
            @ np.swapaxes(gps_direction_jacobian, -1, -2)
        )
        combined_direction_covariance = los_covariance + gps_direction_covariance
        range_variance = range_variance + np.einsum(
            "...i,ij,...j->...", los, self._gps_covariance, los,
        )
        radial_variance = radial_variance + np.einsum(
            "ni,nmij,nj->nm", command_velocity_mps,
            gps_direction_covariance, command_velocity_mps,
        )
        direction_variance = 0.5 * np.einsum(
            "...ij,...ji->...", combined_direction_covariance,
            combined_direction_covariance,
        )
        scales = (range_variance, radial_variance, direction_variance)
        if any(np.any(~np.isfinite(v)) or np.any(v <= 0.0) for v in scales):
            raise ValueError("reported measurement uncertainty must yield finite positive residual scales")
        return (
            np.sqrt(range_variance), np.sqrt(radial_variance),
            np.sqrt(direction_variance), combined_direction_covariance,
        )


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
    tx_power_dbm: float = 0.0,
    radar_cross_section: float = 0.05,
    num_neighbors: int = 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulation convenience wrapper using the current monostatic CRB model.

    True positions belong to the simulator only. Processing callers should use
    compute_inter_uav_consistency_from_measurement with their measured ranges.
    Remaining peer parameters use ISACConfig reference defaults, including its
    observation time, noise figure, loss and effective receive-array gain.
    """
    true_pos = np.asarray(true_position_m, dtype=np.float64)
    gps_pos = np.asarray(gps_observed_m, dtype=np.float64)
    if true_pos.shape != gps_pos.shape or true_pos.ndim < 2 or true_pos.shape[-1] != 3:
        raise ValueError("positions must have shape (..., num_uavs, 3)")
    if num_neighbors < 0:
        raise ValueError("num_neighbors must be non-negative")
    num_uavs = true_pos.shape[-2]
    k = min(int(num_neighbors), max(num_uavs - 1, 0))
    if k == 0:
        return np.zeros(true_pos.shape[:-1] + (2,), dtype=np.float32)
    model = ISACMeasurementModel(ISACConfig(
        inter_uav_carrier_hz=carrier_hz,
        inter_uav_bandwidth_hz=bandwidth_hz,
        inter_uav_tx_power_dbm=tx_power_dbm,
        radar_cross_section=radar_cross_section,
    ))
    generator = np.random.default_rng(0) if rng is None else rng
    flat_true = true_pos.reshape(-1, num_uavs, 3)
    flat_gps = gps_pos.reshape(-1, num_uavs, 3)
    rows = []
    for position, gps in zip(flat_true, flat_gps):
        measured = model.measure_inter_uav(position, generator)
        rows.append(compute_inter_uav_consistency_from_measurement(
            gps, measured.range_m, measured.valid_mask, num_neighbors=k,
        ))
    return np.stack(rows).reshape(true_pos.shape[:-1] + (2,))

"""Statistical path-loss features used by the Deep Ensemble baseline."""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "delta_mean",
    "delta_variance",
    "delta_skewness",
    "delta_kurtosis",
    "delta_q0",
    "delta_q1",
    "delta_q2",
    "delta_q3",
    "delta_q4",
    "wasserstein_2",
)


def _distribution_statistics(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] < 4:
        raise ValueError("samples must contain at least four packet values")
    mean = np.mean(values, axis=-1)
    centered = values - mean[..., None]
    variance = np.mean(centered**2, axis=-1)
    safe_std = np.sqrt(np.maximum(variance, 1.0e-12))
    standardized = centered / safe_std[..., None]
    skewness = np.mean(standardized**3, axis=-1)
    kurtosis = np.mean(standardized**4, axis=-1)
    q1, q2, q3 = np.quantile(values, (0.25, 0.5, 0.75), axis=-1)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    inlier_low = np.where(values >= lower[..., None], values, np.inf)
    inlier_high = np.where(values <= upper[..., None], values, -np.inf)
    q0 = np.min(inlier_low, axis=-1)
    q4 = np.max(inlier_high, axis=-1)
    q0 = np.where(np.isfinite(q0), np.min(values, axis=-1), q0)
    q4 = np.where(np.isfinite(q4), np.max(values, axis=-1), q4)
    return np.stack(
        (mean, variance, skewness, kurtosis, q0, q1, q2, q3, q4),
        axis=-1,
    )


def deep_ensemble_features(
    actual_pathloss_db: np.ndarray,
    gps_pathloss_db: np.ndarray,
) -> np.ndarray:
    """Compute the ten paper-defined features for each UAV-BS observation."""

    actual = np.asarray(actual_pathloss_db, dtype=np.float64)
    gps_implied = np.asarray(gps_pathloss_db, dtype=np.float64)
    if actual.shape != gps_implied.shape:
        raise ValueError("actual and GPS-implied path-loss sets must match")
    actual_statistics = _distribution_statistics(actual)
    gps_statistics = _distribution_statistics(gps_implied)
    statistic_differences = np.abs(actual_statistics - gps_statistics)
    sorted_actual = np.sort(actual, axis=-1)
    sorted_gps = np.sort(gps_implied, axis=-1)
    wasserstein_2 = np.sqrt(
        np.mean((sorted_actual - sorted_gps) ** 2, axis=-1)
    )
    result = np.concatenate(
        (statistic_differences, wasserstein_2[..., None]),
        axis=-1,
    )
    return result.astype(np.float32)

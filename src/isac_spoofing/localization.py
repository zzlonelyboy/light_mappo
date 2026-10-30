from __future__ import annotations

import numpy as np


class RangeDirectionLocalizer:
    """Fixed-covariance distance/angle WLS with a legacy no-quality path."""

    def __init__(
        self,
        iterations: int = 8,
        damping: float = 1.0e-3,
        tolerance_m: float = 1.0e-4,
        step_clip_m: float = 10.0,
        line_search_steps: int = 8,
        relative_improvement_tolerance: float = 1.0e-3,
        min_anchors: int = 3,
    ):
        self.iterations = int(iterations)
        self.damping = float(damping)
        self.tolerance_m = float(tolerance_m)
        self.step_clip_m = float(step_clip_m)
        self.line_search_steps = int(line_search_steps)
        self.relative_improvement_tolerance = float(
            relative_improvement_tolerance
        )
        self.min_anchors = int(min_anchors)

        if self.step_clip_m <= 0.0:
            raise ValueError("step_clip_m must be positive")
        if self.line_search_steps < 1:
            raise ValueError("line_search_steps must be positive")
        if self.relative_improvement_tolerance < 0.0:
            raise ValueError("relative_improvement_tolerance must be non-negative")

    @staticmethod
    def _objective(
        position_m: np.ndarray,
        anchors_m: np.ndarray,
        range_m: np.ndarray,
        los_unit: np.ndarray,
    ) -> float:
        displacement = anchors_m - position_m
        distance = np.linalg.norm(displacement, axis=1)
        range_residual = distance - range_m
        longitudinal = np.sum(displacement * los_unit, axis=1, keepdims=True)
        los_residual = displacement - longitudinal * los_unit
        return float(
            np.sum(range_residual * range_residual)
            + np.sum(los_residual * los_residual)
        )

    def initialize(
        self,
        anchors_m: np.ndarray,
        range_m: np.ndarray,
        los_unit: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Return the legacy range-only linearized multilateration estimate.

        ``los_unit`` is retained in the interface because the subsequent
        nonlinear estimator uses both range and LoS measurements.  The legacy
        initializer itself deliberately uses only ranges.
        """
        anchors = np.asarray(anchors_m, dtype=np.float64)
        ranges = np.asarray(range_m, dtype=np.float64)
        directions = np.asarray(los_unit)
        mask = np.asarray(valid_mask, dtype=bool)
        if ranges.ndim != 2:
            raise ValueError("range_m must have shape (N, M)")
        num_uavs, num_anchors = ranges.shape
        if anchors.shape != (num_anchors, 3):
            raise ValueError(f"anchors_m must have shape ({num_anchors}, 3)")
        if directions.shape != (num_uavs, num_anchors, 3):
            raise ValueError(
                f"los_unit must have shape ({num_uavs}, {num_anchors}, 3)"
            )
        if mask.shape != (num_uavs, num_anchors):
            raise ValueError(
                f"valid_mask must have shape ({num_uavs}, {num_anchors})"
            )

        result = np.zeros((num_uavs, 3), dtype=np.float64)
        anchor_center = np.mean(anchors, axis=0)
        finite_anchors = np.all(np.isfinite(anchors), axis=1)
        for i in range(num_uavs):
            valid = mask[i] & np.isfinite(ranges[i]) & (ranges[i] > 0.0)
            valid &= finite_anchors
            indices = np.flatnonzero(valid)
            if indices.size < self.min_anchors:
                result[i] = anchor_center
                continue

            reference = indices[np.argmin(ranges[i, indices])]
            others = indices[indices != reference]
            if others.size < 3:
                result[i] = np.mean(anchors[indices], axis=0)
                continue

            reference_anchor = anchors[reference]
            reference_range = float(ranges[i, reference])
            h_rows = []
            b_values = []
            for anchor_index in others:
                anchor = anchors[anchor_index]
                measured_range = float(ranges[i, anchor_index])
                h_rows.append(2.0 * (anchor - reference_anchor))
                b_values.append(
                    reference_range**2
                    - measured_range**2
                    + float(
                        np.dot(anchor, anchor)
                        - np.dot(reference_anchor, reference_anchor)
                    )
                )

            h_matrix = np.asarray(h_rows, dtype=np.float64)
            b_vector = np.asarray(b_values, dtype=np.float64)
            regularization = 1.0e-3 * np.eye(3, dtype=np.float64)
            try:
                result[i] = np.linalg.solve(
                    h_matrix.T @ h_matrix + regularization,
                    h_matrix.T @ b_vector,
                )
            except np.linalg.LinAlgError:
                result[i] = np.mean(anchors[indices], axis=0)
        return result

    def estimate(
        self,
        anchors_m: np.ndarray,
        range_m: np.ndarray,
        los_unit: np.ndarray,
        valid_mask: np.ndarray,
        initial_position_m: np.ndarray | None = None,
        *,
        measurement_covariance: np.ndarray | None = None,
        azimuth_rad: np.ndarray | None = None,
        elevation_rad: np.ndarray | None = None,
    ) -> np.ndarray:
        if measurement_covariance is not None:
            return self._estimate_angle_wls(
                anchors_m, range_m, los_unit, valid_mask,
                measurement_covariance, azimuth_rad, elevation_rad,
                initial_position_m,
            )
        anchors = np.asarray(anchors_m, dtype=np.float64)
        ranges = np.asarray(range_m, dtype=np.float64)
        directions = np.asarray(los_unit, dtype=np.float64)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        position = (
            self.initialize(anchors, ranges, directions, valid_mask)
            if initial_position_m is None
            else np.asarray(initial_position_m, dtype=np.float64).copy()
        )
        identity = np.eye(3, dtype=np.float64)

        for i in range(ranges.shape[0]):
            valid = (
                valid_mask[i]
                & np.isfinite(ranges[i])
                & (ranges[i] > 0.0)
                & np.all(np.isfinite(directions[i]), axis=1)
                & (np.linalg.norm(directions[i], axis=1) > 0.0)
            )
            if int(np.sum(valid)) < self.min_anchors:
                continue
            selected_anchors = anchors[valid]
            selected_ranges = ranges[i, valid]
            selected_directions = directions[i, valid]
            selected_directions /= np.linalg.norm(
                selected_directions, axis=1, keepdims=True
            )

            for _ in range(self.iterations):
                cost_before = self._objective(
                    position[i],
                    selected_anchors,
                    selected_ranges,
                    selected_directions,
                )
                hessian = self.damping * identity.copy()
                gradient = np.zeros(3, dtype=np.float64)
                for anchor, measured_range, measured_direction in zip(
                    selected_anchors, selected_ranges, selected_directions
                ):
                    displacement = anchor - position[i]
                    distance = max(float(np.linalg.norm(displacement)), 1.0e-9)
                    range_residual = distance - measured_range
                    range_jacobian = -displacement / distance
                    hessian += np.outer(range_jacobian, range_jacobian)
                    gradient += range_jacobian * range_residual

                    projector = identity - np.outer(measured_direction, measured_direction)
                    los_residual = projector @ displacement
                    los_jacobian = -projector
                    hessian += los_jacobian.T @ los_jacobian
                    gradient += los_jacobian.T @ los_residual
                try:
                    update = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    break

                update_norm = float(np.linalg.norm(update))
                if update_norm > self.step_clip_m:
                    update *= self.step_clip_m / max(update_norm, 1.0e-9)

                accepted = False
                step_scale = 1.0
                candidate = position[i]
                cost_after = cost_before
                for _ in range(self.line_search_steps):
                    trial = position[i] + step_scale * update
                    trial_cost = self._objective(
                        trial,
                        selected_anchors,
                        selected_ranges,
                        selected_directions,
                    )
                    if trial_cost <= cost_before:
                        candidate = trial
                        cost_after = trial_cost
                        accepted = True
                        break
                    step_scale *= 0.5

                if not accepted:
                    break

                actual_update_norm = float(np.linalg.norm(candidate - position[i]))
                position[i] = candidate
                if actual_update_norm < self.tolerance_m:
                    break

                relative_improvement = (cost_before - cost_after) / (
                    abs(cost_before) + 1.0e-9
                )
                if relative_improvement < self.relative_improvement_tolerance:
                    break
        return position.astype(np.float32)

    @staticmethod
    def _range_angle_geometry(
        position_m: np.ndarray, anchors_m: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return [distance, world azimuth, world elevation] and its Jacobian."""
        delta = np.asarray(position_m, dtype=np.float64) - anchors_m
        x, y, z = np.moveaxis(delta, -1, 0)
        horizontal_squared = x * x + y * y
        horizontal = np.sqrt(horizontal_squared)
        squared_range = horizontal_squared + z * z
        distance = np.sqrt(squared_range)
        if np.any(distance <= 0.0) or np.any(horizontal <= 0.0):
            raise ValueError("distance/angle geometry is singular at an anchor or zenith")
        predicted = np.stack(
            [distance, np.arctan2(y, x), np.arctan2(z, horizontal)], axis=-1,
        )
        jacobian = np.stack(
            [
                delta / distance[:, None],
                np.stack([-y / horizontal_squared, x / horizontal_squared,
                          np.zeros_like(x)], axis=-1),
                np.stack([-x * z / (squared_range * horizontal),
                          -y * z / (squared_range * horizontal),
                          horizontal / squared_range], axis=-1),
            ], axis=1,
        )
        return predicted, jacobian

    @staticmethod
    def _whitened_residual(
        predicted: np.ndarray, observed: np.ndarray, inverse_cholesky: np.ndarray,
    ) -> np.ndarray:
        difference = predicted - observed
        difference[:, 1] = np.arctan2(
            np.sin(difference[:, 1]), np.cos(difference[:, 1]),
        )
        return np.einsum("mij,mj->mi", inverse_cholesky, difference)

    def _estimate_angle_wls(
        self,
        anchors_m: np.ndarray,
        range_m: np.ndarray,
        los_unit: np.ndarray,
        valid_mask: np.ndarray,
        measurement_covariance: np.ndarray,
        azimuth_rad: np.ndarray | None,
        elevation_rad: np.ndarray | None,
        initial_position_m: np.ndarray | None,
    ) -> np.ndarray:
        """Minimize one fixed reported covariance objective per observation batch.

        No GPS, physical RCS, received-power truth or covariance recomputation
        is needed. Each link's Cartesian reconstruction supplies initialization,
        including when all anchors lie in the same horizontal plane.
        """
        anchors = np.asarray(anchors_m, dtype=np.float64)
        ranges = np.asarray(range_m, dtype=np.float64)
        directions = np.asarray(los_unit, dtype=np.float64)
        mask = np.asarray(valid_mask, dtype=bool)
        if ranges.ndim != 2:
            raise ValueError("range_m must have shape (N, M)")
        n, m = ranges.shape
        if anchors.shape != (m, 3) or directions.shape != (n, m, 3):
            raise ValueError("anchor/range/direction shapes do not agree")
        if mask.shape != ranges.shape:
            raise ValueError("valid_mask must have shape (N, M)")
        covariance = np.asarray(measurement_covariance, dtype=np.float64)
        if covariance.shape == (n, m, 4, 4):
            covariance = covariance[..., :3, :3]
        if covariance.shape != (n, m, 3, 3):
            raise ValueError("reported covariance must have shape (N, M, 3, 3) or (N, M, 4, 4)")
        outward = -directions
        azimuth = (
            np.arctan2(outward[..., 1], outward[..., 0])
            if azimuth_rad is None else np.asarray(azimuth_rad, dtype=np.float64)
        )
        elevation = (
            np.arctan2(outward[..., 2], np.linalg.norm(outward[..., :2], axis=-1))
            if elevation_rad is None else np.asarray(elevation_rad, dtype=np.float64)
        )
        if azimuth.shape != ranges.shape or elevation.shape != ranges.shape:
            raise ValueError("angles must have shape (N, M)")
        observed = np.stack([ranges, azimuth, elevation], axis=-1)
        if initial_position_m is None:
            result = np.empty((n, 3), dtype=np.float64)
        else:
            result = np.asarray(initial_position_m, dtype=np.float64).copy()
            if result.shape != (n, 3) or not np.all(np.isfinite(result)):
                raise ValueError("initial_position_m must be finite with shape (N, 3)")
        identity = np.eye(3, dtype=np.float64)
        for i in range(n):
            active = mask[i] & np.all(np.isfinite(observed[i]), axis=-1)
            active &= ranges[i] > 0.0
            active &= np.all(np.isfinite(covariance[i]), axis=(-1, -2))
            selected = np.flatnonzero(active)
            if selected.size < self.min_anchors:
                raise ValueError("too few valid observations for the configured WLS localizer")
            station = anchors[selected]
            measurement = observed[i, selected]
            fixed_covariance = covariance[i, selected].copy()
            if not np.allclose(
                fixed_covariance, np.swapaxes(fixed_covariance, -1, -2),
                rtol=1.0e-8, atol=1.0e-14,
            ):
                raise ValueError("reported covariance must be symmetric")
            try:
                cholesky = np.linalg.cholesky(fixed_covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("reported covariance must be positive definite") from exc
            inverse_cholesky = np.linalg.solve(cholesky, np.broadcast_to(identity, cholesky.shape))
            if initial_position_m is None:
                d, theta, phi = np.moveaxis(measurement, -1, 0)
                ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
                ell = np.stack([cp * ct, cp * st, sp], axis=-1)
                cartesian_jacobian = np.stack(
                    [
                        ell,
                        d[:, None] * np.stack([-cp * st, cp * ct, np.zeros_like(d)], axis=-1),
                        d[:, None] * np.stack([-sp * ct, -sp * st, cp], axis=-1),
                    ], axis=-1,
                )
                position_covariance = (
                    cartesian_jacobian @ fixed_covariance
                    @ np.swapaxes(cartesian_jacobian, -1, -2)
                )
                precision_weight = 1.0 / np.trace(position_covariance, axis1=-2, axis2=-1)
                result[i] = np.average(station + d[:, None] * ell, axis=0, weights=precision_weight)
            for _ in range(self.iterations):
                predicted, jacobian = self._range_angle_geometry(result[i], station)
                residual = self._whitened_residual(predicted, measurement, inverse_cholesky)
                whitened_jacobian = inverse_cholesky @ jacobian
                cost_before = float(np.sum(residual * residual))
                hessian = (
                    np.einsum("mji,mjk->ik", whitened_jacobian, whitened_jacobian)
                    + self.damping * identity
                )
                gradient = np.einsum("mji,mj->i", whitened_jacobian, residual)
                try:
                    update = -np.linalg.solve(hessian, gradient)
                except np.linalg.LinAlgError:
                    break
                norm = float(np.linalg.norm(update))
                if norm > self.step_clip_m:
                    update *= self.step_clip_m / norm
                accepted = False
                scale = 1.0
                for _ in range(self.line_search_steps):
                    trial = result[i] + scale * update
                    try:
                        trial_prediction, _ = self._range_angle_geometry(trial, station)
                    except ValueError:
                        scale *= 0.5
                        continue
                    trial_residual = self._whitened_residual(
                        trial_prediction, measurement, inverse_cholesky,
                    )
                    cost_after = float(np.sum(trial_residual * trial_residual))
                    if cost_after <= cost_before:
                        result[i] = trial
                        accepted = True
                        break
                    scale *= 0.5
                if not accepted or scale * norm < self.tolerance_m:
                    break
                improvement = (cost_before - cost_after) / max(cost_before, np.finfo(float).tiny)
                if improvement < self.relative_improvement_tolerance:
                    break
        return result.astype(np.float32)

from __future__ import annotations

import numpy as np


class RangeDirectionLocalizer:
    """Unweighted damped Gauss-Newton ISAC position estimator."""

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
    ) -> np.ndarray:
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

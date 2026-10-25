from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .types import TrajectoryData


def _unit(vectors: np.ndarray, eps: float = 1.0e-9) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(
        vectors,
        norm,
        out=np.zeros_like(vectors, dtype=np.float64),
        where=norm > eps,
    )


def nominal_formation_offsets(
    num_uavs: int,
    formation_type: str = "cubic",
    cube_edge_length_m: float = 8.0,
) -> np.ndarray:
    """Return centered, attack-independent nominal formation offsets."""

    n = int(num_uavs)
    if n < 1:
        raise ValueError("num_uavs must be positive")
    if formation_type == "cubic":
        if n != 8:
            raise ValueError("cubic formation requires exactly 8 UAVs")
        edge_length = float(cube_edge_length_m)
        if not np.isfinite(edge_length) or edge_length <= 0.0:
            raise ValueError("cube_edge_length_m must be finite and positive")
        half_edge = 0.5 * edge_length
        signs = np.asarray(
            [
                (-1.0, -1.0, -1.0),
                (-1.0, -1.0, 1.0),
                (-1.0, 1.0, -1.0),
                (-1.0, 1.0, 1.0),
                (1.0, -1.0, -1.0),
                (1.0, -1.0, 1.0),
                (1.0, 1.0, -1.0),
                (1.0, 1.0, 1.0),
            ],
            dtype=np.float64,
        )
        offsets = half_edge * signs
    elif formation_type == "compact_ring":
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        radii = 3.0 + 0.7 * np.arange(n)
        offsets = np.column_stack(
            [
                radii * np.cos(angles),
                radii * np.sin(angles),
                1.5 * ((np.arange(n) % 3) - 1),
            ]
        )
    else:
        raise ValueError(f"unsupported formation type: {formation_type!r}")
    return offsets - np.mean(offsets, axis=0, keepdims=True)


def fixed_formation_neighbor_indices(
    num_uavs: int,
    num_neighbors: int,
    formation_type: str = "cubic",
    cube_edge_length_m: float = 8.0,
) -> np.ndarray:
    """Build one fixed K-nearest-neighbor table from the planned formation."""

    n = int(num_uavs)
    k = min(int(num_neighbors), max(n - 1, 0))
    if k < 0:
        raise ValueError("num_neighbors must be non-negative")
    if k == 0:
        return np.empty((n, 0), dtype=np.int64)
    offsets = nominal_formation_offsets(
        n,
        formation_type=formation_type,
        cube_edge_length_m=cube_edge_length_m,
    )
    distance = np.linalg.norm(
        offsets[:, None, :] - offsets[None, :, :],
        axis=-1,
    )
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1)[:, :k].astype(np.int64)


class PredefinedTrajectoryGenerator:
    """Generate complete UAV trajectories before an episode starts.

    There is deliberately no action input or navigation policy here.  The
    trajectory is exogenous data used to drive the sensor simulators.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

    def generate(
        self,
        rng: np.random.Generator,
        trajectory_type: str | None = None,
    ) -> TrajectoryData:
        kind = trajectory_type or str(rng.choice(self.config.trajectory_types))
        if kind not in self.config.trajectory_types:
            raise ValueError(f"trajectory type {kind!r} is not enabled")

        time_s = np.arange(self.config.num_steps, dtype=np.float64) * self.config.dt_s
        builders = {
            "straight": self._straight,
            "circle": self._circle,
            "helix": self._helix,
            "s_curve": self._s_curve,
        }
        # Sample the formation before any path-specific parameters.  Therefore,
        # a fixed seed gives every trajectory type the same compact formation;
        # only the centroid path changes between builders.
        formation_offsets = self._formation_offsets(rng)
        position, velocity = builders[kind](time_s, rng, formation_offsets)
        heading = _unit(velocity)
        return TrajectoryData(
            time_s=time_s.astype(np.float32),
            position_m=position.astype(np.float32),
            velocity_mps=velocity.astype(np.float32),
            heading=heading.astype(np.float32),
            trajectory_type=kind,
        )

    def _formation_offsets(self, rng: np.random.Generator) -> np.ndarray:
        n = self.config.num_uavs
        offsets = nominal_formation_offsets(
            n,
            formation_type=self.config.formation_type,
            cube_edge_length_m=self.config.cube_edge_length_m,
        )
        offsets += rng.normal(
            0.0,
            self.config.formation_jitter_std_m,
            size=(n, 3),
        )
        # Keep the reference path equal to the exact formation centroid.
        return offsets - np.mean(offsets, axis=0, keepdims=True)

    def _apply_formation(
        self,
        base_position: np.ndarray,
        base_velocity: np.ndarray,
        formation_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place one rigid compact formation on a centroid trajectory."""

        expected_path_shape = (base_position.shape[0], 3)
        if base_position.shape != expected_path_shape:
            raise ValueError("base_position must have shape (time, 3)")
        if base_velocity.shape != expected_path_shape:
            raise ValueError("base_velocity must have shape (time, 3)")
        expected_offset_shape = (self.config.num_uavs, 3)
        if formation_offsets.shape != expected_offset_shape:
            raise ValueError(
                f"formation_offsets must have shape {expected_offset_shape}"
            )

        position = base_position[:, None, :] + formation_offsets[None, :, :]
        velocity = np.broadcast_to(
            base_velocity[:, None, :],
            position.shape,
        ).copy()
        return position, velocity

    def _straight(
        self,
        time_s: np.ndarray,
        rng: np.random.Generator,
        formation_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Repeated straight waypoint legs keep a long episode inside ISAC
        # coverage without introducing a navigation controller.
        x_min, x_max = 14.0, 90.0
        leg_length = x_max - x_min
        path_distance = self.config.speed_mps * time_s + rng.uniform(0.0, leg_length)
        folded = np.mod(path_distance, 2.0 * leg_length)
        going_forward = folded <= leg_length
        base_x = x_min + np.where(going_forward, folded, 2.0 * leg_length - folded)
        base_velocity_x = np.where(
            going_forward, self.config.speed_mps, -self.config.speed_mps
        )
        start = np.array([0.0, 48.0, 22.0]) + rng.normal(0.0, [0.0, 2.0, 1.0])
        base_position = np.column_stack(
            [
                base_x,
                np.full_like(time_s, start[1]),
                np.full_like(time_s, start[2]),
            ]
        )
        base_velocity = np.zeros_like(base_position)
        base_velocity[:, 0] = base_velocity_x
        return self._apply_formation(
            base_position,
            base_velocity,
            formation_offsets,
        )

    def _circle(
        self,
        time_s: np.ndarray,
        rng: np.random.Generator,
        formation_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = rng.uniform(20.0, 30.0)
        omega = self.config.speed_mps / radius
        center = np.array([52.0, 55.0, 23.0]) + rng.normal(0.0, [2.0, 2.0, 1.0])
        initial_phase = rng.uniform(0.0, 2.0 * np.pi)
        theta = omega * time_s + initial_phase
        base_position = np.column_stack(
            [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                np.full_like(time_s, center[2]),
            ]
        )
        base_velocity = np.column_stack(
            [
                -radius * omega * np.sin(theta),
                radius * omega * np.cos(theta),
                np.zeros_like(time_s),
            ]
        )
        return self._apply_formation(
            base_position,
            base_velocity,
            formation_offsets,
        )

    def _helix(
        self,
        time_s: np.ndarray,
        rng: np.random.Generator,
        formation_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        radius = rng.uniform(18.0, 25.0)
        vertical_speed = rng.uniform(0.12, 0.3)
        horizontal_speed_squared = self.config.speed_mps**2 - vertical_speed**2
        if horizontal_speed_squared <= 0.0:
            raise ValueError("speed_mps must exceed the helix vertical speed")
        horizontal_speed = np.sqrt(horizontal_speed_squared)
        omega = horizontal_speed / radius
        center = np.array([52.0, 55.0, 16.0]) + rng.normal(0.0, [2.0, 2.0, 0.5])
        initial_phase = rng.uniform(0.0, 2.0 * np.pi)
        theta = omega * time_s + initial_phase
        base_position = np.column_stack(
            [
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                center[2] + vertical_speed * time_s,
            ]
        )
        base_velocity = np.column_stack(
            [
                -radius * omega * np.sin(theta),
                radius * omega * np.cos(theta),
                np.full_like(time_s, vertical_speed),
            ]
        )
        return self._apply_formation(
            base_position,
            base_velocity,
            formation_offsets,
        )

    def _s_curve(
        self,
        time_s: np.ndarray,
        rng: np.random.Generator,
        formation_offsets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Open S path: linear start-to-target progress plus one lateral sine
        # cycle.  Arc-length parameterisation preserves constant speed.
        start = np.asarray(self.config.s_curve_start_m, dtype=np.float64)
        target = np.asarray(self.config.s_curve_target_m, dtype=np.float64)
        displacement = target - start
        horizontal = displacement[:2]
        if float(np.linalg.norm(horizontal)) > 1.0e-9:
            lateral = np.array([-horizontal[1], horizontal[0], 0.0])
        else:
            lateral = np.array([1.0, 0.0, 0.0])
        lateral /= np.linalg.norm(lateral)
        amplitude = self.config.s_curve_lateral_amplitude_m

        dense_parameter = np.linspace(0.0, 1.0, 8193)
        dense_tangent = (
            displacement[None, :]
            + (
                2.0
                * np.pi
                * amplitude
                * np.cos(2.0 * np.pi * dense_parameter)
            )[:, None]
            * lateral[None, :]
        )
        ds = np.linalg.norm(dense_tangent, axis=1)
        increments = 0.5 * (ds[:-1] + ds[1:]) * np.diff(dense_parameter)
        cumulative_length = np.concatenate([[0.0], np.cumsum(increments)])
        total_length = cumulative_length[-1]
        path_distance = self.config.speed_mps * time_s
        folded_distance = np.mod(path_distance, 2.0 * total_length)
        moving_forward = folded_distance <= total_length
        target_distance = np.where(
            moving_forward, folded_distance, 2.0 * total_length - folded_distance
        )
        parameter = np.interp(target_distance, cumulative_length, dense_parameter)
        base_position = (
            start[None, :]
            + parameter[:, None] * displacement[None, :]
            + amplitude
            * np.sin(2.0 * np.pi * parameter)[:, None]
            * lateral[None, :]
        )
        tangent = (
            displacement[None, :]
            + (
                2.0 * np.pi * amplitude * np.cos(2.0 * np.pi * parameter)
            )[:, None]
            * lateral[None, :]
        )
        direction = np.where(moving_forward[:, None], 1.0, -1.0)
        base_velocity = self.config.speed_mps * _unit(tangent) * direction
        return self._apply_formation(
            base_position,
            base_velocity,
            formation_offsets,
        )

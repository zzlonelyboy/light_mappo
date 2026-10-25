from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    num_uavs: int = 8
    dt_s: float = 0.1
    duration_s: float = 60.0
    speed_mps: float = 4.0
    trajectory_types: tuple[str, ...] = ("s_curve",)
    formation_type: str = "cubic"
    cube_edge_length_m: float = 8.0
    formation_jitter_std_m: float = 0.5
    s_curve_start_m: tuple[float, float, float] = (0.0, 0.0, 10.0)
    s_curve_target_m: tuple[float, float, float] = (100.0, 100.0, 50.0)
    s_curve_lateral_amplitude_m: float = 44.0

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    def validate(self) -> None:
        if self.num_uavs < 1:
            raise ValueError("num_uavs must be positive")
        if self.dt_s <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("dt_s and duration_s must be positive")
        if self.num_steps < 3:
            raise ValueError("the simulation must contain at least three time steps")
        allowed = {"straight", "circle", "helix", "s_curve"}
        unknown = set(self.trajectory_types) - allowed
        if not self.trajectory_types or unknown:
            raise ValueError(f"invalid trajectory types: {sorted(unknown)}")
        allowed_formations = {"cubic", "compact_ring"}
        if self.formation_type not in allowed_formations:
            raise ValueError(
                f"invalid formation type: {self.formation_type!r}; "
                f"expected one of {sorted(allowed_formations)}"
            )
        if self.formation_type == "cubic" and self.num_uavs != 8:
            raise ValueError("cubic formation requires exactly 8 UAVs")
        if (
            not math.isfinite(self.cube_edge_length_m)
            or self.cube_edge_length_m <= 0.0
        ):
            raise ValueError("cube_edge_length_m must be finite and positive")
        if (
            not math.isfinite(self.formation_jitter_std_m)
            or self.formation_jitter_std_m < 0.0
        ):
            raise ValueError(
                "formation_jitter_std_m must be finite and non-negative"
            )
        if len(self.s_curve_start_m) != 3 or len(self.s_curve_target_m) != 3:
            raise ValueError("S-curve start and target must be three-dimensional")
        endpoints = (*self.s_curve_start_m, *self.s_curve_target_m)
        if not all(math.isfinite(float(value)) for value in endpoints):
            raise ValueError("S-curve start and target must be finite")
        displacement = tuple(
            float(target) - float(start)
            for start, target in zip(self.s_curve_start_m, self.s_curve_target_m)
        )
        if math.sqrt(sum(value * value for value in displacement)) <= 0.0:
            raise ValueError("S-curve start and target must be distinct")
        if (
            not math.isfinite(self.s_curve_lateral_amplitude_m)
            or self.s_curve_lateral_amplitude_m <= 0.0
        ):
            raise ValueError("s_curve_lateral_amplitude_m must be finite and positive")


@dataclass(frozen=True)
class GPSConfig:
    # The original environment restores nominal GPS directly to true_p.
    noise_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        if len(self.noise_std_m) != 3 or any(value < 0.0 for value in self.noise_std_m):
            raise ValueError("noise_std_m must contain three non-negative values")


@dataclass(frozen=True)
class AttackConfig:
    # With the three enabled attack types sampled uniformly, 0.25 gives the
    # four scenarios (clean/random/replay/stealth) equal marginal probability.
    # The dataset CLI additionally enforces exact balancing per data split.
    clean_episode_probability: float = 0.25
    attack_types: tuple[str, ...] = ("random", "replay", "stealth")
    duration_fraction_range: tuple[float, float] = (0.2, 0.4)
    target_count_range: tuple[int, int] = (1, 4)
    severity_levels: tuple[str, ...] = ("weak", "medium", "strong")
    random_sigma_ranges_m: tuple[tuple[float, float], ...] = (
        (0.25, 0.75),
        (0.75, 1.5),
        (1.5, 3.0),
    )
    replay_delay_ranges_s: tuple[tuple[float, float], ...] = (
        (0.2, 0.5),
        (0.5, 1.5),
        (1.5, 3.0),
    )
    stealth_rate_ranges_mps: tuple[tuple[float, float], ...] = (
        (0.1, 0.3),
        (0.3, 0.8),
        (0.8, 1.2),
    )

    @staticmethod
    def _validate_strength_ranges(
        name: str,
        ranges: tuple[tuple[float, float], ...],
        expected_count: int,
    ) -> None:
        if len(ranges) != expected_count:
            raise ValueError(
                f"{name} must contain one interval for each severity level"
            )
        for bounds in ranges:
            if len(bounds) != 2:
                raise ValueError(f"each {name} interval must contain two bounds")
            low, high = (float(bounds[0]), float(bounds[1]))
            if (
                not math.isfinite(low)
                or not math.isfinite(high)
                or low <= 0.0
                or high <= 0.0
                or high < low
            ):
                raise ValueError(
                    f"each {name} interval must be finite, positive, and ordered"
                )

    def validate(self) -> None:
        if not 0.0 <= self.clean_episode_probability <= 1.0:
            raise ValueError("clean_episode_probability must lie in [0, 1]")
        allowed = {"random", "replay", "stealth"}
        unknown = set(self.attack_types) - allowed
        if not self.attack_types or unknown:
            raise ValueError(f"invalid attack types: {sorted(unknown)}")
        if (
            len(self.duration_fraction_range) != 2
            or min(self.duration_fraction_range) <= 0.0
            or max(self.duration_fraction_range) > 1.0
        ):
            raise ValueError("duration_fraction_range must lie in (0, 1]")
        if self.target_count_range[0] < 1 or self.target_count_range[1] < 1:
            raise ValueError("target_count_range must contain positive counts")
        if not self.severity_levels or len(set(self.severity_levels)) != len(
            self.severity_levels
        ):
            raise ValueError("severity_levels must be non-empty and unique")
        self._validate_strength_ranges(
            "random_sigma_ranges_m",
            self.random_sigma_ranges_m,
            len(self.severity_levels),
        )
        self._validate_strength_ranges(
            "replay_delay_ranges_s",
            self.replay_delay_ranges_s,
            len(self.severity_levels),
        )
        self._validate_strength_ranges(
            "stealth_rate_ranges_mps",
            self.stealth_rate_ranges_mps,
            len(self.severity_levels),
        )


@dataclass(frozen=True)
class ISACConfig:
    anchors_m: tuple[tuple[float, float, float], ...] = (
        (25.0, 35.0, 5.0),
        (25.0, 70.0, 25.0),
        (60.0, 35.0, 35.0),
        (85.0, 70.0, 15.0),
    )
    carrier_hz: float = 3.0e9
    bandwidth_hz: float = 30.0e6
    tx_power_dbm: float = 27.0
    radar_cross_section: float = 6.0
    coherent_time_s: float = 0.05
    effective_aperture_m: float = 0.35

    def validate(self) -> None:
        if len(self.anchors_m) < 3 or any(len(a) != 3 for a in self.anchors_m):
            raise ValueError("anchors_m must contain at least three 3-D anchors")
        physical_values = (
            self.carrier_hz,
            self.bandwidth_hz,
            self.tx_power_dbm,
            self.radar_cross_section,
            self.coherent_time_s,
            self.effective_aperture_m,
        )
        if not all(math.isfinite(float(value)) for value in physical_values):
            raise ValueError("ISAC physical parameters must be finite")
        if (
            self.carrier_hz <= 0.0
            or self.bandwidth_hz <= 0.0
            or self.radar_cross_section <= 0.0
            or self.coherent_time_s <= 0.0
            or self.effective_aperture_m <= 0.0
        ):
            raise ValueError("ISAC physical scale parameters must be positive")


@dataclass(frozen=True)
class FeatureConfig:
    warmup_steps: int = 0
    position_scale_m: float = 1.0
    num_neighbors: int = 3
    neighbor_feature_mode: str = "inter_uav"
    neighbor_selection_mode: str = "dynamic_isac"

    def validate(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.position_scale_m <= 0.0 or self.num_neighbors < 0:
            raise ValueError("invalid position scale or neighbour count")
        if self.neighbor_feature_mode not in {
            "current",
            "formation_deformation",
            "uncertainty_robust",
            "inter_uav",
        }:
            raise ValueError(
                "neighbor_feature_mode must be 'current', "
                "'formation_deformation', 'uncertainty_robust', or 'inter_uav'"
            )
        if self.neighbor_selection_mode not in {
            "dynamic_isac",
            "fixed_formation",
        }:
            raise ValueError(
                "neighbor_selection_mode must be 'dynamic_isac' or "
                "'fixed_formation'"
            )


@dataclass(frozen=True)
class LocalizationConfig:
    iterations: int = 50
    damping: float = 1.0e-5
    tolerance_m: float = 1.0e-5
    step_clip_m: float = 10.0
    line_search_steps: int = 8
    relative_improvement_tolerance: float = 1.0e-3
    min_anchors: int = 4

    def validate(self) -> None:
        if (
            self.iterations < 1
            or self.line_search_steps < 1
            or self.min_anchors < 1
        ):
            raise ValueError(
                "localization iterations, line_search_steps, and min_anchors "
                "must be positive"
            )
        if (
            self.damping < 0.0
            or self.tolerance_m <= 0.0
            or self.step_clip_m <= 0.0
            or self.relative_improvement_tolerance < 0.0
        ):
            raise ValueError("invalid localization numerical parameters")


@dataclass(frozen=True)
class ProjectConfig:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    isac: ISACConfig = field(default_factory=ISACConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)

    def validate(self) -> None:
        self.simulation.validate()
        self.gps.validate()
        self.attack.validate()
        self.isac.validate()
        self.features.validate()
        self.localization.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_tuple(data: dict[str, Any], key: str) -> None:
    if key in data:
        data[key] = tuple(data[key])


def load_config(path: str | Path | None = None) -> ProjectConfig:
    if path is None:
        config = ProjectConfig()
        config.validate()
        return config

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    simulation = dict(raw.get("simulation", {}))
    gps = dict(raw.get("gps", {}))
    attack = dict(raw.get("attack", {}))
    isac = dict(raw.get("isac", {}))
    features = dict(raw.get("features", {}))
    # Accept earlier configuration files while ensuring these retired values
    # cannot silently re-enable hand-crafted temporal features.
    features.pop("history_steps", None)
    features.pop("ewma_beta", None)
    localization = dict(raw.get("localization", {}))

    _as_tuple(simulation, "trajectory_types")
    _as_tuple(simulation, "s_curve_start_m")
    _as_tuple(simulation, "s_curve_target_m")
    _as_tuple(gps, "noise_std_m")
    for key in (
        "attack_types",
        "severity_levels",
        "duration_fraction_range",
        "target_count_range",
    ):
        _as_tuple(attack, key)
    for key in (
        "random_sigma_ranges_m",
        "replay_delay_ranges_s",
        "stealth_rate_ranges_mps",
    ):
        if key in attack:
            attack[key] = tuple(tuple(bounds) for bounds in attack[key])
    if "anchors_m" in isac:
        isac["anchors_m"] = tuple(tuple(anchor) for anchor in isac["anchors_m"])

    config = ProjectConfig(
        simulation=SimulationConfig(**simulation),
        gps=GPSConfig(**gps),
        attack=AttackConfig(**attack),
        isac=ISACConfig(**isac),
        features=FeatureConfig(**features),
        localization=LocalizationConfig(**localization),
    )
    config.validate()
    return config

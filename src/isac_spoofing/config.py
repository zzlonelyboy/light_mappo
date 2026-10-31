from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    num_uavs: int = 8
    dt_s: float = 0.02
    duration_s: float = 60.0
    speed_mps: float = 4.0
    trajectory_types: tuple[str, ...] = ("s_curve",)
    formation_type: str = "cubic"
    cube_edge_length_m: float = 8.0
    formation_jitter_std_m: float = 0.5
    s_curve_start_m: tuple[float, float, float] = (0.0, 0.0, 20.0)
    s_curve_target_m: tuple[float, float, float] = (100.0, 100.0, 60.0)
    s_curve_lateral_amplitude_m: float = 44.0

    @property
    def num_steps(self) -> int:
        return int(round(self.duration_s / self.dt_s))

    def validate(self) -> None:
        if not math.isfinite(self.dt_s) or not math.isfinite(self.duration_s):
            raise ValueError("dt_s and duration_s must be finite")
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
class NavigationConfig:
    # Reduced-order simulation settings, not calibrated airframe parameters.
    enabled: bool = True
    # PX4 v1.14 reference position gains, not complete flight dynamics.
    position_gain_xy_s_inv: float = 0.95
    position_gain_z_s_inv: float = 1.0
    max_speed_mps: float = 8.0
    mitigation_enabled: bool = False  # Keep GPS feedback; alarms do not switch navigation by default.
    confirmation_steps: int = 5
    alarm_threshold: float = 0.5

    def confirmation_duration_s(self, dt_s: float) -> float:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("confirmation dt_s must be finite and positive")
        return self.confirmation_steps * dt_s

    def validate(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.mitigation_enabled, bool):
            raise ValueError("navigation enabled/mitigation_enabled must be booleans")
        if isinstance(self.confirmation_steps, bool) or not isinstance(self.confirmation_steps, int) or self.confirmation_steps < 1:
            raise ValueError("navigation.confirmation_steps must be a positive integer")
        if not math.isfinite(self.alarm_threshold) or not 0.0 < self.alarm_threshold < 1.0:
            raise ValueError("navigation.alarm_threshold must lie in (0, 1)")
        for name in (
            "position_gain_xy_s_inv",
            "position_gain_z_s_inv",
            "max_speed_mps",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"navigation.{name} must be finite and positive")


@dataclass(frozen=True)
class GPSConfig:
    # User-selected simulation reference, not a measured receiver specification.
    noise_std_m: tuple[float, float, float] = (1.5, 1.5, 3.0)

    def validate(self) -> None:
        if len(self.noise_std_m) != 3 or any(not math.isfinite(value) or value < 0.0 for value in self.noise_std_m):
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
    replay_source: str = "recorded_normal_episode"
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
        if self.replay_source not in {"recorded_normal_episode", "delayed_clean_stream"}:
            raise ValueError("replay_source must be recorded_normal_episode or delayed_clean_stream")
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
    # Ground-level reference deployment avoids crossing the horizontal array plane.
    anchors_m: tuple[tuple[float, float, float], ...] = (
        (25.0, 35.0, 0.0),
        (25.0, 70.0, 0.0),
        (60.0, 35.0, 0.0),
        (85.0, 70.0, 0.0),
    )
    carrier_hz: float = 3.0e9
    # An empty tuple allocates equally spaced disjoint carriers around carrier_hz.
    bs_carrier_hz: tuple[float, ...] = ()
    bandwidth_hz: float = 40.0e6  # Per-BS total budget, divided across UAVs.
    tx_power_dbm: float = 27.0  # Per-BS total active transmit power.
    radar_cross_section: float = 0.05  # Nominal RCS in m^2, rounded from the 3GPP small-UAV mean.
    coherent_time_s: float = 0.01  # T_B, not the complete navigation frame.
    tx_array_shape: tuple[int, int] = (8, 8)
    rx_array_shape: tuple[int, int] = (8, 8)
    noise_figure_db: float = 8.0
    loss_db: float = 6.0
    thermal_noise_dbm_hz: float = -174.0
    transmit_match_power: float = 1.0
    # Empty means the known upper halfspace at every anchor; never inferred from truth.
    coverage_halfspace: tuple[int, ...] = ()
    inter_uav_carrier_hz: float = 3.0e9
    inter_uav_bandwidth_hz: float = 30.0e6
    inter_uav_tx_power_dbm: float = 0.0
    inter_uav_observation_time_s: float = 0.001
    inter_uav_tx_gain: float = 1.0
    inter_uav_rx_gain: float = 16.0
    inter_uav_noise_figure_db: float = 9.0
    inter_uav_loss_db: float = 6.0
    # Accepted only as an explicitly deprecated constructor field. It is not used.
    effective_aperture_m: float | None = None

    @property
    def carrier_frequencies_hz(self) -> tuple[float, ...]:
        if self.bs_carrier_hz:
            return tuple(float(value) for value in self.bs_carrier_hz)
        count = len(self.anchors_m)
        return tuple(
            self.carrier_hz + (i - (count - 1) / 2.0) * self.bandwidth_hz
            for i in range(count)
        )

    @property
    def coverage_sides(self) -> tuple[int, ...]:
        return self.coverage_halfspace or (1,) * len(self.anchors_m)

    @property
    def total_bandwidth_hz(self) -> float:
        return self.bandwidth_hz * len(self.anchors_m)

    def validate(self) -> None:
        if len(self.anchors_m) < 3 or any(len(a) != 3 for a in self.anchors_m):
            raise ValueError("anchors_m must contain at least three 3-D anchors")
        if not all(math.isfinite(float(v)) for a in self.anchors_m for v in a):
            raise ValueError("anchor coordinates must be finite")
        positive_fields = (
            "carrier_hz", "bandwidth_hz", "radar_cross_section", "coherent_time_s",
            "inter_uav_carrier_hz", "inter_uav_bandwidth_hz",
            "inter_uav_observation_time_s", "inter_uav_tx_gain", "inter_uav_rx_gain",
        )
        for name in positive_fields:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"isac.{name} must be finite and positive")
        for name in (
            "tx_power_dbm", "noise_figure_db", "loss_db", "thermal_noise_dbm_hz",
            "inter_uav_tx_power_dbm", "inter_uav_noise_figure_db", "inter_uav_loss_db",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"isac.{name} must be finite")
        for name in ("noise_figure_db", "loss_db", "inter_uav_noise_figure_db", "inter_uav_loss_db"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"isac.{name} must be non-negative")
        if not math.isfinite(self.transmit_match_power) or not 0.0 < self.transmit_match_power <= 1.0:
            raise ValueError("transmit_match_power must lie in (0, 1]")
        for name in ("tx_array_shape", "rx_array_shape"):
            shape = getattr(self, name)
            if len(shape) != 2 or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in shape
            ):
                raise ValueError(f"{name} must contain two integer dimensions >= 2")
        carriers = self.carrier_frequencies_hz
        if len(carriers) != len(self.anchors_m) or any(not math.isfinite(v) or v <= 0.0 for v in carriers):
            raise ValueError("one positive BS carrier is required per anchor")
        ordered = sorted(carriers)
        if any(b - a < self.bandwidth_hz * (1.0 - 1.0e-12) for a, b in zip(ordered, ordered[1:])):
            raise ValueError("BS carrier bands must be disjoint")
        if len(self.coverage_sides) != len(self.anchors_m) or any(v not in (-1, 1) for v in self.coverage_sides):
            raise ValueError("one known coverage halfspace (+1 or -1) is required per anchor")
        if self.effective_aperture_m is not None and (
            not math.isfinite(self.effective_aperture_m) or self.effective_aperture_m <= 0.0
        ):
            raise ValueError("deprecated effective_aperture_m must be finite and positive")


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
    model_schema: str = "crb_received_power_v1"
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    gps: GPSConfig = field(default_factory=GPSConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    isac: ISACConfig = field(default_factory=ISACConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)

    def validate(self) -> None:
        if self.model_schema != "crb_received_power_v1":
            raise ValueError("only crb_received_power_v1 is supported by the current model")
        self.simulation.validate()
        self.gps.validate()
        self.attack.validate()
        self.isac.validate()
        self.features.validate()
        self.localization.validate()
        self.navigation.validate()
        if self.simulation.dt_s < 0.02 - 1e-12:
            raise ValueError("the two 10 ms sensing stages require dt_s >= 0.02")
        if self.isac.coherent_time_s > 0.01 + 1e-12:
            raise ValueError("BS observation time must fit the 10 ms BS stage")
        if (
            self.simulation.num_uavs * self.isac.inter_uav_observation_time_s
            > 0.01 + 1e-12
        ):
            raise ValueError("sequential UAV observations must fit the 10 ms peer stage")
        if self.navigation.enabled and (
            not math.isfinite(self.simulation.speed_mps)
            or self.simulation.speed_mps < 0.0
            or self.simulation.speed_mps > self.navigation.max_speed_mps
        ):
            raise ValueError("closed-loop reference speed must lie in [0, max_speed_mps]")

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

    schema = raw.get("model_schema", "crb_received_power_v1")
    if schema != "crb_received_power_v1":
        raise ValueError(
            "Legacy effective-echo configurations cannot be interpreted as physical RCS. "
            "Use configs/default.json or configs/multitrajectory_crb.json."
        )
    simulation = dict(raw.get("simulation", {}))
    gps = dict(raw.get("gps", {}))
    attack = dict(raw.get("attack", {}))
    # A saved configuration without this field describes the former live-delay model.
    attack.setdefault("replay_source", "delayed_clean_stream")
    isac = dict(raw.get("isac", {}))
    if "model_schema" not in raw and "effective_aperture_m" in isac:
        raise ValueError(
            "Legacy aperture/effective-echo JSON needs an explicit model migration. "
            "Do not relabel its effective coefficient as physical RCS."
        )
    features = dict(raw.get("features", {}))
    # Accept earlier configuration files while ensuring these retired values
    # cannot silently re-enable hand-crafted temporal features.
    features.pop("history_steps", None)
    features.pop("ewma_beta", None)
    localization = dict(raw.get("localization", {}))
    navigation = dict(raw.get("navigation", {}))
    # Preserve explicit scalar gains from earlier JSON configurations.
    if "position_gain_s_inv" in navigation:
        legacy_gain = navigation.pop("position_gain_s_inv")
        navigation.setdefault("position_gain_xy_s_inv", legacy_gain)
        navigation.setdefault("position_gain_z_s_inv", legacy_gain)
    # Retired plant parameters are ignored by the ideal velocity model.
    navigation.pop("velocity_time_constant_s", None)
    navigation.pop("max_acceleration_mps2", None)

    # Preserve explicitly configured legacy durations when reading older JSON.
    # New configurations define K directly, so changing dt never changes K.
    if "confirmation_duration_s" in navigation:
        duration = navigation.pop("confirmation_duration_s")
        dt_s = simulation.get("dt_s", SimulationConfig().dt_s)
        for name, value in (("confirmation_duration_s", duration), ("dt_s", dt_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"legacy confirmation {name} must be finite and positive")
        legacy_steps = max(1, int(math.ceil(duration / dt_s - 1.0e-12)))
        if "confirmation_steps" in navigation and navigation["confirmation_steps"] != legacy_steps:
            raise ValueError("navigation.confirmation_steps conflicts with legacy confirmation_duration_s")
        navigation.setdefault("confirmation_steps", legacy_steps)

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
    for key in ("bs_carrier_hz", "tx_array_shape", "rx_array_shape", "coverage_halfspace"):
        _as_tuple(isac, key)
    if "anchors_m" in isac:
        isac["anchors_m"] = tuple(tuple(anchor) for anchor in isac["anchors_m"])

    config = ProjectConfig(
        model_schema=schema,
        simulation=SimulationConfig(**simulation),
        gps=GPSConfig(**gps),
        attack=AttackConfig(**attack),
        isac=ISACConfig(**isac),
        features=FeatureConfig(**features),
        localization=LocalizationConfig(**localization),
        navigation=NavigationConfig(**navigation),
    )
    config.validate()
    return config

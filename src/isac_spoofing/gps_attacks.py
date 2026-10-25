from __future__ import annotations

import numpy as np

from .config import AttackConfig, GPSConfig
from .types import AttackScenario


def generate_clean_gps(
    true_position_m: np.ndarray,
    config: GPSConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    std = np.asarray(config.noise_std_m, dtype=np.float64)
    if std.shape != (3,) or np.any(std < 0.0):
        raise ValueError("GPS noise_std_m must contain three non-negative values")
    noise = rng.normal(0.0, std, size=true_position_m.shape)
    return (np.asarray(true_position_m, dtype=np.float64) + noise).astype(np.float32)


class GPSAttackInjector:
    """Sample an episode-level attack and apply it to clean GPS data.

    Attack severity and its continuous strength are sampled exactly once when
    the event is created.  The resulting parameter is then reused for every
    active step and every targeted UAV in that event.
    """

    def __init__(
        self,
        config: AttackConfig,
        dt_s: float,
    ):
        self.config = config
        self.dt_s = float(dt_s)

    @staticmethod
    def _sample_fraction(
        rng: np.random.Generator, bounds: tuple[float, float]
    ) -> float:
        low, high = sorted((float(bounds[0]), float(bounds[1])))
        return float(rng.uniform(low, high)) if high > low else low

    def _sample_event_strength(
        self,
        attack_type: str,
        rng: np.random.Generator,
    ) -> dict[str, object]:
        severity_index = int(rng.integers(0, len(self.config.severity_levels)))
        severity_level = self.config.severity_levels[severity_index]

        if attack_type == "random":
            bounds = self.config.random_sigma_ranges_m[severity_index]
            value = self._sample_fraction(rng, bounds)
            return {
                "severity_level": severity_level,
                "strength_value": value,
                "strength_unit": "m",
                "sigma_m": value,
            }
        if attack_type == "replay":
            bounds = self.config.replay_delay_ranges_s[severity_index]
            value = self._sample_fraction(rng, bounds)
            delay_steps = max(1, int(np.round(value / self.dt_s)))
            return {
                "severity_level": severity_level,
                "strength_value": value,
                "strength_unit": "s",
                "delay_s": value,
                "delay_steps": delay_steps,
                "effective_delay_s": delay_steps * self.dt_s,
            }
        if attack_type == "stealth":
            bounds = self.config.stealth_rate_ranges_mps[severity_index]
            value = self._sample_fraction(rng, bounds)
            return {
                "severity_level": severity_level,
                "strength_value": value,
                "strength_unit": "m/s",
                "rate_mps": value,
            }
        raise ValueError(f"unsupported attack type: {attack_type}")

    def sample(
        self,
        num_steps: int,
        num_uavs: int,
        rng: np.random.Generator,
        attack_type: str | None = None,
    ) -> AttackScenario:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if num_uavs < 1:
            raise ValueError("num_uavs must be positive")
        if attack_type is not None:
            requested_type = str(attack_type)
            allowed = {"clean", *self.config.attack_types}
            if requested_type not in allowed:
                raise ValueError(
                    f"attack_type must be one of {sorted(allowed)}, got {requested_type!r}"
                )
        elif rng.random() < self.config.clean_episode_probability:
            requested_type = "clean"
        else:
            requested_type = str(rng.choice(self.config.attack_types))

        if requested_type == "clean":
            return AttackScenario(
                attack_type="clean",
                start_step=0,
                end_step=0,
                target_mask=np.zeros(num_uavs, dtype=bool),
                parameters={},
                nominal_duration_steps=0,
                actual_duration_steps=0,
                was_truncated=False,
            )

        attack_type = requested_type
        duration_fraction = self._sample_fraction(rng, self.config.duration_fraction_range)
        # The original interval is inclusive [t0, t1].  This project stores an
        # exclusive end index, hence the additional active step.
        nominal_duration_steps = min(
            num_steps,
            max(1, int(duration_fraction * num_steps) + 1),
        )
        # Sample the onset only from the interval in which the complete event
        # fits.  This makes every generated attack fully observable and avoids
        # treating an episode-boundary censoring artefact as a detector miss.
        latest_start_step = num_steps - nominal_duration_steps
        start_step = int(rng.integers(0, latest_start_step + 1))
        end_step = start_step + nominal_duration_steps
        actual_duration_steps = nominal_duration_steps
        was_truncated = False

        low_count, high_count = sorted(self.config.target_count_range)
        low_count = max(1, int(low_count))
        high_count = min(num_uavs, int(high_count))
        if high_count < low_count:
            raise ValueError("target_count_range is incompatible with num_uavs")
        target_count = int(rng.integers(low_count, high_count + 1))
        targets = rng.choice(num_uavs, size=target_count, replace=False)
        target_mask = np.zeros(num_uavs, dtype=bool)
        target_mask[targets] = True

        parameters = self._sample_event_strength(attack_type, rng)

        return AttackScenario(
            attack_type=attack_type,
            start_step=start_step,
            end_step=end_step,
            target_mask=target_mask,
            parameters=parameters,
            nominal_duration_steps=nominal_duration_steps,
            actual_duration_steps=actual_duration_steps,
            was_truncated=was_truncated,
        )

    def apply(
        self,
        gps_clean_m: np.ndarray,
        scenario: AttackScenario,
        rng: np.random.Generator,
        heading: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Attacks act on the nominal GPS observation, not directly on simulator
        # ground truth.  This is identical to the legacy equations when nominal
        # GPS noise is zero, while preserving that noise in robustness runs.
        nominal = np.asarray(gps_clean_m, dtype=np.float32)
        observed = nominal.copy()
        num_steps, num_uavs, _ = observed.shape
        labels = np.zeros((num_steps, num_uavs), dtype=np.uint8)
        if scenario.attack_type == "clean":
            return observed, labels

        active = slice(scenario.start_step, scenario.end_step)
        targets = np.flatnonzero(scenario.target_mask)
        labels[active, targets] = 1

        if scenario.attack_type == "random":
            sigma = float(scenario.parameters["sigma_m"])
            count = scenario.end_step - scenario.start_step
            noise = rng.normal(0.0, sigma, size=(count, targets.size, 3))
            observed[active, targets, :] = nominal[active, targets, :] + noise
        elif scenario.attack_type == "replay":
            delay = int(scenario.parameters["delay_steps"])
            for time_index in range(scenario.start_step, scenario.end_step):
                source_index = max(0, time_index - delay)
                observed[time_index, targets] = nominal[source_index, targets]
        elif scenario.attack_type == "stealth":
            if heading is None:
                directions = np.zeros((num_uavs, 3), dtype=np.float32)
                directions[:, 0] = 1.0
            else:
                directions = np.asarray(heading[scenario.start_step], dtype=np.float32)
                direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
                if np.any(direction_norms <= 0.0):
                    raise ValueError("stealth attack requires non-zero headings")
                directions = directions / direction_norms
            scenario.parameters["directions"] = directions.tolist()
            rate = float(scenario.parameters["rate_mps"])
            elapsed = np.arange(
                scenario.end_step - scenario.start_step, dtype=np.float32
            ) * self.dt_s
            drift = (
                elapsed[:, None, None]
                * rate
                * directions[targets][None, :, :]
            )
            observed[active, targets, :] = nominal[active, targets, :] + drift
        else:
            raise ValueError(f"unsupported attack type: {scenario.attack_type}")

        return observed.astype(np.float32), labels

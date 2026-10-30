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

    def _replay_mode(self, scenario: AttackScenario | None = None) -> str:
        mode = (
            self.config.replay_source
            if scenario is None else
            scenario.parameters.get("replay_source", self.config.replay_source)
        )
        if mode not in ("recorded_normal_episode", "delayed_clean_stream"):
            raise ValueError(f"unsupported replay source: {mode!r}")
        return str(mode)

    def _replay_values(
        self,
        gps_clean_m: np.ndarray,
        scenario: AttackScenario,
        source_indices: int | np.ndarray,
        targets: np.ndarray,
        replay_source_gps_m: np.ndarray | None,
    ) -> np.ndarray:
        mode = self._replay_mode(scenario)
        if mode == "recorded_normal_episode":
            if replay_source_gps_m is None:
                raise ValueError("recorded_normal_episode requires replay_source_gps_m")
            source = np.asarray(replay_source_gps_m)
        else:
            source = gps_clean_m
        if source.ndim != 3 or source.shape[1:] != gps_clean_m.shape[1:]:
            raise ValueError("replay source must have shape (T_source,N,3) matching the victim UAVs")
        indices = np.asarray(source_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= source.shape[0]):
            raise ValueError("replay source does not cover the requested recording indices")
        if indices.ndim == 0:
            values = source[int(indices), targets]
        else:
            values = source[indices[:, None], targets[None, :]]
        # Inspect only the samples being used, not the entire recording on
        # every streaming call. A float32 recording is copied without new noise.
        values = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("replay source contains non-finite requested GPS samples")
        return values

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
            mode = self._replay_mode()
            parameters = {
                "severity_level": severity_level,
                "strength_value": value,
                "strength_unit": "s",
                "replay_source": mode,
                "offset_semantics": (
                    "trajectory_phase_offset" if mode == "recorded_normal_episode"
                    else "measurement_delay"
                ),
                # Preserve the historical parameter names for old consumers.
                "delay_s": value,
                "delay_steps": delay_steps,
                "effective_delay_s": delay_steps * self.dt_s,
            }
            if mode == "recorded_normal_episode":
                parameters.update({
                    "phase_offset_s": value,
                    "phase_offset_steps": delay_steps,
                    "effective_phase_offset_s": delay_steps * self.dt_s,
                })
            return parameters
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
        if attack_type == "replay":
            delay_steps = int(parameters["delay_steps"])
            if delay_steps > latest_start_step:
                raise ValueError(
                    "episode is too short for the sampled replay delay and complete "
                    "attack duration; increase duration_s or reduce replay_delay_ranges_s"
                )
            if start_step < delay_steps:
                start_step = int(rng.integers(delay_steps, latest_start_step + 1))
                end_step = start_step + nominal_duration_steps

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

    def apply_step(
        self,
        gps_clean_history_m: np.ndarray,
        scenario: AttackScenario,
        time_index: int,
        rng: np.random.Generator,
        heading: np.ndarray | None = None,
        *,
        replay_source_gps_m: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inject into the current simulated observation using available history.

        Victim history ends at the current step. Recorded replay reads a
        completed normal flight, while delayed mode reads this plant's past
        clean observations. Neither mode reads future victim observations.
        """
        history = np.asarray(gps_clean_history_m, dtype=np.float32)
        if (
            history.ndim != 3
            or history.shape[0] != time_index + 1
            or history.shape[2] != 3
        ):
            raise ValueError("GPS history must have shape (time_index + 1, N, 3)")
        observed = history[-1].copy()
        num_uavs = observed.shape[0]
        if np.asarray(scenario.target_mask).shape != (num_uavs,):
            raise ValueError("attack target mask must match the UAV count")
        labels = np.zeros(num_uavs, dtype=np.uint8)
        if scenario.attack_type == "clean" or not (
            scenario.start_step <= time_index < scenario.end_step
        ):
            return observed, labels
        targets = np.flatnonzero(scenario.target_mask)
        labels[targets] = 1
        if scenario.attack_type == "random":
            observed[targets] += rng.normal(
                0.0, float(scenario.parameters["sigma_m"]), (targets.size, 3)
            )
        elif scenario.attack_type == "replay":
            delay = int(scenario.parameters["delay_steps"])
            if delay < 1:
                raise ValueError("replay delay_steps must be positive")
            if time_index < delay:
                raise ValueError("replay requires a complete historical GPS report")
            observed[targets] = self._replay_values(
                history, scenario, time_index - delay, targets, replay_source_gps_m,
            )
        elif scenario.attack_type == "stealth":
            if "directions" not in scenario.parameters:
                if heading is None:
                    directions = np.zeros((num_uavs, 3), dtype=np.float32)
                    directions[:, 0] = 1.0
                else:
                    directions = np.asarray(heading, dtype=np.float32).copy()
                if directions.shape != (num_uavs, 3):
                    raise ValueError("stealth headings must have shape (N, 3)")
                norms = np.linalg.norm(directions, axis=1, keepdims=True)
                if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
                    raise ValueError("stealth attack requires non-zero headings")
                scenario.parameters["directions"] = (directions / norms).tolist()
            directions = np.asarray(scenario.parameters["directions"], dtype=np.float32)
            if directions.shape != (num_uavs, 3) or not np.all(np.isfinite(directions)):
                raise ValueError("stealth directions must be finite with shape (N, 3)")
            elapsed_s = (time_index - scenario.start_step) * self.dt_s
            observed[targets] += (
                elapsed_s * float(scenario.parameters["rate_mps"]) * directions[targets]
            )
        else:
            raise ValueError(f"unsupported attack type: {scenario.attack_type}")
        return observed, labels

    def apply(
        self,
        gps_clean_m: np.ndarray,
        scenario: AttackScenario,
        rng: np.random.Generator,
        heading: np.ndarray | None = None,
        *,
        replay_source_gps_m: np.ndarray | None = None,
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
            if delay < 1 or scenario.start_step < delay:
                raise ValueError("replay requires a positive delay and sufficient history")
            source_indices = np.arange(scenario.start_step, scenario.end_step) - delay
            observed[active, targets, :] = self._replay_values(
                nominal, scenario, source_indices, targets, replay_source_gps_m,
            )
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

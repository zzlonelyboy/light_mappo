from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _binary_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only hard binary values 0 or 1")
    return array.astype(bool, copy=False)


def binary_detection_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Secondary time-step metrics for hard 0/1 model outputs.

    The detector converts one attack logit to a Sigmoid probability and applies
    the fixed 0.5 threshold before this helper receives the resulting hard labels.
    """
    truth = _binary_array(labels, "labels").ravel()
    prediction = _binary_array(predictions, "predictions").ravel()
    if truth.shape != prediction.shape:
        raise ValueError("labels and predictions must have the same shape")
    true_positive = int(np.sum(truth & prediction))
    false_positive = int(np.sum(~truth & prediction))
    false_negative = int(np.sum(truth & ~prediction))
    true_negative = int(np.sum(~truth & ~prediction))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    false_alarm_rate = false_positive / max(false_positive + true_negative, 1)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
    }


def first_alarm_steps(
    predictions: np.ndarray,
    warmup_steps: int = 0,
) -> np.ndarray:
    """Return one latched first-alarm step per UAV, or ``-1`` if absent."""
    hard = _binary_array(predictions, "predictions")
    if hard.ndim != 2:
        raise ValueError("predictions must have shape (T,N)")
    if not 0 <= warmup_steps <= hard.shape[0]:
        raise ValueError("warmup_steps must lie in [0,T]")
    alarms = np.full(hard.shape[1], -1, dtype=np.int32)
    for uav in range(hard.shape[1]):
        indices = np.flatnonzero(hard[warmup_steps:, uav])
        if indices.size:
            alarms[uav] = int(indices[0] + warmup_steps)
    return alarms


def event_detection_metrics(
    predictions: np.ndarray,
    has_attack: np.ndarray,
    attack_start_step: np.ndarray,
    attack_end_step_exclusive: np.ndarray,
    dt_s: float,
    warmup_steps: int = 0,
    detection_deadlines_s: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
) -> dict[str, float | int | np.ndarray]:
    """Evaluate the per-UAV first alarm with one-to-one event matching.

    A hard alarm inside ``[attack_start_step, attack_end_step_exclusive)`` is a
    matched detection.  An alarm outside that interval is an unmatched false
    alarm, and the true attack remains missed.  Continuous positive predictions
    can therefore produce at most one alarm for each UAV and episode.
    """
    hard = _binary_array(predictions, "predictions")
    if hard.ndim != 2:
        raise ValueError("predictions must have shape (T,N)")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be positive")
    num_steps, num_uavs = hard.shape
    attacked = np.asarray(has_attack, dtype=bool)
    starts = np.asarray(attack_start_step, dtype=np.int64)
    ends = np.asarray(attack_end_step_exclusive, dtype=np.int64)
    if attacked.shape != (num_uavs,) or starts.shape != (num_uavs,) or ends.shape != (num_uavs,):
        raise ValueError("event metadata must have shape (N,)")
    if np.any(attacked & ((starts < 0) | (ends <= starts) | (ends > num_steps))):
        raise ValueError("attacked UAVs require a valid [start,end) interval")
    if np.any(~attacked & ((starts != -1) | (ends != -1))):
        raise ValueError("unattacked UAVs must use -1 event indices")

    alarms = first_alarm_steps(hard, warmup_steps=warmup_steps)
    status = np.full(num_uavs, "correct_rejection", dtype="<U24")
    delays: list[float] = []
    false_alarms = 0
    false_alarms_unattacked = 0
    premature_false_alarms = 0
    late_false_alarms = 0
    detected = 0
    negative_exposure_steps = 0

    for uav, alarm in enumerate(alarms):
        exposure_end = num_steps if alarm < 0 else int(alarm) + 1
        monitored_start = min(warmup_steps, exposure_end)
        monitored_steps = exposure_end - monitored_start

        if not attacked[uav]:
            negative_exposure_steps += monitored_steps
            if alarm >= 0:
                status[uav] = "false_alarm_unattacked"
                false_alarms += 1
                false_alarms_unattacked += 1
            continue

        attack_overlap = max(
            0,
            min(exposure_end, int(ends[uav]))
            - max(monitored_start, int(starts[uav])),
        )
        negative_exposure_steps += monitored_steps - attack_overlap
        if alarm < 0:
            status[uav] = "silent_miss"
        elif alarm < starts[uav]:
            status[uav] = "premature_false_alarm"
            false_alarms += 1
            premature_false_alarms += 1
        elif alarm < ends[uav]:
            status[uav] = "detected"
            detected += 1
            delays.append((int(alarm) - int(starts[uav])) * float(dt_s))
        else:
            status[uav] = "late_false_alarm"
            false_alarms += 1
            late_false_alarms += 1

    num_attacked = int(np.sum(attacked))
    num_unattacked = num_uavs - num_attacked
    missed = num_attacked - detected
    predicted_alarms = int(np.sum(alarms >= 0))
    precision = detected / max(detected + false_alarms, 1)
    recall = detected / max(num_attacked, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    delay_array = np.asarray(delays, dtype=np.float64)
    negative_exposure_hours = negative_exposure_steps * float(dt_s) / 3600.0

    result: dict[str, float | int | np.ndarray] = {
        "num_uavs": num_uavs,
        "num_attacked_events": num_attacked,
        "num_unattacked_uavs": num_unattacked,
        "num_predicted_alarms": predicted_alarms,
        "num_detected_events": detected,
        "num_missed_events": missed,
        "num_false_alarms": false_alarms,
        "num_false_alarms_unattacked": false_alarms_unattacked,
        "num_premature_false_alarms": premature_false_alarms,
        "num_late_false_alarms": late_false_alarms,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "event_miss_rate": missed / max(num_attacked, 1),
        "unattacked_event_false_alarm_probability": (
            false_alarms_unattacked / num_unattacked
            if num_unattacked
            else float("nan")
        ),
        "premature_false_alarm_probability": (
            premature_false_alarms / num_attacked
            if num_attacked
            else float("nan")
        ),
        "false_alarms_per_uav_hour": (
            false_alarms / negative_exposure_hours
            if negative_exposure_hours > 0.0
            else float("nan")
        ),
        "delay_mean_s": (
            float(np.mean(delay_array)) if delay_array.size else float("nan")
        ),
        "delay_median_s": (
            float(np.median(delay_array)) if delay_array.size else float("nan")
        ),
        "delay_q1_s": (
            float(np.percentile(delay_array, 25.0))
            if delay_array.size
            else float("nan")
        ),
        "delay_q3_s": (
            float(np.percentile(delay_array, 75.0))
            if delay_array.size
            else float("nan")
        ),
        "first_alarm_step": alarms,
        "event_status": status,
        "detected_delays_s": delay_array,
    }
    for deadline in detection_deadlines_s:
        deadline = float(deadline)
        if deadline < 0.0:
            raise ValueError("detection deadlines must be non-negative")
        key = str(deadline).replace(".", "p")
        result[f"detection_rate_within_{key}_s"] = (
            float(np.sum(delay_array <= deadline)) / max(num_attacked, 1)
        )
    return result


@dataclass(frozen=True)
class AlarmEvent:
    uav: int
    trigger_step: int
    end_step_exclusive: int


@dataclass(frozen=True)
class AttackEvent:
    uav: int
    start_step: int
    end_step_exclusive: int
    event_id: str = ""


def k_step_alarm_events(
    predictions: np.ndarray,
    confirmation_steps: int,
    warmup_steps: int = 0,
) -> tuple[tuple[AlarmEvent, ...], ...]:
    """Convert hard predictions into re-armable alarm events.

    The state machine confirms an alarm after ``confirmation_steps`` consecutive
    ones and recovers after the same number of consecutive zeros. Ground-truth
    labels are deliberately absent from this interface.
    """

    hard = _binary_array(predictions, "predictions")
    if hard.ndim != 2:
        raise ValueError("predictions must have shape (T,N)")
    if confirmation_steps < 1:
        raise ValueError("confirmation_steps must be positive")
    if not 0 <= warmup_steps <= hard.shape[0]:
        raise ValueError("warmup_steps must lie in [0,T]")

    num_steps, num_uavs = hard.shape
    events: list[list[AlarmEvent]] = [[] for _ in range(num_uavs)]
    for uav in range(num_uavs):
        active = False
        positive_run = 0
        negative_run = 0
        trigger_step = -1
        for step in range(warmup_steps, num_steps):
            positive = bool(hard[step, uav])
            if not active:
                if positive:
                    positive_run += 1
                    if positive_run >= confirmation_steps:
                        active = True
                        trigger_step = step
                        positive_run = 0
                        negative_run = 0
                else:
                    positive_run = 0
                continue

            if positive:
                negative_run = 0
            else:
                negative_run += 1
                if negative_run >= confirmation_steps:
                    events[uav].append(
                        AlarmEvent(
                            uav=uav,
                            trigger_step=trigger_step,
                            end_step_exclusive=step + 1,
                        )
                    )
                    active = False
                    positive_run = 0
                    negative_run = 0
                    trigger_step = -1
        if active:
            events[uav].append(
                AlarmEvent(
                    uav=uav,
                    trigger_step=trigger_step,
                    end_step_exclusive=num_steps,
                )
            )
    return tuple(tuple(uav_events) for uav_events in events)


def apply_alarm_cooldown(
    predictions: np.ndarray,
    confirmation_steps: int,
    cooldown_steps: int,
    warmup_steps: int = 0,
) -> np.ndarray:
    """Force negative predictions for ``cooldown_steps`` after each alarm ends.

    This is a deployment-time post-processing step.  After the K-step alarm
    state machine closes an alarm (K consecutive negatives), the model may
    still output sporadic positives that prevent the state machine from fully
    resetting.  Forcing a brief cooldown window guarantees that the next
    attack can be detected from a clean alarm state.

    Parameters
    ----------
    predictions : (T, N) binary array
    confirmation_steps : K used by the alarm state machine
    cooldown_steps : number of forced-negative steps after alarm end
    warmup_steps : same as in ``k_step_alarm_events``

    Returns
    -------
    modified : (T, N) predictions with cooldown zeros injected
    """
    hard = _binary_array(predictions, "predictions")
    if hard.ndim != 2:
        raise ValueError("predictions must have shape (T,N)")
    if cooldown_steps < 0:
        raise ValueError("cooldown_steps must be non-negative")
    if cooldown_steps == 0:
        return hard.copy()

    # First pass: locate alarm boundaries
    alarms = k_step_alarm_events(hard, confirmation_steps, warmup_steps)

    modified = hard.copy()
    num_steps = hard.shape[0]
    for uav_events in alarms:
        for event in uav_events:
            start = event.end_step_exclusive
            end = min(start + cooldown_steps, num_steps)
            modified[start:end, event.uav] = 0
    return modified


def attack_events_from_arrays(
    has_attack: np.ndarray,
    attack_start_step: np.ndarray,
    attack_end_step_exclusive: np.ndarray,
) -> tuple[AttackEvent, ...]:
    attacked = np.asarray(has_attack, dtype=bool)
    starts = np.asarray(attack_start_step, dtype=np.int64)
    ends = np.asarray(attack_end_step_exclusive, dtype=np.int64)
    if attacked.ndim != 1 or starts.shape != attacked.shape or ends.shape != attacked.shape:
        raise ValueError("per-UAV event arrays must have matching shape (N,)")
    events = []
    for uav in range(attacked.shape[0]):
        if attacked[uav]:
            events.append(
                AttackEvent(
                    uav=uav,
                    start_step=int(starts[uav]),
                    end_step_exclusive=int(ends[uav]),
                    event_id=f"uav_{uav}_event_0",
                )
            )
        elif starts[uav] != -1 or ends[uav] != -1:
            raise ValueError("unattacked UAVs must use -1 event indices")
    return tuple(events)


def _deadline_key(deadline: float) -> str:
    return str(float(deadline)).replace(".", "p")


def match_alarm_events(
    alarms_by_uav: Sequence[Sequence[AlarmEvent]],
    attack_events: Sequence[AttackEvent],
    num_steps: int,
    dt_s: float,
    warmup_steps: int = 0,
    detection_deadlines_s: Sequence[float] = (0.5, 1.0, 2.0, 5.0),
) -> dict[str, object]:
    """Match alarm triggers to non-overlapping ground-truth attack events.

    The first alarm whose trigger lies inside an attack interval detects that
    event.  Later re-armed alarms inside the same already-detected interval are
    retained for diagnostics but ignored by TP/FP/FN statistics.  Alarm triggers
    outside every attack interval remain false positives.
    """

    if num_steps < 1 or dt_s <= 0.0:
        raise ValueError("num_steps and dt_s must be positive")
    num_uavs = len(alarms_by_uav)
    if not 0 <= warmup_steps <= num_steps:
        raise ValueError("warmup_steps must lie in [0,num_steps]")

    attacks_by_uav: list[list[AttackEvent]] = [[] for _ in range(num_uavs)]
    for event in attack_events:
        if not 0 <= event.uav < num_uavs:
            raise ValueError("attack event references an invalid UAV")
        if not 0 <= event.start_step < event.end_step_exclusive <= num_steps:
            raise ValueError("attack events require a valid [start,end) interval")
        attacks_by_uav[event.uav].append(event)
    for uav_events in attacks_by_uav:
        uav_events.sort(key=lambda event: (event.start_step, event.end_step_exclusive))
        if any(
            current.end_step_exclusive > following.start_step
            for current, following in zip(uav_events, uav_events[1:])
        ):
            raise ValueError("attack events on one UAV must not overlap")
    event_ids: dict[tuple[int, int], str] = {}
    used_event_ids: set[str] = set()
    for uav, uav_events in enumerate(attacks_by_uav):
        for attack_index, event in enumerate(uav_events):
            event_id = event.event_id or f"uav_{uav}_event_{attack_index}"
            if event_id in used_event_ids:
                raise ValueError(f"attack event_id must be unique: {event_id!r}")
            used_event_ids.add(event_id)
            event_ids[(uav, attack_index)] = event_id

    normal_exposure_steps = 0
    for uav in range(num_uavs):
        attack_mask = np.zeros(num_steps, dtype=bool)
        for event in attacks_by_uav[uav]:
            attack_mask[event.start_step : event.end_step_exclusive] = True
        normal_exposure_steps += int(np.sum(~attack_mask[warmup_steps:]))

    matched_attack_ids: set[tuple[int, int]] = set()
    alarm_records: list[dict[str, object]] = []
    attack_records: list[dict[str, object]] = []
    detected_delays_s: list[float] = []
    false_alarms = 0
    ignored_duplicate_alarms = 0

    for uav, alarms in enumerate(alarms_by_uav):
        ordered_alarms = sorted(alarms, key=lambda event: event.trigger_step)
        for alarm in ordered_alarms:
            if alarm.uav != uav:
                raise ValueError("alarm UAV does not match its container index")
            if not warmup_steps <= alarm.trigger_step < num_steps:
                raise ValueError("alarm trigger lies outside the monitored interval")
            if not alarm.trigger_step < alarm.end_step_exclusive <= num_steps:
                raise ValueError("alarm event requires trigger < end <= num_steps")
            containing = [
                (attack_index, attack)
                for attack_index, attack in enumerate(attacks_by_uav[uav])
                if attack.start_step <= alarm.trigger_step < attack.end_step_exclusive
            ]
            match = next(
                (
                    (attack_index, attack)
                    for attack_index, attack in containing
                    if (uav, attack_index) not in matched_attack_ids
                ),
                None,
            )
            if match is not None:
                attack_index, attack = match
                matched_attack_ids.add((uav, attack_index))
                delay_s = (alarm.trigger_step - attack.start_step) * float(dt_s)
                detected_delays_s.append(delay_s)
                alarm_records.append(
                    {
                        "uav": uav,
                        "trigger_step": alarm.trigger_step,
                        "end_step_exclusive": alarm.end_step_exclusive,
                        "status": "true_positive",
                        "matched_event_id": event_ids[(uav, attack_index)],
                        "related_event_id": event_ids[(uav, attack_index)],
                        "delay_s": delay_s,
                    }
                )
            elif containing:
                ignored_duplicate_alarms += 1
                attack_index, _ = containing[0]
                alarm_records.append(
                    {
                        "uav": uav,
                        "trigger_step": alarm.trigger_step,
                        "end_step_exclusive": alarm.end_step_exclusive,
                        "status": "ignored_duplicate_alarm",
                        "matched_event_id": None,
                        "related_event_id": event_ids[(uav, attack_index)],
                        "delay_s": None,
                    }
                )
            else:
                false_alarms += 1
                alarm_records.append(
                    {
                        "uav": uav,
                        "trigger_step": alarm.trigger_step,
                        "end_step_exclusive": alarm.end_step_exclusive,
                        "status": "unmatched_false_alarm",
                        "matched_event_id": None,
                        "related_event_id": None,
                        "delay_s": None,
                    }
                )

    for uav, uav_attacks in enumerate(attacks_by_uav):
        for attack_index, attack in enumerate(uav_attacks):
            matched = (uav, attack_index) in matched_attack_ids
            event_id = event_ids[(uav, attack_index)]
            matched_alarm = next(
                (
                    record
                    for record in alarm_records
                    if record["matched_event_id"] == event_id
                ),
                None,
            )
            attack_records.append(
                {
                    "event_id": event_id,
                    "uav": uav,
                    "start_step": attack.start_step,
                    "end_step_exclusive": attack.end_step_exclusive,
                    "status": "detected" if matched else "missed",
                    "alarm_trigger_step": (
                        matched_alarm["trigger_step"] if matched_alarm else None
                    ),
                    "delay_s": matched_alarm["delay_s"] if matched_alarm else None,
                }
            )

    true_positives = len(matched_attack_ids)
    false_negatives = len(attack_events) - true_positives
    deadline_counts: dict[str, int] = {}
    delay_array = np.asarray(detected_delays_s, dtype=np.float64)
    for deadline in detection_deadlines_s:
        deadline = float(deadline)
        if deadline < 0.0:
            raise ValueError("detection deadlines must be non-negative")
        deadline_counts[_deadline_key(deadline)] = int(np.sum(delay_array <= deadline))

    raw_alarm_events = len(alarm_records)
    evaluated_alarm_events = true_positives + false_alarms
    return {
        "num_uavs": num_uavs,
        "num_attack_events": len(attack_events),
        # Backward-compatible alias: this is the number of raw K-step alarm
        # triggers, including duplicates ignored by event-level metrics.
        "num_alarm_events": raw_alarm_events,
        "num_raw_alarm_events": raw_alarm_events,
        "num_evaluated_alarm_events": evaluated_alarm_events,
        "num_ignored_duplicate_alarms": ignored_duplicate_alarms,
        "true_positives": true_positives,
        "false_positives": false_alarms,
        "false_negatives": false_negatives,
        "normal_exposure_steps": normal_exposure_steps,
        "normal_exposure_hours": normal_exposure_steps * float(dt_s) / 3600.0,
        "detected_delays_s": detected_delays_s,
        "deadline_detection_counts": deadline_counts,
        "clean_episode": len(attack_events) == 0,
        "clean_episode_had_alarm": len(attack_events) == 0 and false_alarms > 0,
        "alarm_events": alarm_records,
        "attack_events": attack_records,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("nan")


def _aggregate_without_ci(results: Sequence[dict[str, object]]) -> dict[str, object]:
    true_positives = int(sum(int(result["true_positives"]) for result in results))
    false_positives = int(sum(int(result["false_positives"]) for result in results))
    false_negatives = int(sum(int(result["false_negatives"]) for result in results))
    num_alarm_events = int(sum(int(result["num_alarm_events"]) for result in results))
    num_raw_alarm_events = int(
        sum(
            int(result.get("num_raw_alarm_events", result["num_alarm_events"]))
            for result in results
        )
    )
    num_evaluated_alarm_events = int(
        sum(
            int(
                result.get(
                    "num_evaluated_alarm_events",
                    int(result["true_positives"]) + int(result["false_positives"]),
                )
            )
            for result in results
        )
    )
    num_ignored_duplicate_alarms = int(
        sum(
            int(result.get("num_ignored_duplicate_alarms", 0))
            for result in results
        )
    )
    num_attack_events = int(sum(int(result["num_attack_events"]) for result in results))
    exposure_hours = float(
        sum(float(result["normal_exposure_hours"]) for result in results)
    )
    delays = np.asarray(
        [
            float(delay)
            for result in results
            for delay in result["detected_delays_s"]
        ],
        dtype=np.float64,
    )
    deadline_keys = sorted(
        {
            str(key)
            for result in results
            for key in result["deadline_detection_counts"]
        }
    )
    deadline_rates = {
        key: _safe_ratio(
            sum(int(result["deadline_detection_counts"].get(key, 0)) for result in results),
            num_attack_events,
        )
        for key in deadline_keys
    }
    clean_results = [result for result in results if bool(result["clean_episode"])]
    clean_with_alarm = sum(
        bool(result["clean_episode_had_alarm"]) for result in clean_results
    )
    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0.0
        else float("nan")
    )
    return {
        "num_episodes": len(results),
        "num_attack_events": num_attack_events,
        "num_alarm_events": num_alarm_events,
        "num_raw_alarm_events": num_raw_alarm_events,
        "num_evaluated_alarm_events": num_evaluated_alarm_events,
        "num_ignored_duplicate_alarms": num_ignored_duplicate_alarms,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "event_detection_rate": _safe_ratio(true_positives, num_attack_events),
        "event_miss_rate": _safe_ratio(false_negatives, num_attack_events),
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "normal_exposure_hours": exposure_hours,
        "false_alarms_per_uav_hour": _safe_ratio(false_positives, exposure_hours),
        "num_clean_episodes": len(clean_results),
        "clean_episode_alarm_probability": _safe_ratio(
            clean_with_alarm, len(clean_results)
        ),
        "delay_mean_s": float(np.mean(delays)) if delays.size else float("nan"),
        "delay_median_s": float(np.median(delays)) if delays.size else float("nan"),
        "delay_q1_s": (
            float(np.percentile(delays, 25.0)) if delays.size else float("nan")
        ),
        "delay_q3_s": (
            float(np.percentile(delays, 75.0)) if delays.size else float("nan")
        ),
        "detection_rate_within_s": deadline_rates,
    }


def aggregate_event_results(
    episode_results: Sequence[dict[str, object]],
    bootstrap_samples: int = 0,
    seed: int = 0,
) -> dict[str, object]:
    """Aggregate counts first, with optional episode-cluster bootstrap CIs."""

    results = tuple(episode_results)
    if not results:
        raise ValueError("at least one episode result is required")
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    aggregate = _aggregate_without_ci(results)
    aggregate["bootstrap_samples"] = int(bootstrap_samples)
    aggregate["bootstrap_seed"] = int(seed)
    if bootstrap_samples == 0:
        aggregate["confidence_intervals_95"] = {}
        return aggregate

    rng = np.random.default_rng(int(seed))
    scalar_keys = (
        "event_detection_rate",
        "event_miss_rate",
        "event_precision",
        "event_recall",
        "event_f1",
        "false_alarms_per_uav_hour",
        "clean_episode_alarm_probability",
        "delay_mean_s",
        "delay_median_s",
    )
    samples: dict[str, list[float]] = {key: [] for key in scalar_keys}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(results), size=len(results))
        summary = _aggregate_without_ci(tuple(results[int(index)] for index in indices))
        for key in scalar_keys:
            value = float(summary[key])
            if np.isfinite(value):
                samples[key].append(value)
    aggregate["confidence_intervals_95"] = {
        key: {
            "lower": float(np.percentile(values, 2.5)),
            "upper": float(np.percentile(values, 97.5)),
        }
        for key, values in samples.items()
        if values
    }
    return aggregate

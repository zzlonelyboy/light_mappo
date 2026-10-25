from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np

from .learning_data import (
    FeatureStandardizer,
    LoadedEpisode,
    UAVSequenceDataset,
    load_split,
    select_episode_features,
    training_label_counts,
)


ATTACK_PROBABILITY_THRESHOLD = 0.5
LOSS_OBJECTIVES = (
    "bce",
    "event_likelihood",
    "focal",
)
SOFT_CONFIRMATION_METHOD = "normalized_soft_min_k_hazard_v1"
CHECKPOINT_SELECTION_METHOD = "minimum_validation_total_loss_v1"


@dataclass(frozen=True)
class TrainingConfig:
    model_type: str = "mlp"
    class_weighting: str = "balanced"
    loss_objective: str = "bce"
    confirmation_steps: int = 5
    confirmation_temperature: float = 0.1
    time_step_s: float = 0.1
    detection_loss_weight: float = 0.1
    delay_loss_weight: float = 0.01
    false_alarm_loss_weight: float = 0.05
    early_detection_deadline_s: float = 3.0
    event_ramp_epochs: int = 20
    initial_checkpoint: str | None = None
    focal_gamma: float = 2.0
    focal_alpha: float | None = None
    hidden_dim: int = 64
    num_layers: int = 1
    input_encoder_dim: int = 0
    mlp_hidden_dims: tuple[int, ...] | None = None
    batch_size: int = 16
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 50
    patience: int = 8
    seed: int = 20260716
    device: str = "auto"
    num_threads: int = 1
    feature_names: tuple[str, ...] | None = None

    def validate(self) -> None:
        if self.model_type not in {"mlp", "gru"}:
            raise ValueError("model_type must be 'mlp' or 'gru'")
        if self.class_weighting not in {"balanced", "none"}:
            raise ValueError("class_weighting must be 'balanced' or 'none'")
        if self.loss_objective not in LOSS_OBJECTIVES:
            raise ValueError(
                f"loss_objective must be one of {LOSS_OBJECTIVES}, "
                f"got {self.loss_objective!r}"
            )
        if not 1 <= self.confirmation_steps <= 5:
            raise ValueError("confirmation_steps must lie in [1, 5]")
        if (
            not math.isfinite(self.confirmation_temperature)
            or self.confirmation_temperature <= 0.0
        ):
            raise ValueError("confirmation_temperature must be finite and positive")
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be finite and positive")
        for name, value in (
            ("detection_loss_weight", self.detection_loss_weight),
            ("delay_loss_weight", self.delay_loss_weight),
            ("false_alarm_loss_weight", self.false_alarm_loss_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.early_detection_deadline_s)
            or self.early_detection_deadline_s <= 0.0
        ):
            raise ValueError(
                "early_detection_deadline_s must be finite and positive"
            )
        if self.event_ramp_epochs < 0:
            raise ValueError("event_ramp_epochs must be non-negative")
        if self.loss_objective == "focal":
            if not math.isfinite(self.focal_gamma) or self.focal_gamma < 0.0:
                raise ValueError("focal_gamma must be finite and non-negative")
            if self.focal_alpha is not None and (
                not math.isfinite(self.focal_alpha)
                or not 0.0 < self.focal_alpha < 1.0
            ):
                raise ValueError(
                    "focal_alpha must be None or in (0, 1), "
                    f"got {self.focal_alpha!r}"
                )
        if (
            self.loss_objective == "event_likelihood"
            and self.detection_loss_weight == 0.0
            and self.delay_loss_weight == 0.0
            and self.false_alarm_loss_weight == 0.0
        ):
            raise ValueError(
                "event_likelihood requires a positive detection, early-detection, "
                "or false-alarm loss weight"
            )
        if self.hidden_dim < 1 or self.num_layers < 1 or self.batch_size < 1:
            raise ValueError("model dimensions and batch_size must be positive")
        if self.input_encoder_dim < 0:
            raise ValueError("input_encoder_dim must be non-negative")
        if self.input_encoder_dim > 0 and self.model_type != "gru":
            raise ValueError("input_encoder_dim is supported only for the GRU model")
        if self.mlp_hidden_dims is not None and (
            not self.mlp_hidden_dims or any(width < 1 for width in self.mlp_hidden_dims)
        ):
            raise ValueError("mlp_hidden_dims must contain positive layer widths")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.max_epochs < 1 or self.patience < 1:
            raise ValueError("max_epochs and patience must be positive")
        if self.num_threads < 1:
            raise ValueError("num_threads must be positive")
        if self.feature_names is not None and (
            not self.feature_names or len(set(self.feature_names)) != len(self.feature_names)
        ):
            raise ValueError("feature_names must be a non-empty unique sequence")


@dataclass
class DetectorBundle:
    model: Any
    standardizer: FeatureStandardizer
    device: Any
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class DetectionObjectiveResult:
    total: Any
    base_bce: Any
    detection: Any
    early: Any
    false_alarm: Any
    attack_events: int
    unobservable_attack_events: int
    normal_intervals: int


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Install the learning extra with: pip install -e '.[learning]'"
        ) from exc
    return torch, nn, DataLoader


def _select_device(requested: str):
    torch, _, _ = _import_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def binary_predictions_from_logits(
    logits,
    threshold: float = ATTACK_PROBABILITY_THRESHOLD,
):
    """Convert attack logits to hard labels using a fixed probability threshold."""
    attack_probability = attack_probabilities_from_logits(logits)
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("probability threshold must lie in [0, 1]")
    return (attack_probability >= float(threshold)).long()


def attack_probabilities_from_logits(logits):
    """Return p(attack), supporting new one-logit and legacy two-logit models."""
    torch, _, _ = _import_torch()
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits[..., 0])
    if logits.shape[-1] == 2:
        return torch.softmax(logits, dim=-1)[..., 1]
    raise ValueError(
        "binary detector logits must have final dimension 1 "
        "(or 2 for a legacy checkpoint)"
    )


def set_reproducible_seed(seed: int) -> None:
    torch, _, _ = _import_torch()
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _classification_summary(
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int,
) -> dict[str, float | int]:
    total = true_positive + false_positive + false_negative + true_negative
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)
    return {
        "accuracy": (true_positive + true_negative) / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def _soft_first_alarm_log_distribution(
    interval_logits,
    confirmation_steps: int,
    temperature: float,
    valid_mask=None,
):
    """Return log first-alarm hazards and log no-alarm probabilities.

    Each valid K-step window is reduced to a smooth approximation of its
    minimum logit. That soft minimum is converted into a conditional alarm
    hazard whose zero boundary matches the deployed ``p >= 0.5`` rule as the
    temperature decreases. Exclusive cumulative log survival then produces
    the first-alarm distribution without recurrent state normalization.

    Inputs may be a single interval ``(T,)`` or a padded interval batch
    ``(B,T)``. A K-window is valid only when all of its steps are valid, so
    padded steps and interval boundaries cannot create a confirmation.
    """

    torch, nn, _ = _import_torch()
    if interval_logits.ndim not in (1, 2):
        raise ValueError("interval_logits must have shape (time,) or (batch, time)")
    if confirmation_steps < 1:
        raise ValueError("confirmation_steps must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not interval_logits.is_floating_point():
        raise ValueError("interval_logits must be floating point")

    single_interval = interval_logits.ndim == 1
    batched_logits = interval_logits.unsqueeze(0) if single_interval else interval_logits
    if valid_mask is None:
        batched_mask = torch.ones_like(batched_logits, dtype=torch.bool)
    else:
        batched_mask = torch.as_tensor(
            valid_mask,
            dtype=torch.bool,
            device=batched_logits.device,
        )
        if single_interval and batched_mask.ndim == 1:
            batched_mask = batched_mask.unsqueeze(0)
        if batched_mask.shape != batched_logits.shape:
            raise ValueError("valid_mask must have the same shape as interval_logits")

    batch_size, time_steps = batched_logits.shape
    if time_steps < confirmation_steps:
        log_first_alarm = batched_logits.new_full(
            (batch_size, time_steps),
            -float("inf"),
        )
        log_no_alarm = batched_logits.sum(dim=1) * 0.0
        if single_interval:
            return log_first_alarm[0], log_no_alarm[0]
        return log_first_alarm, log_no_alarm

    window_logits = batched_logits.unfold(
        dimension=1,
        size=int(confirmation_steps),
        step=1,
    )
    window_is_valid = batched_mask.unfold(
        dimension=1,
        size=int(confirmation_steps),
        step=1,
    ).all(dim=-1)
    scaled_negative_logits = -window_logits / float(temperature)
    soft_minimum = -float(temperature) * (
        torch.logsumexp(scaled_negative_logits, dim=-1)
        - math.log(float(confirmation_steps))
    )

    scaled_soft_minimum = soft_minimum / float(temperature)
    log_hazard = nn.functional.logsigmoid(scaled_soft_minimum)
    log_no_hazard = nn.functional.logsigmoid(-scaled_soft_minimum)
    log_hazard = torch.where(
        window_is_valid,
        log_hazard,
        torch.full_like(log_hazard, -float("inf")),
    )
    log_no_hazard = torch.where(
        window_is_valid,
        log_no_hazard,
        torch.zeros_like(log_no_hazard),
    )
    log_survival_before = torch.cumsum(log_no_hazard, dim=1) - log_no_hazard
    log_first_window_alarm = log_hazard + log_survival_before
    prefix = batched_logits.new_full(
        (batch_size, int(confirmation_steps) - 1),
        -float("inf"),
    )
    log_first_alarm = torch.cat((prefix, log_first_window_alarm), dim=1)
    log_no_alarm = torch.sum(log_no_hazard, dim=1)
    if single_interval:
        return log_first_alarm[0], log_no_alarm[0]
    return log_first_alarm, log_no_alarm


def soft_first_alarm_distribution(
    interval_logits,
    confirmation_steps: int,
    temperature: float,
    valid_mask=None,
):
    """Return Soft-Min K-step first-alarm and no-alarm probabilities."""

    torch, _, _ = _import_torch()
    log_first_alarm, log_no_alarm = _soft_first_alarm_log_distribution(
        interval_logits,
        confirmation_steps=confirmation_steps,
        temperature=temperature,
        valid_mask=valid_mask,
    )
    return torch.exp(log_first_alarm), log_no_alarm


def _pack_interval_logits(
    logits,
    intervals: list[tuple[int, int, int]],
):
    """Gather variable-length intervals into one padded differentiable batch."""

    torch, _, _ = _import_torch()
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, time)")
    if not intervals:
        return (
            logits.new_zeros((0, 0)),
            torch.zeros((0, 0), dtype=torch.bool, device=logits.device),
            torch.zeros(0, dtype=torch.long, device=logits.device),
        )
    batch_indices = torch.as_tensor(
        [item[0] for item in intervals],
        dtype=torch.long,
        device=logits.device,
    )
    starts = torch.as_tensor(
        [item[1] for item in intervals],
        dtype=torch.long,
        device=logits.device,
    )
    lengths = torch.as_tensor(
        [item[2] - item[1] for item in intervals],
        dtype=torch.long,
        device=logits.device,
    )
    if torch.any(lengths <= 0):
        raise ValueError("packed intervals must have positive length")
    maximum_length = int(lengths.max().item())
    offsets = torch.arange(maximum_length, device=logits.device)
    valid_mask = offsets.unsqueeze(0) < lengths.unsqueeze(1)
    time_indices = (starts.unsqueeze(1) + offsets.unsqueeze(0)).clamp_max(
        logits.shape[1] - 1
    )
    packed = logits[batch_indices.unsqueeze(1), time_indices]
    packed = torch.where(valid_mask, packed, torch.zeros_like(packed))
    return packed, valid_mask, lengths


def _label_intervals(labels) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Return positive and normal half-open intervals as ``(batch, start, end)``."""

    if labels.ndim != 2:
        raise ValueError("labels must have shape (batch, time)")
    labels_cpu = labels.detach().to(device="cpu").numpy()
    if not np.all((labels_cpu == 0) | (labels_cpu == 1)):
        raise ValueError("labels must be binary")
    attack_events: list[tuple[int, int, int]] = []
    normal_intervals: list[tuple[int, int, int]] = []
    for batch_index, sequence in enumerate(labels_cpu):
        change_steps = np.flatnonzero(np.diff(sequence) != 0) + 1
        starts = np.concatenate(([0], change_steps))
        ends = np.concatenate((change_steps, [len(sequence)]))
        values = sequence[starts]
        attack_events.extend(
            (batch_index, int(start), int(end))
            for start, end, value in zip(starts, ends, values)
            if int(value) == 1
        )
        normal_intervals.extend(
            (batch_index, int(start), int(end))
            for start, end, value in zip(starts, ends, values)
            if int(value) == 0
        )
    return attack_events, normal_intervals


def _event_likelihood_losses(
    logits,
    attack_events: list[tuple[int, int, int]],
    normal_intervals: list[tuple[int, int, int]],
    confirmation_steps: int,
    temperature: float,
    time_step_s: float,
    early_detection_deadline_s: float,
    warmup_steps: int = 0,
):
    """Compute event terms with one vectorized Soft-Min K-hazard pass."""

    torch, _, _ = _import_torch()
    zero = logits.sum() * 0.0
    observable_events = [
        interval
        for interval in attack_events
        if interval[2] - interval[1] >= confirmation_steps
    ]
    unobservable_count = len(attack_events) - len(observable_events)
    monitored_intervals = []
    for batch_index, start, end in normal_intervals:
        monitored_start = max(int(start), int(warmup_steps))
        if int(end) - monitored_start >= int(confirmation_steps):
            monitored_intervals.append((batch_index, monitored_start, int(end)))

    combined_intervals = observable_events + monitored_intervals
    if not combined_intervals:
        return zero, zero, zero, 0, unobservable_count, 0

    packed_logits, valid_mask, lengths = _pack_interval_logits(
        logits,
        combined_intervals,
    )
    log_first_alarm, log_no_alarm = _soft_first_alarm_log_distribution(
        packed_logits,
        confirmation_steps=confirmation_steps,
        temperature=temperature,
        valid_mask=valid_mask,
    )
    observable_count = len(observable_events)
    if observable_count:
        event_log_first_alarm = log_first_alarm[:observable_count]
        log_detection_probability = torch.logsumexp(
            event_log_first_alarm,
            dim=1,
        )
        detection_loss = -log_detection_probability.mean()
        delay_s = torch.arange(
            packed_logits.shape[1],
            dtype=logits.dtype,
            device=logits.device,
        ).unsqueeze(0) * float(time_step_s)
        normalized_delay = torch.clamp(
            delay_s / float(early_detection_deadline_s),
            max=1.0,
        )
        conditional_first_alarm = torch.exp(
            event_log_first_alarm - log_detection_probability.unsqueeze(1)
        )
        early_loss = torch.sum(
            normalized_delay * conditional_first_alarm,
            dim=1,
        ).mean()
    else:
        detection_loss = zero
        early_loss = zero

    normal_interval_count = len(monitored_intervals)
    if normal_interval_count:
        normal_log_no_alarm = log_no_alarm[observable_count:]
        normal_lengths = lengths[observable_count:].to(dtype=logits.dtype)
        false_alarm_loss = (-normal_log_no_alarm / normal_lengths).mean()
    else:
        false_alarm_loss = zero
    return (
        detection_loss,
        early_loss,
        false_alarm_loss,
        observable_count,
        unobservable_count,
        normal_interval_count,
    )


def _binary_focal_loss(
    logits,
    labels,
    *,
    gamma: float,
    alpha: float | None,
):
    """Numerically stable Binary Focal Loss.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(z) for y=1, and p_t = 1 - sigmoid(z) for y=0.
    alpha_t = alpha for y=1, (1-alpha) for y=0 (when alpha is not None).

    Uses BCEWithLogitsLoss(reduction='none') internally for stability,
    then applies the modulating factor (1-p_t)^gamma and alpha weighting.

    When gamma=0 and alpha=None, this reduces to ordinary BCE.
    """
    torch, _, _ = _import_torch()
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != 1:
        raise ValueError("logits and labels must have compatible shapes")
    if gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative")

    targets = labels.to(dtype=logits.dtype).unsqueeze(-1)

    # Numerically stable BCE (element-wise, no reduction)
    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none",
    )

    if gamma == 0.0 and alpha is None:
        return bce_loss.mean()

    # Compute p_t = probability of the correct class
    p = torch.sigmoid(logits)
    p_t = torch.where(targets > 0.5, p, 1.0 - p)

    # Modulating factor: (1 - p_t)^gamma
    modulating_factor = (1.0 - p_t).pow(gamma)

    # Alpha weighting
    if alpha is not None:
        alpha_t = torch.where(targets > 0.5, float(alpha), 1.0 - float(alpha))
        focal_loss = alpha_t * modulating_factor * bce_loss
    else:
        focal_loss = modulating_factor * bce_loss

    return focal_loss.mean()


def binary_detection_objective(
    logits,
    labels,
    criterion,
    training_config: TrainingConfig,
    label_intervals: tuple[
        list[tuple[int, int, int]], list[tuple[int, int, int]]
    ]
    | None = None,
    monitoring_warmup_steps: int = 0,
    event_loss_scale: float = 1.0,
):
    """Return BCE or the Soft-Min K-hazard event-likelihood objective."""

    torch, _, _ = _import_torch()
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != 1:
        raise ValueError(
            f"logits {tuple(logits.shape)} are incompatible with "
            f"labels {tuple(labels.shape)}"
        )
    if not 0 <= int(monitoring_warmup_steps) <= int(labels.shape[1]):
        raise ValueError("monitoring_warmup_steps must lie in [0, sequence length]")
    if not math.isfinite(event_loss_scale) or not 0.0 <= event_loss_scale <= 1.0:
        raise ValueError("event_loss_scale must lie in [0, 1]")
    targets = labels.to(dtype=logits.dtype).unsqueeze(-1)
    elementwise_bce = criterion(logits, targets)
    if elementwise_bce.shape != logits.shape:
        raise ValueError("criterion must return one loss value per model logit")
    base_bce_loss = elementwise_bce.mean()
    if training_config.loss_objective == "bce":
        zero = base_bce_loss * 0.0
        return DetectionObjectiveResult(
            total=base_bce_loss,
            base_bce=base_bce_loss,
            detection=zero,
            early=zero,
            false_alarm=zero,
            attack_events=0,
            unobservable_attack_events=0,
            normal_intervals=0,
        )
    if training_config.loss_objective == "focal":
        focal_loss = _binary_focal_loss(
            logits,
            labels,
            gamma=float(training_config.focal_gamma),
            alpha=(
                float(training_config.focal_alpha)
                if training_config.focal_alpha is not None
                else None
            ),
        )
        zero = focal_loss * 0.0
        return DetectionObjectiveResult(
            total=focal_loss,
            base_bce=focal_loss,
            detection=zero,
            early=zero,
            false_alarm=zero,
            attack_events=0,
            unobservable_attack_events=0,
            normal_intervals=0,
        )
    if training_config.loss_objective != "event_likelihood":
        raise ValueError(f"unsupported loss objective: {training_config.loss_objective}")
    attack_events, normal_intervals = (
        _label_intervals(labels) if label_intervals is None else label_intervals
    )
    sequence_logits = logits[..., 0]
    (
        detection_loss,
        early_loss,
        false_alarm_loss,
        attack_event_count,
        unobservable_count,
        normal_interval_count,
    ) = _event_likelihood_losses(
        sequence_logits,
        attack_events,
        normal_intervals,
        confirmation_steps=training_config.confirmation_steps,
        temperature=training_config.confirmation_temperature,
        time_step_s=training_config.time_step_s,
        early_detection_deadline_s=training_config.early_detection_deadline_s,
        warmup_steps=monitoring_warmup_steps,
    )
    event_loss = (
        # training_config.detection_loss_weight * detection_loss
        + training_config.delay_loss_weight * early_loss
        + training_config.false_alarm_loss_weight * false_alarm_loss
    )
    total = base_bce_loss + float(event_loss_scale) * event_loss
    return DetectionObjectiveResult(
        total=total,
        base_bce=base_bce_loss,
        detection=detection_loss,
        early=early_loss,
        false_alarm=false_alarm_loss,
        attack_events=attack_event_count,
        unobservable_attack_events=unobservable_count,
        normal_intervals=normal_interval_count,
    )


def _hard_event_summary_from_sequences(
    predictions_by_sequence: list[np.ndarray],
    labels_by_sequence: list[np.ndarray],
    confirmation_steps: int,
    time_step_s: float,
    warmup_steps: int,
) -> dict[str, object]:
    """Evaluate hard K-step alarms on independent per-UAV validation sequences."""

    from .metrics import (
        AttackEvent,
        aggregate_event_results,
        k_step_alarm_events,
        match_alarm_events,
    )

    if len(predictions_by_sequence) != len(labels_by_sequence):
        raise ValueError("prediction and label sequence counts differ")
    episode_results: list[dict[str, object]] = []
    for sequence_index, (predictions, labels) in enumerate(
        zip(predictions_by_sequence, labels_by_sequence)
    ):
        hard_predictions = np.asarray(predictions, dtype=np.uint8).reshape(-1, 1)
        hard_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
        positive_intervals, _ = _label_intervals(
            _import_torch()[0].as_tensor(hard_labels.reshape(1, -1))
        )
        attack_events = tuple(
            AttackEvent(
                uav=0,
                start_step=int(start),
                end_step_exclusive=int(end),
                event_id=f"sequence_{sequence_index}_event_{event_index}",
            )
            for event_index, (_, start, end) in enumerate(positive_intervals)
        )
        alarms = k_step_alarm_events(
            hard_predictions,
            confirmation_steps=confirmation_steps,
            warmup_steps=warmup_steps,
        )
        episode_results.append(
            match_alarm_events(
                alarms,
                attack_events,
                num_steps=hard_predictions.shape[0],
                dt_s=time_step_s,
                warmup_steps=warmup_steps,
            )
        )
    return aggregate_event_results(episode_results, bootstrap_samples=0, seed=0)


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    training_config: TrainingConfig,
    optimizer=None,
    monitoring_warmup_steps: int = 0,
    compute_hard_event_metrics: bool = False,
    event_loss_scale: float = 1.0,
    progress_label: str | None = None,
    progress_interval_batches: int = 100,
) -> dict[str, object]:
    torch, _, _ = _import_torch()
    training = optimizer is not None
    model.train(training)
    objective_loss_sum = 0.0
    base_bce_loss_sum = 0.0
    detection_loss_sum = 0.0
    early_loss_sum = 0.0
    false_alarm_loss_sum = 0.0
    attack_event_count = 0
    unobservable_attack_event_count = 0
    normal_interval_count = 0
    observation_count = 0
    true_positive = false_positive = false_negative = true_negative = 0
    hard_prediction_sequences: list[np.ndarray] = []
    hard_label_sequences: list[np.ndarray] = []
    total_batches = len(loader)
    started_at = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_number, (features, labels) in enumerate(loader, start=1):
            label_intervals = (
                _label_intervals(labels)
                if training_config.loss_objective == "event_likelihood"
                else None
            )
            features = features.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            if logits.shape[:-1] != labels.shape or logits.shape[-1] != 1:
                raise RuntimeError(
                    f"model output {tuple(logits.shape)} is incompatible with "
                    f"labels {tuple(labels.shape)}"
                )
            objective = binary_detection_objective(
                logits,
                labels,
                criterion,
                training_config,
                label_intervals=label_intervals,
                monitoring_warmup_steps=monitoring_warmup_steps,
                event_loss_scale=event_loss_scale,
            )
            if not bool(torch.isfinite(objective.total).detach().item()):
                component_values = {
                    "total": float(objective.total.detach().item()),
                    "base_bce": float(objective.base_bce.detach().item()),
                    "detection": float(objective.detection.detach().item()),
                    "early": float(objective.early.detach().item()),
                    "false_alarm": float(objective.false_alarm.detach().item()),
                }
                raise FloatingPointError(
                    "non-finite detection objective before optimizer step: "
                    f"{component_values}"
                )
            if training:
                objective.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            count = labels.numel()
            objective_loss_sum += float(objective.total.detach().item()) * count
            base_bce_loss_sum += float(objective.base_bce.detach().item()) * count
            detection_loss_sum += (
                float(objective.detection.detach().item()) * objective.attack_events
            )
            early_loss_sum += (
                float(objective.early.detach().item()) * objective.attack_events
            )
            false_alarm_loss_sum += (
                float(objective.false_alarm.detach().item())
                * objective.normal_intervals
            )
            attack_event_count += objective.attack_events
            unobservable_attack_event_count += objective.unobservable_attack_events
            normal_interval_count += objective.normal_intervals
            observation_count += count
            predictions = binary_predictions_from_logits(logits.detach())
            true_positive += int(torch.sum((predictions == 1) & (labels == 1)).item())
            false_positive += int(torch.sum((predictions == 1) & (labels == 0)).item())
            false_negative += int(torch.sum((predictions == 0) & (labels == 1)).item())
            true_negative += int(torch.sum((predictions == 0) & (labels == 0)).item())
            if compute_hard_event_metrics:
                hard_prediction_sequences.extend(
                    np.asarray(sequence, dtype=np.uint8)
                    for sequence in predictions.detach().to(device="cpu").numpy()
                )
                hard_label_sequences.extend(
                    np.asarray(sequence, dtype=np.uint8)
                    for sequence in labels.detach().to(device="cpu").numpy()
                )
            if progress_label is not None and (
                batch_number % max(int(progress_interval_batches), 1) == 0
                or batch_number == total_batches
            ):
                elapsed = time.perf_counter() - started_at
                print(
                    f"{progress_label} | batch {batch_number}/{total_batches} "
                    f"| elapsed={elapsed:.1f}s",
                    flush=True,
                )
    summary = _classification_summary(
        true_positive, false_positive, false_negative, true_negative
    )
    summary["loss"] = objective_loss_sum / max(observation_count, 1)
    summary["base_bce_loss"] = base_bce_loss_sum / max(observation_count, 1)
    summary["detection_loss"] = detection_loss_sum / max(attack_event_count, 1)
    summary["early_loss"] = early_loss_sum / max(attack_event_count, 1)
    summary["false_alarm_loss"] = false_alarm_loss_sum / max(
        normal_interval_count, 1
    )
    summary["attack_events_in_objective"] = attack_event_count
    summary["unobservable_attack_events"] = unobservable_attack_event_count
    summary["normal_intervals_in_objective"] = normal_interval_count
    summary["event_loss_scale"] = float(event_loss_scale)
    if compute_hard_event_metrics:
        summary["hard_event_metrics"] = _hard_event_summary_from_sequences(
            hard_prediction_sequences,
            hard_label_sequences,
            confirmation_steps=training_config.confirmation_steps,
            time_step_s=training_config.time_step_s,
            warmup_steps=monitoring_warmup_steps,
        )
    return summary


def _class_weights(
    episodes: tuple[LoadedEpisode, ...],
    weighting: str,
) -> tuple[np.ndarray, tuple[int, int]]:
    negative, positive = training_label_counts(episodes)
    if negative == 0 or positive == 0:
        raise ValueError("training data must contain both normal and attack labels")
    if weighting == "none":
        return np.ones(2, dtype=np.float32), (negative, positive)
    if weighting != "balanced":
        raise ValueError("weighting must be 'balanced' or 'none'")
    weights = np.asarray([1.0, negative / positive], dtype=np.float32)
    return weights, (negative, positive)


def _write_json(path: Path, values: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2)


def _dataset_time_step_s(data_root: str | Path) -> float:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        value = float(manifest["config"]["simulation"]["dt_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"dataset manifest does not define config.simulation.dt_s: {manifest_path}"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dataset time step must be finite and positive")
    return value


def _dataset_warmup_steps(data_root: str | Path) -> int:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    try:
        value = int(manifest["config"]["features"].get("warmup_steps", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"dataset manifest has invalid config.features.warmup_steps: {manifest_path}"
        ) from exc
    if value < 0:
        raise ValueError("dataset warmup steps must be non-negative")
    return value


def _plot_training_history(
    history: list[dict[str, object]],
    output: Path,
    best_epoch: int,
    loss_objective: str,
) -> tuple[Path, Path]:
    """Save objective, checkpoint-selection, and step-level metric curves."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - learning extra includes matplotlib
        raise ImportError(
            "Install the learning extra with: pip install -e '.[learning]'"
        ) from exc

    epochs = [int(item["epoch"]) for item in history]
    train_metrics = [item["train"] for item in history]
    validation_metrics = [item["validation"] for item in history]

    loss_path = output / "Figure_1.png"
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    if loss_objective == "bce":
        axis.plot(
            epochs,
            [float(metrics["base_bce_loss"]) for metrics in train_metrics],
            marker="o",
            markersize=3,
            label="Train base BCE",
        )
        axis.plot(
            epochs,
            [float(metrics["base_bce_loss"]) for metrics in validation_metrics],
            marker="o",
            markersize=3,
            label="Validation base BCE (selection)",
        )
    else:
        axis.plot(
            epochs,
            [float(metrics["loss"]) for metrics in train_metrics],
            alpha=0.8,
            label="Train ramped event-likelihood objective",
        )
        axis.plot(
            epochs,
            [float(metrics["loss"]) for metrics in validation_metrics],
            alpha=0.8,
            label="Validation full event-likelihood objective (selection)",
        )
        axis.plot(
            epochs,
            [float(metrics["base_bce_loss"]) for metrics in train_metrics],
            linestyle="--",
            label="Train base BCE",
        )
        axis.plot(
            epochs,
            [float(metrics["base_bce_loss"]) for metrics in validation_metrics],
            color="black",
            linewidth=1.4,
            label="Validation base BCE",
        )
    axis.axvline(
        best_epoch,
        color="black",
        linestyle=":",
        alpha=0.6,
        label="Selected epoch",
    )
    axis.set(
        title=f"Training history: {loss_objective}",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(loss_path, dpi=200)
    plt.close(fig)

    metrics_path = output / "Figure_2.png"
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for axis, metric, title in zip(
        axes,
        ("precision", "recall", "f1"),
        ("Precision", "Recall", "F1"),
    ):
        axis.plot(
            epochs,
            [float(values[metric]) for values in train_metrics],
            linestyle="--",
            label="Train",
        )
        axis.plot(
            epochs,
            [float(values[metric]) for values in validation_metrics],
            label="Validation",
        )
        axis.axvline(best_epoch, color="black", linestyle=":", alpha=0.45)
        axis.set(title=title, xlabel="Epoch", ylim=(0.0, 1.0))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Step-level score")
    axes[-1].legend()
    fig.savefig(metrics_path, dpi=200)
    plt.close(fig)
    return loss_path, metrics_path


def _validation_loss_improved(current_loss: float, best_loss: float) -> bool:
    """Return whether a finite validation loss improves the checkpoint."""

    if not math.isfinite(current_loss):
        raise FloatingPointError(
            f"validation loss must be finite for checkpoint selection: {current_loss}"
        )
    return current_loss < best_loss


def train_detector(
    data_root: str | Path,
    output_directory: str | Path,
    config: TrainingConfig | None = None,
) -> dict[str, object]:
    """Train one detector using train/validation only and save the best checkpoint."""

    torch, nn, DataLoader = _import_torch()
    from .models.torch_models import build_detector

    training_config = config or TrainingConfig()
    training_config.validate()
    if training_config.loss_objective == "event_likelihood":
        dataset_dt_s = _dataset_time_step_s(data_root)
        if not math.isclose(
            training_config.time_step_s,
            dataset_dt_s,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "event-risk time_step_s must match the dataset manifest: "
                f"configured={training_config.time_step_s}, dataset={dataset_dt_s}"
            )
    monitoring_warmup_steps = _dataset_warmup_steps(data_root)
    torch.set_num_threads(training_config.num_threads)
    set_reproducible_seed(training_config.seed)
    device = _select_device(training_config.device)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    train_episodes = load_split(data_root, "train")
    validation_episodes = load_split(data_root, "validation")
    if validation_episodes[0].feature_names != train_episodes[0].feature_names:
        raise ValueError("training and validation feature schemas differ")
    selected_feature_names = (
        train_episodes[0].feature_names
        if training_config.feature_names is None
        else training_config.feature_names
    )
    train_episodes = tuple(
        select_episode_features(episode, selected_feature_names)
        for episode in train_episodes
    )
    validation_episodes = tuple(
        select_episode_features(episode, selected_feature_names)
        for episode in validation_episodes
    )
    initial_bundle = None
    if training_config.initial_checkpoint is not None:
        initial_path = Path(training_config.initial_checkpoint)
        if not initial_path.is_file():
            raise FileNotFoundError(
                f"initial checkpoint does not exist: {initial_path}"
            )
        initial_bundle = load_trained_detector(
            initial_path,
            requested_device=str(device),
        )
        if str(initial_bundle.checkpoint["model_type"]) != training_config.model_type:
            raise ValueError(
                "initial checkpoint model_type does not match the requested model"
            )
        if tuple(initial_bundle.standardizer.feature_names) != tuple(
            selected_feature_names
        ):
            raise ValueError(
                "initial checkpoint feature names do not match the selected feature set"
            )
        initial_model_config = initial_bundle.checkpoint["model_config"]
        expected_mlp_dims = (
            None
            if training_config.mlp_hidden_dims is None
            else tuple(training_config.mlp_hidden_dims)
        )
        checkpoint_mlp_dims = (
            None
            if initial_model_config.get("mlp_hidden_dims") is None
            else tuple(int(width) for width in initial_model_config["mlp_hidden_dims"])
        )
        if (
            int(initial_model_config["hidden_dim"]) != training_config.hidden_dim
            or int(initial_model_config["num_layers"]) != training_config.num_layers
            or int(initial_model_config.get("input_encoder_dim", 0))
            != training_config.input_encoder_dim
            or checkpoint_mlp_dims != expected_mlp_dims
            or int(initial_model_config.get("output_dim", 2)) != 1
        ):
            raise ValueError(
                "initial checkpoint architecture does not match the training config"
            )
        standardizer = initial_bundle.standardizer
    else:
        standardizer = FeatureStandardizer.fit(train_episodes)
    train_dataset = UAVSequenceDataset(train_episodes, standardizer)
    validation_dataset = UAVSequenceDataset(validation_episodes, standardizer)
    generator = torch.Generator()
    generator.manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    feature_dim = len(standardizer.feature_names)
    model = (
        initial_bundle.model
        if initial_bundle is not None
        else build_detector(
            training_config.model_type,
            input_dim=feature_dim,
            hidden_dim=training_config.hidden_dim,
            num_layers=training_config.num_layers,
            mlp_hidden_dims=training_config.mlp_hidden_dims,
            output_dim=1,
            input_encoder_dim=training_config.input_encoder_dim,
        ).to(device)
    )
    weights, label_counts = _class_weights(
        train_episodes,
        training_config.class_weighting,
    )
    positive_weight = (
        None
        if training_config.class_weighting == "none"
        else torch.as_tensor([weights[1]], dtype=torch.float32, device=device)
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight,
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    checkpoint_path = output / "best_model.pt"
    history: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_selection_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_event_loss_scale = 0.0
    best_training_event_loss_scale = 0.0
    epochs_without_improvement = 0
    for epoch in range(1, training_config.max_epochs + 1):
        event_loss_scale = (
            0.0
            if training_config.loss_objective == "bce"
            else (
                1.0
                if training_config.event_ramp_epochs == 0
                else min(epoch / float(training_config.event_ramp_epochs), 1.0)
            )
        )
        validation_event_loss_scale = (
            0.0 if training_config.loss_objective == "bce" else 1.0
        )
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            training_config,
            optimizer,
            monitoring_warmup_steps=monitoring_warmup_steps,
            event_loss_scale=event_loss_scale,
            progress_label=f"epoch {epoch:03d} train",
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            training_config,
            optimizer=None,
            monitoring_warmup_steps=monitoring_warmup_steps,
            compute_hard_event_metrics=(
                training_config.loss_objective == "event_likelihood"
            ),
            event_loss_scale=validation_event_loss_scale,
            progress_label=f"epoch {epoch:03d} validation",
        )
        history_item: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(history_item)
        _write_json(output / "training_history.json", {"epochs": history})
        validation_loss = float(validation_metrics["loss"])
        selection_key = (-validation_loss,)
        improved = _validation_loss_improved(
            validation_loss,
            best_validation_loss,
        )
        if improved:
            best_validation_loss = validation_loss
            best_selection_key = selection_key
            best_epoch = epoch
            best_event_loss_scale = validation_event_loss_scale
            best_training_event_loss_scale = event_loss_scale
            epochs_without_improvement = 0
            checkpoint = {
                "schema_version": "1.12",
                "model_type": training_config.model_type,
                "model_config": {
                    "input_dim": feature_dim,
                    "hidden_dim": training_config.hidden_dim,
                    "num_layers": training_config.num_layers,
                    "input_encoder_dim": training_config.input_encoder_dim,
                    "mlp_hidden_dims": training_config.mlp_hidden_dims,
                    "output_dim": 1,
                    "output_mode": "single_attack_logit",
                },
                "model_state_dict": model.state_dict(),
                "standardizer": standardizer.to_dict(),
                "training_config": asdict(training_config),
                "training_label_counts": {
                    "normal": label_counts[0],
                    "attack": label_counts[1],
                },
                "class_weighting": training_config.class_weighting,
                "class_weights": weights.tolist(),
                "positive_class_weight": float(weights[1]),
                "loss_objective": training_config.loss_objective,
                "checkpoint_selection": CHECKPOINT_SELECTION_METHOD,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "best_selection_key": list(selection_key),
                "event_loss_scale": validation_event_loss_scale,
                "training_event_loss_scale": event_loss_scale,
                "validation_event_loss_scale": validation_event_loss_scale,
                "best_validation_hard_event_metrics": validation_metrics.get(
                    "hard_event_metrics"
                ),
                "monitoring_warmup_steps": monitoring_warmup_steps,
            }
            torch.save(checkpoint, checkpoint_path)
            status = "best"
        else:
            epochs_without_improvement += 1
            status = f"wait {epochs_without_improvement}/{training_config.patience}"
        if training_config.loss_objective == "event_likelihood":
            hard_event = validation_metrics["hard_event_metrics"]
            objective_diagnostics = (
                f"det={float(validation_metrics['detection_loss']):.4f} "
                f"early={float(validation_metrics['early_loss']):.4f} "
                f"FA={float(validation_metrics['false_alarm_loss']):.4f} "
                f"trainRamp={event_loss_scale:.3f} "
                f"eventP/R/F1={float(hard_event['event_precision']):.4f}/"
                f"{float(hard_event['event_recall']):.4f}/"
                f"{float(hard_event['event_f1']):.4f} "
                f"hardFAR={float(hard_event['false_alarms_per_uav_hour']):.4f}"
            )
        else:
            objective_diagnostics = ""
        print(
            f"epoch {epoch:03d}/{training_config.max_epochs:03d} | "
            f"train loss={float(train_metrics['loss']):.6f} "
            f"acc={float(train_metrics['accuracy']):.4f} "
            f"P/R/F1={float(train_metrics['precision']):.4f}/"
            f"{float(train_metrics['recall']):.4f}/"
            f"{float(train_metrics['f1']):.4f} | "
            f"validation loss={float(validation_metrics['loss']):.6f} "
            f"base_bce={float(validation_metrics['base_bce_loss']):.6f} "
            f"{objective_diagnostics} "
            f"acc={float(validation_metrics['accuracy']):.4f} "
            f"P/R/F1={float(validation_metrics['precision']):.4f}/"
            f"{float(validation_metrics['recall']):.4f}/"
            f"{float(validation_metrics['f1']):.4f} | {status}",
            flush=True,
        )
        if epochs_without_improvement >= training_config.patience:
            print(
                "early stopping: validation loss did not improve for "
                f"{training_config.patience} consecutive epochs",
                flush=True,
            )
            break

    loss_figure, metrics_figure = _plot_training_history(
        history,
        output,
        best_epoch,
        training_config.loss_objective,
    )
    print(f"saved training curves: {loss_figure} and {metrics_figure}", flush=True)

    bundle = load_trained_detector(checkpoint_path, requested_device=str(device))
    final_validation = _run_epoch(
        bundle.model,
        validation_loader,
        criterion,
        bundle.device,
        training_config,
        optimizer=None,
        monitoring_warmup_steps=monitoring_warmup_steps,
        compute_hard_event_metrics=(
            training_config.loss_objective == "event_likelihood"
        ),
        event_loss_scale=best_event_loss_scale,
    )
    if float(final_validation["accuracy"]) <= 0.0:
        raise RuntimeError("validation accuracy is 0; the training pipeline is invalid")
    result: dict[str, object] = {
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_selection_key": (
            None if best_selection_key is None else list(best_selection_key)
        ),
        "best_event_loss_scale": best_event_loss_scale,
        "best_training_event_loss_scale": best_training_event_loss_scale,
        "epochs_completed": len(history),
        "training_sequences": len(train_dataset),
        "validation_sequences": len(validation_dataset),
        "training_label_counts": {
            "normal": label_counts[0],
            "attack": label_counts[1],
        },
        "class_weighting": training_config.class_weighting,
        "class_weights": weights.tolist(),
        "training_config": asdict(training_config),
        "loss_objective": training_config.loss_objective,
        "monitoring_warmup_steps": monitoring_warmup_steps,
        "event_likelihood_objective": {
            "soft_confirmation_method": SOFT_CONFIRMATION_METHOD,
            "confirmation_steps": training_config.confirmation_steps,
            "confirmation_temperature": training_config.confirmation_temperature,
            "time_step_s": training_config.time_step_s,
            "monitoring_warmup_steps": monitoring_warmup_steps,
            "detection_loss_weight": training_config.detection_loss_weight,
            "delay_loss_weight": training_config.delay_loss_weight,
            "false_alarm_loss_weight": training_config.false_alarm_loss_weight,
            "early_detection_deadline_s": (
                training_config.early_detection_deadline_s
            ),
            "event_ramp_epochs": training_config.event_ramp_epochs,
            "initial_checkpoint": training_config.initial_checkpoint,
            "detection_definition": (
                "negative log likelihood of at least one first K-step alarm "
                "within each observable attack event"
            ),
            "early_definition": (
                "detection-conditional expected first-alarm delay normalized "
                "by early_detection_deadline_s"
            ),
            "soft_confirmation_definition": (
                "vectorized normalized Soft-Min over each K-logit window, "
                "followed by a differentiable conditional alarm hazard and "
                "exclusive cumulative log survival; confirmation_temperature "
                "controls both smoothing stages and logit zero is the deployed "
                "p=0.5 threshold in the zero-temperature limit"
            ),
            "false_alarm_definition": (
                "mean length-normalized negative log no-alarm probability over "
                "maximal monitored normal intervals; cross-boundary transitions excluded"
            ),
            "false_alarm_surrogate": "normal_interval_no_alarm_likelihood",
            "short_attack_policy": (
                "events shorter than K remain in step BCE but are excluded from "
                "event detection and early-delay losses"
            ),
            "checkpoint_selection": bundle.checkpoint["checkpoint_selection"],
        },
        "model_config": bundle.checkpoint["model_config"],
        "final_validation_time_step_metrics": final_validation,
        "feature_names": list(standardizer.feature_names),
        "training_loss_figure": str(loss_figure.resolve()),
        "training_metrics_figure": str(metrics_figure.resolve()),
    }
    _write_json(output / "training_result.json", result)
    _write_json(output / "standardizer.json", standardizer.to_dict())
    _write_json(output / "training_config.json", asdict(training_config))
    return result


def load_trained_detector(
    checkpoint_path: str | Path,
    requested_device: str = "auto",
) -> DetectorBundle:
    torch, _, _ = _import_torch()
    from .models.torch_models import build_detector

    device = _select_device(requested_device)
    try:
        checkpoint = torch.load(
            Path(checkpoint_path), map_location=device, weights_only=False
        )
    except TypeError:  # pragma: no cover - older torch compatibility
        checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model_config = checkpoint["model_config"]
    model = build_detector(
        checkpoint["model_type"],
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        mlp_hidden_dims=(
            None
            if model_config.get("mlp_hidden_dims") is None
            else tuple(int(width) for width in model_config["mlp_hidden_dims"])
        ),
        output_dim=int(model_config.get("output_dim", 2)),
        input_encoder_dim=int(model_config.get("input_encoder_dim", 0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    standardizer = FeatureStandardizer.from_dict(checkpoint["standardizer"])
    return DetectorBundle(
        model=model,
        standardizer=standardizer,
        device=device,
        checkpoint=checkpoint,
    )


def predict_episode(
    bundle: DetectorBundle,
    episode: LoadedEpisode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw logits, p(attack), and hard decisions for one episode."""

    torch, _, _ = _import_torch()
    selected_episode = select_episode_features(
        episode, bundle.standardizer.feature_names
    )
    normalized = bundle.standardizer.transform(selected_episode.features)
    sequence_batch = np.transpose(normalized, (1, 0, 2))
    with torch.no_grad():
        tensor = torch.from_numpy(sequence_batch).to(
            device=bundle.device, dtype=torch.float32
        )
        logits_batch = bundle.model(tensor)
        attack_probability_batch = attack_probabilities_from_logits(logits_batch)
        predictions_batch = binary_predictions_from_logits(logits_batch)
    logits = np.transpose(logits_batch.cpu().numpy(), (1, 0, 2)).astype(
        np.float32, copy=False
    )
    predictions = np.transpose(predictions_batch.cpu().numpy(), (1, 0)).astype(
        np.uint8, copy=False
    )
    attack_probability = np.transpose(
        attack_probability_batch.cpu().numpy(), (1, 0)
    ).astype(np.float32, copy=False)
    return logits, attack_probability, predictions

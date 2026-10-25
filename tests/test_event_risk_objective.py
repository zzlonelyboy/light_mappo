from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from isac_spoofing.training import (
    TrainingConfig,
    _validation_loss_improved,
    binary_detection_objective,
    soft_first_alarm_distribution,
)


def _logits_from_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return torch.logit(probabilities).unsqueeze(-1).detach().requires_grad_(True)


def _reference_soft_min_hazards(
    logits: torch.Tensor,
    confirmation_steps: int,
    temperature: float,
) -> torch.Tensor:
    windows = logits.unfold(0, confirmation_steps, 1)
    soft_minimum = -temperature * (
        torch.logsumexp(-windows / temperature, dim=-1)
        - math.log(confirmation_steps)
    )
    return torch.sigmoid(soft_minimum / temperature)


def test_soft_min_hazard_first_alarm_distribution_handles_overlapping_windows() -> None:
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    logits = torch.logit(probabilities)

    first_alarm, log_no_alarm = soft_first_alarm_distribution(
        logits,
        confirmation_steps=3,
        temperature=1.0,
    )

    hazards = _reference_soft_min_hazards(logits, 3, 1.0)
    expected = torch.stack(
        (
            logits.new_zeros(()),
            logits.new_zeros(()),
            hazards[0],
            (1.0 - hazards[0]) * hazards[1],
        )
    )
    assert torch.allclose(first_alarm, expected, atol=1.0e-7)
    assert math.exp(log_no_alarm.item()) == pytest.approx(1.0 - expected.sum().item())


def test_first_alarm_and_no_alarm_probabilities_sum_to_one() -> None:
    probabilities = torch.tensor(
        [0.7, 0.8, 0.6, 0.9, 0.4, 0.75], dtype=torch.float64
    )
    first_alarm, log_no_alarm = soft_first_alarm_distribution(
        torch.logit(probabilities),
        confirmation_steps=5,
        temperature=1.0,
    )

    total = first_alarm.sum() + torch.exp(log_no_alarm)
    assert total.item() == pytest.approx(1.0, abs=1.0e-12)
    assert torch.all(first_alarm[:4] == 0.0)


def test_batched_padded_soft_min_hazard_matches_individual_intervals() -> None:
    probabilities = torch.tensor(
        [
            [0.1, 0.7, 0.8, 0.6, 0.9, 0.4],
            [0.3, 0.4, 0.5, 0.6, 0.0, 0.0],
            [0.9, 0.9, 0.2, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    lengths = (6, 4, 3)
    valid_mask = torch.arange(6).unsqueeze(0) < torch.tensor(lengths).unsqueeze(1)
    logits = torch.logit(probabilities.clamp(1.0e-6, 1.0 - 1.0e-6))

    batched_first, batched_log_no_alarm = soft_first_alarm_distribution(
        logits,
        confirmation_steps=3,
        temperature=1.0,
        valid_mask=valid_mask,
    )

    for index, length in enumerate(lengths):
        individual_first, individual_log_no_alarm = soft_first_alarm_distribution(
            logits[index, :length],
            confirmation_steps=3,
            temperature=1.0,
        )
        assert torch.allclose(
            batched_first[index, :length], individual_first, atol=1.0e-12
        )
        assert torch.all(batched_first[index, length:] == 0.0)
        assert torch.allclose(
            batched_log_no_alarm[index], individual_log_no_alarm, atol=1.0e-12
        )


def test_batched_event_objective_has_finite_gradients_for_variable_intervals() -> None:
    labels = torch.tensor(
        [
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    logits = torch.randn((3, 7, 1), dtype=torch.float64, requires_grad=True)
    config = TrainingConfig(
        loss_objective="event_likelihood",
        confirmation_steps=2,
        detection_loss_weight=0.1,
        delay_loss_weight=0.01,
        false_alarm_loss_weight=0.05,
    )

    result = binary_detection_objective(
        logits,
        labels,
        nn.BCEWithLogitsLoss(reduction="none"),
        config,
    )
    result.total.backward()

    assert result.attack_events == 2
    assert result.normal_intervals == 3
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


@pytest.mark.parametrize("value", (-50.0, -20.0, 0.0, 20.0, 50.0))
def test_long_soft_min_hazard_sequences_have_finite_values_and_gradients(
    value: float,
) -> None:
    logits = torch.full(
        (4, 600),
        value,
        dtype=torch.float32,
        requires_grad=True,
    )

    first_alarm, log_no_alarm = soft_first_alarm_distribution(
        logits,
        confirmation_steps=5,
        temperature=1.0,
    )
    objective = first_alarm.sum() + log_no_alarm.sum()
    objective.backward()

    assert torch.all(torch.isfinite(first_alarm))
    assert torch.all(torch.isfinite(log_no_alarm))
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_bce_objective_exactly_matches_legacy_value_and_gradient() -> None:
    labels = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    logits = torch.tensor(
        [[[-1.0], [0.2], [1.1], [-0.4]], [[0.7], [-0.8], [0.3], [1.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_logits = logits.detach().clone().requires_grad_(True)
    positive_weight = torch.tensor([2.25])
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight, reduction="none")

    result = binary_detection_objective(
        logits,
        labels,
        criterion,
        TrainingConfig(loss_objective="bce"),
    )
    legacy = nn.BCEWithLogitsLoss(pos_weight=positive_weight)(
        reference_logits,
        labels.float().unsqueeze(-1),
    )

    assert torch.equal(result.total, result.base_bce)
    assert torch.allclose(result.total, legacy)
    assert result.detection.item() == 0.0
    assert result.early.item() == 0.0
    assert result.false_alarm.item() == 0.0
    result.total.backward()
    legacy.backward()
    assert torch.allclose(logits.grad, reference_logits.grad)


def test_event_likelihood_objective_matches_declared_components() -> None:
    labels = torch.tensor(
        [[0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.long
    )
    probabilities = torch.tensor(
        [
            [0.5, 0.5, 0.8, 0.8, 0.8, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    logits = _logits_from_probabilities(probabilities)
    config = TrainingConfig(
        loss_objective="event_likelihood",
        confirmation_steps=2,
        confirmation_temperature=1.0,
        time_step_s=0.1,
        detection_loss_weight=0.2,
        delay_loss_weight=0.3,
        false_alarm_loss_weight=0.4,
        early_detection_deadline_s=1.0,
    )

    result = binary_detection_objective(
        logits,
        labels,
        nn.BCEWithLogitsLoss(reduction="none"),
        config,
        event_loss_scale=0.5,
    )

    # Equal attack logits give Soft-Min hazards [0.8, 0.8], so the first-alarm
    # probabilities are q=[0, 0.8, 0.16] and P(det)=0.96.
    expected_detection = -math.log(0.96)
    expected_early = (0.1 * 0.8 + 0.2 * 0.16) / 0.96
    # Normal intervals have no-alarm probabilities 0.5 (length 2) and
    # 0.9**5 (length 6). The one-step suffix is shorter than K and excluded.
    expected_false_alarm = (
        -math.log(0.5) / 2.0 - math.log(0.9**5) / 6.0
    ) / 2.0
    expected_total = result.base_bce + 0.5 * (
        0.2 * expected_detection
        + 0.3 * expected_early
        + 0.4 * expected_false_alarm
    )

    assert result.attack_events == 1
    assert result.unobservable_attack_events == 0
    assert result.normal_intervals == 2
    assert result.detection.item() == pytest.approx(expected_detection, rel=1.0e-5)
    assert result.early.item() == pytest.approx(expected_early, rel=1.0e-5)
    assert result.false_alarm.item() == pytest.approx(
        expected_false_alarm, rel=1.0e-5
    )
    assert torch.allclose(result.total, expected_total)
    result.total.backward()
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))


def test_attack_shorter_than_k_is_excluded_from_event_terms() -> None:
    labels = torch.tensor([[0, 1, 0, 0]], dtype=torch.long)
    logits = torch.zeros((1, 4, 1), dtype=torch.float32, requires_grad=True)
    config = TrainingConfig(
        loss_objective="event_likelihood",
        confirmation_steps=2,
        detection_loss_weight=1.0,
        delay_loss_weight=1.0,
        false_alarm_loss_weight=0.0,
    )

    result = binary_detection_objective(
        logits,
        labels,
        nn.BCEWithLogitsLoss(reduction="none"),
        config,
    )

    assert result.attack_events == 0
    assert result.unobservable_attack_events == 1
    assert result.detection.item() == 0.0
    assert result.early.item() == 0.0
    assert torch.equal(result.total, result.base_bce)


def test_normal_interval_loss_uses_whole_interval_no_alarm_probability() -> None:
    labels = torch.zeros((1, 4), dtype=torch.long)
    probabilities = torch.tensor([[0.1, 0.9, 0.9, 0.1]], dtype=torch.float32)
    logits = _logits_from_probabilities(probabilities)
    config = TrainingConfig(
        loss_objective="event_likelihood",
        confirmation_steps=2,
        confirmation_temperature=1.0,
        detection_loss_weight=0.0,
        delay_loss_weight=0.0,
        false_alarm_loss_weight=0.2,
    )

    result = binary_detection_objective(
        logits,
        labels,
        nn.BCEWithLogitsLoss(reduction="none"),
        config,
    )

    interval_logits = torch.logit(probabilities[0])
    hazards = _reference_soft_min_hazards(interval_logits, 2, 1.0)
    no_alarm_probability = torch.prod(1.0 - hazards).item()
    expected_false_alarm = -math.log(no_alarm_probability) / 4.0
    assert result.normal_intervals == 1
    assert result.false_alarm.item() == pytest.approx(expected_false_alarm)
    assert result.total.item() == pytest.approx(
        result.base_bce.item() + 0.2 * expected_false_alarm
    )


def test_event_loss_scale_zero_exactly_recovers_bce() -> None:
    labels = torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.long)
    logits = torch.zeros((1, 5, 1), dtype=torch.float32, requires_grad=True)
    config = TrainingConfig(
        loss_objective="event_likelihood",
        confirmation_steps=2,
    )
    result = binary_detection_objective(
        logits,
        labels,
        nn.BCEWithLogitsLoss(reduction="none"),
        config,
        event_loss_scale=0.0,
    )
    assert torch.equal(result.total, result.base_bce)


def test_checkpoint_selection_uses_only_lower_finite_validation_loss() -> None:
    assert _validation_loss_improved(0.8, 0.9)
    assert not _validation_loss_improved(0.9, 0.8)
    assert not _validation_loss_improved(0.9, 0.9)
    with pytest.raises(FloatingPointError, match="validation loss"):
        _validation_loss_improved(float("nan"), 0.9)


@pytest.mark.parametrize(
    "config, message",
    [
        (TrainingConfig(loss_objective="unknown"), "loss_objective"),
        (TrainingConfig(confirmation_steps=0), "confirmation_steps"),
        (TrainingConfig(confirmation_steps=6), "confirmation_steps"),
        (TrainingConfig(confirmation_temperature=0.0), "confirmation_temperature"),
        (TrainingConfig(time_step_s=0.0), "time_step_s"),
        (TrainingConfig(detection_loss_weight=float("nan")), "detection_loss_weight"),
        (TrainingConfig(delay_loss_weight=float("nan")), "delay_loss_weight"),
        (TrainingConfig(false_alarm_loss_weight=-1.0), "false_alarm_loss_weight"),
        (TrainingConfig(event_ramp_epochs=-1), "event_ramp_epochs"),
        (
            TrainingConfig(early_detection_deadline_s=0.0),
            "early_detection_deadline_s",
        ),
        (TrainingConfig(learning_rate=float("nan")), "learning_rate"),
        (TrainingConfig(weight_decay=float("inf")), "weight_decay"),
        (
            TrainingConfig(
                loss_objective="event_likelihood",
                detection_loss_weight=0.0,
                delay_loss_weight=0.0,
                false_alarm_loss_weight=0.0,
            ),
            "positive detection, early-detection, or false-alarm",
        ),
    ],
)
def test_event_likelihood_training_config_rejects_invalid_values(
    config: TrainingConfig,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()

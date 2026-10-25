from __future__ import annotations

import torch
from torch import nn

from isac_spoofing.models.torch_models import build_detector
from isac_spoofing.training import (
    ATTACK_PROBABILITY_THRESHOLD,
    attack_probabilities_from_logits,
    binary_predictions_from_logits,
)


def test_fixed_probability_threshold_decision() -> None:
    logits = torch.tensor(
        [
            [-1.0],
            [1.0],
            [0.0],
        ],
        dtype=torch.float32,
    )
    predictions = binary_predictions_from_logits(logits)
    probabilities = attack_probabilities_from_logits(logits)
    assert ATTACK_PROBABILITY_THRESHOLD == 0.5
    assert predictions.tolist() == [0, 1, 1]
    assert torch.allclose(
        probabilities,
        torch.tensor([0.26894143, 0.7310586, 0.5]),
    )


def test_legacy_two_logit_checkpoint_decision_is_supported() -> None:
    logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    assert binary_predictions_from_logits(logits).tolist() == [0, 1]


def test_probability_threshold_rejects_non_binary_logits() -> None:
    logits = torch.zeros((2, 3), dtype=torch.float32)
    try:
        binary_predictions_from_logits(logits)
    except ValueError as error:
        assert "final dimension 1" in str(error)
    else:
        raise AssertionError("non-binary logits must be rejected")


def test_mlp_and_gru_train_with_one_attack_logit() -> None:
    features = torch.randn(2, 5, 15)
    targets = torch.randint(0, 2, (2, 5, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
    for model_type in ("mlp", "gru"):
        model = build_detector(model_type, input_dim=15, output_dim=1)
        logits = model(features)
        assert logits.shape == (2, 5, 1)
        loss = criterion(logits, targets)
        assert torch.isfinite(loss)
        loss.backward()


def test_gru_optional_input_encoder_is_causal_and_trainable() -> None:
    model = build_detector(
        "gru",
        input_dim=10,
        hidden_dim=64,
        num_layers=1,
        output_dim=1,
        input_encoder_dim=16,
    )
    features = torch.randn(2, 7, 10, requires_grad=True)
    logits = model(features)

    assert logits.shape == (2, 7, 1)
    assert model.gru.input_size == 16
    assert sum(parameter.numel() for parameter in model.parameters()) == 15_985

    changed_future = features.detach().clone()
    changed_future[:, 4:] += 100.0
    future_logits = model(changed_future)
    assert torch.allclose(logits[:, :4], future_logits[:, :4], atol=1.0e-6)

    logits.sum().backward()
    encoder_weight = model.input_encoder[0].weight
    assert encoder_weight.grad is not None
    assert torch.isfinite(encoder_weight.grad).all()


def test_original_gru_remains_the_default_architecture() -> None:
    model = build_detector("gru", input_dim=10, hidden_dim=64, output_dim=1)

    assert isinstance(model.input_encoder, nn.Identity)
    assert model.gru.input_size == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 14_657

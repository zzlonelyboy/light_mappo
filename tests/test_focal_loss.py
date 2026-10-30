"""Unit checks for Binary Focal Loss implementation.

Run: python tests/test_focal_loss.py
"""

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from isac_spoofing.training import (
    TrainingConfig,
    _binary_focal_loss,
    binary_detection_objective,
)


def _make_labels_and_logits(
    batch_size: int = 2,
    time_steps: int = 100,
    attack_start: int = 30,
    attack_end: int = 70,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create labels and logits with one attack event per batch item."""
    labels = torch.zeros(batch_size, time_steps, dtype=torch.long)
    logits = torch.zeros(batch_size, time_steps, 1, dtype=torch.float32)
    for b in range(batch_size):
        labels[b, attack_start:attack_end] = 1
        # Logits: positive for attack steps, negative for normal
        logits[b, :, 0] = -2.0
        logits[b, attack_start:attack_end, 0] = 1.0
    return labels, logits


def test_focal_equals_bce_when_gamma_zero_alpha_none():
    """gamma=0, alpha=None must equal ordinary BCE loss."""
    labels, logits = _make_labels_and_logits()
    logits_param = logits.clone().detach().requires_grad_(True)

    loss_focal = _binary_focal_loss(
        logits_param, labels, gamma=0.0, alpha=None,
    )

    targets = labels.to(dtype=logits.dtype).unsqueeze(-1)
    loss_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits_param, targets, reduction="mean",
    )

    assert torch.allclose(loss_focal, loss_bce, atol=1e-6), (
        f"focal(γ=0)={loss_focal.item():.8f} != BCE={loss_bce.item():.8f}"
    )
    print(f"OK: focal(γ=0, α=None)={loss_focal.item():.8f} == BCE={loss_bce.item():.8f}")


def test_focal_downweights_easy_examples():
    """As gamma increases, loss contribution from easy examples decreases."""
    labels, _ = _make_labels_and_logits()
    # Create a mix: easy (high-confidence correct) and hard (wrong) predictions
    logits = torch.zeros(2, 100, 1, dtype=torch.float32)
    # Easy positive: high logit
    logits[0, 30:50, 0] = 5.0  # very confident attack
    labels[0, 30:50] = 1
    # Hard positive: low logit
    logits[0, 50:70, 0] = 0.1  # uncertain attack
    labels[0, 50:70] = 1
    # Easy negative: strongly negative logit
    logits[0, :30, 0] = -5.0  # very confident normal
    # Hard negative: near-zero logit
    logits[1, :30, 0] = -0.1

    loss_g0 = float(_binary_focal_loss(logits, labels, gamma=0.0, alpha=None).item())
    loss_g1 = float(_binary_focal_loss(logits, labels, gamma=1.0, alpha=None).item())
    loss_g2 = float(_binary_focal_loss(logits, labels, gamma=2.0, alpha=None).item())
    loss_g3 = float(_binary_focal_loss(logits, labels, gamma=3.0, alpha=None).item())

    # Loss should decrease as gamma increases (easy examples downweighted more)
    assert loss_g0 > loss_g1 > loss_g2 > loss_g3, (
        f"Expected monotonic decrease: {loss_g0:.6f} > {loss_g1:.6f} > {loss_g2:.6f} > {loss_g3:.6f}"
    )
    print(f"OK: loss decreases with gamma: {loss_g0:.6f} → {loss_g1:.6f} → {loss_g2:.6f} → {loss_g3:.6f}")


def test_focal_loss_and_gradient_are_finite():
    """Loss and gradients must be finite for extreme logits."""
    labels, _ = _make_labels_and_logits()
    # Extreme logits (very large positive and negative)
    logits = torch.zeros(2, 100, 1, dtype=torch.float32)
    logits[0, :, 0] = 10.0   # very confident attack
    logits[1, :, 0] = -10.0  # very confident normal
    logits_param = logits.clone().detach().requires_grad_(True)

    for gamma in [0.0, 1.0, 2.0, 3.0]:
        loss = _binary_focal_loss(logits_param, labels, gamma=gamma, alpha=0.75)
        assert math.isfinite(float(loss.item())), f"γ={gamma}: loss not finite: {loss.item()}"
        loss.backward(retain_graph=True)
        assert logits_param.grad is not None
        assert torch.all(torch.isfinite(logits_param.grad)), f"γ={gamma}: gradient not finite"
        logits_param.grad.zero_()
    print("OK: loss and gradients finite for γ ∈ {0,1,2,3} with extreme logits")


def test_focal_alpha_weights_positives():
    """alpha should weight positive class differently from negative class."""
    labels, _ = _make_labels_and_logits()
    logits = torch.zeros(2, 100, 1, dtype=torch.float32)
    # All positive labels
    labels_pos = torch.ones(2, 100, dtype=torch.long)
    # All negative labels
    labels_neg = torch.zeros(2, 100, dtype=torch.long)

    # With alpha=0.9, positive loss should be weighted 0.9, negative 0.1
    loss_pos = float(_binary_focal_loss(logits, labels_pos, gamma=1.0, alpha=0.9).item())
    loss_neg = float(_binary_focal_loss(logits, labels_neg, gamma=1.0, alpha=0.9).item())
    # At z=0, p=0.5, BCE=log(2) ≈ 0.693. FL = alpha * 1 * 0.693
    # pos: 0.9 * 0.5^1 * 0.693 = 0.312
    # neg: 0.1 * 0.5^1 * 0.693 = 0.035
    assert loss_pos > loss_neg * 5, (
        f"alpha=0.9 should weight positive much more: pos={loss_pos:.6f} neg={loss_neg:.6f}"
    )
    print(f"OK: alpha weighting: pos loss={loss_pos:.6f} >> neg loss={loss_neg:.6f}")


def test_no_nan_inf():
    """Focal loss must never produce NaN or Inf."""
    labels, _ = _make_labels_and_logits()
    for gamma in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        for alpha in [None, 0.25, 0.5, 0.75, 0.9]:
            logits = torch.randn(labels.shape[0], 50, 1) * 3.0  # aligned batch size
            loss = _binary_focal_loss(logits, labels[:, :50], gamma=gamma, alpha=alpha)
            val = float(loss.item())
            assert math.isfinite(val), f"γ={gamma} α={alpha}: loss={val}"
            assert not math.isnan(val), f"γ={gamma} α={alpha}: loss is NaN"
    print("OK: no NaN/Inf for any (γ,α) combination")


def test_binary_detection_objective_focal_mode():
    """binary_detection_objective with focal produces valid result."""
    labels, logits = _make_labels_and_logits()
    config = TrainingConfig(
        loss_objective="focal",
        focal_gamma=2.0,
        focal_alpha=0.75,
        class_weighting="none",
    )
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    result = binary_detection_objective(logits, labels, criterion, config)
    assert math.isfinite(float(result.total.item())), "Total loss must be finite"
    assert float(result.total.item()) > 0, "Loss should be positive"
    print(f"OK: binary_detection_objective focal mode: loss={float(result.total.item()):.6f}")


if __name__ == "__main__":
    print("Running Binary Focal Loss unit checks...\n")
    test_focal_equals_bce_when_gamma_zero_alpha_none()
    test_focal_downweights_easy_examples()
    test_focal_loss_and_gradient_are_finite()
    test_focal_alpha_weights_positives()
    test_no_nan_inf()
    test_binary_detection_objective_focal_mode()
    print("\nAll Focal Loss unit checks passed!")

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - only used with optional dependency
    raise ImportError(
        "Install the learning extra with: pip install -e '.[learning]'"
    ) from exc


class TimeStepMLP(nn.Module):
    """Non-temporal binary baseline for one 11-D UAV feature vector."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        hidden_dims: tuple[int, ...] | None = None,
        output_dim: int = 1,
    ):
        super().__init__()
        if output_dim not in {1, 2}:
            raise ValueError("output_dim must be 1, or 2 for a legacy checkpoint")
        widths = (hidden_dim, hidden_dim) if hidden_dims is None else hidden_dims
        if not widths or any(width < 1 for width in widths):
            raise ValueError("hidden_dims must contain positive layer widths")
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for index, width in enumerate(widths):
            layers.extend((nn.Linear(previous_dim, width), nn.ReLU()))
            if index == 0:
                layers.append(nn.Dropout(0.1))
            previous_dim = width
        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return attack logits with shape ``(..., output_dim)``."""
        return self.network(features)


class CausalGRUDetector(nn.Module):
    """Causal sequence detector; output[t] only uses observations up to t."""

    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 64,
        num_layers: int = 1,
        output_dim: int = 1,
        input_encoder_dim: int = 0,
    ):
        super().__init__()
        if output_dim not in {1, 2}:
            raise ValueError("output_dim must be 1, or 2 for a legacy checkpoint")
        if input_encoder_dim < 0:
            raise ValueError("input_encoder_dim must be non-negative")
        self.input_encoder_dim = int(input_encoder_dim)
        if self.input_encoder_dim > 0:
            self.input_encoder = nn.Sequential(
                nn.Linear(input_dim, self.input_encoder_dim),
                nn.ReLU(),
            )
            gru_input_dim = self.input_encoder_dim
        else:
            self.input_encoder = nn.Identity()
            gru_input_dim = input_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return causal attack logits with shape ``(batch, time, output_dim)``."""
        encoded = self.input_encoder(sequences)
        hidden, _ = self.gru(encoded)
        return self.classifier(hidden)


def build_detector(
    model_type: str,
    input_dim: int = 11,
    hidden_dim: int = 64,
    num_layers: int = 1,
    mlp_hidden_dims: tuple[int, ...] | None = None,
    output_dim: int = 1,
    input_encoder_dim: int = 0,
) -> nn.Module:
    normalized = str(model_type).lower()
    if normalized == "mlp":
        return TimeStepMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_dims=mlp_hidden_dims,
            output_dim=output_dim,
        )
    if normalized == "gru":
        return CausalGRUDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            input_encoder_dim=input_encoder_dim,
        )
    raise ValueError("model_type must be 'mlp' or 'gru'")

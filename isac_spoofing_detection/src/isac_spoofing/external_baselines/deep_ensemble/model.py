"""Per-BS MLP models for the Deep Ensemble baseline."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MLPTrainingConfig:
    hidden_dims: tuple[int, ...] = (10, 20, 10)
    learning_rate: float = 1.0e-3
    batch_size: int = 1024
    epochs: int = 500
    patience: int = 100
    weight_decay: float = 0.0
    seed: int = 20260714

    def validate(self) -> None:
        if not self.hidden_dims or any(width < 1 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.learning_rate <= 0.0 or self.batch_size < 1:
            raise ValueError("learning rate and batch size must be positive")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "FeatureStandardizer":
        values = np.asarray(features, dtype=np.float64)
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1.0e-8, scale, 1.0)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        ).astype(np.float32)

    def to_json(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, list[float]]) -> "FeatureStandardizer":
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
        )


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Deep Ensemble training requires the optional learning dependencies"
        ) from error
    return torch, nn


def build_mlp(input_dim: int, hidden_dims: tuple[int, ...]):
    torch, nn = _torch()
    layers: list[object] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_per_bs_models(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    output_dir: Path,
    config: MLPTrainingConfig,
    device: str = "auto",
    resume: bool = True,
) -> dict[str, object]:
    """Train one independent binary MLP per base station."""

    config.validate()
    torch, nn = _torch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if train_features.ndim != 3:
        raise ValueError("features must have shape (samples,BS,features)")
    if validation_features.shape[1:] != train_features.shape[1:]:
        raise ValueError("training and validation feature dimensions must match")
    if train_labels.shape != (train_features.shape[0],):
        raise ValueError("training labels must have shape (samples,)")
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    num_bs = train_features.shape[1]
    histories: list[dict[str, object]] = []
    standardizers: list[FeatureStandardizer] = []

    for bs_index in range(num_bs):
        checkpoint_path = output_dir / f"bs_{bs_index}.pt"
        standardizer_path = output_dir / f"bs_{bs_index}_standardizer.json"
        history_path = output_dir / f"bs_{bs_index}_history.json"
        if (
            resume
            and checkpoint_path.exists()
            and standardizer_path.exists()
            and history_path.exists()
        ):
            standardizer = FeatureStandardizer.from_json(
                json.loads(standardizer_path.read_text(encoding="utf-8"))
            )
            standardizers.append(standardizer)
            histories.append(
                json.loads(history_path.read_text(encoding="utf-8"))
            )
            print(
                f"[Per-BS MLP][BS {bs_index + 1}/{num_bs}] reused checkpoint",
                flush=True,
            )
            continue
        print(
            f"[Per-BS MLP][BS {bs_index + 1}/{num_bs}] training "
            f"{train_features.shape[-1]}-{list(config.hidden_dims)}-1 model",
            flush=True,
        )
        seed = config.seed + bs_index
        _seed_everything(seed)
        standardizer = FeatureStandardizer.fit(train_features[:, bs_index])
        standardizers.append(standardizer)
        x_train = torch.from_numpy(
            standardizer.transform(train_features[:, bs_index])
        )
        y_train = torch.from_numpy(train_labels.astype(np.float32))
        x_validation = torch.from_numpy(
            standardizer.transform(validation_features[:, bs_index])
        ).to(selected_device)
        y_validation = torch.from_numpy(
            validation_labels.astype(np.float32)
        ).to(selected_device)
        generator = torch.Generator().manual_seed(seed)
        dataset = torch.utils.data.TensorDataset(x_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
        )
        model = build_mlp(train_features.shape[-1], config.hidden_dims).to(
            selected_device
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        loss_function = nn.BCEWithLogitsLoss()
        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        epoch_records: list[dict[str, float | int]] = []
        for epoch in range(config.epochs):
            model.train()
            training_loss_sum = 0.0
            training_count = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(selected_device)
                batch_labels = batch_labels.to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features).squeeze(-1)
                loss = loss_function(logits, batch_labels)
                loss.backward()
                optimizer.step()
                training_loss_sum += float(loss.detach()) * batch_labels.numel()
                training_count += batch_labels.numel()
            model.eval()
            with torch.no_grad():
                validation_logits = model(x_validation).squeeze(-1)
                validation_loss = float(
                    loss_function(validation_logits, y_validation).detach()
                )
            training_loss = training_loss_sum / max(training_count, 1)
            epoch_records.append(
                {
                    "epoch": epoch + 1,
                    "train_bce": training_loss,
                    "validation_bce": validation_loss,
                }
            )
            print(
                f"[Per-BS MLP][BS {bs_index + 1}/{num_bs}] "
                f"epoch {epoch + 1}/{config.epochs} "
                f"train_bce={training_loss:.6f} "
                f"validation_bce={validation_loss:.6f}",
                flush=True,
            )
            if validation_loss < best_loss - 1.0e-6:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    print(
                        f"[Per-BS MLP][BS {bs_index + 1}/{num_bs}] "
                        f"early stopping at epoch {epoch + 1}",
                        flush=True,
                    )
                    break
        if best_state is None:
            raise RuntimeError("MLP training did not produce a checkpoint")
        checkpoint = {
            "state_dict": best_state,
            "input_dim": int(train_features.shape[-1]),
            "hidden_dims": tuple(config.hidden_dims),
            "bs_index": bs_index,
            "training_config": asdict(config),
        }
        torch.save(checkpoint, checkpoint_path)
        history = {
            "bs_index": bs_index,
            "best_validation_bce": best_loss,
            "epochs_ran": len(epoch_records),
            "epochs": epoch_records,
        }
        histories.append(history)
        standardizer_path.write_text(
            json.dumps(standardizer.to_json(), indent=2),
            encoding="utf-8",
        )
        history_path.write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )
        print(
            f"[Per-BS MLP][BS {bs_index + 1}/{num_bs}] completed "
            f"best_validation_bce={best_loss:.6f}",
            flush=True,
        )
    (output_dir / "standardizers.json").write_text(
        json.dumps([item.to_json() for item in standardizers], indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(histories, indent=2),
        encoding="utf-8",
    )
    return {
        "num_bs": num_bs,
        "device": selected_device,
        "histories": histories,
    }


class PerBSModelSet:
    def __init__(
        self,
        models: list[object],
        standardizers: list[FeatureStandardizer],
        device: str,
    ) -> None:
        self.models = models
        self.standardizers = standardizers
        self.device = device

    @classmethod
    def load(cls, model_dir: Path, device: str = "auto") -> "PerBSModelSet":
        torch, _ = _torch()
        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        payload = json.loads(
            (Path(model_dir) / "standardizers.json").read_text(encoding="utf-8")
        )
        standardizers = [FeatureStandardizer.from_json(item) for item in payload]
        models = []
        for bs_index in range(len(standardizers)):
            checkpoint = torch.load(
                Path(model_dir) / f"bs_{bs_index}.pt",
                map_location=selected_device,
                weights_only=False,
            )
            model = build_mlp(
                int(checkpoint["input_dim"]),
                tuple(checkpoint["hidden_dims"]),
            ).to(selected_device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            models.append(model)
        return cls(models, standardizers, selected_device)

    def predict_proba(self, features: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        torch, _ = _torch()
        values = np.asarray(features, dtype=np.float32)
        if values.shape[-2] != len(self.models):
            raise ValueError("feature BS dimension does not match loaded models")
        flat = values.reshape(-1, values.shape[-2], values.shape[-1])
        probabilities = np.empty((flat.shape[0], len(self.models)), dtype=np.float32)
        with torch.no_grad():
            for bs_index, (model, standardizer) in enumerate(
                zip(self.models, self.standardizers)
            ):
                standardized = standardizer.transform(flat[:, bs_index])
                chunks: list[np.ndarray] = []
                for start in range(0, standardized.shape[0], batch_size):
                    tensor = torch.from_numpy(
                        standardized[start : start + batch_size]
                    ).to(self.device)
                    chunks.append(
                        torch.sigmoid(model(tensor).squeeze(-1))
                        .cpu()
                        .numpy()
                    )
                probabilities[:, bs_index] = np.concatenate(chunks)
        return probabilities.reshape(values.shape[:-2] + (len(self.models),))

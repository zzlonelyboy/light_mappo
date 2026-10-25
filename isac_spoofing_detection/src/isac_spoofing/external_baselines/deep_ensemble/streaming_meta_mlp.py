"""Full-train streaming MLP for multi-BS probability fusion."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .dataset import discover_sidecars
from .model import PerBSModelSet


@dataclass(frozen=True)
class StreamingMetaMLPConfig:
    hidden_dim: int = 4
    learning_rate: float = 5.0e-3
    batch_size: int = 8192
    epochs: int = 500
    patience: int = 100
    weight_decay: float = 0.0
    seed: int = 20260714

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.batch_size < 1:
            raise ValueError("hidden_dim and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.epochs < 1 or self.patience < 1:
            raise ValueError("epochs and patience must be positive")


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Streaming meta-MLP training requires the learning dependencies"
        ) from error
    return torch, nn


def build_streaming_meta_mlp(input_dim: int, hidden_dim: int):
    _, nn = _torch()
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _ = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(path: Path, payload: dict[str, object]) -> None:
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def generate_meta_probability_cache(
    feature_dir: Path,
    output_dir: Path,
    per_bs_models: PerBSModelSet,
    splits: tuple[str, ...] = ("train", "validation"),
    force: bool = False,
    max_episodes_per_split: int | None = None,
) -> dict[str, object]:
    """Generate sharded four-BS probabilities without dropping slot samples."""

    summary: dict[str, object] = {"splits": {}}
    for split in splits:
        processed_episodes = 0
        processed_samples = 0
        records: list[dict[str, object]] = []
        for source_path in discover_sidecars(feature_dir, split):
            if (
                max_episodes_per_split is not None
                and processed_episodes >= max_episodes_per_split
            ):
                break
            target_path = Path(output_dir) / split / source_path.name
            with np.load(source_path, allow_pickle=False) as source:
                available = int(source["episode_ids"].shape[0])
                take = available
                if max_episodes_per_split is not None:
                    take = min(
                        take,
                        max_episodes_per_split - processed_episodes,
                    )
                if take < 1:
                    break
                if target_path.exists() and not force:
                    with np.load(target_path, allow_pickle=False) as cached:
                        cached_episodes = int(cached["episode_ids"].shape[0])
                        cached_samples = int(cached["labels"].size)
                    if cached_episodes == take:
                        processed_episodes += cached_episodes
                        processed_samples += cached_samples
                        records.append(
                            {
                                "shard": target_path.name,
                                "episodes": cached_episodes,
                                "samples": cached_samples,
                                "status": "reused",
                            }
                        )
                        continue
                features = np.asarray(source["features"][:take], dtype=np.float32)
                labels = np.asarray(source["labels"][:take], dtype=np.float32)
                episode_ids = np.asarray(source["episode_ids"][:take])
            print(
                f"[Meta cache][{split}] {source_path.name} "
                f"episodes={take} samples={labels.size}",
                flush=True,
            )
            probabilities = per_bs_models.predict_proba(features)
            if probabilities.shape[:-1] != labels.shape:
                raise ValueError("cached probability and label shapes do not match")
            _atomic_save_npz(
                target_path,
                probabilities=probabilities.astype(np.float32),
                labels=labels,
                episode_ids=episode_ids,
            )
            processed_episodes += take
            processed_samples += int(labels.size)
            records.append(
                {
                    "shard": target_path.name,
                    "episodes": int(take),
                    "samples": int(labels.size),
                    "status": "generated",
                }
            )
        if not records:
            raise ValueError(f"no probability cache was generated for {split}")
        summary["splits"][split] = {
            "episodes": int(processed_episodes),
            "samples": int(processed_samples),
            "shards": records,
        }
    _atomic_write_json(Path(output_dir) / "cache_summary.json", summary)
    return summary


def discover_probability_cache(cache_dir: Path, split: str) -> tuple[Path, ...]:
    paths = tuple(sorted((Path(cache_dir) / split).glob("shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"no meta-probability cache found for {split}")
    return paths


def _streaming_standardizer(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray, int]:
    total = np.zeros(4, dtype=np.float64)
    total_square = np.zeros(4, dtype=np.float64)
    count = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["probabilities"], dtype=np.float64).reshape(-1, 4)
        total += np.sum(values, axis=0)
        total_square += np.sum(values * values, axis=0)
        count += values.shape[0]
    if count < 1:
        raise ValueError("the probability cache contains no training samples")
    mean = total / count
    variance = np.maximum(total_square / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return mean.astype(np.float32), scale.astype(np.float32), int(count)


def _evaluate_bce(
    model,
    paths: tuple[Path, ...],
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
    device: str,
) -> tuple[float, float, int]:
    torch, nn = _torch()
    loss_function = nn.BCEWithLogitsLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    count = 0
    model.eval()
    with torch.no_grad():
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["probabilities"], dtype=np.float32).reshape(-1, 4)
                targets = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
            values = (values - mean) / scale
            for start in range(0, targets.size, batch_size):
                stop = min(start + batch_size, targets.size)
                batch_values = torch.from_numpy(values[start:stop]).to(device)
                batch_targets = torch.from_numpy(targets[start:stop]).to(device)
                logits = model(batch_values).squeeze(-1)
                loss_sum += float(loss_function(logits, batch_targets).detach())
                correct += int(
                    torch.sum((logits >= 0.0) == (batch_targets >= 0.5)).detach()
                )
                count += int(batch_targets.numel())
    return loss_sum / max(count, 1), correct / max(count, 1), count


def train_streaming_meta_mlp(
    cache_dir: Path,
    output_dir: Path,
    config: StreamingMetaMLPConfig,
    decision_threshold: float = 0.5,
    device: str = "auto",
    resume: bool = True,
) -> dict[str, object]:
    """Train one 4-4-1 MLP by traversing every cached train sample."""

    config.validate()
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError("decision_threshold must lie in (0,1)")
    output_dir = Path(output_dir)
    selection_path = output_dir / "selection.json"
    model_path = output_dir / "selected_ensemble.pt"
    state_path = output_dir / "training_state.pt"
    if resume and selection_path.exists() and model_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    train_paths = discover_probability_cache(cache_dir, "train")
    validation_paths = discover_probability_cache(cache_dir, "validation")
    mean, scale, train_sample_count = _streaming_standardizer(train_paths)
    torch, nn = _torch()
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    _seed_everything(config.seed)
    model = build_streaming_meta_mlp(4, config.hidden_dim).to(selected_device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_accuracy = 0.0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    start_epoch = 0
    validation_sample_count = 0
    for path in validation_paths:
        with np.load(path, allow_pickle=False) as data:
            validation_sample_count += int(data["labels"].size)
    if resume and state_path.exists():
        training_state = torch.load(
            state_path,
            map_location=selected_device,
            weights_only=False,
        )
        if training_state["training_config"] != asdict(config):
            raise ValueError(
                "the saved meta-MLP state uses a different configuration. "
                "Use --force to start a new run"
            )
        model.load_state_dict(training_state["state_dict"])
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        best_loss = float(training_state["best_validation_bce"])
        best_accuracy = float(training_state["best_validation_accuracy"])
        best_state = training_state["best_state_dict"]
        epochs_without_improvement = int(
            training_state["epochs_without_improvement"]
        )
        history = list(training_state["history"])
        start_epoch = int(training_state["completed_epochs"])
        rng.bit_generator.state = training_state["numpy_rng_state"]
        print(
            f"[Streaming meta MLP] Resuming after epoch {start_epoch}",
            flush=True,
        )
    print(
        "[Streaming meta MLP] Training 4-"
        f"{config.hidden_dim}-1 model on all {train_sample_count} train samples",
        flush=True,
    )
    for epoch in range(start_epoch, config.epochs):
        model.train()
        training_loss_sum = 0.0
        training_count = 0
        for shard_index in rng.permutation(len(train_paths)):
            path = train_paths[int(shard_index)]
            with np.load(path, allow_pickle=False) as data:
                values = np.asarray(data["probabilities"], dtype=np.float32).reshape(-1, 4)
                targets = np.asarray(data["labels"], dtype=np.float32).reshape(-1)
            values = (values - mean) / scale
            order = rng.permutation(targets.size)
            for start in range(0, targets.size, config.batch_size):
                indices = order[start : start + config.batch_size]
                batch_values = torch.from_numpy(values[indices]).to(selected_device)
                batch_targets = torch.from_numpy(targets[indices]).to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_values).squeeze(-1)
                loss = loss_function(logits, batch_targets)
                loss.backward()
                optimizer.step()
                training_loss_sum += float(loss.detach()) * batch_targets.numel()
                training_count += int(batch_targets.numel())
        train_bce = training_loss_sum / max(training_count, 1)
        validation_bce, validation_accuracy, validation_count = _evaluate_bce(
            model,
            validation_paths,
            mean,
            scale,
            config.batch_size,
            selected_device,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_bce": train_bce,
                "validation_bce": validation_bce,
                "validation_step_accuracy_at_0p5": validation_accuracy,
            }
        )
        print(
            f"[Streaming meta MLP] epoch {epoch + 1}/{config.epochs} "
            f"train_bce={train_bce:.6f} "
            f"validation_bce={validation_bce:.6f} "
            f"step_accuracy@0.50={validation_accuracy:.6f}",
            flush=True,
        )
        should_stop = False
        if validation_bce < best_loss - 1.0e-6:
            best_loss = validation_bce
            best_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                should_stop = True
                print(
                    f"[Streaming meta MLP] early stopping at epoch {epoch + 1}",
                    flush=True,
                )
        _atomic_torch_save(
            state_path,
            {
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_bce": float(best_loss),
                "best_validation_accuracy": float(best_accuracy),
                "best_state_dict": best_state,
                "epochs_without_improvement": int(epochs_without_improvement),
                "completed_epochs": int(epoch + 1),
                "numpy_rng_state": rng.bit_generator.state,
                "training_config": asdict(config),
                "history": history,
            },
        )
        _atomic_write_json(
            output_dir / "training_progress.json",
            {
                "completed_epochs": int(epoch + 1),
                "best_validation_bce": float(best_loss),
                "epochs_without_improvement": int(epochs_without_improvement),
                "train_samples_per_epoch": int(train_sample_count),
                "validation_samples": int(validation_sample_count),
            },
        )
        if should_stop:
            break
    if best_state is None:
        raise RuntimeError("streaming meta MLP did not produce a checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": best_state,
        "input_dim": 4,
        "hidden_dim": int(config.hidden_dim),
        "mean": mean,
        "scale": scale,
        "training_config": asdict(config),
    }
    torch.save(checkpoint, model_path)
    history_payload = {
        "candidate": "mlp_full_train_streaming",
        "train_samples_per_epoch": int(train_sample_count),
        "validation_samples": int(validation_sample_count),
        "best_validation_bce": float(best_loss),
        "best_validation_step_accuracy_at_0p5": float(best_accuracy),
        "epochs_ran": len(history),
        "epochs": history,
    }
    _atomic_write_json(output_dir / "training_history.json", history_payload)
    selection = {
        "selection_metric": "validation_binary_cross_entropy",
        "decision_threshold": float(decision_threshold),
        "model_format": "torch_streaming_meta_mlp",
        "model_file": model_path.name,
        "selected": {
            "candidate": "mlp_full_train_streaming",
            "average_bce": float(best_loss),
            "step_accuracy_at_fixed_threshold": float(best_accuracy),
            "threshold": float(decision_threshold),
        },
        "all_results": [
            {
                "candidate": "mlp_full_train_streaming",
                "average_bce": float(best_loss),
                "step_accuracy_at_fixed_threshold": float(best_accuracy),
            }
        ],
    }
    _atomic_write_json(selection_path, selection)
    return selection


class StreamingMetaMLPPredictor:
    """Scikit-compatible probability interface for the PyTorch meta MLP."""

    def __init__(self, model, mean: np.ndarray, scale: np.ndarray, device: str):
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.device = device
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def predict_proba(self, values: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        torch, _ = _torch()
        inputs = np.asarray(values, dtype=np.float32)
        if inputs.ndim != 2 or inputs.shape[1] != self.mean.size:
            raise ValueError("meta-MLP inputs must have shape (samples,BS)")
        standardized = (inputs - self.mean) / self.scale
        positive_chunks: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0, inputs.shape[0], batch_size):
                tensor = torch.from_numpy(
                    standardized[start : start + batch_size]
                ).to(self.device)
                positive_chunks.append(
                    torch.sigmoid(self.model(tensor).squeeze(-1)).cpu().numpy()
                )
        positive = np.concatenate(positive_chunks).astype(np.float64)
        return np.column_stack((1.0 - positive, positive))


def load_streaming_meta_mlp(path: Path, device: str = "auto") -> StreamingMetaMLPPredictor:
    torch, _ = _torch()
    selected_device = (
        "cuda" if device == "auto" and torch.cuda.is_available() else
        "cpu" if device == "auto" else device
    )
    checkpoint = torch.load(
        Path(path),
        map_location=selected_device,
        weights_only=False,
    )
    model = build_streaming_meta_mlp(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(selected_device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return StreamingMetaMLPPredictor(
        model,
        np.asarray(checkpoint["mean"], dtype=np.float32),
        np.asarray(checkpoint["scale"], dtype=np.float32),
        selected_device,
    )

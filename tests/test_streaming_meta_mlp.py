from __future__ import annotations

import json

import numpy as np

from isac_spoofing.external_baselines.deep_ensemble.streaming_meta_mlp import (
    StreamingMetaMLPConfig,
    generate_meta_probability_cache,
    load_streaming_meta_mlp,
    train_streaming_meta_mlp,
)


class DummyPerBSModels:
    def predict_proba(self, features):
        values = np.asarray(features, dtype=np.float32)
        return np.full(values.shape[:-2] + (values.shape[-2],), 0.5, dtype=np.float32)


def _write_feature_shard(root, split: str) -> None:
    target = root / split / "shard_00000.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    features = np.zeros((2, 3, 2, 4, 10), dtype=np.float32)
    labels = (np.arange(12).reshape(2, 3, 2) % 2).astype(np.float32)
    np.savez_compressed(
        target,
        features=features,
        labels=labels,
        episode_ids=np.asarray([f"{split}_0", f"{split}_1"]),
    )


def test_streaming_meta_mlp_traverses_all_cached_samples(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    _write_feature_shard(feature_dir, "train")
    _write_feature_shard(feature_dir, "validation")
    cache_dir = tmp_path / "cache"
    summary = generate_meta_probability_cache(
        feature_dir=feature_dir,
        output_dir=cache_dir,
        per_bs_models=DummyPerBSModels(),
    )
    assert summary["splits"]["train"]["episodes"] == 2
    assert summary["splits"]["train"]["samples"] == 12
    assert summary["splits"]["validation"]["samples"] == 12

    selection_dir = tmp_path / "selection"
    selection = train_streaming_meta_mlp(
        cache_dir=cache_dir,
        output_dir=selection_dir,
        config=StreamingMetaMLPConfig(
            hidden_dim=4,
            learning_rate=0.005,
            batch_size=4,
            epochs=2,
            patience=2,
            seed=7,
        ),
        device="cpu",
    )
    assert selection["model_format"] == "torch_streaming_meta_mlp"
    history = json.loads(
        (selection_dir / "training_history.json").read_text(encoding="utf-8")
    )
    assert history["train_samples_per_epoch"] == 12
    assert history["validation_samples"] == 12

    predictor = load_streaming_meta_mlp(
        selection_dir / "selected_ensemble.pt",
        device="cpu",
    )
    probabilities = predictor.predict_proba(np.full((5, 4), 0.5, dtype=np.float32))
    assert probabilities.shape == (5, 2)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)

    (selection_dir / "selection.json").unlink()
    (selection_dir / "selected_ensemble.pt").unlink()
    resumed = train_streaming_meta_mlp(
        cache_dir=cache_dir,
        output_dir=selection_dir,
        config=StreamingMetaMLPConfig(
            hidden_dim=4,
            learning_rate=0.005,
            batch_size=4,
            epochs=2,
            patience=2,
            seed=7,
        ),
        device="cpu",
        resume=True,
    )
    assert resumed["selected"]["candidate"] == "mlp_full_train_streaming"

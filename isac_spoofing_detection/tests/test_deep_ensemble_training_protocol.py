from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from isac_spoofing.cli import run_deep_ensemble_baseline as cli


def test_full_train_meta_mlp_uses_all_cached_slots(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_collect_slot_samples(
        feature_dir,
        split,
        max_samples,
        seed,
        predicate=None,
    ):
        del feature_dir, seed, predicate
        calls.append((split, max_samples))
        features = np.zeros((max_samples, 4, 10), dtype=np.float32)
        labels = (np.arange(max_samples) % 2).astype(np.float32)
        return features, labels

    trained: dict[str, object] = {}

    def fake_train_per_bs_models(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        **kwargs,
    ):
        del train_labels, validation_labels, kwargs
        trained["per_bs_train"] = train_features.shape[0]
        trained["per_bs_validation"] = validation_features.shape[0]

    class DummyPerBSModels:
        pass

    def fake_generate_meta_probability_cache(**kwargs):
        trained["cache_splits"] = kwargs["splits"]
        return {
            "splits": {
                "train": {"episodes": 4000, "samples": 19_200_000},
                "validation": {"episodes": 500, "samples": 2_400_000},
            }
        }

    def fake_train_streaming_meta_mlp(**kwargs):
        trained["meta_cache_dir"] = kwargs["cache_dir"]
        return {
            "selected": {"candidate": "mlp_full_train_streaming"},
            "selection_metric": "validation_binary_cross_entropy",
            "decision_threshold": 0.5,
        }

    monkeypatch.setattr(cli, "collect_slot_samples", fake_collect_slot_samples)
    monkeypatch.setattr(cli, "train_per_bs_models", fake_train_per_bs_models)
    monkeypatch.setattr(
        cli,
        "PerBSModelSet",
        SimpleNamespace(load=lambda *args, **kwargs: DummyPerBSModels()),
    )
    monkeypatch.setattr(
        cli,
        "generate_meta_probability_cache",
        fake_generate_meta_probability_cache,
    )
    monkeypatch.setattr(
        cli,
        "train_streaming_meta_mlp",
        fake_train_streaming_meta_mlp,
    )

    config = {
        "training": {
            "seed": 17,
            "device": "cpu",
            "max_train_samples": 12,
            "max_validation_samples": 6,
            "decision_threshold": 0.5,
            "mlp": {
                "hidden_dims": [10, 20, 10],
                "learning_rate": 0.005,
                "batch_size": 4,
                "epochs": 2,
                "patience": 2,
                "weight_decay": 0.0,
                "seed": 17,
            },
            "meta_mlp": {
                "hidden_dim": 4,
                "learning_rate": 0.005,
                "batch_size": 8,
                "epochs": 2,
                "patience": 2,
                "weight_decay": 0.0,
                "seed": 17,
            },
        }
    }
    run_dir = tmp_path / "run"
    cli._run_train(
        config=config,
        feature_dir=tmp_path / "features",
        run_dir=run_dir,
        pretrained_per_bs_dir=None,
        quick=False,
        force=False,
    )

    assert calls == [("validation", 6), ("train", 12)]
    assert trained["per_bs_train"] == 12
    assert trained["per_bs_validation"] == 6
    assert trained["cache_splits"] == ("train", "validation")
    assert trained["meta_cache_dir"] == run_dir / "meta_probability_cache"
    completion = json.loads(
        (run_dir / "training_complete.json").read_text(encoding="utf-8")
    )
    assert completion["training_protocol"] == {
        "per_bs_training_split": "train",
        "per_bs_early_stopping_split": "validation",
        "meta_training_split": "train",
        "meta_early_stopping_split": "validation",
        "meta_training_uses_all_slots": True,
        "test_split_used_during_training": False,
    }
    assert completion["sample_counts"]["meta_training"] == 19_200_000
    assert completion["sample_counts"]["meta_validation"] == 2_400_000

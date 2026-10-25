"""Run the adapted Deep Ensemble baseline as a resumable pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from isac_spoofing.external_baselines.common.pathloss import A2GPathLossConfig
from isac_spoofing.external_baselines.deep_ensemble.dataset import (
    collect_slot_samples,
    generate_feature_shards,
)
from isac_spoofing.external_baselines.deep_ensemble.evaluation import (
    evaluate_selected_model,
)
from isac_spoofing.external_baselines.deep_ensemble.model import (
    MLPTrainingConfig,
    PerBSModelSet,
    train_per_bs_models,
)
from isac_spoofing.external_baselines.deep_ensemble.streaming_meta_mlp import (
    StreamingMetaMLPConfig,
    generate_meta_probability_cache,
    train_streaming_meta_mlp,
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_environment(dataset_dir: Path) -> tuple[dict[str, object], np.ndarray, float, int]:
    manifest = _load_json(Path(dataset_dir) / "manifest.json")
    config = manifest["config"]
    anchors = np.asarray(config["isac"]["anchors_m"], dtype=np.float64)
    carrier_ghz = float(config["isac"]["carrier_hz"]) / 1.0e9
    base_seed = int(manifest["base_seed"])
    return manifest, anchors, carrier_ghz, base_seed


def _pathloss_config(payload: dict[str, object], carrier_ghz: float) -> A2GPathLossConfig:
    values = dict(payload)
    values["carrier_ghz"] = carrier_ghz
    return A2GPathLossConfig(**values)


def _mlp_config(payload: dict[str, object]) -> MLPTrainingConfig:
    values = dict(payload)
    values["hidden_dims"] = tuple(int(value) for value in values["hidden_dims"])
    return MLPTrainingConfig(**values)


def _meta_mlp_config(payload: dict[str, object]) -> StreamingMetaMLPConfig:
    return StreamingMetaMLPConfig(**dict(payload))


def _run_generate(
    config: dict[str, object],
    dataset_dir: Path,
    feature_dir: Path,
    max_episodes_per_split: int | None,
    quick: bool,
) -> None:
    _, anchors, carrier_ghz, base_seed = _load_environment(dataset_dir)
    pathloss = _pathloss_config(config["pathloss"], carrier_ghz)
    if quick:
        pathloss = replace(pathloss, packet_samples=min(pathloss.packet_samples, 16))
        if max_episodes_per_split is None:
            max_episodes_per_split = 2
    summary = generate_feature_shards(
        dataset_dir=dataset_dir,
        output_dir=feature_dir,
        splits=("train", "validation", "test"),
        bs_positions_m=anchors,
        config=pathloss,
        base_seed=base_seed,
        time_chunk_steps=int(config["feature_generation"]["time_chunk_steps"]),
        max_episodes_per_split=max_episodes_per_split,
    )
    print(json.dumps(summary, indent=2))


def _run_train(
    config: dict[str, object],
    feature_dir: Path,
    run_dir: Path,
    pretrained_per_bs_dir: Path | None,
    quick: bool,
    force: bool,
) -> None:
    complete_path = run_dir / "training_complete.json"
    if complete_path.exists() and not force:
        print(f"Reusing completed training stage at {complete_path}")
        return
    training = config["training"]
    seed = int(training["seed"])
    max_train_samples = int(training["max_train_samples"])
    max_validation_samples = int(training["max_validation_samples"])
    mlp_config = _mlp_config(training["mlp"])
    meta_mlp_config = _meta_mlp_config(training["meta_mlp"])
    print(
        "[Train] Full-train MLP ensemble. Four frozen BS-level MLPs feed one "
        "streaming 4-4-1 meta MLP.",
        flush=True,
    )
    if quick:
        max_train_samples = min(max_train_samples, 3000)
        max_validation_samples = min(max_validation_samples, 1500)
        mlp_config = replace(
            mlp_config,
            epochs=min(mlp_config.epochs, 2),
            patience=min(mlp_config.patience, 2),
            batch_size=min(mlp_config.batch_size, 256),
        )
        meta_mlp_config = replace(
            meta_mlp_config,
            epochs=min(meta_mlp_config.epochs, 2),
            patience=min(meta_mlp_config.patience, 2),
            batch_size=min(meta_mlp_config.batch_size, 256),
        )

    print("[Train] Loading validation samples", flush=True)
    validation_features, validation_labels = collect_slot_samples(
        feature_dir,
        "validation",
        max_samples=max_validation_samples,
        seed=seed + 1,
    )

    if pretrained_per_bs_dir is None:
        print("[Train] Loading BS-level training samples", flush=True)
        train_features, train_labels = collect_slot_samples(
            feature_dir,
            "train",
            max_samples=max_train_samples,
            seed=seed,
        )
        model_dir = run_dir / "per_bs_models"
        train_per_bs_models(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            output_dir=model_dir,
            config=mlp_config,
            device=str(training["device"]),
            resume=not force,
        )
    else:
        model_dir = Path(pretrained_per_bs_dir)
        print(f"[Train] Reusing BS-level MLPs from {model_dir}", flush=True)

    per_bs_models = PerBSModelSet.load(model_dir, device=str(training["device"]))
    cache_dir = run_dir / "meta_probability_cache"
    print(
        "[Train] Generating four-BS probabilities for every train and "
        "validation slot",
        flush=True,
    )
    cache_summary = generate_meta_probability_cache(
        feature_dir=feature_dir,
        output_dir=cache_dir,
        per_bs_models=per_bs_models,
        splits=("train", "validation"),
        force=force,
        max_episodes_per_split=2 if quick else None,
    )
    selection_dir = run_dir / "selection"
    print("[Train] Training the full-train streaming meta MLP", flush=True)
    selection = train_streaming_meta_mlp(
        cache_dir=cache_dir,
        output_dir=selection_dir,
        config=meta_mlp_config,
        decision_threshold=float(training["decision_threshold"]),
        device=str(training["device"]),
        resume=not force,
    )
    complete_path.parent.mkdir(parents=True, exist_ok=True)
    complete_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_candidate": selection["selected"]["candidate"],
                "selection_metric": selection["selection_metric"],
                "decision_threshold": selection["decision_threshold"],
                "per_bs_model_dir": str(model_dir.resolve()),
                "training_protocol": {
                    "per_bs_training_split": "train",
                    "per_bs_early_stopping_split": "validation",
                    "meta_training_split": "train",
                    "meta_early_stopping_split": "validation",
                    "meta_training_uses_all_slots": True,
                    "test_split_used_during_training": False,
                },
                "sample_counts": {
                    "per_bs_training": int(train_labels.size)
                    if pretrained_per_bs_dir is None
                    else None,
                    "per_bs_validation": int(validation_labels.size),
                    "meta_training": int(
                        cache_summary["splits"]["train"]["samples"]
                    ),
                    "meta_validation": int(
                        cache_summary["splits"]["validation"]["samples"]
                    ),
                },
                "quick": bool(quick),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[Train] Completed. Artifacts saved to {run_dir}", flush=True)


def _run_evaluate(
    config: dict[str, object],
    feature_dir: Path,
    run_dir: Path,
    pretrained_per_bs_dir: Path | None,
    quick: bool,
) -> None:
    evaluation = config["evaluation"]
    bootstrap_samples = (
        0 if quick else int(evaluation["bootstrap_samples"])
    )
    model_dir = (
        Path(pretrained_per_bs_dir)
        if pretrained_per_bs_dir is not None
        else run_dir / "per_bs_models"
    )
    threshold_candidates = tuple(
        float(value) for value in evaluation["threshold_candidates"]
    )
    print(
        "[Evaluation][validation] selecting the event threshold",
        flush=True,
    )
    validation_payload = evaluate_selected_model(
        feature_dir=feature_dir,
        model_dir=model_dir,
        selection_dir=run_dir / "selection",
        split="validation",
        confirmation_steps=int(evaluation["confirmation_steps"]),
        output_path=run_dir / "evaluation_validation.json",
        device=str(config["training"]["device"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(config["training"]["seed"]),
        threshold_candidates=threshold_candidates,
        threshold_search_output_path=(
            run_dir / "threshold_search_validation.json"
        ),
    )
    print(json.dumps(validation_payload, indent=2, default=str))

    selected_threshold = float(validation_payload["threshold"])
    print(
        f"[Evaluation][test] using validation-selected threshold="
        f"{selected_threshold:.1f}",
        flush=True,
    )
    test_payload = evaluate_selected_model(
        feature_dir=feature_dir,
        model_dir=model_dir,
        selection_dir=run_dir / "selection",
        split="test",
        confirmation_steps=int(evaluation["confirmation_steps"]),
        output_path=run_dir / "evaluation_test.json",
        device=str(config["training"]["device"]),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=int(config["training"]["seed"]),
        decision_threshold=selected_threshold,
    )
    print(json.dumps(test_payload, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("generate", "train", "evaluate", "all"),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pretrained-per-bs-dir", type=Path)
    parser.add_argument("--max-episodes-per-split", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_json(args.config)
    if args.stage in {"generate", "all"}:
        _run_generate(
            config,
            args.dataset_dir,
            args.feature_dir,
            args.max_episodes_per_split,
            args.quick,
        )
    if args.stage in {"train", "all"}:
        _run_train(
            config,
            args.feature_dir,
            args.run_dir,
            args.pretrained_per_bs_dir,
            args.quick,
            args.force,
        )
    if args.stage in {"evaluate", "all"}:
        _run_evaluate(
            config,
            args.feature_dir,
            args.run_dir,
            args.pretrained_per_bs_dir,
            args.quick,
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from isac_spoofing.evaluation import evaluate_split, json_ready
from isac_spoofing.feature_sets import FEATURE_SETS, resolve_feature_set
from isac_spoofing.training import LOSS_OBJECTIVES, TrainingConfig, train_detector


def _parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in raw.split(",")))
    if not values or any(value < 1 or value > 5 for value in values):
        raise argparse.ArgumentTypeError(
            "K values must be comma-separated integers in [1, 5]"
        )
    return values


def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(","))
    if not values or any(value < 1 for value in values):
        raise argparse.ArgumentTypeError(
            "MLP hidden dimensions must be comma-separated positive integers"
        )
    return values


def _finite_or(value, fallback: float) -> float:
    if value is None:
        return fallback
    numeric = float(value)
    return numeric if math.isfinite(numeric) else fallback


def _selection_key(result: dict[str, object], confirmation_steps: int) -> tuple[float, ...]:
    aggregate = result["aggregate"]
    return (
        _finite_or(aggregate["event_f1"], -1.0),
        _finite_or(aggregate["event_detection_rate"], -1.0),
        -_finite_or(aggregate["false_alarms_per_uav_hour"], float("inf")),
        -_finite_or(aggregate["delay_median_s"], float("inf")),
        -float(confirmation_steps),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate one D1 ISAC spoofing detector run"
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", choices=("mlp", "gru"), default="mlp")
    parser.add_argument(
        "--feature_set",
        choices=tuple(FEATURE_SETS),
        default="full_f11",
    )
    parser.add_argument(
        "--class_weighting",
        choices=("balanced", "none"),
        default="balanced",
    )
    parser.add_argument(
        "--loss_objective",
        choices=LOSS_OBJECTIVES,
        default="bce",
        help=(
            "bce uses ordinary step-level BCE. event_likelihood adds a "
            "vectorized Soft-Min K-step hazard, first-alarm detection "
            "likelihood, detection-conditional early-delay loss, and "
            "normal-interval no-alarm likelihood."
        ),
    )
    parser.add_argument("--confirmation_temperature", type=float, default=0.1)
    parser.add_argument("--time_step_s", type=float, default=0.1)
    parser.add_argument("--detection_loss_weight", type=float, default=0.1)
    parser.add_argument("--delay_loss_weight", type=float, default=0.01)
    parser.add_argument("--false_alarm_loss_weight", type=float, default=0.05)
    parser.add_argument("--early_detection_deadline_s", type=float, default=3.0)
    parser.add_argument("--event_ramp_epochs", type=int, default=20)
    parser.add_argument("--initial_checkpoint", type=Path)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal Loss gamma parameter (>=0). gamma=0 with alpha=None reduces to BCE.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=str,
        default="none",
        help="Focal Loss alpha class weight in (0,1), or 'none' to disable",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument(
        "--input_encoder_dim",
        type=int,
        default=0,
        help=(
            "Width of an optional per-step Linear+ReLU encoder before the GRU. "
            "Use 0 to retain the original direct-input GRU."
        ),
    )
    parser.add_argument("--mlp_hidden_dims", type=_parse_hidden_dims)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--k_values", type=_parse_k_values, default=(1, 2, 3, 4, 5))
    parser.add_argument(
        "--fixed_k",
        type=int,
        choices=(1, 2, 3, 4, 5),
        help=(
            "Use a pre-specified K instead of selecting K by validation event F1. "
            "The validation K sweep is still saved for diagnostics."
        ),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Run a train/validation pilot without opening the frozen test split",
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap_samples must be non-negative")
    if args.input_encoder_dim < 0:
        parser.error("--input_encoder_dim must be non-negative")
    if args.input_encoder_dim > 0 and args.model != "gru":
        parser.error("--input_encoder_dim is supported only with --model gru")
    if args.loss_objective == "event_likelihood" and args.fixed_k is None:
        parser.error(
            f"{args.loss_objective} requires --fixed_k so training and inference "
            "use the same K"
        )
    if args.fixed_k is not None and args.fixed_k not in args.k_values:
        parser.error("--fixed_k must also be included in --k_values")
    # Parse focal-alpha: accept "none" (case-insensitive) or a float in (0,1)
    focal_alpha: float | None = None
    if args.loss_objective == "focal":
        raw = args.focal_alpha.strip().lower()
        if raw != "none":
            try:
                focal_alpha = float(args.focal_alpha)
            except ValueError:
                parser.error(
                    "--focal-alpha must be 'none' or a float in (0,1), "
                    f"got {args.focal_alpha!r}"
                )
            if not math.isfinite(focal_alpha) or not 0.0 < focal_alpha < 1.0:
                parser.error(
                    "--focal-alpha must be None or in (0, 1), "
                    f"got {focal_alpha}"
                )
        if not math.isfinite(args.focal_gamma) or args.focal_gamma < 0.0:
            parser.error("--focal-gamma must be finite and >= 0")

    output = args.out_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    dataset_feature_names = tuple(
        str(value) for value in dataset_manifest["feature_names"]
    )
    selected_feature_names = resolve_feature_set(
        args.feature_set,
        available_feature_names=dataset_feature_names,
    )
    print(
        "This run tests whether the selected detector feature set can learn from complete "
        "per-UAV sequences and produce event-level detections after validation-only "
        "model selection and the configured K-step alarm persistence rule."
    )
    training_config = TrainingConfig(
        model_type=args.model,
        class_weighting=args.class_weighting,
        loss_objective=args.loss_objective,
        confirmation_steps=args.fixed_k if args.fixed_k is not None else 5,
        confirmation_temperature=args.confirmation_temperature,
        time_step_s=args.time_step_s,
        detection_loss_weight=args.detection_loss_weight,
        delay_loss_weight=args.delay_loss_weight,
        false_alarm_loss_weight=args.false_alarm_loss_weight,
        early_detection_deadline_s=args.early_detection_deadline_s,
        event_ramp_epochs=args.event_ramp_epochs,
        initial_checkpoint=(
            None
            if args.initial_checkpoint is None
            else str(args.initial_checkpoint.resolve())
        ),
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        input_encoder_dim=args.input_encoder_dim,
        mlp_hidden_dims=args.mlp_hidden_dims,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
        num_threads=args.num_threads,
        feature_names=selected_feature_names,
    )
    training_result = train_detector(
        args.data_dir,
        output / "training",
        training_config,
    )
    checkpoint = Path(str(training_result["checkpoint"]))

    validation_sweep: dict[str, object] = {}
    best_k = args.fixed_k
    best_key = None
    for confirmation_steps in args.k_values:
        validation_result = evaluate_split(
            checkpoint,
            args.data_dir,
            "validation",
            confirmation_steps=confirmation_steps,
            output_directory=None,
            bootstrap_samples=0,
            seed=args.seed,
            device=args.device,
            save_predictions=False,
        )
        validation_sweep[str(confirmation_steps)] = validation_result["aggregate"]
        key = _selection_key(validation_result, confirmation_steps)
        if args.fixed_k is None and (best_key is None or key > best_key):
            best_key = key
            best_k = confirmation_steps
    assert best_k is not None
    with (output / "validation_k_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(validation_sweep), handle, ensure_ascii=False, indent=2)

    validation_result = evaluate_split(
        checkpoint,
        args.data_dir,
        "validation",
        confirmation_steps=best_k,
        output_directory=output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        device=args.device,
        save_predictions=True,
    )
    test_result = None
    if not args.skip_test:
        test_result = evaluate_split(
            checkpoint,
            args.data_dir,
            "test",
            confirmation_steps=best_k,
            output_directory=output,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 1,
            device=args.device,
            save_predictions=True,
        )
    final_info = {
        "schema_version": "1.0",
        "run_type": "D1_single_event_detection",
        "model": args.model,
        "model_config": training_result["model_config"],
        "class_weighting": args.class_weighting,
        "loss_objective": args.loss_objective,
        "event_likelihood_objective": training_result[
            "event_likelihood_objective"
        ],
        "feature_set": args.feature_set,
        "data_directory": str(args.data_dir.resolve()),
        "training": training_result,
        "validation_k_sweep": validation_sweep,
        "selected_confirmation_steps": best_k,
        "validation": validation_result["aggregate"],
        "selection_policy": (
            f"pre-specified fixed K={best_k}; validation K sweep is diagnostic only"
            if args.fixed_k is not None
            else (
                "maximize validation event F1, then detection rate, then minimize "
                "false alarms per UAV-hour and delay; test is read only after K freezes"
            )
        ),
    }
    if test_result is not None:
        final_info["test"] = test_result["aggregate"]
        final_info["test_time_step_metrics"] = test_result["time_step_metrics"]
    with (output / "final_info.json").open("w", encoding="utf-8") as handle:
        json.dump(json_ready(final_info), handle, ensure_ascii=False, indent=2)
    reported_split = "validation" if test_result is None else "test"
    reported_metrics = final_info[reported_split]
    notes = (
        f"Run: {output.name}\n"
        f"Model: {args.model}\n"
        f"Feature set: {args.feature_set}\n"
        f"Model config: {training_result['model_config']}\n"
        f"Class weighting: {args.class_weighting}\n"
        f"Loss objective: {args.loss_objective}\n"
        f"Event-likelihood objective: "
        f"{training_result['event_likelihood_objective']}\n"
        f"Selected K: {best_k}\n"
        f"Best epoch: {training_result['best_epoch']}\n"
        f"Reported split: {reported_split}\n"
        f"Event detection rate: {reported_metrics['event_detection_rate']}\n"
        f"Event precision: {reported_metrics['event_precision']}\n"
        f"Event F1: {reported_metrics['event_f1']}\n"
        f"False alarms per UAV-hour: "
        f"{reported_metrics['false_alarms_per_uav_hour']}\n"
        f"Mean detection delay (s): {reported_metrics['delay_mean_s']}\n"
        f"Median detection delay (s): {reported_metrics['delay_median_s']}\n"
        f"Frozen test split opened: {test_result is not None}\n"
        "Interpretation: smoke or formal status depends on the supplied dataset; "
        "performance claims require the frozen formal dataset and multiple seeds.\n"
    )
    (output / "notes.txt").write_text(notes, encoding="utf-8")
    print(
        f"The final {reported_split} event metrics show detection/false-alarm/delay "
        "behavior after validation-only selection of the model and K."
    )
    print(json.dumps(json_ready(reported_metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

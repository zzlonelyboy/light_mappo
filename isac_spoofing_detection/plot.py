from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_runs(root: Path) -> list[tuple[str, dict[str, object], dict[str, object]]]:
    runs = []
    for final_path in sorted(root.glob("run_*/final_info.json")):
        with final_path.open("r", encoding="utf-8") as handle:
            final_info = json.load(handle)
        history_path = final_path.parent / "training" / "training_history.json"
        with history_path.open("r", encoding="utf-8") as handle:
            history = json.load(handle)
        runs.append((final_path.parent.name, final_info, history))
    if not runs:
        raise FileNotFoundError(f"no run_*/final_info.json files found under {root}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot D1 detector experiment runs")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("."))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Install the learning extra with: pip install -e '.[learning]'"
        ) from exc

    runs = _load_runs(args.root)
    args.output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for run_name, final_info, history in runs:
        model = str(final_info.get("model", "model")).upper()
        label = f"{run_name} ({model})"
        epochs = [int(item["epoch"]) for item in history["epochs"]]
        train_loss = [float(item["train"]["loss"]) for item in history["epochs"]]
        validation_loss = [
            float(item["validation"]["loss"]) for item in history["epochs"]
        ]
        axes[0].plot(epochs, train_loss, label=label)
        axes[1].plot(epochs, validation_loss, label=label)
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(args.output / "Figure_1.png", dpi=200)
    plt.close(fig)

    labels = []
    detection_rates = []
    false_alarm_rates = []
    for run_name, final_info, _ in runs:
        labels.append(f"{run_name}\n{str(final_info.get('model', '')).upper()}")
        detection_rates.append(final_info["test"].get("event_detection_rate"))
        false_alarm_rates.append(final_info["test"].get("false_alarms_per_uav_hour"))
    detection_rates = [float(value) if value is not None else 0.0 for value in detection_rates]
    false_alarm_rates = [float(value) if value is not None else 0.0 for value in false_alarm_rates]
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].bar(positions, detection_rates, color="#0072B2")
    axes[0].set(
        title="Test event detection rate",
        ylabel="Detection rate",
        xticks=positions,
        xticklabels=labels,
        ylim=(0.0, 1.0),
    )
    axes[1].bar(positions, false_alarm_rates, color="#D55E00")
    axes[1].set(
        title="Test false alarms",
        ylabel="Events per UAV-hour",
        xticks=positions,
        xticklabels=labels,
    )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(args.output / "Figure_2.png", dpi=200)
    plt.close(fig)
    print(f"wrote {args.output / 'Figure_1.png'}")
    print(f"wrote {args.output / 'Figure_2.png'}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_early_detection_objective_ablation.py"
SPEC = importlib.util.spec_from_file_location("early_detection_ablation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)


def _record(objective: str) -> dict[str, object]:
    record: dict[str, object] = {
        "objective": objective,
        "seed": 7,
        "reported_split": "validation",
    }
    for metric in ABLATION.SUMMARY_METRICS:
        record[metric] = None if metric.startswith("delay_") else 0.0
    return record


def test_zero_true_positive_delay_remains_undefined_during_aggregation() -> None:
    summary = ABLATION._aggregate(
        [_record("bce"), _record("event_likelihood")]
    )

    assert len(summary) == 2
    for row in summary:
        assert row["delay_mean_s_mean"] is None
        assert row["delay_mean_s_std"] is None
        assert row["delay_median_s_mean"] is None
        assert row["delay_median_s_std"] is None
        assert row["event_recall_mean"] == 0.0
        assert ABLATION._format_metric(row["delay_mean_s_mean"]) == "undefined"


def test_plot_value_maps_only_undefined_values_to_nan() -> None:
    assert ABLATION._plot_value(0.0) == 0.0
    assert ABLATION.math.isnan(ABLATION._plot_value(None))

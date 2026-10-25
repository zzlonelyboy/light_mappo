from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "analyze_ablation_by_attack_count.py"
SPEC = importlib.util.spec_from_file_location("attack_count_ablation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_attack_uav_count_uses_distinct_uav_indices() -> None:
    episode = {
        "attack_events": [
            {"uav": 1},
            {"uav": 3},
            {"uav": 3},
        ]
    }
    assert ANALYSIS._attack_uav_count(episode) == 2


def test_summary_uses_seed_paired_delta_against_reference() -> None:
    records = []
    for feature_set, values in (
        ("reference", (0.8, 0.9)),
        ("ablation", (0.7, 0.85)),
    ):
        for seed, event_f1 in zip((1, 2), values):
            record = {
                "feature_set": feature_set,
                "feature_dim": 2,
                "seed": seed,
                "attack_uav_count": 1,
                "num_episodes": 4,
                "num_attack_events": 4,
            }
            for metric in ANALYSIS.METRICS:
                record[metric] = event_f1
            records.append(record)

    summaries = ANALYSIS._summarize(records, "reference")
    ablation = next(row for row in summaries if row["feature_set"] == "ablation")
    assert abs(ablation["event_f1_mean"] - 0.775) < 1.0e-12
    assert abs(ablation["event_f1_delta_vs_reference_mean"] + 0.075) < 1.0e-12

"""Batch cross-evaluate all D1-MA models on their own test split."""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

EXPERIMENTS = {
    "BCE": "d1ma_bce_baseline_rcs6",
    "BCE+Delay": "d1ma_feature_ablation_rcs6_delay",
    "BCE+FA": "d1ma_feature_ablation_rcs6_FA",
    "Full EL": "d1ma_feature_ablation_rcs6",
}
FEATURES = ["no_neighbor", "signed_anomaly_contrast_only", "no_bs_residual_signed_only"]
F_LABELS = {
    "no_neighbor": "F8",
    "signed_anomaly_contrast_only": "F9",
    "no_bs_residual_signed_only": "NoBS",
}
SEEDS = [20260716, 20260717, 20260718]
DATA_DIR = "data/d1_2000_multitattack_rcs6_seed20260731"

results = {}

for exp_name, exp_dir in EXPERIMENTS.items():
    print(f"\n=== {exp_name} ===", flush=True)
    results[exp_name] = {}
    for feat in FEATURES:
        for seed in SEEDS:
            run = f"experiments/{exp_dir}/runs/{feat}_seed{seed}"
            cmd = [
                sys.executable, "-u", "scripts/evaluate_cross_dataset.py",
                "--run-dir", run,
                "--data-dir", DATA_DIR,
                "--device", "cuda",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  FAILED: {feat}_seed{seed}", flush=True)
                print(f"  stderr: {r.stderr[-200:]}", flush=True)
                continue

            # Parse the summary line from stdout
            summary = {}
            for line in r.stdout.split("\n"):
                line = line.strip()
                if line.startswith("Event F1"):
                    summary["event_f1"] = float(line.split(":")[1].strip())
                elif line.startswith("Event Recall"):
                    summary["event_recall"] = float(line.split(":")[1].strip())
                elif line.startswith("Event Precision"):
                    summary["event_precision"] = float(line.split(":")[1].strip())
                elif line.startswith("FAR / UAV"):
                    summary["far"] = float(line.split(":")[1].strip())
                elif line.startswith("Mean Delay"):
                    summary["delay_mean_s"] = float(line.split(":")[1].strip().replace("s", ""))
                elif line.startswith("Median Delay"):
                    summary["delay_median_s"] = float(line.split(":")[1].strip().replace("s", ""))

            if summary:
                results[exp_name][f"{feat}_seed{seed}"] = summary
                print(f"  OK: {feat}_seed{seed}  F1={summary.get('event_f1', '?')}", flush=True)
            else:
                print(f"  PARSE_FAILED: {feat}_seed{seed}", flush=True)

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY: Multi-Attack Training → Test Split Results")
print("=" * 80)

for exp_name in EXPERIMENTS:
    print(f"\n--- {exp_name} ---")
    for feat in FEATURES:
        vals = [
            v for k, v in results.get(exp_name, {}).items()
            if k.startswith(feat)
        ]
        if len(vals) < 3:
            print(f"  {F_LABELS[feat]}: INCOMPLETE ({len(vals)}/3 seeds)")
            continue

        def mean_std(xs):
            m = sum(xs) / len(xs)
            s = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
            return m, s

        recalls = [v["event_recall"] for v in vals]
        precs = [v["event_precision"] for v in vals]
        f1s = [v["event_f1"] for v in vals]
        fars = [v["far"] for v in vals]
        delays = [v["delay_mean_s"] for v in vals]

        rm, rs = mean_std(recalls)
        pm, ps = mean_std(precs)
        fm, fs = mean_std(f1s)
        farm, fars_std = mean_std(fars)
        dm, ds = mean_std(delays)

        print(
            f"  {F_LABELS[feat]}: "
            f"R={rm:.3f}±{rs:.3f}  "
            f"P={pm:.3f}±{ps:.3f}  "
            f"F1={fm:.3f}±{fs:.3f}  "
            f"FAR={farm:.3f}±{fars_std:.3f}  "
            f"delay={dm:.2f}±{ds:.2f}s"
        )

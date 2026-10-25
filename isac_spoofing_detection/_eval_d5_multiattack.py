import subprocess, sys
from collections import defaultdict

EXP_DIRS = {
    "BCE": "d1ma_bce_baseline_rcs6",
    "Full EL": "d1ma_feature_ablation_rcs6",
}
FEATURES = ["no_neighbor", "signed_anomaly_contrast_only"]
F_LABELS = {"no_neighbor": "F8", "signed_anomaly_contrast_only": "F9"}
SEEDS = [20260716, 20260717, 20260718]
DATA = "data/d5_long_1h_multiattack_rcs6_sloop_seed20270725"

def parse_output(stdout):
    s = {}
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("Event F1"):
            s["f1"] = float(line.split(":")[1])
        elif line.startswith("Event Recall"):
            s["recall"] = float(line.split(":")[1])
        elif line.startswith("Event Precision"):
            s["precision"] = float(line.split(":")[1])
        elif line.startswith("FAR / UAV"):
            s["far"] = float(line.split(":")[1])
        elif line.startswith("Mean Delay"):
            s["delay"] = float(line.split(":")[1].replace("s", ""))
    return s

all_results = {}

for exp_name, exp_dir in EXP_DIRS.items():
    print(f"\n=== {exp_name} ===", flush=True)
    all_results[exp_name] = {}
    for feat in FEATURES:
        vals = []
        for seed in SEEDS:
            run = f"experiments/{exp_dir}/runs/{feat}_seed{seed}"
            r = subprocess.run(
                [sys.executable, "-u", "scripts/evaluate_d5_long_multiattack.py",
                 "--run-dir", run, "--data-dir", DATA, "--device", "cuda",
                 "--gru-state-reset-s", "30", "--gru-context-replay-s", "30",
                 "--bootstrap-samples", "0", "--force"],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode != 0:
                print(f"  FAIL: {feat}_seed{seed}", flush=True)
                continue
            s = parse_output(r.stdout)
            if s:
                vals.append(s)
                print(f"  OK: {feat}_seed{seed}  F1={s['f1']:.3f} R={s['recall']:.3f} FAR={s['far']:.3f}", flush=True)
        all_results[exp_name][feat] = vals

print("\n" + "=" * 80)
print("D5 LONG-DURATION EVALUATION RESULTS")
print("=" * 80)

# Include D1 baseline for comparison
print("\n--- D1 Baseline (single-attack training, from earlier) ---")
print("  F8: R=0.846  F1=0.903  FAR=0.172  delay=4.10s")
print("  F9: R=0.852  F1=0.906  FAR=0.171  delay=4.12s")

for exp_name in EXP_DIRS:
    print(f"\n--- D1-MA {exp_name} ---")
    for feat in FEATURES:
        vals = all_results[exp_name].get(feat, [])
        if len(vals) < 3:
            print(f"  {F_LABELS[feat]}: INCOMPLETE ({len(vals)}/3)")
            continue

        def ms(x):
            m = sum(x) / len(x)
            s = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5
            return m, s

        rm, rs = ms([v["recall"] for v in vals])
        pm, ps = ms([v["precision"] for v in vals])
        fm, fs = ms([v["f1"] for v in vals])
        farm, fars = ms([v["far"] for v in vals])
        dm, ds = ms([v["delay"] for v in vals])
        print(
            f"  {F_LABELS[feat]}: "
            f"R={rm:.3f}+-{rs:.3f}  P={pm:.3f}+-{ps:.3f}  "
            f"F1={fm:.3f}+-{fs:.3f}  FAR={farm:.3f}+-{fars:.3f}  "
            f"delay={dm:.2f}+-{ds:.2f}s"
        )

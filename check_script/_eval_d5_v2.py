"""Batch D5 eval: D1 models + D1-MA models on enlarged D5 dataset."""
import subprocess, sys
from collections import defaultdict

# D1 single-attack trained
D1_RUNS = {
    "D1-F8": "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/no_neighbor_seed",
    "D1-F9": "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed",
}

# D1-MA multi-attack trained
D1MA_RUNS = {
    "MA-BCE-F8": "experiments/d1ma_bce_baseline_rcs6/runs/no_neighbor_seed",
    "MA-BCE-F9": "experiments/d1ma_bce_baseline_rcs6/runs/signed_anomaly_contrast_only_seed",
    "MA-EL-F8": "experiments/d1ma_feature_ablation_rcs6/runs/no_neighbor_seed",
    "MA-EL-F9": "experiments/d1ma_feature_ablation_rcs6/runs/signed_anomaly_contrast_only_seed",
}

SEEDS = [20260716, 20260717, 20260718]
DATA = "data/d5_long_1h_multiattack_rcs6_sloop_seed20270725"

def parse_output(stdout):
    s = {}
    for line in stdout.split("\n"):
        line = line.strip()
        if line.startswith("Event F1"): s["f1"] = float(line.split(":")[1])
        elif line.startswith("Event Recall"): s["recall"] = float(line.split(":")[1])
        elif line.startswith("Event Precision"): s["precision"] = float(line.split(":")[1])
        elif line.startswith("FAR / UAV"): s["far"] = float(line.split(":")[1])
        elif line.startswith("Mean Delay"): s["delay"] = float(line.split(":")[1].replace("s", ""))
    return s

all_results = {}

for label, run_template in {**D1_RUNS, **D1MA_RUNS}.items():
    print(f"\n=== {label} ===", flush=True)
    vals = []
    for seed in SEEDS:
        run = f"{run_template}{seed}"
        r = subprocess.run(
            [sys.executable, "-u", "scripts/evaluate_d5_long_multiattack.py",
             "--run-dir", run, "--data-dir", DATA, "--device", "cuda",
             "--gru-state-reset-s", "30", "--gru-context-replay-s", "30",
             "--bootstrap-samples", "0", "--force"],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            print(f"  FAIL: seed={seed}", flush=True)
            continue
        s = parse_output(r.stdout)
        if s:
            vals.append(s)
            print(f"  seed={seed}  F1={s['f1']:.3f} R={s['recall']:.3f} FAR={s['far']:.3f}", flush=True)
    all_results[label] = vals

# Summary
print("\n" + "=" * 80)
print("D5 RESULTS (219 group events, 9+3 episodes)")
print("=" * 80)

# D1 baselines first
print("\n--- D1 Single-Attack Training ---")
for label in D1_RUNS:
    vals = all_results.get(label, [])
    if len(vals) < 3:
        print(f"  {label}: INCOMPLETE")
        continue
    def ms(x): m=sum(x)/len(x); s=(sum((v-m)**2 for v in x)/(len(x)-1))**0.5; return m,s
    rm,rs=ms([v["recall"] for v in vals])
    fm,fs=ms([v["f1"] for v in vals])
    farm,fars=ms([v["far"] for v in vals])
    dm,ds=ms([v["delay"] for v in vals])
    print(f"  {label}: R={rm:.3f}+-{rs:.3f} F1={fm:.3f}+-{fs:.3f} FAR={farm:.3f}+-{fars:.3f} delay={dm:.1f}s")

print("\n--- D1-MA Multi-Attack Training ---")
for label in D1MA_RUNS:
    vals = all_results.get(label, [])
    if len(vals) < 3:
        print(f"  {label}: INCOMPLETE")
        continue
    def ms(x): m=sum(x)/len(x); s=(sum((v-m)**2 for v in x)/(len(x)-1))**0.5; return m,s
    rm,rs=ms([v["recall"] for v in vals])
    fm,fs=ms([v["f1"] for v in vals])
    farm,fars=ms([v["far"] for v in vals])
    dm,ds=ms([v["delay"] for v in vals])
    print(f"  {label}: R={rm:.3f}+-{rs:.3f} F1={fm:.3f}+-{fs:.3f} FAR={farm:.3f}+-{fars:.3f} delay={dm:.1f}s")

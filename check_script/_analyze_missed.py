"""Analyze missed detections in D5 evaluation: by attack type, severity,
target count, event order, and GRU state segment."""
import json
from pathlib import Path
from collections import defaultdict, Counter

# Compare D1-trained vs D1-MA-trained on D5
RUNS = {
    "D1-F9": "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716",
    "D1MA-F9": "experiments/d1ma_feature_ablation_rcs6/runs/signed_anomaly_contrast_only_seed20260716",
    "D1MA-BCE": "experiments/d1ma_bce_baseline_rcs6/runs/signed_anomaly_contrast_only_seed20260716",
}
DATA_DIR = "d5_long_1h_multiattack_rcs6_sloop_seed20270725"
POLICY = "gru_segment_300_steps_context_300_steps"

def load_events(label, run_dir):
    path = Path(run_dir) / "cross_test" / DATA_DIR / POLICY / "evaluation_d5_long_multiattack.json"
    if not path.exists():
        print(f"  {label}: MISSING {path}")
        return None
    with open(path) as f:
        d = json.load(f)

    attack_events = []
    for ep in d["episodes"]:
        for evt in ep["attack_events"]:
            attack_events.append(evt)

    total = len(attack_events)
    missed = [e for e in attack_events if e["status"] != "detected"]
    detected = [e for e in attack_events if e["status"] == "detected"]

    print(f"\n{'='*60}")
    print(f"{label}: {total} events, {len(detected)} detected, {len(missed)} missed")
    print(f"{'='*60}")

    # By attack type
    by_type = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        t = e["attack_type"]
        by_type[t]["total"] += 1
        if e["status"] != "detected":
            by_type[t]["missed"] += 1
    print("\nBy attack type:")
    for t in ["random", "replay", "stealth"]:
        v = by_type[t]
        print(f"  {t:8s}: {v['missed']}/{v['total']} missed (recall={1-v['missed']/v['total']:.3f})")

    # By severity
    by_sev = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        s = e["severity"]
        by_sev[s]["total"] += 1
        if e["status"] != "detected":
            by_sev[s]["missed"] += 1
    print("\nBy severity:")
    for s in ["weak", "medium", "strong"]:
        v = by_sev[s]
        print(f"  {s:8s}: {v['missed']}/{v['total']} missed (recall={1-v['missed']/v['total']:.3f})")

    # By type × severity
    print("\nBy type × severity:")
    by_ts = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        key = f"{e['attack_type']}+{e['severity']}"
        by_ts[key]["total"] += 1
        if e["status"] != "detected":
            by_ts[key]["missed"] += 1
    for key in sorted(by_ts.keys()):
        v = by_ts[key]
        print(f"  {key:20s}: {v['missed']:2d}/{v['total']:2d} missed")

    # By target count
    print("\nBy target UAV count:")
    by_tc = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        tc = e["target_uav_count"]
        by_tc[tc]["total"] += 1
        if e["status"] != "detected":
            by_tc[tc]["missed"] += 1
    for tc in sorted(by_tc.keys()):
        v = by_tc[tc]
        print(f"  {tc} UAVs: {v['missed']}/{v['total']} missed")

    # By attack ordinal (first vs subsequent)
    print("\nBy attack ordinal on UAV:")
    by_ord = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        ord_key = "first" if e.get("is_first_attack_on_uav") else "subsequent"
        by_ord[ord_key]["total"] += 1
        if e["status"] != "detected":
            by_ord[ord_key]["missed"] += 1
    for k in ["first", "subsequent"]:
        v = by_ord[k]
        print(f"  {k:10s}: {v['missed']}/{v['total']} missed")

    # By GRU state segment
    print("\nBy GRU state segment:")
    by_seg = defaultdict(lambda: {"total": 0, "missed": 0})
    for e in attack_events:
        seg = e.get("gru_state_segment", "unknown")
        by_seg[seg]["total"] += 1
        if e["status"] != "detected":
            by_seg[seg]["missed"] += 1
    for seg in sorted(by_seg.keys()):
        v = by_seg[seg]
        print(f"  {seg:30s}: {v['missed']}/{v['total']} missed")

    # Top missed events detail
    print("\nMissed events detail (first 15):")
    for i, e in enumerate(missed[:15]):
        print(f"  [{i}] type={e['attack_type']:8s} sev={e['severity']:8s} "
              f"UAVs={e['target_uav_count']} start={e['start_step']} "
              f"ord={'1st' if e.get('is_first_attack_on_uav') else 'nth'} "
              f"seg={e.get('gru_state_segment','?')}")

    return {"missed": missed, "detected": detected, "total": total}


for label, run_dir in RUNS.items():
    load_events(label, run_dir)

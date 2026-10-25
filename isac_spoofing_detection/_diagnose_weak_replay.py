"""Diagnose: Weak Replay - why recall dropped from 0.996 to 0.606?"""
import json
from pathlib import Path
from collections import defaultdict

# D1-F9 (single-attack trained) on D5
RUN_DIR = "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716"
DATA_DIR = "d5_long_1h_multiattack_rcs6_sloop_seed20270725"
POLICY = "gru_segment_300_steps_context_300_steps"

path = Path(RUN_DIR) / "cross_test" / DATA_DIR / POLICY / "evaluation_d5_long_multiattack.json"
with open(path) as f:
    d = json.load(f)

# Collect all Weak Replay events
weak_replay = []
for ep in d["episodes"]:
    for evt in ep["attack_events"]:
        if evt["attack_type"] == "replay" and evt["severity"] == "weak":
            weak_replay.append(evt)

print(f"Weak Replay total: {len(weak_replay)}")
detected = [e for e in weak_replay if e["status"] == "detected"]
missed = [e for e in weak_replay if e["status"] != "detected"]
print(f"Detected: {len(detected)}, Missed: {len(missed)}")
print(f"Recall: {len(detected)/len(weak_replay):.3f}")

# Check: is there a relationship with attack parameter (delay)?
print("\n--- Attack delay parameter analysis ---")
print("(sampled_attack_parameter = delay in seconds)")

for label, events in [("DETECTED", detected), ("MISSED", missed)]:
    params = []
    for e in events:
        p = e.get("sampled_attack_parameter")
        if p is not None:
            params.append(float(p))
    if params:
        import numpy as np
        params = np.array(params)
        print(f"  {label}: n={len(params)} mean={params.mean():.3f}s "
              f"min={params.min():.3f}s max={params.max():.3f}s "
              f"median={np.median(params):.3f}s")

# Also check: inter-attack gaps for Weak Replay specifically
print("\n--- Gap before Weak Replay attacks ---")
uav_timelines = defaultdict(list)
for ep in d["episodes"]:
    for evt in ep["attack_events"]:
        uav_timelines[evt["uav"]].append({
            "start": evt["start_step"],
            "end": evt["end_step_exclusive"],
            "type": evt["attack_type"],
            "severity": evt["severity"],
            "status": evt["status"],
        })

for uav in uav_timelines:
    uav_timelines[uav].sort(key=lambda x: x["start"])

for label, events in [("DETECTED", detected), ("MISSED", missed)]:
    gaps = []
    for e in events:
        uav = e["uav"]
        timeline = uav_timelines[uav]
        for i, t in enumerate(timeline):
            if t["start"] == e["start_step"] and t["type"] == "replay" and t["severity"] == "weak":
                if i > 0:
                    prev = timeline[i-1]
                    gap = (e["start_step"] - prev["end"]) * 0.1
                    gaps.append(gap)
                break
    if gaps:
        import numpy as np
        gaps = np.array(gaps)
        print(f"  {label}: n={len(gaps)} mean_gap={gaps.mean():.0f}s "
              f"min={gaps.min():.0f}s max={gaps.max():.0f}s")

# Check target UAV count distribution
print("\n--- Target UAV count distribution ---")
for label, events in [("DETECTED", detected), ("MISSED", missed)]:
    counts = defaultdict(int)
    for e in events:
        counts[e["target_uav_count"]] += 1
    print(f"  {label}: {dict(sorted(counts.items()))}")

# Cross-episode: is first attack on UAV more detectable?
print("\n--- First vs subsequent on UAV ---")
for label, events in [("DETECTED", detected), ("MISSED", missed)]:
    first = sum(1 for e in events if e.get("is_first_attack_on_uav"))
    subseq = sum(1 for e in events if not e.get("is_first_attack_on_uav"))
    print(f"  {label}: first={first}, subsequent={subseq}")

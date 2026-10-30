"""Diagnose: does inter-attack gap predict missed detection?"""
import json
from pathlib import Path
from collections import defaultdict

RUN_DIR = "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716"
DATA_DIR = "d5_long_1h_multiattack_rcs6_sloop_seed20270725"
POLICY = "gru_segment_300_steps_context_300_steps"

path = Path(RUN_DIR) / "cross_test" / DATA_DIR / POLICY / "evaluation_d5_long_multiattack.json"
with open(path) as f:
    d = json.load(f)

# Build per-UAV attack timeline across all episodes
uav_timelines = defaultdict(list)  # uav -> [(episode_id, start_step, end_step, attack_type, severity, status), ...]

for ep in d["episodes"]:
    for evt in ep["attack_events"]:
        uav = evt["uav"]
        uav_timelines[uav].append({
            "episode": ep["episode_id"],
            "start": evt["start_step"],
            "end": evt["end_step_exclusive"],
            "type": evt["attack_type"],
            "severity": evt["severity"],
            "status": evt["status"],
            "ordinal": "first" if evt.get("is_first_attack_on_uav") else "subsequent",
        })

# Sort each UAV's timeline by time
for uav in uav_timelines:
    uav_timelines[uav].sort(key=lambda x: (x["episode"], x["start"]))

# Analysis 1: Inter-attack gap vs miss rate
print("=" * 60)
print("ANALYSIS 1: Inter-attack gap vs detection rate")
print("=" * 60)

gap_buckets = {
    "0-60s": (0, 600),
    "60-120s": (600, 1200),
    "120-300s": (1200, 3000),
    "300-600s": (3000, 6000),
    "600s+": (6000, 999999),
}

for label, (lo, hi) in gap_buckets.items():
    pairs = {"total": 0, "detected": 0}
    for uav, timeline in uav_timelines.items():
        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]
            # Only consider same-episode pairs
            if prev["episode"] != curr["episode"]:
                continue
            gap = curr["start"] - prev["end"]
            if lo <= gap < hi:
                pairs["total"] += 1
                if curr["status"] == "detected":
                    pairs["detected"] += 1
    if pairs["total"] > 0:
        rate = pairs["detected"] / pairs["total"]
        print(f"  Gap {label:10s}: {pairs['detected']}/{pairs['total']} detected ({rate:.3f})")

# Analysis 2: Preceding attack strength vs subsequent weak attack detection
print("\n" + "=" * 60)
print("ANALYSIS 2: Preceding attack severity → subsequent WEAK attack detection")
print("=" * 60)

for prev_sev in ["weak", "medium", "strong"]:
    for prev_type in ["random", "replay", "stealth"]:
        pairs = {"total": 0, "detected": 0}
        for uav, timeline in uav_timelines.items():
            for i in range(1, len(timeline)):
                prev = timeline[i - 1]
                curr = timeline[i]
                if prev["episode"] != curr["episode"]:
                    continue
                if curr["severity"] != "weak":
                    continue  # Only look at weak subsequent attacks
                if prev["severity"] == prev_sev and prev["type"] == prev_type:
                    pairs["total"] += 1
                    if curr["status"] == "detected":
                        pairs["detected"] += 1
        if pairs["total"] > 0:
            rate = pairs["detected"] / pairs["total"]
            print(f"  {prev_type}+{prev_sev} → WEAK: {pairs['detected']}/{pairs['total']} ({rate:.3f})")

# Analysis 3: Detailed - for each missed Weak Random, what was the gap?
print("\n" + "=" * 60)
print("ANALYSIS 3: Gap before each MISSED Weak Random/Replay attack")
print("=" * 60)

for uav, timeline in uav_timelines.items():
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        if prev["episode"] != curr["episode"]:
            continue
        if curr["severity"] != "weak":
            continue
        if curr["type"] not in ("random", "replay"):
            continue
        if curr["status"] == "detected":
            continue
        gap = curr["start"] - prev["end"]
        gap_s = gap * 0.1
        print(f"  UAV={uav} {prev['type']}+{prev['severity']} → {curr['type']}+{curr['severity']} "
              f"gap={gap_s:.1f}s start={curr['start']} status=MISSED")

# Analysis 4: Detect rate by gap when preceding was STRONG
print("\n" + "=" * 60)
print("ANALYSIS 4: After STRONG attack, detection of next WEAK by gap")
print("=" * 60)

for label, (lo, hi) in gap_buckets.items():
    pairs = {"total": 0, "detected": 0}
    for uav, timeline in uav_timelines.items():
        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]
            if prev["episode"] != curr["episode"]:
                continue
            if prev["severity"] != "strong":
                continue
            if curr["severity"] != "weak":
                continue
            gap = curr["start"] - prev["end"]
            if lo <= gap < hi:
                pairs["total"] += 1
                if curr["status"] == "detected":
                    pairs["detected"] += 1
    if pairs["total"] > 0:
        rate = pairs["detected"] / pairs["total"]
        print(f"  Gap {label:10s}: {pairs['detected']}/{pairs['total']} ({rate:.3f})")

"""Apples-to-apples: per-UAV event evaluation of D1 model on D1 test split."""
import json, pathlib, numpy as np, sys
from collections import defaultdict
try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isac_spoofing.training import load_trained_detector
from isac_spoofing.learning_data import load_split, select_episode_features
from isac_spoofing.metrics import k_step_alarm_events, match_alarm_events, AttackEvent

RUN = "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716"
DATA = "data/d1_5000_rcs6_neighbor_probe_seed20260724"

bundle = load_trained_detector(RUN + "/training/best_model.pt")
episodes = load_split(DATA, "test")
print(f"Loaded {len(episodes)} test episodes")

# Per-UAV events with attack parameters
all_events = []

for ep in episodes:
    # Get attack metadata from episode
    meta = ep.metadata
    atk = meta.get("attack", {})
    atype = atk.get("attack_type", "clean")
    params = atk.get("parameters", {})
    sev = params.get("severity_level", "unknown")
    targets = atk.get("target_uavs", [])

    if atype == "clean" or not targets:
        continue

    # Predict
    selected = select_episode_features(ep, bundle.standardizer.feature_names)
    normalized = bundle.standardizer.transform(selected.features)
    with torch.no_grad():
        tensor = torch.from_numpy(np.transpose(normalized, (1, 0, 2))).float()
        tensor = tensor.to(device=bundle.device)
        logits = bundle.model(tensor)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).int().cpu().numpy()
    # preds shape: (N, T) or (N, T, 1) → transpose to (T, N)
    if preds.ndim == 3:
        preds = preds.squeeze(-1)
    preds = np.transpose(preds, (1, 0))

    # K-step alarms
    alarms = k_step_alarm_events(preds, confirmation_steps=5)

    # Per-UAV attack events
    per_uav_events = []
    for uav in targets:
        per_uav_events.append({
            "uav": uav,
            "start_step": atk["start_step"],
            "end_step_exclusive": atk["end_step"],
            "event_id": f"{ep.metadata.get('episode_id','?')}_uav{uav}",
        })

    # Build AttackEvent objects for matching
    attack_events_list = [
        AttackEvent(uav=e["uav"], start_step=e["start_step"],
                    end_step_exclusive=e["end_step_exclusive"],
                    event_id=e["event_id"])
        for e in per_uav_events
    ]

    result = match_alarm_events(alarms, attack_events_list,
                                num_steps=ep.features.shape[0], dt_s=0.1)

    for rec in result["attack_events"]:
        rec["attack_type"] = atype
        rec["severity"] = sev
        rec["sigma_m"] = params.get("sigma_m")
        rec["delay_s"] = params.get("strength_value")
        rec["rate_mps"] = params.get("strength_value")
        rec["target_uav_count"] = len(targets)
        rec["episode_id"] = meta.get("episode_id", "?")
        all_events.append(rec)

print(f"Total per-UAV events: {len(all_events)}")

# --- Analysis ---
total = len(all_events)
missed = [e for e in all_events if e["status"] != "detected"]
detected = [e for e in all_events if e["status"] == "detected"]
print(f"Overall: {len(detected)}/{total} detected → Recall={len(detected)/total:.3f}")
print()

# By type x severity
by_ts = defaultdict(lambda: {"total": 0, "missed": 0})
for e in all_events:
    key = e["attack_type"] + "+" + e["severity"]
    by_ts[key]["total"] += 1
    if e["status"] != "detected":
        by_ts[key]["missed"] += 1

print("By type x severity:")
for key in sorted(by_ts.keys()):
    v = by_ts[key]
    r = 1 - v["missed"] / v["total"]
    bar = "#" * int(r * 20) + "-" * (20 - int(r * 20))
    print(f"  {key:20s}: {v['missed']:3d}/{v['total']:3d} [{bar}] {r:.3f}")

# Weak Random detail
print("\n=== D1 Weak Random detail ===")
wr = [e for e in all_events if e["attack_type"] == "random" and e["severity"] == "weak"]
wr_det = [e for e in wr if e["status"] == "detected"]
wr_mis = [e for e in wr if e["status"] != "detected"]
print(f"Total: {len(wr)}, Detected: {len(wr_det)}, Missed: {len(wr_mis)}")
for label, subset in [("DETECTED", wr_det), ("MISSED", wr_mis)]:
    sigmas = [e.get("sigma_m", 0) for e in subset if e.get("sigma_m")]
    if sigmas:
        s = np.array(sigmas)
        print(f"  {label}: n={len(s)} sigma mean={s.mean():.3f} range=[{s.min():.3f},{s.max():.3f}]")

# Weak Replay detail
print("\n=== D1 Weak Replay detail ===")
wrp = [e for e in all_events if e["attack_type"] == "replay" and e["severity"] == "weak"]
wrp_det = [e for e in wrp if e["status"] == "detected"]
wrp_mis = [e for e in wrp if e["status"] != "detected"]
print(f"Total: {len(wrp)}, Detected: {len(wrp_det)}, Missed: {len(wrp_mis)}")
for label, subset in [("DETECTED", wrp_det), ("MISSED", wrp_mis)]:
    delays = [e.get("delay_s", 0) for e in subset if e.get("delay_s")]
    if delays:
        d = np.array(delays)
        print(f"  {label}: n={len(d)} delay mean={d.mean():.3f}s range=[{d.min():.3f},{d.max():.3f}]")

# By target count
print("\n=== D1 Miss rate by target count ===")
for tc in [1, 2, 3, 4]:
    subset = [e for e in all_events if e["target_uav_count"] == tc]
    missed_n = sum(1 for e in subset if e["status"] != "detected")
    print(f"  {tc} UAV: {missed_n}/{len(subset)} ({missed_n/len(subset)*100:.0f}%)" if subset else f"  {tc} UAV: N/A")

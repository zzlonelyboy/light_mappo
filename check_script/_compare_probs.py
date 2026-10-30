"""Compare model probability outputs: D1 vs D5 for Weak attacks."""
import json, pathlib, numpy as np, sys
from collections import defaultdict

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import torch
from isac_spoofing.training import load_trained_detector
from isac_spoofing.learning_data import load_split, select_episode_features

RUN = "experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716"
bundle = load_trained_detector(RUN + "/training/best_model.pt")

def episode_probs(ep):
    selected = select_episode_features(ep, bundle.standardizer.feature_names)
    normalized = bundle.standardizer.transform(selected.features)
    with torch.no_grad():
        tensor = torch.from_numpy(np.transpose(normalized, (1, 0, 2))).float()
        tensor = tensor.to(device=bundle.device)
        logits = bundle.model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    if probs.ndim == 3:
        probs = probs.squeeze(-1)
    return np.transpose(probs, (1, 0))

# --- D5: load predictions (already saved by evaluate_d5_long_multiattack) ---
d5_dir = pathlib.Path(RUN) / "cross_test" / "d5_long_1h_multiattack_rcs6_sloop_seed20270725" / "gru_segment_300_steps_context_300_steps"
d5_eval = json.load(open(d5_dir / "evaluation_d5_long_multiattack.json"))

# D5 predictions are saved as NPZ files
d5_pred_dir = d5_dir / ".." / ".."  # not saved without --save-predictions
# We need to recompute D5 probs. Load D5 test episodes.
print("Loading D5 episodes...")
d5_eps = load_split("data/d5_long_1h_multiattack_rcs6_sloop_seed20270725", "test")
d5_ep_map = {ep.metadata.get("episode_id", str(i)): ep for i, ep in enumerate(d5_eps)}

# D1 episodes
print("Loading D1 episodes...")
d1_eps = load_split("data/d1_5000_rcs6_neighbor_probe_seed20260724", "test")
d1_ep_map = {ep.metadata.get("episode_id", str(i)): ep for i, ep in enumerate(d1_eps)}

# Collect attack-interval probabilities
def collect_probs(eval_data, ep_map, label):
    results = {"random+weak": [], "replay+weak": [], "stealth+weak": []}
    for ep_result in eval_data["episodes"]:
        ep_id = ep_result["episode_id"]
        ep = ep_map.get(ep_id)
        if ep is None:
            continue
        probs = episode_probs(ep)
        for evt in ep_result["attack_events"]:
            atype = evt["attack_type"]
            sev = evt["severity"]
            if sev != "weak":
                continue
            key = atype + "+weak"
            start = evt["start_step"]
            end = evt["end_step_exclusive"]
            uav = evt["uav"]
            if start < probs.shape[0] and uav < probs.shape[1]:
                interval_probs = probs[start:end, uav]
                mean_p = float(np.mean(interval_probs))
                max_p = float(np.max(interval_probs))
                results[key].append({
                    "mean_p": mean_p,
                    "max_p": max_p,
                    "status": evt["status"],
                    "sigma": evt.get("sampled_attack_parameter", evt.get("sigma_m", 0)),
                })
    return results

# --- D5 ---
print("Computing D5 probabilities...")
d5_probs = collect_probs(d5_eval, d5_ep_map, "D5")

# --- D1 ---
# Build D1 eval data from _d1_per_uav_eval results
# We already have the per-event data. Let me rebuild it.
print("Computing D1 probabilities...")
from isac_spoofing.metrics import k_step_alarm_events, match_alarm_events, AttackEvent

d1_all_events = []
for ep in d1_eps:
    meta = ep.metadata
    atk = meta.get("attack", {})
    atype = atk.get("attack_type", "clean")
    params = atk.get("parameters", {})
    sev = params.get("severity_level", "unknown")
    targets = atk.get("target_uavs", [])
    if atype == "clean" or not targets:
        continue

    probs = episode_probs(ep)
    preds = (probs >= 0.5).astype(np.int32)
    alarms = k_step_alarm_events(preds, confirmation_steps=5)

    attack_events_list = [
        AttackEvent(uav=u, start_step=atk["start_step"],
                    end_step_exclusive=atk["end_step"],
                    event_id=f"{meta.get('episode_id','?')}_uav{u}")
        for u in targets
    ]
    result = match_alarm_events(alarms, attack_events_list,
                                num_steps=ep.features.shape[0], dt_s=0.1)

    for rec in result["attack_events"]:
        rec["attack_type"] = atype
        rec["severity"] = sev
        rec["sampled_attack_parameter"] = params.get("sigma_m") or params.get("strength_value")
        uav = rec["uav"]
        start = rec["start_step"]
        end = rec["end_step_exclusive"]
        interval_probs = probs[start:end, uav]
        rec["mean_p"] = float(np.mean(interval_probs))
        rec["max_p"] = float(np.max(interval_probs))
        d1_all_events.append(rec)

# Organize D1 results
d1_probs = {"random+weak": [], "replay+weak": [], "stealth+weak": []}
for e in d1_all_events:
    if e["severity"] != "weak":
        continue
    key = e["attack_type"] + "+weak"
    d1_probs[key].append({
        "mean_p": e["mean_p"],
        "max_p": e["max_p"],
        "status": e["status"],
        "sigma": e.get("sampled_attack_parameter", 0),
    })

# --- Compare ---
print()
for key in ["random+weak", "replay+weak", "stealth+weak"]:
    d1 = d1_probs[key]
    d5 = d5_probs[key]
    if not d1 or not d5:
        continue
    d1_mean = np.array([x["mean_p"] for x in d1])
    d1_max = np.array([x["max_p"] for x in d1])
    d5_mean = np.array([x["mean_p"] for x in d5])
    d5_max = np.array([x["max_p"] for x in d5])
    print(f"=== {key} ===")
    print(f"  D1 (n={len(d1)}): mean_p={d1_mean.mean():.3f}  max_p={d1_max.mean():.3f}  frac>0.5={np.mean(d1_max>=0.5):.2%}")
    print(f"  D5 (n={len(d5)}): mean_p={d5_mean.mean():.3f}  max_p={d5_max.mean():.3f}  frac>0.5={np.mean(d5_max>=0.5):.2%}")
    print(f"  D5/D1 ratio: mean_p={d5_mean.mean()/d1_mean.mean():.2f}  max_p={d5_max.mean()/d1_max.mean():.2f}")
    print()

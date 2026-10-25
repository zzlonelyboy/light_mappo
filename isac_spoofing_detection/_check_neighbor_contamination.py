"""Check if group attacks contaminate neighbor references."""
import json, pathlib, numpy as np
from collections import defaultdict

p = pathlib.Path('experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716/cross_test/d5_long_1h_multiattack_rcs6_sloop_seed20270725/gru_segment_300_steps_context_300_steps/evaluation_d5_long_multiattack.json')
d = json.load(open(p))

# Formation geometry (same as trajectories.py)
n = 8
angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
radii = 3.0 + 0.7 * np.arange(n)
positions = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), 1.5 * ((np.arange(n) % 3) - 1)])
positions -= np.mean(positions, axis=0)

# K=3 neighbors from formation
k = 3
dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
np.fill_diagonal(dists, np.inf)
neighbors = np.argsort(dists, axis=1)[:, :k]

print("Formation neighbor table (K=3):")
for i in range(n):
    print(f"  UAV {i}: neighbors = {list(neighbors[i])}")

# Group events by start_step+type+severity
group_events = defaultdict(list)
for ep in d['episodes']:
    for e in ep['attack_events']:
        key = ep['episode_id'] + '_' + str(e['start_step']) + '_' + e['attack_type'] + '_' + e['severity']
        group_events[key].append(e)

# Check contamination
total = len(group_events)
contaminated_count = 0
for key, events in group_events.items():
    targets = set(e['uav'] for e in events)
    if len(targets) <= 1:
        continue
    for uav in targets:
        uav_nbs = set(neighbors[uav].tolist())
        others = targets - {uav}
        if uav_nbs & others:
            contaminated_count += 1
            break

print(f"\nTotal group events: {total}")
print(f"Groups with >=1 UAV whose neighbor is also targeted: {contaminated_count}/{total}")

# For each target count
for tc in [1, 2, 3, 4]:
    groups = [(k, v) for k, v in group_events.items() if len(set(e['uav'] for e in v)) == tc]
    cont = 0
    for key, events in groups:
        targets = set(e['uav'] for e in events)
        for uav in targets:
            if set(neighbors[uav].tolist()) & (targets - {uav}):
                cont += 1
                break
    print(f"  {tc}-UAV groups: {cont}/{len(groups)} contaminated ({cont/len(groups)*100:.0f}%)" if groups else f"  {tc}-UAV groups: N/A")

# Example: a 4-UAV group
print("\nExample 4-UAV attacks:")
count = 0
for key, events in group_events.items():
    targets = set(e['uav'] for e in events)
    if len(targets) == 4 and count < 4:
        atype = events[0]['attack_type']
        sev = events[0]['severity']
        statuses = [e['status'] for e in events]
        print(f"  {atype}+{sev}: targets={sorted(targets)}, detected={statuses.count('detected')}/{len(statuses)}")
        for uav in sorted(targets):
            nb = list(neighbors[uav])
            nb_attacked = [x for x in nb if x in targets]
            print(f"    UAV {uav}: neighbors={nb}, attacked_neighbors={nb_attacked}")
        count += 1

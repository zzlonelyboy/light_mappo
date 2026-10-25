"""Deep dive: Weak Random & Weak Replay missed vs detected patterns."""
import json, pathlib, numpy as np
from collections import defaultdict

p = pathlib.Path('experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716/cross_test/d5_long_1h_multiattack_rcs6_sloop_seed20270725/gru_segment_300_steps_context_300_steps/evaluation_d5_long_multiattack.json')
d = json.load(open(p))

events = []
for ep in d['episodes']:
    for e in ep['attack_events']:
        events.append(e)

# ====== Weak Random ======
wr = [e for e in events if e['attack_type']=='random' and e['severity']=='weak']
wr_detected = [e for e in wr if e['status']=='detected']
wr_missed = [e for e in wr if e['status']!='detected']

print(f'Weak Random: {len(wr_detected)} detected, {len(wr_missed)} missed ({len(wr_missed)/len(wr):.0%} missed)')
print()

# Sigma analysis
print('--- Sigma (m) distribution ---')
for label, subset in [('DETECTED', wr_detected), ('MISSED', wr_missed)]:
    sigmas = [float(e.get('sampled_attack_parameter', 0)) for e in subset]
    if sigmas:
        sigmas = np.array(sigmas)
        print(f'  {label}: n={len(sigmas)} mean={sigmas.mean():.3f} min={sigmas.min():.3f} max={sigmas.max():.3f}')
        for lo, hi in [(0.25,0.35),(0.35,0.50),(0.50,0.75)]:
            n = int(np.sum((sigmas>=lo)&(sigmas<hi)))
            miss_n = int(np.sum((sigmas>=lo)&(sigmas<hi)))-n if label=='DETECTED' else n
            print(f'    [{lo:.2f},{hi:.2f}): {n} events')

# Target count
print('\n--- Target UAV count ---')
for label, subset in [('DETECTED', wr_detected), ('MISSED', wr_missed)]:
    tcs = defaultdict(int)
    for e in subset:
        tcs[e['target_uav_count']] += 1
    print(f'  {label}: {dict(sorted(tcs.items()))}')

# Gap from previous attack
print('\n--- Gap from previous attack (s) ---')
uav_timelines = defaultdict(list)
for e in events:
    uav_timelines[e['uav']].append(e)
for uav in uav_timelines:
    uav_timelines[uav].sort(key=lambda x: x['start_step'])

for label, subset in [('DETECTED', wr_detected), ('MISSED', wr_missed)]:
    gaps = []
    for e in subset:
        uav = e['uav']
        tl = uav_timelines[uav]
        for i, t in enumerate(tl):
            if t['start_step']==e['start_step'] and t['attack_type']==e['attack_type'] and t['severity']==e['severity']:
                if i > 0:
                    gaps.append((e['start_step'] - tl[i-1]['end_step_exclusive']) * 0.1)
                break
    if gaps:
        gaps = np.array(gaps)
        print(f'  {label}: n={len(gaps)} mean={gaps.mean():.0f}s min={gaps.min():.0f}s max={gaps.max():.0f}s')

# ====== Weak Replay ======
print('\n' + '='*60)
wrp = [e for e in events if e['attack_type']=='replay' and e['severity']=='weak']
wrp_detected = [e for e in wrp if e['status']=='detected']
wrp_missed = [e for e in wrp if e['status']!='detected']

print(f'Weak Replay: {len(wrp_detected)} detected, {len(wrp_missed)} missed ({len(wrp_missed)/len(wrp):.0%} missed)')
print()

# Delay analysis
print('--- Delay (s) distribution ---')
for label, subset in [('DETECTED', wrp_detected), ('MISSED', wrp_missed)]:
    delays = [float(e.get('sampled_attack_parameter', 0)) for e in subset]
    if delays:
        delays = np.array(delays)
        print(f'  {label}: n={len(delays)} mean={delays.mean():.3f} min={delays.min():.3f} max={delays.max():.3f}')
        for lo, hi in [(0.20,0.30),(0.30,0.40),(0.40,0.50)]:
            n = int(np.sum((delays>=lo)&(delays<hi)))
            print(f'    [{lo:.2f},{hi:.2f}): {n} events')

# Target count
print('\n--- Target UAV count ---')
for label, subset in [('DETECTED', wrp_detected), ('MISSED', wrp_missed)]:
    tcs = defaultdict(int)
    for e in subset:
        tcs[e['target_uav_count']] += 1
    print(f'  {label}: {dict(sorted(tcs.items()))}')

# Miss rate by target count
print('\n--- Miss rate by target UAV count ---')
for tc in [1,2,3,4]:
    total = sum(1 for e in wrp if e['target_uav_count']==tc)
    missed = sum(1 for e in wrp if e['target_uav_count']==tc and e['status']!='detected')
    if total > 0:
        print(f'  {tc} UAV: {missed}/{total} ({missed/total:.0%})')

# Same for Random
print('\n--- Weak Random miss rate by target count ---')
for tc in [1,2,3,4]:
    total = sum(1 for e in wr if e['target_uav_count']==tc)
    missed = sum(1 for e in wr if e['target_uav_count']==tc and e['status']!='detected')
    if total > 0:
        print(f'  {tc} UAV: {missed}/{total} ({missed/total:.0%})')

# Cross-tab: sigma x target count for Weak Random
print('\n--- Weak Random: sigma x target count x status ---')
for tc in [1,2,3,4]:
    for lo, hi in [(0.25,0.40),(0.40,0.55),(0.55,0.75)]:
        subset = [e for e in wr if e['target_uav_count']==tc and lo<=e.get('sampled_attack_parameter',0)<hi]
        if subset:
            missed = sum(1 for e in subset if e['status']!='detected')
            print(f'  {tc}UAV sigma[{lo:.2f},{hi:.2f}): {missed}/{len(subset)} missed')

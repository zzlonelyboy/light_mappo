import json, pathlib, numpy as np
from collections import defaultdict

p = pathlib.Path('experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716/cross_test/d5_long_1h_multiattack_rcs6_sloop_seed20270725/gru_segment_300_steps_context_300_steps/evaluation_d5_long_multiattack.json')
d = json.load(open(p))

events = []
for ep in d['episodes']:
    for e in ep['attack_events']:
        events.append(e)

timelines = defaultdict(list)
for e in events:
    timelines[e['uav']].append(e)
for uav in timelines:
    timelines[uav].sort(key=lambda x: x['start_step'])

# Weak Random: preceding attack type
print('=== Weak Random: preceding attack type ===')
for label, status_match in [('DETECTED', 'detected'), ('MISSED', 'not detected')]:
    prec = defaultdict(int)
    for uav, tl in timelines.items():
        for i in range(1, len(tl)):
            curr = tl[i]
            if curr['attack_type'] != 'random' or curr['severity'] != 'weak':
                continue
            is_ok = (status_match == 'detected' and curr['status'] == 'detected') or \
                    (status_match == 'not detected' and curr['status'] != 'detected')
            if not is_ok:
                continue
            prev = tl[i-1]
            key = prev['attack_type'] + '+' + prev['severity']
            prec[key] += 1
    print(f'  {label}: {dict(sorted(prec.items()))}')

# Weak Random: attack duration
print('\n=== Weak Random: attack duration (s) ===')
for label, status_match in [('DETECTED', 'detected'), ('MISSED', 'not detected')]:
    durs = []
    for e in events:
        if e['attack_type'] == 'random' and e['severity'] == 'weak':
            ok = (status_match == 'detected' and e['status'] == 'detected') or \
                 (status_match == 'not detected' and e['status'] != 'detected')
            if ok:
                durs.append((e['end_step_exclusive'] - e['start_step']) * 0.1)
    durs = np.array(durs)
    print(f'  {label}: n={len(durs)} mean={durs.mean():.1f}s')

# Replay: delay by target count
print('\n=== Weak Replay: delay by target count ===')
for tc in [1, 2, 3, 4]:
    subset = [e for e in events if e['attack_type'] == 'replay' and e['severity'] == 'weak' and e['target_uav_count'] == tc]
    det_d = [e.get('sampled_attack_parameter', 0) for e in subset if e['status'] == 'detected']
    mis_d = [e.get('sampled_attack_parameter', 0) for e in subset if e['status'] != 'detected']
    if det_d:
        print(f'  {tc}UAV DET: delay={np.mean(det_d):.3f}s n={len(det_d)}')
    if mis_d:
        print(f'  {tc}UAV MIS: delay={np.mean(mis_d):.3f}s n={len(mis_d)}')

# Miss rate by delay bin for Replay
print('\n=== Weak Replay: miss rate by delay bin ===')
for lo, hi in [(0.20, 0.25), (0.25, 0.30), (0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50)]:
    total = [e for e in events if e['attack_type'] == 'replay' and e['severity'] == 'weak' and lo <= e.get('sampled_attack_parameter', 0) < hi]
    missed = [e for e in total if e['status'] != 'detected']
    if total:
        print(f'  [{lo:.2f},{hi:.2f})s: {len(missed)}/{len(total)} missed ({len(missed)/len(total):.0%})')

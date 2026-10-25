import json, pathlib, numpy as np

p = pathlib.Path('experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716/cross_test/d5_long_1h_multiattack_rcs6_sloop_seed20270725/gru_segment_300_steps_context_300_steps/evaluation_d5_long_multiattack.json')
d = json.load(open(p))
events = []
for ep in d['episodes']:
    for e in ep['attack_events']:
        events.append(e)

w1 = [e for e in events if e['severity'] == 'weak' and e['target_uav_count'] == 1]
det = [e for e in w1 if e['status'] == 'detected']
mis = [e for e in w1 if e['status'] != 'detected']
print(f'1-UAV Weak: {len(w1)} total, {len(det)} det, {len(mis)} missed')
print()

print('=== MISSED ===')
for e in sorted(mis, key=lambda x: x['start_step']):
    param = e.get('sampled_attack_parameter', 0)
    atype = e['attack_type']
    first = e.get('is_first_attack_on_uav', False)
    seg = e.get('gru_state_segment', '?')
    step = e['start_step']
    uav = e['uav']
    unit = {'random': 'm', 'replay': 's', 'stealth': 'm/s'}.get(atype, '?')
    print(f'  {atype:8s} {param:.3f}{unit}  step={step:5d}  uav={uav}  first={first}  seg={seg}')

print()
print('=== DETECTED ===')
for e in sorted(det, key=lambda x: x['start_step']):
    param = e.get('sampled_attack_parameter', 0)
    atype = e['attack_type']
    first = e.get('is_first_attack_on_uav', False)
    seg = e.get('gru_state_segment', '?')
    step = e['start_step']
    uav = e['uav']
    unit = {'random': 'm', 'replay': 's', 'stealth': 'm/s'}.get(atype, '?')
    print(f'  {atype:8s} {param:.3f}{unit}  step={step:5d}  uav={uav}  first={first}  seg={seg}')

# Summary by type
print()
print('=== Summary by type ===')
for atype in ['random', 'replay', 'stealth']:
    subset = [e for e in w1 if e['attack_type'] == atype]
    missed = [e for e in subset if e['status'] != 'detected']
    if subset:
        params_missed = [e.get('sampled_attack_parameter', 0) for e in missed]
        params_det = [e.get('sampled_attack_parameter', 0) for e in subset if e['status'] == 'detected']
        print(f'{atype}: {len(missed)}/{len(subset)} missed')
        if params_missed:
            print('  missed params: ' + str([round(p, 3) for p in params_missed]))
        if params_det:
            print('  det params:    ' + str([round(p, 3) for p in params_det]))

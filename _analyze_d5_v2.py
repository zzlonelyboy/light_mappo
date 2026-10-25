import json, pathlib
from collections import defaultdict

p = pathlib.Path('experiments/neighbor_3way_rcs6_stage2_event_likelihood/runs/signed_anomaly_contrast_only_seed20260716/cross_test/d5_long_1h_multiattack_rcs6_sloop_seed20270725/gru_segment_300_steps_context_300_steps/evaluation_d5_long_multiattack.json')
d = json.load(open(p))

events = []
for ep in d['episodes']:
    for e in ep['attack_events']:
        events.append(e)

total = len(events)
missed = [e for e in events if e['status'] != 'detected']
print(f'Total: {total}, Missed: {len(missed)}, Recall: {1-len(missed)/total:.3f}')
print()

print('By type x severity:')
by_ts = defaultdict(lambda: {'total':0,'missed':0})
for e in events:
    key = e['attack_type'] + '+' + e['severity']
    by_ts[key]['total'] += 1
    if e['status'] != 'detected':
        by_ts[key]['missed'] += 1
for key in sorted(by_ts.keys()):
    v = by_ts[key]
    r = 1 - v['missed']/v['total']
    bar = '#' * int(r * 20) + '-' * (20 - int(r * 20))
    print(f'  {key:20s}: {v["missed"]:3d}/{v["total"]:3d} [{bar}] {r:.3f}')

print()
print('By type:')
by_type = defaultdict(lambda: {'total':0,'missed':0})
for e in events:
    by_type[e['attack_type']]['total'] += 1
    if e['status'] != 'detected': by_type[e['attack_type']]['missed'] += 1
for t in ['random','replay','stealth']:
    v = by_type[t]
    print(f'  {t:8s}: {v["missed"]}/{v["total"]} recall={1-v["missed"]/v["total"]:.3f}')

print()
print('By severity:')
by_sev = defaultdict(lambda: {'total':0,'missed':0})
for e in events:
    by_sev[e['severity']]['total'] += 1
    if e['status'] != 'detected': by_sev[e['severity']]['missed'] += 1
for s in ['weak','medium','strong']:
    v = by_sev[s]
    print(f'  {s:8s}: {v["missed"]}/{v["total"]} recall={1-v["missed"]/v["total"]:.3f}')

print()
print('By target UAVs:')
by_tc = defaultdict(lambda: {'total':0,'missed':0})
for e in events:
    tc = e['target_uav_count']
    by_tc[tc]['total'] += 1
    if e['status'] != 'detected': by_tc[tc]['missed'] += 1
for tc in sorted(by_tc.keys()):
    v = by_tc[tc]
    print(f'  {tc} UAVs: {v["missed"]}/{v["total"]} recall={1-v["missed"]/v["total"]:.3f}')

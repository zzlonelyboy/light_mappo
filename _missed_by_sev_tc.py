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
print(f'Total: {total}  Missed: {len(missed)} ({len(missed)/total*100:.1f}%)')
print()

# Missed absolute counts
print('=== Missed count by severity x target UAVs ===')
header = '            '
for tc in [1, 2, 3, 4]:
    header += f'  {tc}UAV '
header += '  TOTAL'
print(header)
for sev in ['weak', 'medium', 'strong']:
    row = f'{sev:>10s}  '
    row_total = 0
    for tc in [1, 2, 3, 4]:
        n = sum(1 for e in missed if e['severity'] == sev and e['target_uav_count'] == tc)
        denom = sum(1 for e in events if e['severity'] == sev and e['target_uav_count'] == tc)
        row += f'  {n:3d}  '
        row_total += n
    row += f'  {row_total:5d}'
    print(row)

row = 'TOTAL       '
for tc in [1, 2, 3, 4]:
    n = sum(1 for e in missed if e['target_uav_count'] == tc)
    row += f'  {n:3d}  '
row += f'  {len(missed):5d}'
print(row)

print()
print('=== Miss RATE (%) by severity x target UAVs ===')
header = '            '
for tc in [1, 2, 3, 4]:
    header += f'  {tc}UAV'
header += '   ALL'
print(header)
for sev in ['weak', 'medium', 'strong']:
    row = f'{sev:>10s}  '
    for tc in [1, 2, 3, 4]:
        n = sum(1 for e in missed if e['severity'] == sev and e['target_uav_count'] == tc)
        denom = sum(1 for e in events if e['severity'] == sev and e['target_uav_count'] == tc)
        if denom > 0:
            row += f'  {n/denom*100:3.0f}%'
        else:
            row += f'    -'
    n_all = sum(1 for e in missed if e['severity'] == sev)
    denom_all = sum(1 for e in events if e['severity'] == sev)
    row += f'  {n_all/denom_all*100:3.0f}%'
    print(row)

# Total per target count
row = 'TOTAL       '
for tc in [1, 2, 3, 4]:
    n = sum(1 for e in missed if e['target_uav_count'] == tc)
    denom = sum(1 for e in events if e['target_uav_count'] == tc)
    if denom > 0:
        row += f'  {n/denom*100:3.0f}%'
    else:
        row += f'    -'
row += f'  {len(missed)/total*100:3.0f}%'
print(row)

# Denominator table for reference
print()
print('=== Total events by severity x target UAVs (denominator) ===')
for sev in ['weak', 'medium', 'strong']:
    row = f'{sev:>10s}  '
    for tc in [1, 2, 3, 4]:
        denom = sum(1 for e in events if e['severity'] == sev and e['target_uav_count'] == tc)
        row += f'  {denom:3d}  '
    row += f'  {sum(1 for e in events if e["severity"] == sev):5d}'
    print(row)

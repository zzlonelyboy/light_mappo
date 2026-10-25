import subprocess, sys

tasks = {
    'single-atk': ('experiments/d1_shifted_rcs6/runs', 'data/d1_2000_shifted_rcs6_seed20260801'),
    'multi-atk': ('experiments/d1_ma_shifted_rcs6/runs', 'data/d1_2000_ma_shifted_rcs6_seed20260801'),
}
features = ['no_neighbor', 'signed_anomaly_contrast_only', 'no_bs_residual_signed_only']
f_labels = {'no_neighbor': 'F8', 'signed_anomaly_contrast_only': 'F9', 'no_bs_residual_signed_only': 'NoBS'}
seeds = [20260716, 20260717, 20260718]

def ms(x):
    m = sum(x) / len(x)
    s = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** 0.5
    return m, s

for label, (run_base, data_dir) in tasks.items():
    print('\n' + '=' * 60)
    print('=== ' + label + ' ===')
    for feat in features:
        vals = []
        for seed in seeds:
            run = run_base + '/' + feat + '_seed' + str(seed)
            r = subprocess.run(
                [sys.executable, '-u', 'scripts/evaluate_cross_dataset.py',
                 '--run-dir', run, '--data-dir', data_dir, '--device', 'cuda'],
                capture_output=True, text=True, timeout=180,
            )
            if r.returncode != 0:
                print('  FAIL: ' + feat + '_seed' + str(seed))
                continue
            s = {}
            for line in r.stdout.split('\n'):
                line = line.strip()
                if line.startswith('Event F1'):
                    s['f1'] = float(line.split(':')[1])
                elif line.startswith('Event Recall'):
                    s['recall'] = float(line.split(':')[1])
                elif line.startswith('Event Precision'):
                    s['precision'] = float(line.split(':')[1])
                elif line.startswith('FAR / UAV'):
                    s['far'] = float(line.split(':')[1])
                elif line.startswith('Mean Delay'):
                    s['delay'] = float(line.split(':')[1].replace('s', ''))
            if s:
                vals.append(s)
                print('  ' + f_labels[feat] + '_s' + str(seed) + ': F1=' + str(round(s['f1'], 3)) + ' R=' + str(round(s['recall'], 3)), flush=True)
        if len(vals) >= 3:
            rm, rs = ms([v['recall'] for v in vals])
            fm, fs = ms([v['f1'] for v in vals])
            farm, fars = ms([v['far'] for v in vals])
            dm, ds = ms([v['delay'] for v in vals])
            print('  => ' + f_labels[feat] + ': R=' + str(round(rm, 3)) + '+-' + str(round(rs, 3)) + ' F1=' + str(round(fm, 3)) + '+-' + str(round(fs, 3)) + ' FAR=' + str(round(farm, 3)) + ' delay=' + str(round(dm, 1)) + 's')

"""One pre-registered GRU pilot on the stopped, partially generated dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def allocate(total: int, group_sizes: dict[str, int]) -> dict[str, int]:
    keys = sorted(group_sizes)
    quota = {key: total * group_sizes[key] / sum(group_sizes.values()) for key in keys}
    result = {key: int(np.floor(quota[key])) for key in keys}
    order = sorted(keys, key=lambda key: (-(quota[key] - result[key]), key))
    for key in order[:total - sum(result.values())]:
        result[key] += 1
    assert sum(result.values()) == total
    return result


def prepare_dataset(source: Path, output: Path, seed: int, confirmation_steps: int = 5) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError('pilot data directory must be new or empty')
    records = []
    by_file = {}
    for meta_path in sorted((source / 'train').glob('shard_*.json')):
        npz_path = meta_path.with_suffix('.npz')
        if not npz_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        by_file[npz_path] = []
        for index, episode in enumerate(meta['episodes']):
            attack = episode['attack']
            scenario = attack['attack_type']
            severity = attack.get('parameters', {}).get('severity_level', 'none')
            record = dict(episode_id=episode['episode_id'], source_npz=str(npz_path),
                          source_index=index, scenario=scenario, severity=severity,
                          stratum=f'{scenario}:{severity}', metadata=episode)
            records.append(record)
            by_file[npz_path].append(record)
    assert len(records) == 444, len(records)
    assert len({r['episode_id'] for r in records}) == 444
    groups = defaultdict(list)
    for record in records:
        groups[record['stratum']].append(record)
    sizes = {name: len(rows) for name, rows in groups.items()}
    test_counts = allocate(45, sizes)
    remaining = {name: sizes[name] - test_counts[name] for name in sizes}
    validation_counts = allocate(44, remaining)
    rng = np.random.default_rng(seed)
    split_records = {name: [] for name in ('train', 'validation', 'test')}
    for name in sorted(groups):
        rows = groups[name]
        rng.shuffle(rows)
        nt, nv = test_counts[name], validation_counts[name]
        split_records['test'].extend(rows[:nt])
        split_records['validation'].extend(rows[nt:nt + nv])
        split_records['train'].extend(rows[nt + nv:])
    expected = {'train': 355, 'validation': 44, 'test': 45}
    maps = {}
    events = {}
    metadata = {}
    event_names = ('has_attack', 'attack_start_step', 'attack_end_step_exclusive',
                   'attack_was_truncated')
    for split, rows in split_records.items():
        rows.sort(key=lambda row: row['episode_id'])
        assert len(rows) == expected[split]
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        maps[split] = (
            np.lib.format.open_memmap(directory / 'features.npy', mode='w+',
                                      dtype=np.float32, shape=(len(rows), 3000, 8, 10)),
            np.lib.format.open_memmap(directory / 'labels.npy', mode='w+',
                                      dtype=np.uint8, shape=(len(rows), 3000, 8)),
        )
        events[split] = {
            name: np.empty((len(rows), 8), dtype=bool if name in {
                'has_attack', 'attack_was_truncated'} else np.int32)
            for name in event_names
        }
        metadata[split] = []
        for index, record in enumerate(rows):
            record.update(split=split, destination_index=index)
            row = deepcopy(record['metadata'])
            row.update(split=split, original_split='train',
                       source_npz=record['source_npz'],
                       source_episode_index=record['source_index'],
                       pilot_derived_from_stopped_dataset=True)
            metadata[split].append(row)
    feature_names = None
    completed = 0
    for file, rows in by_file.items():
        with np.load(file, allow_pickle=False) as stored:
            x, y = stored['features'], stored['labels']
            names = stored['feature_names'].tolist()
            if feature_names is None:
                feature_names = names
            assert names == feature_names
            assert x.shape == (len(rows), 3000, 8, 10)
            assert np.isfinite(x).all()
            for record in rows:
                source_index, destination = record['source_index'], record['destination_index']
                split = record['split']
                maps[split][0][destination] = x[source_index]
                maps[split][1][destination] = y[source_index]
                for name in event_names:
                    events[split][name][destination] = stored[name][source_index]
                completed += 1
        if completed % 100 == 0:
            print(f'Prepared {completed}/444 pilot episode records', flush=True)
    split_summary = {}
    for split, rows in split_records.items():
        for mapping in maps[split]:
            mapping.flush()
        np.savez(output / split / 'events.npz', **events[split])
        write_json(output / split / 'metadata.json', {'episodes': metadata[split]})
        split_summary[split] = {
            'episodes': len(rows),
            'scenarios': dict(Counter(row['scenario'] for row in rows)),
            'strata': dict(Counter(row['stratum'] for row in rows)),
            'episode_ids': [row['episode_id'] for row in rows],
        }
    id_sets = [set(row['episode_ids']) for row in split_summary.values()]
    assert all(not a.intersection(b) for i, a in enumerate(id_sets) for b in id_sets[i + 1:])
    first_meta = records[0]['metadata']
    manifest = {
        'schema_version': '3.0', 'episodes': 444, 'feature_dim': 10,
        'feature_names': feature_names, 'config': first_meta['config'],
        'measurement_timing': first_meta['measurement_timing'],
        'source_dataset': str(source.resolve()), 'source_status': 'stopped_by_user',
        'source_original_planned_episodes': 5000,
        'split_policy': 'fixed episode-level scenario/severity stratification, seed 20261030',
        'split_seed': seed, 'episodes_by_split': expected,
        'storage': {'format': 'npy_feature_memmap_v1', 'episode_axis': 0,
                    'note': 'Features and event labels only; source raw observations are retained separately'},
        'sampling_policy': 'all 444 complete source episodes, no simulation regenerated',
        'collection_mode': 'shared_gps_feedback_without_alarm_feedback',
        'confirmation_steps': confirmation_steps,
        'confirmation_duration_s': confirmation_steps * first_meta['config']['simulation']['dt_s'],
    }
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'split_assignment.json', split_summary)
    del maps
    gc.collect()
    print('Pilot split frozen: train355 / validation44 / test45', flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-output', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--confirmation-steps', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.confirmation_steps < 1:
        parser.error('--confirmation-steps must be positive')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('run output directory must be new or empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    write_json(args.out_dir / 'status.json', {'status': 'preparing_data'})
    try:
        manifest = prepare_dataset(args.source, args.data_output, 20261030, args.confirmation_steps)
        from isac_spoofing.training import TrainingConfig, train_detector
        from isac_spoofing.evaluation import evaluate_split, json_ready
        from isac_spoofing.learning_data import load_split
        from isac_spoofing.metrics import aggregate_event_results

        config = TrainingConfig(
            model_type='gru', input_encoder_dim=16, hidden_dim=64, num_layers=1,
            loss_objective='bce', class_weighting='balanced',
            confirmation_steps=args.confirmation_steps, time_step_s=.02,
            batch_size=16, max_epochs=args.epochs, patience=4,
            learning_rate=.001, weight_decay=.0001,
            seed=20260716, num_threads=1, device=args.device,
        )
        notes = (
            'Pilot plan fixed before training and opening held-out test outcomes.\n'
            'Use all444 episodes with an episode-level scenario/severity stratified 355/44/45 split.\n'
            'Fit normalization on training only. Train FC16-GRU64 with balanced ordinary BCE.\n'
            f'Select checkpoint by validation BCE. Fix threshold0.5 and K{args.confirmation_steps} at dt0.02.\n'
            'This is a learnability pilot, not the full event-aligned paper objective.\n'
            'Test is evaluated once after checkpoint selection; no test-based retuning.\n'
        )
        (args.out_dir / 'notes.txt').write_text(notes, encoding='utf-8')
        write_json(args.out_dir / 'status.json', {'status': 'training', 'epochs_planned': args.epochs})
        print('Training one balanced-BCE GRU pilot on the frozen episode split.', flush=True)
        training = train_detector(args.data_output, args.out_dir / 'training', config)
        checkpoint = training['checkpoint']
        gc.collect()
        write_json(args.out_dir / 'status.json', {'status': 'evaluating'})
        validation = evaluate_split(
            checkpoint, args.data_output, 'validation', args.confirmation_steps,
            args.out_dir, bootstrap_samples=0, device=args.device,
        )
        test = evaluate_split(
            checkpoint, args.data_output, 'test', args.confirmation_steps,
            args.out_dir, bootstrap_samples=500, seed=20261030, device=args.device,
        )
        test_metadata = {ep.episode_id: ep.metadata for ep in load_split(args.data_output, 'test')}
        groups = defaultdict(list)
        for row in test['episodes']:
            attack = test_metadata[row['episode_id']]['attack']
            severity = attack.get('parameters', {}).get('severity_level', 'none')
            groups[f"{row['scenario']}:{severity}"].append(row)
        by_severity = {
            name: aggregate_event_results(rows, bootstrap_samples=0)
            for name, rows in groups.items()
        }
        result = {
            'study_type': 'single_seed_short_training_pilot', 'training': training,
            'data_manifest': manifest, 'validation': validation, 'test': test,
            'test_by_scenario_and_severity': by_severity,
            'elapsed_s': time.perf_counter() - started,
            'interpretation_limits': [
                '444 source episodes rather than the planned5000',
                'One seed and at most10epochs; fixed scenario geometry and ordinary BCE',
                'Episode bootstrap quantifies test-sample variability, not training-seed variability',
                'Detection delay is conditional on successful detection; deadline DR includes misses',
                'No online alarm-driven navigation evaluation in this pilot',
            ],
        }
        write_json(args.out_dir / 'final_info.json', json_ready(result))
        with (args.out_dir / 'notes.txt').open('a', encoding='utf-8') as handle:
            handle.write('\nActual held-out test summary\n')
            handle.write(json.dumps(json_ready(test['aggregate']), indent=2))
        write_json(args.out_dir / 'status.json', {
            'status': 'completed', 'elapsed_s': result['elapsed_s'],
            'best_epoch': training['best_epoch'], 'test': json_ready(test['aggregate']),
        })
        print(json.dumps(json_ready(test['aggregate']), indent=2), flush=True)
    except BaseException as exc:
        write_json(args.out_dir / 'status.json', {
            'status': 'failed', 'error': repr(exc), 'elapsed_s': time.perf_counter() - started,
        })
        raise


if __name__ == '__main__':
    main()

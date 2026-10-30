"""Generate and verify the replacement 5000-episode GPS/CRB training asset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def verify_dataset(output: Path, expected: int, seed: int) -> dict:
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['episodes'] == expected
    assert manifest['base_seed'] == seed
    assert manifest['config']['gps']['noise_std_m'] == [1.5, 1.5, 3.0]
    assert manifest['config']['simulation']['dt_s'] == 0.02
    assert manifest['config']['simulation']['duration_s'] == 60.0
    assert manifest['feature_dim'] == 10
    assert manifest['trajectory_counts'] == {'s_curve': expected}
    expected_splits = {'train': int(expected * .8), 'validation': int(expected * .1)}
    expected_splits['test'] = expected - sum(expected_splits.values())
    counts = {}
    identifiers = set()
    recording_ids = set()
    recording_ids_by_split = {split: [] for split in expected_splits}
    recorded_mode = manifest.get('replay_source') == 'recorded_normal_episode'
    total_noise_sum = np.zeros(3, dtype=np.float64)
    total_noise_square = np.zeros(3, dtype=np.float64)
    sample_count = 0
    byte_count = 0
    for split, required in expected_splits.items():
        count = 0
        files = sorted((output / split).glob('*.npz'))
        assert len(files) == len(list((output / split).glob('*.json')))
        for file in files:
            meta = json.loads(file.with_suffix('.json').read_text(encoding='utf-8'))
            rows = meta['episodes']
            count += len(rows)
            for row in rows:
                episode_id = row['episode_id']
                assert episode_id not in identifiers, episode_id
                identifiers.add(episode_id)
                assert row['split'] == split
                assert row['config']['gps']['noise_std_m'] == [1.5, 1.5, 3.0]
            with np.load(file, allow_pickle=False) as data:
                assert data['features'].shape == (len(rows), 3000, 8, 10)
                assert np.isfinite(data['features']).all()
                if recorded_mode:
                    assert data['replay_source_gps_m'].shape == (len(rows), 3000, 8, 3)
                    for row_index, row in enumerate(rows):
                        attack = row['attack']
                        recording = row['replay_recording']
                        source = data['replay_source_gps_m'][row_index]
                        if attack['attack_type'] == 'replay':
                            assert recording['present']
                            source_id = recording['source_id']
                            assert source_id not in recording_ids, source_id
                            recording_ids.add(source_id)
                            recording_ids_by_split[split].append(source_id)
                            assert recording['split'] == split
                            assert recording['victim_episode_id'] == row['episode_id']
                            assert recording['gps_noise_reused_without_resampling']
                            start, end = attack['start_step'], attack['end_step']
                            offset = attack['parameters']['delay_steps']
                            targets = np.asarray(attack['target_uavs'], dtype=np.int64)
                            np.testing.assert_array_equal(
                                data['gps_observed_m'][row_index, start:end][:, targets],
                                source[start-offset:end-offset][:, targets],
                            )
                        else:
                            assert not recording['present']
                            assert not np.any(source)
                noise = data['gps_clean_m'].astype(np.float64) - data['true_position_m']
                assert np.isfinite(noise).all()
                flat = noise.reshape(-1, 3)
                total_noise_sum += flat.sum(axis=0)
                total_noise_square += np.square(flat).sum(axis=0)
                sample_count += flat.shape[0]
                assert data['available_time_s'].shape == (len(rows), 3000)
                np.testing.assert_allclose(
                    data['available_time_s'],
                    np.broadcast_to((np.arange(3000) + 1) * .02, (len(rows), 3000)),
                    atol=1e-10,
                )
                for name in ('isac_measurement_covariance', 'isac_reported_covariance',
                             'isac_reported_los_covariance'):
                    assert name not in data.files
            byte_count += file.stat().st_size
        assert count == required, (split, count, required)
        counts[split] = count
        assert manifest['scenario_counts_by_split'][split] == {
            name: required // 4 for name in ('clean', 'random', 'replay', 'stealth')
        }
    assert len(identifiers) == expected
    if recorded_mode:
        assert len(recording_ids) == manifest['scenario_counts']['replay']
        assert len(recording_ids) == manifest['replay_recording_count']
        assert recording_ids_by_split == manifest['replay_recording_sources_by_split']
    mean = total_noise_sum / sample_count
    std = np.sqrt(total_noise_square / sample_count - mean**2)
    np.testing.assert_allclose(std, [1.5, 1.5, 3.0], rtol=.01)
    return {
        'status': 'verified', 'episodes': expected, 'unique_episode_ids': len(identifiers),
        'episodes_by_split': counts, 'feature_shape_per_episode': [3000, 8, 10],
        'gps_noise_sample_count': sample_count, 'gps_noise_mean_m': mean.tolist(),
        'gps_noise_std_m': std.tolist(), 'npz_bytes': byte_count,
        'replay_recording_count': len(recording_ids),
        'scope': 'all shard metadata, finite features, GPS noise, availability times and replay sources',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--replay-source', choices=('recorded_normal_episode', 'delayed_clean_stream'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be new or empty')
    if args.workers < 1:
        parser.error('workers must be positive')
    args.run_directory.mkdir(parents=True, exist_ok=True)
    status_path = args.run_directory / 'status.json'
    if status_path.exists():
        parser.error('run directory already contains a job status')
    config = json.loads((project / 'configs/default.json').read_text(encoding='utf-8'))
    if args.replay_source is not None:
        config['attack']['replay_source'] = args.replay_source
    assert config['gps']['noise_std_m'] == [1.5, 1.5, 3.0]
    assert config['simulation']['trajectory_types'] == ['s_curve']
    snapshot = args.run_directory / 'config.json'
    write_json(snapshot, config)
    command = [
        sys.executable, '-m', 'isac_spoofing.cli.generate_dataset',
        '--config', str(snapshot.resolve()), '--output', str(args.output.resolve()),
        '--episodes', '5000', '--seed', '20260714', '--episodes-per-shard', '2',
        '--workers', str(args.workers), '--audit-level', 'standard',
    ]
    started = time.perf_counter()
    status = {
        'status': 'starting', 'supervisor_pid': os.getpid(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'output': str(args.output.resolve()), 'command': command,
        'expected_episodes': 5000, 'expected_splits': [4000, 500, 500],
    }
    write_json(status_path, status)
    environment = dict(os.environ)
    environment.update({
        'PYTHONPATH': str(project / 'src'), 'PYTHONDONTWRITEBYTECODE': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
    })
    try:
        with (args.run_directory / 'stdout.log').open('w', encoding='utf-8') as out:
            with (args.run_directory / 'stderr.log').open('w', encoding='utf-8') as err:
                process = subprocess.Popen(command, cwd=project, env=environment,
                                           stdout=out, stderr=err)
                status.update(status='generating', generator_pid=process.pid)
                write_json(status_path, status)
                exit_code = process.wait()
        status['generator_exit_code'] = exit_code
        if exit_code:
            raise RuntimeError(f'dataset generator exited with code {exit_code}')
        status.update(status='verifying', elapsed_s=time.perf_counter() - started)
        write_json(status_path, status)
        verification = verify_dataset(args.output, 5000, 20260714)
        write_json(args.run_directory / 'verification.json', verification)
        status.update(status='completed', verification=verification)
    except BaseException as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        status.update(elapsed_s=time.perf_counter() - started,
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)


if __name__ == '__main__':
    main()

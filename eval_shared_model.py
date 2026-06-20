#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import get_config
from envs.env_discrete import DiscreteActionEnv
from envs.env_wrappers import DummyVecEnv
from runner.shared.env_runner import EnvRunner as Runner


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default="results/stage2DetectEnvISAC6/MyEnv/mappo/check/run3/models",
        # default="results/stage2DetectEnvISAC5/MyEnv/mappo/check/run2/models",
        # required=True,
        help="Directory containing shared-policy actor.pt and critic.pt.",
    )
    parser.add_argument("--n_eval_envs", type=int, default=2, help="Number of seeded test environments.")
    parser.add_argument("--episodes_per_env", type=int, default=1000)
    parser.add_argument("--seed_start", type=int, default=20001, help="Seed of the first test environment.")
    parser.add_argument("--seed_stride", type=int, default=1000, help="Seed offset between test environments.")
    parser.add_argument("--max_steps", type=int, default=400, help="Maximum length of each evaluation episode.")
    parser.add_argument("--eval_log_interval", type=int, default=5, help="TensorBoard/progress interval in episodes.")
    parser.add_argument("--deterministic_eval", action="store_true", default=True)
    parser.add_argument("--csv_path", type=str, default="results/test/eval_shared.csv")
    parser.add_argument("--json_path", type=str, default="results/test/eval_shared.json")
    parser.add_argument("--num_agents", type=int, default=8)
    return parser


def _t2n(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def load_shared_checkpoint(runner: Runner, model_dir: str, device: torch.device) -> None:
    model_dir = Path(model_dir)
    actor_path = model_dir / "actor.pt"
    critic_path = model_dir / "critic.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Missing shared actor checkpoint: {actor_path}")
    if not critic_path.exists():
        raise FileNotFoundError(f"Missing shared critic checkpoint: {critic_path}")

    runner.trainer.policy.actor.load_state_dict(_safe_torch_load(actor_path, device))
    runner.trainer.policy.critic.load_state_dict(_safe_torch_load(critic_path, device))
    runner.trainer.policy.actor.to(device).eval()
    runner.trainer.policy.critic.to(device).eval()


def make_eval_env(seed: int) -> DummyVecEnv:
    return DummyVecEnv([lambda: DiscreteActionEnv(seed=int(seed))])


def eval_seeds(args: argparse.Namespace) -> List[int]:
    return [
        int(args.seed_start + env_idx * args.seed_stride)
        for env_idx in range(int(args.n_eval_envs))
    ]


def reduce_done(dones) -> bool:
    arr = np.asarray(dones)
    if arr.ndim == 0:
        return bool(arr)
    if arr.ndim == 1:
        return bool(arr[0])
    if arr.ndim == 2:
        return bool(arr[0, 0])
    if arr.ndim == 3:
        return bool(arr[0, 0, 0])
    raise ValueError(f"Unsupported dones shape: {arr.shape}")


def first_info(infos) -> Dict:
    try:
        return infos[0][0]
    except Exception:
        return {}


def update_confusion(acc: Dict[str, int], info: Dict) -> None:
    counts = info.get("det_tp_fp_fn_tn", None)
    if counts is None:
        return
    acc["tp"] += int(counts[0])
    acc["fp"] += int(counts[1])
    acc["fn"] += int(counts[2])
    acc["tn"] += int(counts[3])


def confusion_metrics(acc: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
    return {
        "det_precision": precision,
        "det_recall": recall,
        "det_f1": f1,
        "det_accuracy": accuracy,
        "det_fpr": fpr,
        "det_tp": int(tp),
        "det_fp": int(fp),
        "det_fn": int(fn),
        "det_tn": int(tn),
    }


def detection_log_metrics(
    confusion: Dict[str, int],
    delay_steps: List[float],
    delay_seconds: List[float],
    event_started: int,
    event_detected: int,
    event_missed: int,
) -> Dict[str, float]:
    metrics = confusion_metrics(confusion)
    rate_den = max(event_started, event_detected + event_missed, 1)
    return {
        **metrics,
        "det_delay_mean_steps": float(np.mean(delay_steps)) if delay_steps else 0.0,
        "det_delay_max_steps": float(np.max(delay_steps)) if delay_steps else 0.0,
        "det_delay_mean_seconds": float(np.mean(delay_seconds)) if delay_seconds else 0.0,
        "det_delay_max_seconds": float(np.max(delay_seconds)) if delay_seconds else 0.0,
        "det_event_detection_rate": event_detected / rate_den,
        "det_event_started": int(event_started),
        "det_event_detected": int(event_detected),
        "det_event_missed": int(event_missed),
    }


def action_to_env(eval_actions: np.ndarray, action_space) -> np.ndarray:
    space_name = action_space.__class__.__name__
    if space_name == "MultiDiscrete":
        action_env = None
        for i in range(action_space.shape):
            uc = np.eye(action_space.high[i] + 1)[eval_actions[:, :, i]]
            action_env = uc if action_env is None else np.concatenate((action_env, uc), axis=2)
        return action_env
    if space_name == "Discrete":
        return np.squeeze(np.eye(action_space.n)[eval_actions], 2)
    if space_name in ("Box", "Tuple"):
        return eval_actions
    raise NotImplementedError(f"Unsupported action space: {space_name}")


@torch.no_grad()
def policy_action(runner: Runner, obs: np.ndarray, rnn_states: np.ndarray, masks: np.ndarray):
    runner.trainer.prep_rollout()
    action, next_rnn_states = runner.trainer.policy.act(
        np.concatenate(obs),
        np.concatenate(rnn_states),
        np.concatenate(masks),
        deterministic=True,
    )
    eval_actions = np.array(np.split(_t2n(action), runner.n_rollout_threads))
    next_rnn_states = np.array(np.split(_t2n(next_rnn_states), runner.n_rollout_threads))
    actions_env = action_to_env(eval_actions, runner.envs.action_space[0])
    return actions_env, next_rnn_states


def evaluate(runner: Runner, args: argparse.Namespace) -> Tuple[Dict, List[Dict]]:
    rows: List[Dict] = []
    environment_summaries: List[Dict] = []
    confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    delay_steps: List[float] = []
    delay_seconds: List[float] = []
    event_started = 0
    event_detected = 0
    event_missed = 0
    start_time = time.time()
    seeds = eval_seeds(args)
    total_episodes = int(args.episodes_per_env) * len(seeds)
    completed_episodes = 0

    for env_index, seed in enumerate(seeds, start=1):
        envs = make_eval_env(seed)
        runner.envs = envs
        runner.n_rollout_threads = 1
        env_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        env_delay_steps: List[float] = []
        env_delay_seconds: List[float] = []
        env_event_started = env_event_detected = env_event_missed = 0
        auto_reset_obs = None

        try:
            for ep_in_env in range(int(args.episodes_per_env)):
                obs = auto_reset_obs if auto_reset_obs is not None else envs.reset()
                auto_reset_obs = None
                rnn_states = np.zeros(
                    (1, args.num_agents, runner.recurrent_N, runner.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((1, args.num_agents, 1), dtype=np.float32)

                ep_steps = 0
                ep_conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                ep_delay_steps: List[float] = []
                ep_delay_seconds: List[float] = []
                ep_event_started = 0
                ep_event_detected = 0
                ep_event_missed = 0

                while ep_steps < int(args.max_steps):
                    actions_env, rnn_states = policy_action(runner, obs, rnn_states, masks)
                    obs, _rewards, dones, infos = envs.step(actions_env)
                    ep_steps += 1
                    info = first_info(infos)

                    update_confusion(confusion, info)
                    update_confusion(env_confusion, info)
                    update_confusion(ep_conf, info)

                    ds = [float(x) for x in info.get("det_delay_steps", [])]
                    dsec = [float(x) for x in info.get("det_delay_seconds", [])]
                    delay_steps.extend(ds)
                    delay_seconds.extend(dsec)
                    env_delay_steps.extend(ds)
                    env_delay_seconds.extend(dsec)
                    ep_delay_steps.extend(ds)
                    ep_delay_seconds.extend(dsec)

                    started = int(info.get("det_event_started", 0))
                    detected = int(info.get("det_event_detected", 0))
                    missed = int(info.get("det_event_missed", 0))
                    event_started += started
                    event_detected += detected
                    event_missed += missed
                    env_event_started += started
                    env_event_detected += detected
                    env_event_missed += missed
                    ep_event_started += started
                    ep_event_detected += detected
                    ep_event_missed += missed

                    done = reduce_done(dones)
                    masks = np.ones((1, args.num_agents, 1), dtype=np.float32)
                    if done:
                        masks[:] = 0.0
                        auto_reset_obs = obs
                        break

                completed_episodes += 1
                ep_metrics = confusion_metrics(ep_conf)
                rows.append({
                    "episode": completed_episodes,
                    "environment": env_index,
                    "episode_in_environment": ep_in_env + 1,
                    "seed": seed,
                    "steps": ep_steps,
                    "det_precision": ep_metrics["det_precision"],
                    "det_recall": ep_metrics["det_recall"],
                    "det_f1": ep_metrics["det_f1"],
                    "det_accuracy": ep_metrics["det_accuracy"],
                    "det_fpr": ep_metrics["det_fpr"],
                    "det_tp": ep_metrics["det_tp"],
                    "det_fp": ep_metrics["det_fp"],
                    "det_fn": ep_metrics["det_fn"],
                    "det_tn": ep_metrics["det_tn"],
                    "det_delay_mean_steps": float(np.mean(ep_delay_steps)) if ep_delay_steps else 0.0,
                    "det_delay_mean_seconds": float(np.mean(ep_delay_seconds)) if ep_delay_seconds else 0.0,
                    "det_event_started": ep_event_started,
                    "det_event_detected": ep_event_detected,
                    "det_event_missed": ep_event_missed,
                })

                if completed_episodes % max(1, int(args.eval_log_interval)) == 0 or completed_episodes == total_episodes:
                    log_metrics = detection_log_metrics(
                        confusion,
                        delay_steps,
                        delay_seconds,
                        event_started,
                        event_detected,
                        event_missed,
                    )
                    runner.log_train(log_metrics, completed_episodes)
                    elapsed = time.time() - start_time
                    print(
                        f"[EVAL] {completed_episodes}/{total_episodes} "
                        f"env={env_index}/{len(seeds)} seed={seed} "
                        f"F1={log_metrics['det_f1']:.4f} P={log_metrics['det_precision']:.4f} "
                        f"R={log_metrics['det_recall']:.4f} FPR={log_metrics['det_fpr']:.4f} "
                        f"delay_steps={log_metrics['det_delay_mean_steps']:.2f} "
                        f"eps/s={completed_episodes / max(elapsed, 1e-9):.2f}",
                        flush=True,
                    )
        finally:
            envs.close()

        env_metrics = detection_log_metrics(
            env_confusion,
            env_delay_steps,
            env_delay_seconds,
            env_event_started,
            env_event_detected,
            env_event_missed,
        )
        environment_summaries.append({
            "environment": env_index,
            "seed": seed,
            "episodes": int(args.episodes_per_env),
            **env_metrics,
        })

    metrics = detection_log_metrics(
        confusion,
        delay_steps,
        delay_seconds,
        event_started,
        event_detected,
        event_missed,
    )
    summary = {
        "n_eval_envs": len(seeds),
        "environment_seeds": seeds,
        "episodes_per_env": int(args.episodes_per_env),
        "episodes": total_episodes,
        "max_steps": int(args.max_steps),
        **metrics,
        "environments": environment_summaries,
    }
    return summary, rows


def write_csv(path: str, rows: List[Dict]) -> None:
    if not path:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path_obj.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str, payload: Dict) -> None:
    if not path:
        return
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def print_detection_summary(summary: Dict) -> None:
    print("[EVAL-SHARED][DETECTION-BY-ENV]")
    for env in summary["environments"]:
        print(
            f"env={env['environment']} seed={env['seed']} episodes={env['episodes']} "
            f"precision={env['det_precision']:.6f} recall={env['det_recall']:.6f} "
            f"f1={env['det_f1']:.6f} accuracy={env['det_accuracy']:.6f} "
            f"fpr={env['det_fpr']:.6f} "
            f"delay_steps={env['det_delay_mean_steps']:.6f} "
            f"delay_seconds={env['det_delay_mean_seconds']:.6f} "
            f"event_detection_rate={env['det_event_detection_rate']:.6f} "
            f"tp={env['det_tp']} fp={env['det_fp']} "
            f"fn={env['det_fn']} tn={env['det_tn']}"
        )

    print("[EVAL-SHARED][DETECTION-TOTAL]")
    print(
        f"environments={summary['n_eval_envs']} episodes={summary['episodes']} "
        f"precision={summary['det_precision']:.6f} recall={summary['det_recall']:.6f} "
        f"f1={summary['det_f1']:.6f} accuracy={summary['det_accuracy']:.6f} "
        f"fpr={summary['det_fpr']:.6f} "
        f"mean_delay_steps={summary['det_delay_mean_steps']:.6f} "
        f"mean_delay_seconds={summary['det_delay_mean_seconds']:.6f} "
        f"event_detection_rate={summary['det_event_detection_rate']:.6f} "
        f"tp={summary['det_tp']} fp={summary['det_fp']} "
        f"fn={summary['det_fn']} tn={summary['det_tn']}"
    )


def main() -> None:
    parser = add_eval_args(get_config())
    args = parser.parse_args()

    args.share_policy = True
    args.n_rollout_threads = 1
    args.n_eval_rollout_threads = 1
    args.model_dir = None

    device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")
    torch.set_num_threads(getattr(args, "n_training_threads", 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    bootstrap_envs = make_eval_env(args.seed_start)
    # run_dir = Path("./results/test/full_shared_k_7_stealth")
    run_dir = Path("./results/test2/local_shared_3_bak2")
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(
        {
            "all_args": args,
            "envs": bootstrap_envs,
            "eval_envs": bootstrap_envs,
            "num_agents": args.num_agents,
            "device": device,
            "run_dir": run_dir,
        }
    )
    load_shared_checkpoint(runner, args.saved_model_dir, device)
    bootstrap_envs.close()

    summary, rows = evaluate(runner, args)
    write_csv(args.csv_path, rows)
    write_json(args.json_path, summary)

    print("[EVAL-SHARED][FINAL]")
    print_detection_summary(summary)
    if args.csv_path:
        print(f"csv: {args.csv_path}")
    if args.json_path:
        print(f"json: {args.json_path}")
    runner.writter.export_scalars_to_json(str(Path(runner.log_dir) / "summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main()

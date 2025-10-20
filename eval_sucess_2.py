#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_under_attack.py
评测：已训练好的 Stage-1 导航策略 在 GPS 欺骗攻击下的到达率（无检测/缓解）。
- 关键修复：RNN 隐状态四维 (n_envs, n_agents, rec_N, hidden_size)；
           按 agent 清零 rnn/masks；稳健 done 规约；可选 GIF 录制。
- 新增：将每回合指标与最终汇总写入 CSV（--csv_path / --csv_append）。
"""
import os
import sys
import time
from pathlib import Path
import argparse
from typing import Tuple, Dict, List, Optional
import csv
from datetime import datetime

import numpy as np
import torch

from utils.load import _load_agent_ckpt

# ==== 复用工程结构 ====
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))
from config import get_config
try:
    from train import make_eval_env          # 若 make_eval_env 在根目录 train.py
except Exception:
    from train.train import make_eval_env    # 若在包 train/train.py

from runner.separated.env_runner import EnvRunner as Runner


# ========== 可选攻击器（优先导入你的实现；否则用兜底） ==========
def _import_attack_manager():
    try:
        from gps_attack_manager import GPSAttackManager, GPSAttackScenario  # type: ignore
        return GPSAttackManager, GPSAttackScenario
    except Exception:
        from dataclasses import dataclass

        @dataclass
        class GPSAttackScenario:
            t0: int
            t1: int
            mask: np.ndarray
            atk_type: str

        class GPSAttackManager:
            def __init__(self,
                         rng: np.random.RandomState,
                         N: int,
                         p_global: float = 0.7,
                         atk_types: Tuple[str, ...] = ("random", "replay", "stealth"),
                         t_min_ratio: float = 0.2,
                         t_max_ratio: float = 0.7,
                         dur_ratio: Tuple[float, float] = (0.2, 0.6),
                         random_sigma: float = 15.0,
                         stealth_rate: float = 0.4,
                         replay_delay_range: Tuple[float, float] = (5.0, 20.0)):
                self.rng = rng
                self.N = int(N)
                self.p_global = float(p_global)
                self.atk_types = atk_types
                self.t_min_ratio = float(t_min_ratio)
                self.t_max_ratio = float(t_max_ratio)
                self.dur_ratio = (float(dur_ratio[0]), float(dur_ratio[1]))
                self.params: Dict[str, object] = dict(
                    random_sigma=float(random_sigma),
                    stealth_rate=float(stealth_rate),
                    replay_delay_range=(float(replay_delay_range[0]), float(replay_delay_range[1])),
                )

            def sample_scenario(self, T_max: int, dt: float) -> GPSAttackScenario:
                t0 = int(self.rng.uniform(self.t_min_ratio, self.t_max_ratio) * T_max)
                dur = int(self.rng.uniform(*self.dur_ratio) * T_max)
                t1 = min(T_max - 1, t0 + max(1, dur))
                # 全局或局部攻击掩码
                if self.rng.rand() < self.p_global:
                    mask = np.ones(self.N, dtype=bool)
                else:
                    upper = max(2, self.N // 2 + 1)
                    k = self.rng.randint(1, min(upper, self.N + 1))
                    idx = self.rng.choice(self.N, size=k, replace=False)
                    mask = np.zeros(self.N, dtype=bool)
                    mask[idx] = True
                atk_type = str(self.rng.choice(self.atk_types))
                return GPSAttackScenario(t0=t0, t1=t1, mask=mask, atk_type=atk_type)

            def apply(self,
                      t: int,
                      true_pos: np.ndarray,
                      hist_true_pos: List[np.ndarray],
                      hist_gps_pos: List[np.ndarray],
                      scenario: GPSAttackScenario,
                      dt: float,
                      heading_now: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
                z = true_pos.copy().astype(np.float32)
                active = np.zeros(self.N, dtype=bool)
                if not (scenario.t0 <= t <= scenario.t1):
                    return z, active
                active = scenario.mask
                typ = scenario.atk_type
                D = true_pos.shape[1]
                if typ == "random":
                    sigma = float(self.params["random_sigma"])
                    noise = self.rng.normal(0.0, sigma, size=(int(active.sum()), D)).astype(np.float32)
                    z[active] += noise
                elif typ == "replay":
                    lo, hi = self.params["replay_delay_range"]
                    delay = float(self.rng.uniform(lo, hi))
                    k = max(0, t - int(round(delay / dt)))
                    src = hist_true_pos[k] if k < len(hist_true_pos) else true_pos
                    z[active] = src[active]
                elif typ == "stealth":
                    rate = float(self.params["stealth_rate"])
                    drift = rate * dt * max(0, t - scenario.t0)
                    if heading_now is not None:
                        z[active] = true_pos[active] + (drift * heading_now[active])
                    else:
                        v = np.zeros((D,), dtype=np.float32)
                        v[0] = 1.0
                        z[active] = true_pos[active] + (drift * v)
                return z, active

        return GPSAttackManager, GPSAttackScenario


GPSAttackManager, GPSAttackScenario = _import_attack_manager()


# ========== CLI ==========
def add_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--saved_model_dir", type=str,
                        default="D:\\Reproduction_of_the_paper\\light_mappo\\results\\MyEnv\\MyEnv\\mappo\\check\\run2\\models",
                        help="包含 actor_agent*.pt / critic_agent*.pt 的目录")
    parser.add_argument("--n_eval_episodes", type=int, default=200)
    parser.add_argument("--deterministic_eval", action="store_true", default=True)
    parser.add_argument("--num_agents", type=int, default=8)

    # 攻击参数（默认不启用攻击请改为不传本 flag；此处默认 True 以便演示）e
    parser.add_argument("--enable_attack", action="store_true",
                        help="启用则在动作前注入GPS攻击并重算观测", default=True)
    parser.add_argument("--atk_types", type=str, default="random,replay,stealth")
    parser.add_argument("--p_global", type=float, default=0.7)
    parser.add_argument("--t_min_ratio", type=float, default=0.2)
    parser.add_argument("--t_max_ratio", type=float, default=0.7)
    parser.add_argument("--dur_lo_ratio", type=float, default=0.2)
    parser.add_argument("--dur_hi_ratio", type=float, default=0.6)
    parser.add_argument("--random_sigma", type=float, default=15.0)
    parser.add_argument("--stealth_rate", type=float, default=0.4)
    parser.add_argument("--replay_delay_lo", type=float, default=5.0)
    parser.add_argument("--replay_delay_hi", type=float, default=20.0)
    parser.add_argument("--sample_scenario_each_ep", action="store_true",
                        help="每回合重新采样攻击场景（默认：仅首回合采样并贯穿本回合）")

    # 渲染/录制（可选）
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record_dir", type=str, default="",
                        help="若指定则保存 eval.gif 到该目录")

    # 日志 & CSV
    parser.add_argument("--csv_path", type=str, default="./eval_with_attack.csv",
                        help="若指定，则将每回合与最终统计写入该 CSV 文件")
    parser.add_argument("--csv_append", action="store_true",
                        help="CSV 已存在时采用追加模式（否则重写表头）")
    return parser


# ========== 工具函数 ==========
def _t2n(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def unwrap_single_env(obj):
    """尽量拿到底层单环境实例（便于访问 true_p / p / _get_obs等）"""
    seen = set()
    cur = obj
    while True:
        for name in ("envs", "venv", "env", "_env"):
            if hasattr(cur, name):
                nxt = getattr(cur, name)
                if isinstance(nxt, (list, tuple)) and len(nxt) > 0:
                    cur = nxt[0]
                else:
                    cur = nxt
                if id(cur) in seen:
                    return cur
                seen.add(id(cur))
                break
        else:
            return cur


def _reduce_done_flags(dones) -> List[bool]:
    """
    将 dones 规约为 [n_envs] 的布尔列表（每个 env 是否结束）。
    规则：以每个 env 的第 0 个 agent 的 done 作为该 env 的终止标志（与你训练时代码一致）。
    """
    arr = np.asarray(dones)
    if arr.ndim == 0:
        return [bool(arr)]
    if arr.ndim == 1:
        return [bool(x) for x in arr]
    if arr.ndim == 2:      # [n_envs, n_agents]
        return [bool(arr[i, 0]) for i in range(arr.shape[0])]
    if arr.ndim == 3:      # [n_envs, n_agents, 1]
        return [bool(arr[i, 0, 0]) for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported dones shape: {arr.shape}")


def _open_csv_writer(path: str, append: bool, fieldnames: list):
    """打开 CSV 写入器；不存在或非 append 时写表头。返回 (file_obj, writer)。"""
    if not path:
        return None, None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    mode = "a" if (append and file_exists) else "w"
    f = open(path, mode, newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
    return f, writer


# ========== 评测（动作前注入 + 观测维度自动对齐，启用攻击时生效） ==========
def evaluate_under_attack(runner, args, csv_path: str = "", csv_append: bool = False) -> dict:
    envs = runner.envs
    num_agents = runner.num_agents
    n_envs = 1
    rec_N = runner.recurrent_N
    hidden_size = runner.hidden_size

    # —— 强制单环境 —— #
    try:
        assert getattr(envs, "num_envs", 1) == 1, "请将 n_rollout_threads 设为 1"
    except Exception:
        pass

    # —— 攻击组件 —— #
    uav = unwrap_single_env(envs)
    rng = np.random.RandomState(args.seed)
    atk_enabled = bool(args.enable_attack)
    if atk_enabled:
        atk_types = tuple([s.strip() for s in str(args.atk_types).split(",") if s.strip()])
        atk_mgr = GPSAttackManager(
            rng=rng, N=num_agents, p_global=args.p_global, atk_types=atk_types,
            t_min_ratio=args.t_min_ratio, t_max_ratio=args.t_max_ratio,
            dur_ratio=(args.dur_lo_ratio, args.dur_hi_ratio),
            random_sigma=args.random_sigma, stealth_rate=args.stealth_rate,
            replay_delay_range=(args.replay_delay_lo, args.replay_delay_hi),
        )

    # ===== CSV 初始化 =====
    csv_fields = [
        "ts", "episode", "term_reason", "episode_return",
        "success_so_far", "collision_so_far", "timeout_so_far",
        "succ_rate", "coll_rate", "tout_rate",
        "mean_return_so_far",
        "enable_attack", "attack_type", "attack_t0", "attack_t1",
        "N", "obs_dim", "deterministic_eval"
    ]
    csv_f, csv_w = _open_csv_writer(csv_path, csv_append, csv_fields)

    # —— 统计 —— #
    success = collision = timeout = 0
    team_returns: List[float] = []
    frames: List[np.ndarray] = []

    # reset
    obs = envs.reset()  # [1, N, obs_dim_train]
    # 训练期期望 obs 维度（由 reset 首次返回推断）
    expected_obs_dim = int(obs.shape[-1])

    # RNN states/masks（关键：四维）
    rnn_states = np.zeros((n_envs, num_agents, rec_N, hidden_size), np.float32)
    masks = np.ones((n_envs, num_agents, 1), np.float32)

    # 攻击历史
    hist_true: List[np.ndarray] = []
    hist_gps: List[np.ndarray] = []

    # 时序
    Tm = int(getattr(uav, "T_max", 1000))
    dt = float(getattr(uav, "dt", 0.1))
    scenario = None

    finished_eps = 0
    ep_return_acc = 0.0
    start_t = time.time()
    LOG_INT = max(1, int(args.log_interval))

    while finished_eps < args.n_eval_episodes:
        # === 1) 若启用攻击：在动作前注入并重算观测 ===
        if atk_enabled:
            # t_now / true_pos / heading
            t_now = int(getattr(uav, "t", 0))
            true_pos = np.asarray(getattr(uav, "true_p", None), dtype=np.float32)
            if true_pos is None:
                raise RuntimeError("环境缺少 true_p，请确保 true_p 表示真实位置。")

            heading_now = None
            if hasattr(uav, "h_hist") and isinstance(uav.h_hist, (list, tuple)) and len(uav.h_hist) > 0:
                heading_now = np.asarray(uav.h_hist[-1], dtype=np.float32)

            # 起始或每回合采样新场景
            if (t_now == 0) and (finished_eps == 0 or args.sample_scenario_each_ep):
                scenario = atk_mgr.sample_scenario(T_max=Tm, dt=dt)
                hist_true.clear()
                hist_gps.clear()

            # 应用攻击 -> 覆盖GPS观测位置到 uav.p
            z_gps, active = atk_mgr.apply(
                t=t_now, true_pos=true_pos,
                hist_true_pos=hist_true, hist_gps_pos=hist_gps,
                scenario=scenario, dt=dt, heading_now=heading_now
            )
            uav.p = np.asarray(z_gps, dtype=np.float32)

            # 维护历史
            if (len(hist_true) == 0) or (not np.array_equal(hist_true[-1], true_pos)):
                hist_true.append(true_pos.copy())
            if (len(hist_gps) == 0) or (not np.array_equal(hist_gps[-1], z_gps)):
                hist_gps.append(z_gps.copy())

            # 立刻重算观测，并对齐到训练维度
            attacked = uav._get_obs()["obs"].astype(np.float32)  # (N, d_now)
            d_now = attacked.shape[1]
            if d_now < expected_obs_dim:
                pad = np.zeros((num_agents, expected_obs_dim - d_now), dtype=np.float32)
                attacked = np.concatenate([attacked, pad], axis=1)
            elif d_now > expected_obs_dim:
                attacked = attacked[:, :expected_obs_dim]
            obs = attacked[None, :, :]  # -> [1, N, expected_obs_dim]

        # === 2) 生成动作 ===
        temp_actions_env = []
        for aid in range(num_agents):
            runner.trainer[aid].prep_rollout()
            action, rnn_state = runner.trainer[aid].policy.act(
                np.array(list(obs[:, aid])),     # [n_envs, obs_dim]
                rnn_states[:, aid],              # [n_envs, rec_N, hidden]
                masks[:, aid],                   # [n_envs, 1]
                deterministic=args.deterministic_eval,
            )
            action = action.detach().cpu().numpy()

            # 动作空间映射
            space_name = envs.action_space[aid].__class__.__name__
            if space_name in ["Box", "Tuple"]:
                action_env = action  # 连续空间直通
            elif space_name == "MultiDiscrete":
                for i in range(envs.action_space[aid].shape):
                    uc = np.eye(envs.action_space[aid].high[i] + 1)[action[:, i]]
                    action_env = uc if i == 0 else np.concatenate((action_env, uc), axis=1)
            elif space_name == "Discrete":
                action_env = np.squeeze(np.eye(envs.action_space[aid].n)[action], 1)
            else:
                raise NotImplementedError(f"未适配的动作空间: {space_name}")

            temp_actions_env.append(action_env)
            rnn_states[:, aid] = _t2n(rnn_state)

        # [n_envs, n_agents, act_dim]
        actions_env = []
        for i in range(n_envs):
            one_env_actions = []
            for temp in temp_actions_env:
                one_env_actions.append(temp[i])
            actions_env.append(one_env_actions)

        # === 3) 环境前进一步 ===
        obs, rewards, dones, infos = envs.step(actions_env)

        # 团队回报（各 agent 求和）
        r = np.array(rewards)  # [n_envs, n_agents]
        ep_return_acc += float(r.sum())

        # === 4) RNN/masks 按 agent 清零，且判断整局结束 ===
        dones_arr = np.asarray(dones, dtype=bool)
        if dones_arr.ndim == 3:      # [n_envs, n_agents, 1]
            agent_done = dones_arr[:, :, 0]
        elif dones_arr.ndim == 2:    # [n_envs, n_agents]
            agent_done = dones_arr
        else:
            agent_done = np.zeros((n_envs, num_agents), dtype=bool)

        # 清零对应 agent 的 rnn 状态；masks=0
        rnn_states[agent_done] = 0.0
        masks = np.ones((n_envs, num_agents, 1), np.float32)
        masks[agent_done] = 0.0

        # 当前回合是否结束
        env_done = all(_reduce_done_flags(dones))

        # 可选渲染/录制
        if args.render or args.record_dir:
            try:
                frame = envs.render(mode="rgb_array")
                if args.record_dir:
                    # 兼容返回嵌套(list/tuple)的情况
                    frame = frame[0][0] if isinstance(frame, (list, tuple)) else frame
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)
            except Exception:
                pass

        if env_done:
            # 终止原因
            term = None
            try:
                term = infos[0][0].get("term_reason", None)
            except Exception:
                pass
            if term == "success":
                success += 1
            elif term == "collision":
                collision += 1
            elif term == "timeout":
                timeout += 1

            team_returns.append(ep_return_acc)
            finished_eps += 1

            # === 写每回合行到 CSV ===
            if csv_w is not None:
                # 解析攻击信息
                atk_type, atk_t0, atk_t1 = "", "", ""
                try:
                    _info = infos[0][0] if isinstance(infos, (list, tuple)) else infos
                    if _info is not None:
                        atk_type = str(_info.get("attack_type", ""))
                        t0t1 = _info.get("attack_t0_t1", ("", ""))
                        atk_t0 = t0t1[0] if isinstance(t0t1, (list, tuple)) and len(t0t1) > 0 else ""
                        atk_t1 = t0t1[1] if isinstance(t0t1, (list, tuple)) and len(t0t1) > 1 else ""
                except Exception:
                    pass

                elapsed = time.time() - start_t
                mean_ret_so_far = float(np.mean(team_returns)) if team_returns else 0.0
                succ_rate = success / finished_eps if finished_eps else 0.0
                coll_rate = collision / finished_eps if finished_eps else 0.0
                tout_rate = timeout / finished_eps if finished_eps else 0.0

                csv_w.writerow({
                    "episode": finished_eps,
                    "succ_rate": f"{succ_rate:.6f}",
                    "coll_rate": f"{coll_rate:.6f}",
                    "tout_rate": f"{tout_rate:.6f}",
                })
                csv_f.flush()

            # 进度
            if (finished_eps % LOG_INT == 0) or (finished_eps == args.n_eval_episodes):
                elapsed = time.time() - start_t
                eps_per_sec = finished_eps / max(1e-9, elapsed)
                mean_ret = float(np.mean(team_returns)) if team_returns else 0.0
                print(
                    f"[EVAL] {finished_eps}/{args.n_eval_episodes} | "
                    f"succ={success/finished_eps:.3f} coll={collision/finished_eps:.3f} "
                    f"timeout={timeout/finished_eps:.3f} | mean_return={mean_ret:.3f} | "
                    f"{eps_per_sec:.2f} eps/s",
                    flush=True,
                )

            # reset
            obs = envs.reset()
            rnn_states[:] = 0.0
            masks[:] = 1.0
            ep_return_acc = 0.0
            # 更新期望 obs 维度（以防 reset 后变动）
            expected_obs_dim = int(obs.shape[-1])
            # 清历史
            if atk_enabled:
                hist_true.clear()
                hist_gps.clear()

    # 保存 gif
    if args.record_dir and len(frames) > 0:
        Path(args.record_dir).mkdir(parents=True, exist_ok=True)
        import imageio
        imageio.mimsave(str(Path(args.record_dir) / "eval.gif"), frames, fps=20)

    stats = {
        "episodes": int(args.n_eval_episodes),
        "mean_team_return": float(np.mean(team_returns)) if team_returns else 0.0,
        "success": int(success),
        "collision": int(collision),
        "timeout": int(timeout),
        "success_rate": success / max(1, finished_eps),
    }

    # === CSV：最终汇总 ===
    if csv_w is not None:
        csv_w.writerow({
            "episode": "FINAL",
            "succ_rate": f"{stats['success_rate']:.6f}",
            "coll_rate": f"{(stats['collision']/max(1, stats['episodes'])):.6f}",
            "tout_rate": f"{(stats['timeout']/max(1, stats['episodes'])):.6f}",
        })
        csv_f.flush()
        csv_f.close()

    return stats


# ========== 主程序 ==========
def main():
    parser = get_config()
    parser = add_eval_args(parser)
    args = parser.parse_args()

    # 设备 & 种子
    device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")
    torch.set_num_threads(getattr(args, "n_training_threads", 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 单环境评测
    args.n_rollout_threads = 1
    args.test_n_rollout_threads = 1
    eval_envs = make_eval_env(args)

    # Runner（仅复用 policy/trainer）
    run_dir = Path("./eval_attack_tmp")
    run_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner({
        "all_args": args,
        "envs": eval_envs,
        "eval_envs": eval_envs,
        "num_agents": args.num_agents,
        "device": device,
        "run_dir": run_dir,
    })

    # 加载权重
    for aid in range(args.num_agents):
        _load_agent_ckpt(runner.trainer[aid], aid, args.saved_model_dir, device)

    # 评测
    stats = evaluate_under_attack(
        runner,
        args,
        csv_path=args.csv_path,
        csv_append=args.csv_append
    )
    print("[EVAL-ATTACK][FINAL]", stats)

    eval_envs.close()


if __name__ == "__main__":
    main()

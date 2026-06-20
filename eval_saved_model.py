# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import os, sys, time
# from pathlib import Path
# import argparse
# import numpy as np
# import torch

# # 复用你的工程结构
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

# from config import get_config
# from train.train import make_eval_env           # 直接用你贴出来的函数
# from runner.separated.env_runner import EnvRunner as Runner

# def add_eval_args(parser):
#     parser.add_argument("--saved_model_dir", type=str, required=False,
#                         help="包含 actor_agent*.pt / critic_agent*.pt 的目录",default="/home/zz/project_Uavs_gps_spoofing_detect/light_mappo/results/MyEnv/MyEnv/mappo/check/run7/models")
#     parser.add_argument("--n_eval_episodes", type=int, default=2000)
#     parser.add_argument("--deterministic_eval", action="store_true",
#                         help="评测使用确定性动作（建议打开）",default=True)
#     parser.add_argument("--render", action="store_true")
#     parser.add_argument("--record_dir", type=str, default="",
#                         help="如需保存动图，指定目录（保存为 eval.gif）")
#     parser.add_argument("--num_agents", type=int, default=8, help="number of players")
#     parser.add_argument("--test_n_rollout_threads", type=int, default=1,
#                         help="评测推荐=1")
#     return parser

# def _t2n(x): return x.detach().cpu().numpy()

# def _load_agent_ckpt(trainer, agent_id, model_dir, device):
#     actor_p = Path(model_dir) / f"actor_agent{agent_id}.pt"
#     critic_p = Path(model_dir) / f"critic_agent{agent_id}.pt"
#     if not actor_p.exists():
#         raise FileNotFoundError(f"缺少 {actor_p}")
#     if not critic_p.exists():
#         raise FileNotFoundError(f"缺少 {critic_p}")
#     actor_sd  = torch.load(str(actor_p),  map_location=device)
#     critic_sd = torch.load(str(critic_p), map_location=device)
#     trainer.policy.actor.load_state_dict(actor_sd)
#     trainer.policy.critic.load_state_dict(critic_sd)
#     trainer.policy.actor.to(device).eval()
#     trainer.policy.critic.to(device).eval()

# def evaluate(runner, n_eps=20000, deterministic=True, render=False, record_dir=""):
#     import imageio

#     envs          = runner.envs
#     device        = runner.device
#     num_agents    = runner.num_agents
#     T             = runner.episode_length
#     n_envs        = 1   # 这里我们会设为1
#     rec_N         = runner.recurrent_N
#     hidden_size   = runner.hidden_size

#     assert n_envs == 1, "评测推荐 --n_rollout_threads=1"

#     frames = []
#     team_returns = []          # 每回合团队总回报
#     success = collision = timeout = 0

#     # reset
#     obs = envs.reset()  # [n_envs, n_agents, obs_dim]

#     # RNN states/masks
#     rnn_states = np.zeros((n_envs, num_agents, rec_N, hidden_size), dtype=np.float32)
#     masks      = np.ones((n_envs, num_agents, 1), dtype=np.float32)

#     finished_eps = 0
#     ep_return_acc = 0.0

#     while finished_eps < n_eps:
#         temp_actions_env = []
#         for agent_id in range(num_agents):
#             runner.trainer[agent_id].prep_rollout()
#             # 使用确定性动作
#             action, rnn_state = runner.trainer[agent_id].policy.act(
#                 np.array(list(obs[:, agent_id])),   # [n_envs, obs_dim]
#                 rnn_states[:, agent_id],
#                 masks[:, agent_id],
#                 deterministic=deterministic
#             )
#             action = action.detach().cpu().numpy()

#             # === 关键：连续动作直通 ===
#             if envs.action_space[agent_id].__class__.__name__ in ["Box", "Tuple"]:
#                 action_env = action
#             elif envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
#                 for i in range(envs.action_space[agent_id].shape):
#                     uc = np.eye(envs.action_space[agent_id].high[i] + 1)[action[:, i]]
#                     action_env = uc if i == 0 else np.concatenate((action_env, uc), axis=1)
#             elif envs.action_space[agent_id].__class__.__name__ == "Discrete":
#                 action_env = np.squeeze(np.eye(envs.action_space[agent_id].n)[action], 1)
#             else:
#                 raise NotImplementedError(f"未适配的动作空间: {envs.action_space[agent_id].__class__.__name__}")

#             temp_actions_env.append(action_env)
#             rnn_states[:, agent_id] = _t2n(rnn_state)

#         # 拼成 [n_envs, n_agents, act_dim]
#         actions_env = []
#         for i in range(n_envs):
#             one_env_actions = []
#             for temp in temp_actions_env:
#                 one_env_actions.append(temp[i])
#             actions_env.append(one_env_actions)

#         # step
#         obs, rewards, dones, infos = envs.step(actions_env)

#         # 统计团队回报（把各agent reward求和）
#         r = np.array(rewards)            # [n_envs, n_agents]
#         ep_return_acc += float(r.sum())

#         # 渲染/录制
#         if render or record_dir:
#             try:
#                 frame = envs.render(mode="rgb_array")
#                 if record_dir:
#                     frames.append(frame[0][0] if isinstance(frame, (list, tuple)) else frame)
#             except Exception:
#                 pass

#         # 处理 done
#         if isinstance(dones, (list, tuple)):
#             done_flags = [dl[0] if isinstance(dl, (list, tuple, np.ndarray)) else dl for dl in dones]
#         else:
#             done_flags = [bool(dones[i, 0]) for i in range(dones.shape[0])]
#         if all(done_flags):
#             # 读取终止原因（你的 env 在 infos[i][0]["term_reason"]）
#             term = None
#             try:
#                 term = infos[0][0].get("term_reason", None)
#             except Exception:
#                 pass
#             if term == "success":    success  += 1
#             elif term == "collision": collision += 1
#             elif term == "timeout":   timeout  += 1

#             team_returns.append(ep_return_acc)
#             finished_eps += 1

#             # reset
#             obs = envs.reset()
#             rnn_states[:] = 0.0
#             masks[:] = 1.0
#             ep_return_acc = 0.0

#         # 更新 masks
#         dones_arr = np.array(dones, dtype=bool)
#         if dones_arr.any():
#             rnn_states[dones_arr] = 0.0
#         masks = np.ones((n_envs, num_agents, 1), dtype=np.float32)
#         masks[dones_arr] = 0.0

#     # 保存gif
#     if record_dir and len(frames) > 0:
#         Path(record_dir).mkdir(parents=True, exist_ok=True)
#         import imageio
#         imageio.mimsave(str(Path(record_dir) / "eval.gif"), frames, fps=20)

#     stats = {
#         "episodes": int(n_eps),
#         "mean_team_return": float(np.mean(team_returns)) if team_returns else 0.0,
#         "success": int(success),
#         "collision": int(collision),
#         "timeout": int(timeout),
#     }
#     return stats

# def main():
#     # 解析参数（继承你的 get_config，再扩展评测参数）
#     parser = get_config()
#     parser = add_eval_args(parser)
#     args = parser.parse_args()

#     # 设备 & 随机种子
#     device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")
#     print(device)
#     torch.set_num_threads(getattr(args, "n_training_threads", 1))
#     torch.manual_seed(args.seed); np.random.seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.seed)
#     print("start eval")
#     # 评测环境（建议单环境）
#     args.test_n_rollout_threads = 1
#     eval_envs = make_eval_env(args)

#     # 构建 Runner（不会调用 run，只为了拿到 trainer/policy 等）
#     dummy_run_dir = Path("./eval_tmp")
#     dummy_run_dir.mkdir(parents=True, exist_ok=True)
#     runner = Runner({
#         "all_args": args,
#         "envs": eval_envs,
#         "eval_envs": eval_envs,
#         "num_agents": args.num_agents,
#         "device": device,
#         "run_dir": dummy_run_dir,
#     })

#     # 加载每个 agent 的权重
#     for aid in range(args.num_agents):
#         _load_agent_ckpt(runner.trainer[aid], aid, args.saved_model_dir, device)

#     # 评测
#     stats = evaluate(
#         runner,
#         n_eps=args.n_eval_episodes,
#         deterministic=args.deterministic_eval,
#         render=args.render,
#         record_dir=args.record_dir
#     )
#     print("[EVAL]", stats)

#     eval_envs.close()

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time
from pathlib import Path
import argparse
import numpy as np
import torch

# 复用工程结构
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from config import get_config

# 兼容两种项目布局：train.py 在根目录 或 包内 train/train.py
try:
    from train import make_eval_env           # 若你的 make_eval_env 在根目录 train.py
except Exception:
    from train.train import make_eval_env     # 若在包 train/train.py

from runner.separated.env_runner import EnvRunner as Runner


def add_eval_args(parser):
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        required=False,
        default="/home/zz/project_Uavs_gps_spoofing_detect/light_mappo/results/MyEnv/MyEnv/mappo/check/run13/models",
        help="包含 actor_agent*.pt / critic_agent*.pt 的目录",
    )
    parser.add_argument("--n_eval_episodes", type=int, default=2000)
    parser.add_argument(
        "--deterministic_eval",
        action="store_true",
        default=True,
        help="评测使用确定性动作（建议打开）",
    )
    parser.add_argument("--render", action="store_true", help="是否弹窗渲染（服务器建议关闭）")
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="保存动图的目录（保存为 eval.gif）；留空则不保存",
    )
    parser.add_argument("--num_agents", type=int, default=8, help="智能体数量")
    parser.add_argument(
        "--test_n_rollout_threads",
        type=int,
        default=1,
        help="评测推荐=1（本脚本会强制使用 1）",
    )
    parser.add_argument(
        "--__log_interval",
        type=int,
        default=20,
        help="每多少回合打印一次评测进度",
    )
    return parser


def _t2n(x):
    return x.detach().cpu().numpy()


def _load_agent_ckpt(trainer, agent_id, model_dir, device):
    actor_p = Path(model_dir) / f"actor_agent{agent_id}.pt"
    critic_p = Path(model_dir) / f"critic_agent{agent_id}.pt"
    if not actor_p.exists():
        raise FileNotFoundError(f"缺少 {actor_p}")
    if not critic_p.exists():
        raise FileNotFoundError(f"缺少 {critic_p}")

    # 优先使用更安全的 weights_only=True；旧版 PyTorch 自动回退
    try:
        actor_sd = torch.load(str(actor_p), map_location=device, weights_only=True)
        critic_sd = torch.load(str(critic_p), map_location=device, weights_only=True)
    except TypeError:
        actor_sd = torch.load(str(actor_p), map_location=device)
        critic_sd = torch.load(str(critic_p), map_location=device)

    trainer.policy.actor.load_state_dict(actor_sd)
    trainer.policy.critic.load_state_dict(critic_sd)
    trainer.policy.actor.to(device).eval()
    trainer.policy.critic.to(device).eval()


def _reduce_done_flags(dones):
    """
    将可能是各种形状/类型的 dones 规约为 [n_envs] 的布尔列表（每个 env 是否结束）。
    规则：以每个 env 的第 0 个 agent 的 done 作为该 env 的终止标志（与你训练时代码一致）。
    """
    arr = np.asarray(dones)
    if arr.ndim == 0:
        return [bool(arr)]
    if arr.ndim == 1:
        # [n_envs]
        return [bool(x) for x in arr]
    if arr.ndim == 2:
        # [n_envs, n_agents]
        return [bool(arr[i, 0]) for i in range(arr.shape[0])]
    if arr.ndim == 3:
        # [n_envs, n_agents, 1]
        return [bool(arr[i, 0, 0]) for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported dones shape: {arr.shape}")


def evaluate(runner, n_eps=2000, deterministic=True, render=False, record_dir="", log_interval=20):
    import imageio

    envs = runner.envs
    device = runner.device
    num_agents = runner.num_agents
    n_envs = 1  # 我们强制单环境
    rec_N = runner.recurrent_N
    hidden_size = runner.hidden_size

    # 安全断言：确保真的是单环境
    try:
        assert getattr(envs, "num_envs", 1) == 1, "请将 n_rollout_threads 设为 1"
    except Exception:
        pass

    frames = []
    team_returns = []  # 每回合团队总回报
    success = collision = timeout = 0

    # reset
    obs = envs.reset()  # [n_envs, n_agents, obs_dim]

    # RNN states / masks
    rnn_states = np.zeros((n_envs, num_agents, rec_N, hidden_size), dtype=np.float32)
    masks = np.ones((n_envs, num_agents, 1), dtype=np.float32)

    finished_eps = 0
    ep_return_acc = 0.0

    start_t = time.time()
    LOG_INTERVAL = max(1, int(log_interval))

    while finished_eps < n_eps:
        temp_actions_env = []
        for agent_id in range(num_agents):
            runner.trainer[agent_id].prep_rollout()
            # 确定性动作
            action, rnn_state = runner.trainer[agent_id].policy.act(
                np.array(list(obs[:, agent_id])),  # [n_envs, obs_dim]
                rnn_states[:, agent_id],
                masks[:, agent_id],
                deterministic=deterministic,
            )
            action = action.detach().cpu().numpy()

            # 动作空间映射
            space_name = envs.action_space[agent_id].__class__.__name__
            if space_name in ["Box", "Tuple"]:
                action_env = action  # 连续空间直通
            elif space_name == "MultiDiscrete":
                for i in range(envs.action_space[agent_id].shape):
                    uc = np.eye(envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    action_env = uc if i == 0 else np.concatenate((action_env, uc), axis=1)
            elif space_name == "Discrete":
                action_env = np.squeeze(np.eye(envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError(f"未适配的动作空间: {space_name}")

            temp_actions_env.append(action_env)
            rnn_states[:, agent_id] = _t2n(rnn_state)

        # 拼成 [n_envs, n_agents, act_dim]
        actions_env = []
        for i in range(n_envs):
            one_env_actions = []
            for temp in temp_actions_env:
                one_env_actions.append(temp[i])
            actions_env.append(one_env_actions)

        # step
        obs, rewards, dones, infos = envs.step(actions_env)

        # 团队回报（各 agent 奖励求和）
        r = np.array(rewards)  # [n_envs, n_agents]
        ep_return_acc += float(r.sum())

        # 处理每个 env 的 done（我们强制 n_envs=1）
        done_flags = _reduce_done_flags(dones)
        env_done = all(done_flags)

        if env_done:
            # 读取终止原因（你的 env: infos[0][0]["term_reason"]）
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

            # —— 每 LOG_INTERVAL 局输出一次进度 —— #
            if (finished_eps % LOG_INTERVAL == 0) or (finished_eps == n_eps):
                elapsed = time.time() - start_t
                eps_per_sec = finished_eps / max(elapsed, 1e-9)
                mean_ret_so_far = float(np.mean(team_returns)) if team_returns else 0.0
                succ_rate = success / finished_eps if finished_eps else 0.0
                coll_rate = collision / finished_eps if finished_eps else 0.0
                tout_rate = timeout / finished_eps if finished_eps else 0.0
                print(
                    f"[EVAL] {finished_eps}/{n_eps} | "
                    f"succ={succ_rate:.3f} coll={coll_rate:.3f} timeout={tout_rate:.3f} | "
                    f"mean_return={mean_ret_so_far:.3f} | {eps_per_sec:.2f} eps/s",
                    flush=True,
                )

            # reset 到下一回合
            obs = envs.reset()
            rnn_states[:] = 0.0
            masks[:] = 1.0
            ep_return_acc = 0.0
        else:
            # 未结束则维持 masks=1（如果你的 env 有“局中单个智能体提前 done”的设计，再按需细化）
            masks[:] = 1.0

    # 保存 gif
    if record_dir and len(frames) > 0:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
        import imageio
        imageio.mimsave(str(Path(record_dir) / "eval.gif"), frames, fps=20)

    stats = {
        "episodes": int(n_eps),
        "mean_team_return": float(np.mean(team_returns)) if team_returns else 0.0,
        "success": int(success),
        "collision": int(collision),
        "timeout": int(timeout),
    }
    return stats


def main():
    # 解析参数（继承 get_config，再扩展评测参数）
    parser = get_config()
    parser = add_eval_args(parser)
    args = parser.parse_args()

    # 设备 & 随机种子
    device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")
    torch.set_num_threads(getattr(args, "n_training_threads", 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[Device] {device}")
    print("[EVAL] start")

    # 评测环境（强制单环境线程）
    args.n_rollout_threads = 1
    args.test_n_rollout_threads = 1  # 兼容你自定义的参数命名
    eval_envs = make_eval_env(args)

    # 构建 Runner（不调用 run，只复用其 policy/trainer）
    dummy_run_dir = Path("./eval_tmp")
    dummy_run_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(
        {
            "all_args": args,
            "envs": eval_envs,
            "eval_envs": eval_envs,
            "num_agents": args.num_agents,
            "device": device,
            "run_dir": dummy_run_dir,
        }
    )

    # 加载每个 agent 的权重
    for aid in range(args.num_agents):
        _load_agent_ckpt(runner.trainer[aid], aid, args.saved_model_dir, device)

    # 评测
    stats = evaluate(
        runner,
        n_eps=args.n_eval_episodes,
        deterministic=args.deterministic_eval,
        render=args.render,
        record_dir=args.record_dir,
        log_interval=args.__log_interval,
    )
    print("[EVAL][FINAL]", stats)

    eval_envs.close()


if __name__ == "__main__":
    main()

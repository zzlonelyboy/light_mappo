from pathlib import Path
from typing import Any

import gym
import numpy as np
import torch

from config import get_config
from runner.separated.env_runner import EnvRunner as Runner
from utils.load import _load_agent_ckpt


class NavPolicyAdapter:
    """
    将已训练好的导航策略（Runner.trainer[*].policy）封装为黑盒控制器。
    - 输入：vecenv 风格的 obs（[1, N, obs_dim]）
    - 输出：动作 [1, N, act_dim]（这里 act_dim=2，对应 [d_yaw, d_pitch]）
    - 内部维护 RNN 状态与 masks（deterministic 推理）
    """
    def __init__(self, runner: Any, deterministic: bool = True):
        self.runner = runner
        self.det = bool(deterministic)
        A = runner.num_agents
        self.rec_N = runner.recurrent_N
        self.hid = runner.hidden_size
        self.rnn_states = np.zeros((1, A, self.rec_N, self.hid), np.float32)
        self.masks = np.ones((1, A, 1), np.float32)

    @staticmethod
    def _t2n(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    @torch.no_grad()
    def act(self, obs_vecenv_style: np.ndarray) -> list:
        """
        obs_vecenv_style: [1, N, obs_dim]
        return: [ [action_of_agent0, action_of_agent1, ...] ]  -> [1, N, 2]
        """
        actions_env = []
        for aid in range(self.runner.num_agents):
            self.runner.trainer[aid].prep_rollout()
            act, rnn_state = self.runner.trainer[aid].policy.act(
                np.asarray(obs_vecenv_style[:, aid]),  # [1, obs_dim]
                self.rnn_states[:, aid],
                self.masks[:, aid],
                deterministic=self.det,
            )
            self.rnn_states[:, aid] = self._t2n(rnn_state)
            actions_env.append(self._t2n(act)[0])  # -> (2,)
        return [np.asarray(actions_env, dtype=np.float32)]  # [1, N, 2]




def make_nav_policy_adapter(runner: Any, env:gym.Env,deterministic: bool = True,) -> NavPolicyAdapter:
    if runner ==None:
        parser=get_config()
        parser.add_argument("--saved_model_dir", type=str,
                            default="D:\\Reproduction_of_the_paper\\light_mappo\\results\\MyEnv\\MyEnv\\mappo\\check\\run2\\models",
                            help="包含 actor_agent*.pt / critic_agent*.pt 的目录")
        parser.add_argument("--num_agents", type=int, default=env.num_agent)
        args=parser.parse_args()
        args.n_rollout_threads=1
        device = torch.device("cuda:0" if torch.cuda.is_available() and getattr(args, "cuda", True) else "cpu")
        args.test_n_rollout_threads = 1
        run_dir=Path("./stageOne_tmp")
        runner=Runner({
            "all_args": args,
            "envs":env,
            "num_agents": args.num_agents,
            "device":device,
            'run_dir':run_dir
        })
        for aid in range(args.num_agents):
            _load_agent_ckpt(runner.trainer[aid], aid, args.saved_model_dir, device)
    return NavPolicyAdapter(runner,deterministic=deterministic)



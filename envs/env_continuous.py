# file: envs/continuous_action_env.py
from __future__ import annotations
import numpy as np
from gym import spaces
from envs.env_core import EnvCore


class ContinuousActionEnv(object):
    """
    直接使用改造后的 EnvCore（已对接 MultiUAVSphereEnv）
    """

    def __init__(self):
        self.env = EnvCore()  # 需要改参时，直接传到 EnvCore(...) 里即可
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        self.discrete_action_input = False
        self.movable = True

        # —— 与官方轻量版一致，直接复用 EnvCore 的 spaces —— #
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.share_observation_space = self.env.share_observation_space

    def step(self, actions):
        """
        单环境情形：actions 形状通常为 (num_agent, action_dim)
        外层若做并行（num_threads），请用 VecEnv 管理，不在此类里处理“线程维”
        """
        obs_list, rews_list, dones_list, infos_list = self.env.step(actions)
        # 官方轻量版期望返回 np.stack 后的数组
        return (
            np.stack(obs_list),           # -> (N, obs_dim)
            np.stack(rews_list),          # -> (N, 1)
            np.stack(dones_list),         # -> (N,)
            infos_list,                   # list(dict)
        )

    def reset(self):
        obs_list = self.env.reset()
        return np.stack(obs_list)         # -> (N, obs_dim)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        pass

# file: envs/env_core.py
from __future__ import annotations
import numpy as np
from typing import List, Any, Dict
from gym import spaces

# 改成你实际的导入路径
from envs.MultiUAVSphereEnv import MultiUAVSphereEnv
from envs.MultiUAVSphereEnv2 import MultiUAVSphereEnvWithObstacle


class EnvCore2(object):
    """
    直接驱动 MultiUAVSphereEnv 的 MAPPO 环境核心（与官方轻量版接口一致）
      - reset() -> List[np.ndarray(obs_dim,)]
      - step(actions) -> [obs_list, rew_list, done_list, info_list]
    """

    def __init__(
        self,
        # —— 透传到 MultiUAVSphereEnv 的参数（按需改默认值） ——
        N: int = 8,
        R: float = 20.0,
        v0: float = 4.0,
        dt: float = 0.1,
        d_safe: float = 2.0,
        K_hist: int = 4,
        k_nbr: int = 4,
        episode_seconds: float = 40.0,
        goal_radius: float = 20.0,
        seed: int | None = 0,
        align_reward: bool = True,

        # —— 适配选项 ——
        include_mask: bool = True,   # 是否把 nbr_mask 拼到每个体观测末尾
        tanh_action: bool = True,    # 是否对输入动作做 tanh 收缩到 [-1,1]
    ):
        self._env = MultiUAVSphereEnvWithObstacle(
            N=N, R=R, v0=v0, dt=dt, d_safe=d_safe,
            K_hist=K_hist, k_nbr=k_nbr,
            episode_seconds=episode_seconds,
            goal_radius=goal_radius,
            align_reward=align_reward,
            seed=seed,
        )

        # 基本元数据
        self.agent_num: int = self._env.N
        self.action_dim: int = 2  # yaw/pitch
        self.include_mask = include_mask
        self._tanh_action = tanh_action

        # 每个体观测维度：per_agent_obs_dim + (可选)k_nbr
        self.obs_dim: int = int(self._env.per_agent_obs_dim + (self._env.k_nbr if include_mask else 0))

        # —— 与官方轻量版一致的 spaces —— #
        # 每个体一个 Box
        self.observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]
        # 连续动作 [-1,1]^2
        self.action_space = [
            spaces.Box(low=-1.0, high=+1.0, shape=(self.action_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]
        # 共享观测：拼接所有个体的 obs
        share_obs_dim = self.agent_num * self.obs_dim
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

        self._last_share_obs: np.ndarray | None = None  # (N*obs_dim,)

    # ================== 必需接口 ==================

    def reset(self) -> List[np.ndarray]:
        obs_dict, _info = self._env.reset()
        obs = self._flatten_obs(obs_dict)              # (N, obs_dim)
        self._last_share_obs = obs.reshape(-1)         # (N*obs_dim,)
        return [obs[i].copy() for i in range(self.agent_num)]

    def step(self, actions: List[np.ndarray]) -> List[Any]:
        """
        actions: 长度 N 的 list/array，每个元素 shape=(action_dim,)
        return:  [obs_list, rew_list, done_list, info_list]
        """
        a = np.asarray(actions, dtype=np.float32)
        if a.shape != (self.agent_num, self.action_dim):
            raise ValueError(f"actions shape must be ({self.agent_num},{self.action_dim}), got {a.shape}")

        # 策略头可能未做边界；这里统一约束到 [-1,1]
        a = np.tanh(a) if self._tanh_action else np.clip(a, -1.0, 1.0)

        obs_dict, rew, terminated, truncated, info = self._env.step(a)
        obs = self._flatten_obs(obs_dict)
        self._last_share_obs = obs.reshape(-1)

        # 官方轻量版返回格式（list of per-agent）
        sub_agent_obs    = [obs[i].copy() for i in range(self.agent_num)]
        sub_agent_reward = [[float(rew[i])] for i in range(self.agent_num)]  # -> stack 后 (N,1)
        done_flag        = bool(terminated or truncated)
        sub_agent_done   = [done_flag for _ in range(self.agent_num)]
        sub_agent_info   = [{} for _ in range(self.agent_num)]
        if "term_reason" in info:
            sub_agent_info[0]["term_reason"] = info["term_reason"]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    # ================== 可选辅助 ==================

    def last_share_obs(self) -> np.ndarray | None:
        return self._last_share_obs

    # ================== 内部工具 ==================

    def _flatten_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        把 {"obs": (N,D), "nbr_mask": (N,k)} -> (N, obs_dim)
        """
        base = obs_dict["obs"]
        if self.include_mask:
            obs = np.concatenate([base, obs_dict["nbr_mask"]], axis=1)
        else:
            obs = base
        return obs.astype(np.float32)

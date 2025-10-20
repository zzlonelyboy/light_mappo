# file: envs/env_core.py
from __future__ import annotations

import gym
import numpy as np
from typing import List, Any, Dict, Optional, Callable
from gym import spaces
from envs.stage2DetectEnv import Stage2DetectEnv
# 改成你实际的导入路径
from envs.MultiUAVSphereEnv import MultiUAVSphereEnv
from envs.MultiUAVSphereEnv2 import MultiUAVSphereEnvWithObstacle
from envs.NavPolicyAdapter import NavPolicyAdapter


class EnvCore(object):
    """
    直接驱动 MultiUAVSphereEnv 的 MAPPO 环境核心（与官方轻量版接口一致）
      - reset() -> List[np.ndarray(obs_dim,)]
      - step(actions) -> [obs_list, rew_list, done_list, info_list]
    """

    def __init__(
            self,
            base_env: gym.Env,
            nav_adapter: NavPolicyAdapter,
            R: Optional[float] = None,
            dt: Optional[float] = None,
            gps_getter: Optional[Callable[[gym.Env], np.ndarray]] = None,
            # 可选：在检测环境内部直接调用攻击管理器并写回 GPS
            atk_manager: Optional[Any] = None,
            atk_scenario: Optional[Any] = None,
            attack_writeback: bool = False,
            # 奖励超参
            reward_tp: float = 1.0,
            reward_fp: float = 1.8,
            reward_fn: float = 3.5,
            flip_penalty: float = 0.01,
            delay_full_seconds: float = 3.0,
    ):
        # 基本元数据
        self._env = Stage2DetectEnv(base_env=base_env,nav_adapter=nav_adapter,
                                    gps_getter=gps_getter,atk_manager=atk_manager,
                                    atk_scenario=atk_scenario,attack_writeback=attack_writeback,
                                    reward_tp=reward_tp,reward_fp=reward_fp,reward_fn=reward_fn,
                                    flip_penalty=flip_penalty,delay_full_seconds=delay_full_seconds)
        self.agent_num: int = self._env.N
        self.action_dim: int = 2  # 【0,1】表示判断是否收到攻击
        self.obs_dim: int = self._env.det_dim
        self.observation_space=[spaces.Box(low=-np.inf,high=np.inf,shape=(self.obs_dim,),dtype=np.float32)
                                for _ in range(self.agent_num)]
        self.action_space=(spaces.Discrete(self.action_dim) for _ in range(self.agent_num))
        share_obs_dim = self.agent_num * self.obs_dim
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32)
            for _ in range(self.agent_num)
        ]

        self._last_share_obs: np.ndarray | None = None  # (N*obs_dim,)
    def reset(self):
        obs,_info=self._env.reset()
        self._last_share_obs = obs.reshape(-1)
        return  [obs[i].copy() for i in range(self.agent_num)]

    def step(self,actions:List[np.ndarray]):
        actions=np.asarray(actions,dtype=np.int32)
        if actions.shape != (self.agent_num, self.action_dim):
            raise ValueError(f"actions shape must be ({self.agent_num},{self.action_dim}), got {actions.shape}")
        obs,rew,terminated,truncated,info=self._env.step(actions)
        self._last_share_obs = obs.reshape(-1)

        sub_agent_obs = [obs[i].copy() for i in range(self.agent_num)]
        sub_agent_reward = [[float(rew[i])] for i in range(self.agent_num)]  # -> stack 后 (N,1)
        done_flag = bool(terminated or truncated)
        sub_agent_done = [done_flag for _ in range(self.agent_num)]
        sub_agent_info = [{} for _ in range(self.agent_num)]
        if "term_reason" in info:
            sub_agent_info[0]["term_reason"] = info["term_reason"]
        if 'det_tp_fp_fn' in info:
            sub_agent_info[0]['det_tp_fp_fn']=info['det_tp_fp_fn']

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


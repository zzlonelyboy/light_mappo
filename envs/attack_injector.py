from __future__ import annotations
from typing import Optional, Tuple, Any

import numpy as np
import gym


class AttackInjectorEnv(gym.Env):
    """
    轻量包装器：在每步调用攻击管理器，生成被篡改的 GPS 并写回 base 环境，
    同时把攻击状态透出到 info（attack_active/attack_mask/attack_type/attack_t0_t1）。

    依赖（从 base 环境读取）：
      - base.N, base.dt, base.p (N,D), base.h_hist[-1] (N,D), base.t
      - base.step / base.reset 的 Gym 接口
    写回（供下游使用）：
      - base.gps = z（N,D）
      - base._atk_active_mask = active（N,）
    """

    def __init__(self, base_env: gym.Env, atk_manager: Any, scenario: Optional[Any] = None):
        super().__init__()
        self.base = base_env
        self.mgr = atk_manager
        self.scenario = scenario
        self.N = int(getattr(base_env, "N"))
        self.dt = float(getattr(base_env, "dt"))

        # 历史缓存（供 replay/分析使用）
        self.hist_true_pos: list[np.ndarray] = []  # 每步 true_pos（N,D）
        self.hist_gps_pos: list[np.ndarray] = []   # 每步 gps（N,D）

        # 透传空间定义
        self.action_space = self.base.action_space
        self.observation_space = getattr(self.base, "observation_space", None)

    # ---------- 用于外部切换/重采样攻击 ----------
    def set_scenario(self, scenario: Any):
        self.scenario = scenario

    def sample_new_scenario(self):
        T_max = int(getattr(self.base, "T_max", 0) or 0)
        if T_max <= 0:
            # 尝试推断一个保守上限
            T_max = 1000
        self.scenario = self.mgr.sample_scenario(T_max=T_max, dt=self.dt)
        return self.scenario

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ret = self.base.reset(seed=seed, options=options)
        # 清空历史
        self.hist_true_pos.clear()
        self.hist_gps_pos.clear()
        # 若尚无场景则采样一个
        if self.scenario is None:
            self.sample_new_scenario()
        return ret

    def step(self, action):
        obs_base, rew_base, terminated, truncated, info_base = self.base.step(action)

        # 采集当前真值与 heading
        true_pos = np.asarray(self.base.p, dtype=np.float32)
        heading_now = np.asarray(self.base.h_hist[-1], dtype=np.float32)

        # 先追加历史真值（与 base.t 对齐）
        self.hist_true_pos.append(true_pos.copy())

        # 调用攻击管理器生成篡改量测
        z, active = self.mgr.apply(
            t=int(getattr(self.base, "t", len(self.hist_true_pos)-1)),
            true_pos=true_pos,
            hist_true_pos=self.hist_true_pos,
            hist_gps_pos=self.hist_gps_pos,
            scenario=self.scenario,
            dt=self.dt,
            heading_now=heading_now,
        )

        # 写回到 base 环境
        setattr(self.base, "gps", z)
        setattr(self.base, "_atk_active_mask", np.asarray(active, dtype=bool))

        # 维护历史 gps
        self.hist_gps_pos.append(z.copy())

        # 组装 info（叠加攻击状态）
        info = dict(info_base)
        info["attack_active"] = bool(np.any(active) and (self.scenario.t0 <= getattr(self.base, "t", 0) <= self.scenario.t1))
        info["attack_mask"] = np.asarray(active, dtype=bool)
        info["attack_type"] = str(getattr(self.scenario, "atk_type", ""))
        info["attack_t0_t1"] = (int(self.scenario.t0), int(self.scenario.t1))

        return obs_base, rew_base, bool(terminated), bool(truncated), info

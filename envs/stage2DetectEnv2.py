# stage2/stage2_detect_env.py
from __future__ import annotations
from typing import Callable, Optional, Any
import numpy as np
import gym
from collections import deque
from gym.spaces import Box, MultiDiscrete, Dict as SpaceDict
from envs.NavPolicyAdapter import NavPolicyAdapter

class Stage2DetectEnv(gym.Env):
    """
    Stage-2 外层"检测"环境：
      - 底层 base 为导航环境（不修改其动力学与奖励）
      - 每步先用冻结的导航策略推进 base，再让检测策略对当前时刻判定攻击（0/1）

    动作空间：
      MultiDiscrete([2]*N)  —— 每个 UAV 二元判定（0=正常，1=欺骗）

    观测：
      Box((N, det_dim))，其中 det_dim = 8

    奖励（逐 agent）：
      +R_TP (a=1,y=1)；-R_FP (a=1,y=0)；-R_FN*w_delay (a=0,y=1)；
      以及轻微翻转惩罚
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        nav_adapter: NavPolicyAdapter,
        R: Optional[float] = None,
        dt: Optional[float] = None,
        gps_getter: Optional[Callable[[gym.Env], np.ndarray]] = None,
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = False,
        resample_attack_on_reset: bool = True,  # 🆕
        reward_tp: float = 1.0,
        reward_fp: float = 1.8,
        reward_fn: float = 3.5,
        flip_penalty: float = 0.01,
        delay_full_seconds: float = 3.0,
    ):
        super().__init__()
        self.base = base_env
        
        # 🔧 健壮地获取最底层环境
        self.unwrapped_env = base_env
        while hasattr(self.unwrapped_env, 'env') or hasattr(self.unwrapped_env, '_env'):
            if hasattr(self.unwrapped_env, 'env'):
                self.unwrapped_env = self.unwrapped_env.env
            elif hasattr(self.unwrapped_env, '_env'):
                self.unwrapped_env = self.unwrapped_env._env
            else:
                break
        
        self.nav = nav_adapter
        self.N = getattr(self.unwrapped_env, "N", None)
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("base_env 必须包含整数属性 N（无人机数量）。")
        
        self.R = float(R if R is not None else getattr(self.unwrapped_env, "R", 20.0))
        self.dt = float(dt if dt is not None else getattr(self.unwrapped_env, "dt", 0.1))
        self.det_dim = 8

        # 动作/观测空间
        self.action_space = MultiDiscrete(np.array([2] * self.N, dtype=np.int64))
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.N, self.det_dim), 
            dtype=np.float32
        )

        # GPS 源获取器
        self.gps_getter = gps_getter

        # 攻击管理
        self.atk_manager = atk_manager
        self.atk_scenario = atk_scenario
        self.attack_writeback = bool(attack_writeback)
        self.resample_attack_on_reset = bool(resample_attack_on_reset)
        self._hist_true_pos = []
        self._hist_gps_pos = []

        # 奖励参数
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_FN = float(reward_fn)
        self.flip_penalty = float(flip_penalty)
        self.delay_full_seconds = float(delay_full_seconds)

        # 历史状态
        self._prev_gps: Optional[np.ndarray] = None
        self._prev_h: Optional[np.ndarray] = None
        self._a_prev = np.zeros((self.N,), np.int32)

    def _get_gps(self) -> np.ndarray:
        """优先从 gps_getter 回调获取；否则 unwrapped_env.gps；否则 unwrapped_env.p"""
        if callable(self.gps_getter):
            z = self.gps_getter(self.unwrapped_env)
            return np.asarray(z, dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            return np.asarray(self.unwrapped_env.gps, dtype=np.float32)
        if hasattr(self.unwrapped_env, "p"):
            return np.asarray(self.unwrapped_env.p, dtype=np.float32)
        raise RuntimeError("无法获取 GPS：请提供 gps_getter 或在 base_env 上设置 .gps/.p")

    def _as_attack_mask(self, info: dict) -> np.ndarray:
        """从 info 中读取 attack_mask，若无则返回全 False"""
        mask = info.get("attack_mask", None)
        if mask is None:
            return np.zeros(self.N, dtype=np.int32)
        return np.asarray(mask, dtype=bool).astype(np.int32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset 导航环境 + 检测历史"""
        # 1. Reset 底层导航环境
        if seed is not None:
            ret = self.base.reset(seed=seed, options=options)
        else:
            ret = self.base.reset()
        
        _obs_base = ret[0] if isinstance(ret, (list, tuple)) else ret
        
        # 2. 初始化检测历史
        self._prev_gps = self._get_gps().copy()
        self._prev_h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32).copy()
        self._a_prev[:] = 0
        
        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()
        
        # 3. 攻击场景管理
        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to sample attack scenario: {e}")
                    self.atk_scenario = None
        
        # 4. 构造检测观测
        x = self._build_det_obs()  # (N, 8) numpy array
        info: dict = {}
        return x, info

    def step(self, a_det) -> tuple:
        """推进检测环境一步"""
        # 🔧 处理输入动作格式
        if isinstance(a_det, dict):
            a = a_det.get("a", None)
            if a is None:
                for k in ("action", "actions", "x", "act"):
                    if k in a_det:
                        a = a_det[k]
                        break
            if a is None:
                raise ValueError("未在字典动作中找到有效键")
        else:
            a = a_det
        
        # 🔧 处理动作格式（支持离散动作和 one-hot）
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2:  # (N, 2) one-hot or logits
            if a.shape[1] == 2:
                a = a.argmax(axis=1).astype(np.int32)
            else:
                raise ValueError(f"Expected action shape (N,) or (N, 2), got {a.shape}")
        elif a.ndim == 1:  # (N,) 离散动作
            a = a.astype(np.int32)
        else:
            raise ValueError(f"Expected action shape (N,) or (N, 2), got {a.shape}")
        
        a = a.reshape(self.N)

        # 1) 用导航策略推进底层环境
        stage_one_obs = self.unwrapped_env._get_obs()
        obs_vec = np.expand_dims(
            np.concatenate([stage_one_obs['obs'], stage_one_obs['nbr_mask']], axis=1),
            axis=0
        )  # [1, N, obs_dim]
        nav_actions = self.nav.act(obs_vec)  # [1, N, 2]
        
        # 调用底层环境 step
        obs_base, rew_base, terminated, truncated, info_base = self.unwrapped_env.step(nav_actions[0])

        # 2) 可选：攻击写回
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None):
            try:
                true_pos = np.asarray(self.unwrapped_env.p, dtype=np.float32)
                heading_now = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
                self._hist_true_pos.append(true_pos.copy())
                
                z, active = self.atk_manager.apply(
                    t=int(getattr(self.unwrapped_env, "t", len(self._hist_true_pos) - 1)),
                    true_pos=true_pos,
                    hist_true_pos=self._hist_true_pos,
                    hist_gps_pos=self._hist_gps_pos,
                    scenario=self.atk_scenario,
                    dt=self.dt,
                    heading_now=heading_now,
                )
                
                # 写回环境
                setattr(self.unwrapped_env, "gps", z)
                setattr(self.unwrapped_env, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))
                
                # 扩充 info
                info_base = dict(info_base)
                current_t = getattr(self.unwrapped_env, "t", 0)
                info_base["attack_active"] = bool(
                    np.any(active) and (self.atk_scenario.t0 <= current_t <= self.atk_scenario.t1)
                )
                info_base["attack_mask"] = np.asarray(active, dtype=bool)
                info_base["attack_type"] = str(getattr(self.atk_scenario, "atk_type", ""))
                info_base["attack_t0_t1"] = (int(self.atk_scenario.t0), int(self.atk_scenario.t1))
            except Exception as e:
                import warnings
                warnings.warn(f"Attack application failed: {e}")

        # 3) 计算检测奖励
        y = self._as_attack_mask(info_base)  # (N,) int in {0,1}
        
        # ✅ 修复：明确括号优先级
        tp = ((a == 1) & (y == 1)) | ((a == 0) & (y == 0))
        fp = (a == 1) & (y == 0)
        fn = (a == 0) & (y == 1)

        # 时延加权
        if info_base.get("attack_active", False):
            t0, _t1 = info_base.get("attack_t0_t1", (self.unwrapped_env.t, self.unwrapped_env.t))
            elapsed = max(0, int(self.unwrapped_env.t) - int(t0))
        else:
            elapsed = 0
        w_delay = min(1.0, elapsed * self.dt / max(self.delay_full_seconds, 1e-6))

        r = (
            + self.R_TP * tp.astype(np.float32)
            - self.R_FP * fp.astype(np.float32)
            - (self.R_FN * w_delay) * fn.astype(np.float32)
        )
        r -= self.flip_penalty * (a != self._a_prev).astype(np.float32)
        self._a_prev = a.copy()

        # 4) 构造检测观测
        x = self._build_det_obs()  # (N, 8) numpy array

        # 汇总信息
        info = dict(info_base)
        info["det_tp_fp_fn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
        
        return x, r.astype(np.float32), bool(terminated), bool(truncated), info

    def _build_det_obs(self) -> np.ndarray:
        """构造检测特征 (N, 8)"""
        gps = self._get_gps()  # (N,3)
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)  # (N,3)

        # 航位推算残差
        pred = self._prev_gps + float(self.unwrapped_env.v0) * self._prev_h * self.dt
        r_dead = (gps - pred) / self.R
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)

        # 速度残差
        speed_res = (
            np.linalg.norm(gps - self._prev_gps, axis=1, keepdims=True) 
            - float(self.unwrapped_env.v0) * self.dt
        )

        # 朝向变化
        dh = np.linalg.norm(h - self._prev_h, axis=1, keepdims=True)

        # 统计特征
        m = np.mean(r_dead, axis=1, keepdims=True)
        mx = np.max(np.abs(r_dead), axis=1, keepdims=True)

        x = np.concatenate([r_dead, r_norm, speed_res, dh, m, mx], axis=1).astype(np.float32)
        
        # 更新历史
        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x
    
    def success_from_info(self, info) -> bool:
        """判断任务是否成功"""
        # 1) 优先检查 term_reason（base 环境在 _check_terminated 中设置）
        if "term_reason" in info:
            return info["term_reason"] == "success"
        
        # 2) 其他标志位
        for k in ["success", "episode_success", "all_reached"]:
            if k in info:
                v = info[k]
                return bool(v.item() if hasattr(v, "item") else v)
        
        # 3) 直接检查环境状态（使用 true_c，因为 base 环境用它判断成功）
        if hasattr(self.unwrapped_env, "true_c") and hasattr(self.unwrapped_env, "g"):
            if hasattr(self.unwrapped_env, "goal_radius"):
                dc = np.linalg.norm(self.unwrapped_env.true_c - self.unwrapped_env.g)
                return bool(dc <= float(self.unwrapped_env.goal_radius))
        
        return False
    
    def set_attack_scenario(self, scenario):
        """手动设置攻击场景"""
        self.atk_scenario = scenario
    
    def clear_attack_scenario(self):
        """清空攻击场景（下次 reset 会重新采样）"""
        self.atk_scenario = None
    
    def eval_task_success_with_attack(
        self, 
        atk_manager=None, 
        episodes=50, 
        sample_scenario_each_ep=True
    ):
        """评估在攻击下的任务成功率"""
        if atk_manager is None:
            atk_manager = self.atk_manager
        
        N = self.N
        success_eps = 0
        
        for ep in range(episodes):
            # 每个 episode 重新采样攻击场景
            if sample_scenario_each_ep and atk_manager is not None:
                self.atk_scenario = atk_manager.sample_scenario(
                    T_max=getattr(self.unwrapped_env, "T_max", 1000), 
                    dt=self.dt
                )
            
            obs, info = self.reset()
            done = False
            last_info = {}
            
            while not done:
                # 不报警（让攻击生效）
                a_det = np.zeros(N, dtype=np.int32)
                obs, r, terminated, truncated, info = self.step(a_det)
                last_info = info
                done = terminated or truncated
            
            if self.success_from_info(last_info):
                success_eps += 1
        
        return success_eps / episodes

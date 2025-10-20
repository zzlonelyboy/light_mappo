# stage2/stage2_detect_env.py
from __future__ import annotations
from typing import Callable, Optional, Any
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete, Dict as SpaceDict
from envs.NavPolicyAdapter import NavPolicyAdapter
class Stage2DetectEnv(gym.Env):
    """
    Stage-2 外层“检测”环境：
      - 底层 base 为导航环境（不修改其动力学与奖励）
      - 每步先用冻结的导航策略推进 base，再让检测策略对当前时刻判定攻击（0/1）
      - 标签从 base.step(...) 返回的 info["attack_mask"] 读取（由外部攻击管理器写入 env 后由 _get_info() 透出）

    动作空间：
      MultiDiscrete([2]*N)  —— 每个 UAV 二元判定（0=正常，1=欺骗）

    观测：
      Dict({"x": Box((N, det_dim))})，其中 det_dim = 8，组成：
        r_dead(3) + ||r_dead||(1) + speed_res(1) + Δh(1) + mean(r_dead)(1) + max|r_dead|(1)

    奖励（逐 agent）：
      +R_TP (a=1,y=1)；-R_FP (a=1,y=0)；-R_FN*w_delay (a=0,y=1)；
      以及轻微翻转惩罚 0.01 * 1[a_t != a_{t-1}]
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

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
        super().__init__()
        self.base = base_env
        self.baseEnv=base_env.env._env
        self.nav = nav_adapter
        self.N = getattr(self.baseEnv, "N", None)
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("base_env 必须包含整数属性 N（无人机数量）。")
        self.R = float(R if R is not None else getattr(self.baseEnv, "R", 20.0))
        self.dt = float(dt if dt is not None else getattr(self.baseEnv, "dt", 0.1))
        self.det_dim = 8

        # 动作/观测空间
        self.action_space = MultiDiscrete(np.array([2] * self.N, dtype=np.int64))
        self.observation_space = SpaceDict({
            "x": Box(low=-np.inf, high=np.inf, shape=(self.N, self.det_dim), dtype=np.float32)
        })

        # GPS 源获取器（优先级：回调 -> base.gps -> base.p）
        self.gps_getter = gps_getter

        # 可选：攻击集成（在检测环境内部直接调用攻击并写回）
        self.atk_manager = atk_manager
        self.atk_scenario = atk_scenario
        self.attack_writeback = bool(attack_writeback)
        self._hist_true_pos = []  # list of (N,D)
        self._hist_gps_pos = []   # list of (N,D)

        # 奖励参数
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_FN = float(reward_fn)
        self.flip_penalty = float(flip_penalty)
        self.delay_full_seconds = float(delay_full_seconds)

        # 历史状态
        self._prev_gps: Optional[np.ndarray] = None  # (N,3)
        self._prev_h: Optional[np.ndarray] = None    # (N,3)
        self._a_prev = np.zeros((self.N,), np.int32)

    # ---------- 工具 ----------
    def _get_gps(self) -> np.ndarray:
        """优先从 gps_getter 回调获取；否则 base.gps；否则 base.p（真值，当作理想 GPS）。"""
        if callable(self.gps_getter):
            z = self.gps_getter(self.baseEnv)
            return np.asarray(z, dtype=np.float32)
        if hasattr(self.baseEnv, "gps"):
            return np.asarray(self.baseEnv.gps, dtype=np.float32)
        if hasattr(self.baseEnv, "p"):
            return np.asarray(self.baseEnv.p, dtype=np.float32)
        raise RuntimeError("无法获取 GPS：请提供 gps_getter 或在 base_env 上设置 .gps/.p")

    def _as_attack_mask(self, info: dict) -> np.ndarray:
        """从 info 中读取 attack_mask，若无则返回全 False。"""
        mask = info.get("attack_mask", None)
        if mask is None:
            return np.zeros(self.N, dtype=np.int32)
        return np.asarray(mask, dtype=bool).astype(np.int32)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        ret = self.base.reset()
        _obs_base = ret[0] if isinstance(ret, (list, tuple)) else ret  # 兼容 gymnasium 风格
        self._prev_gps = self._get_gps().copy()
        self._prev_h = np.asarray(self.baseEnv.h_hist[-1], dtype=np.float32).copy()
        self._a_prev[:] = 0
        # 清空历史（供攻击管理器 replay/stealth 使用）
        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()
        # 若启用内置攻击写回且未提供场景，尝试采样一个
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is None):
            T_max = int(getattr(self.baseEnv, "T_max", 0) or 1000)
            try:
                self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
            except Exception:
                # 不中断 reset，留给外部设置场景
                pass
        x = self._build_det_obs()
        info: dict = {}
        return x, info

    def step(self, a_det) -> tuple:
        # 允许外部传入 ndarray 或 dict（例如 {"a": arr}）
        if isinstance(a_det, dict):
            a = a_det.get("a", None)
            if a is None:
                # 兼容其他键名
                for k in ("action", "actions", "x", "act"):
                    if k in a_det:
                        a = a_det[k]
                        break
            if a is None:
                raise ValueError("未在字典动作中找到有效键（期望 'a' 或 'action'/'actions'/'x'/'act'）")
        else:
            # a = a_det
            a = a_det.argmax(axis=1).astype(np.int32)
        a = np.asarray(a, dtype=np.int32).reshape(self.N)

        # 1) 内环：用冻结导航策略推进底层环境
        stageOneObs=self.baseEnv._get_obs()

        obs_vec = np.expand_dims(np.concatenate([stageOneObs['obs'],stageOneObs['nbr_mask']],axis=1), axis=0)  # [1, N, obs_dim]
        nav_actions = self.nav.act(obs_vec)                            # [1, N, 2]
        obs_base, rew_base, terminated, truncated, info_base = self.baseEnv.step(nav_actions[0])

        # 可选：在此处调用攻击并写回 GPS 与标签
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None):
            try:
                # 采集当前真值/朝向，维护历史
                true_pos = np.asarray(self.baseEnv.p, dtype=np.float32)
                heading_now = np.asarray(self.baseEnv.h_hist[-1], dtype=np.float32)
                self._hist_true_pos.append(true_pos.copy())
                # 调用攻击
                z, active = self.atk_manager.apply(
                    t=int(getattr(self.baseEnv, "t", len(self._hist_true_pos) - 1)),
                    true_pos=true_pos,
                    hist_true_pos=self._hist_true_pos,
                    hist_gps_pos=self._hist_gps_pos,
                    scenario=self.atk_scenario,
                    dt=self.dt,
                    heading_now=heading_now,
                )
                # 写回环境
                setattr(self.baseEnv, "gps", z)
                setattr(self.baseEnv, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))
                # 扩充 info 供奖励/日志使用
                info_base = dict(info_base)
                info_base["attack_active"] = bool(np.any(active) and (self.atk_scenario.t0 <= getattr(self.baseEnv, "t", 0) <= self.atk_scenario.t1))
                info_base["attack_mask"] = np.asarray(active, dtype=bool)
                info_base["attack_type"] = str(getattr(self.atk_scenario, "atk_type", ""))
                info_base["attack_t0_t1"] = (int(self.atk_scenario.t0), int(self.atk_scenario.t1))
            except Exception:
                # 安全回退：若攻击过程出错，不影响主流程
                pass

        # 2) 从 info 读取标签（不泄漏到观测）
        y = self._as_attack_mask(info_base)   # (N,) int in {0,1}
        tp = (a == 1) & (y == 1) | (a == 0) & (y == 0)
        fp = (a == 1) & (y == 0)
        fn = (a == 0) & (y == 1)

        # 攻击开始后的时延加权（鼓励尽早发现）
        if info_base.get("attack_active", False):
            t0, _t1 = info_base.get("attack_t0_t1", (self.baseEnv.t, self.baseEnv.t))
            elapsed = max(0, int(self.baseEnv.t) - int(t0))
        else:
            elapsed = 0
        # 线性权重：delay_full_seconds 内从 0 -> 1
        w_delay = min(1.0, elapsed * self.dt / max(self.delay_full_seconds, 1e-6))

        r = (
            + self.R_TP * tp.astype(np.float32)
            - self.R_FP * fp.astype(np.float32)
            - (self.R_FN * w_delay) * fn.astype(np.float32)
        )
        # 轻微翻转惩罚，避免抖动
        r -= self.flip_penalty * (a != self._a_prev).astype(np.float32)
        self._a_prev = a.copy()

        # 3) 构造检测观测
        x = self._build_det_obs()

        # 汇总信息
        info = dict(info_base)
        info["det_tp_fp_fn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
        # 返回逐 agent 奖励（若算法需要标量，可改为 float(r.mean())）
        return x, r.astype(np.float32), bool(terminated), bool(truncated), info

    # ---------- 检测观测构造 ----------
    def _build_det_obs(self) -> np.ndarray:
        """
        最小检测特征（每 UAV 共 8 维）：
          r_dead(3) + ||r_dead||(1) + speed_res(1) + Δh(1) + mean(r_dead)(1) + max|r_dead|(1)
        """
        gps = self._get_gps()                                # (N,3)
        h = np.asarray(self.baseEnv.h_hist[-1], dtype=np.float32)  # (N,3)

        # 计算位移预测与残差（归一化到 R）
        pred = self._prev_gps + float(self.baseEnv.v0) * self._prev_h * self.dt
        r_dead = (gps - pred) / self.R                        # (N,3)
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)  # (N,1)

        # 速度残差（与 v0*dt 的偏差）
        speed_res = (
            np.linalg.norm(gps - self._prev_gps, axis=1, keepdims=True) - float(self.baseEnv.v0) * self.dt
        )  # (N,1)

        # 朝向变化幅度
        dh = np.linalg.norm(h - self._prev_h, axis=1, keepdims=True)  # (N,1)

        # 简化的邻域/统计：r_dead 的均值与最大绝对值（逐分量统计再取一维）
        m = np.mean(r_dead, axis=1, keepdims=True)   # (N,1)
        mx = np.max(np.abs(r_dead), axis=1, keepdims=True)  # (N,1)

        x = np.concatenate([r_dead, r_norm, speed_res, dh, m, mx], axis=1).astype(np.float32)
        # 更新历史
        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x
    def success_from_info(self, info, base_env) -> bool:
        # 1) 优先用环境返回的标志位（若存在）
        for k in ["success", "episode_success", "all_reached", "done_success"]:
            if k in info:
                v = info[k]
                return bool(v.item() if hasattr(v, "item") else v)
        # 2) 若没有现成标志，使用自定义判据（示例：所有 UAV 到目标半径内）
        # 注意：下面属性名要按你的环境改，比如 base_env.targets / base_env.goal_pos / base_env.goal_radius
        if hasattr(base_env, "goal_pos") and hasattr(base_env, "p") and hasattr(base_env, "goal_radius"):
            d = np.linalg.norm(base_env.p - base_env.goal_pos, axis=1)
            return bool(np.all(d < float(base_env.goal_radius)))
        # 3) 实在没有，就把“正常终止且没有碰撞/越界”当作成功（依赖 info 键名）
        if ("terminated_reason" in info) and (info["terminated_reason"] == "success"):
            return True
        if info.get("collision", False):
            return False
        # 默认失败
        return False

def eval_task_success_with_attack(self,det_env, atk_manager, episodes=50, sample_scenario_each_ep=True):
    N = det_env.N
    success_eps = 0
    for ep in range(episodes):
        if sample_scenario_each_ep and atk_manager is not None:
            det_env.atk_scenario = atk_manager.sample_scenario(
                T_max=getattr(det_env.base, "T_max", 1000), dt=det_env.dt
            )
        obs, info = det_env.reset()
        done = False
        last_info = {}
        while not done:
            # 检测动作不影响底层动力学，这里随便给（全 0：不报警）
            a_det = np.zeros(N, dtype=np.int32)
            obs, r, terminated, truncated, info = det_env.step(a_det)
            last_info = info
            done = terminated or truncated
        if self.success_from_info(last_info, det_env.base):
            success_eps += 1
    return success_eps / episodes
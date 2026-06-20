# stage2/stage2_detect_env_enhanced_fixed.py
from __future__ import annotations
from typing import Callable, Optional, Any
from collections import deque
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete
from envs.NavPolicyAdapter import NavPolicyAdapter


class Stage2DetectEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        nav_adapter: NavPolicyAdapter,
        R: Optional[float] = None,
        dt: Optional[float] = None,
        gps_getter: Optional[Callable] = None,
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = False,
        resample_attack_on_reset: bool = True,
        # 邻域参数
        k_neighbors: int = 3,
        history_window: int = 5,

        # 奖励参数
        reward_tp: float = 16,
        reward_fp: float = 1.8,
        reward_fn: float = 6.0,
        flip_penalty: float = 0.005,
        delay_full_seconds: float = 3.0,
        collaborative_bonus: float = 0.0,

        # CUSUM参数
        cusum_threshold: float = 0.005,
        cusum_drift: float = 0.002,
        cusum_max: float = 0.15,
    ):
        super().__init__()
        self.base = base_env

        # 获取最底层环境
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
        self.v0 = float(getattr(self.unwrapped_env, "v0", 4.0))

        self.k_neighbors = min(int(k_neighbors), self.N - 1)
        self.history_window = int(history_window)

        # 特征维度
        self.det_dim_own = 14
        self.det_dim_neighbor = 10
        self.det_dim_formation = 4
        self.det_dim = 28

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
        self.collaborative_bonus = float(collaborative_bonus)

        # 事件级奖励参数
        # self.R_TN = 0.05
        # self.hold_cost = 0.01
        self.R_TN=0.15
        self.hold_cost=0.02
        self.delay_k = 0.08

        # CUSUM参数
        self.cusum_threshold = float(cusum_threshold)
        self.cusum_drift = float(cusum_drift)
        self.cusum_max = float(cusum_max)

        # 时序历史缓存
        self._r_norm_history = deque(maxlen=self.history_window)
        self._r_dead_history = deque(maxlen=self.history_window)
        self._speed_res_history = deque(maxlen=self.history_window)
        self._cusum = None

        # 历史状态
        self._prev_gps: Optional[np.ndarray] = None
        self._prev_h: Optional[np.ndarray] = None
        self._a_prev = np.zeros((self.N,), np.int32)

        # 事件级计分状态
        self._y_prevprev = np.zeros(self.N, dtype=np.int32)
        self._event_open = np.zeros(self.N, dtype=bool)
        self._detected_in_event = np.zeros(self.N, dtype=bool)
        self._event_t0 = np.zeros(self.N, dtype=np.int32)

        # 智能控制参数
        self.event_scoring_mode = "hybrid"
        self.stay_reward = 0.5 
        self.neg_hold_grace_steps = 3
        self.cusum_waive_threshold = 0.6
        self.fp_cooldown_steps = 3

        self._neg_hold_counter = np.zeros(self.N, dtype=np.int32)
        self._last_fp_step = np.full(self.N, -10**9, dtype=np.int32)

    # -----------------------------
    # 传感读取
    # -----------------------------
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

    # -----------------------------
    # Gym 接口
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset 导航环境 + 检测历史"""
        if seed is not None:
            ret = self.base.reset(seed=seed, options=options)
        else:
            ret = self.base.reset()

        _obs_base = ret[0] if isinstance(ret, (list, tuple)) else ret

        # 初始化检测历史
        gps_init = self._get_gps().copy()
        h_init = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32).copy()

        self._prev_gps = gps_init
        self._prev_h = h_init
        self._a_prev[:] = 0

        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()

        # 预填充时序历史
        self._r_norm_history.clear()
        self._r_dead_history.clear()
        self._speed_res_history.clear()
        for _ in range(self.history_window):
            init_r_norm = np.random.normal(0.02, 0.01, size=(self.N, 1)).astype(np.float32)
            init_r_norm = np.clip(init_r_norm, 0.0, 0.1)
            init_r_dead = np.random.normal(0.0, 0.01, size=(self.N, 3)).astype(np.float32)
            init_r_dead = np.clip(init_r_dead, -0.05, 0.05)
            init_speed_res = np.random.normal(0.0, 0.01, size=(self.N, 1)).astype(np.float32)
            init_speed_res = np.clip(init_speed_res, -0.05, 0.05)
            self._r_norm_history.append(init_r_norm)
            self._r_dead_history.append(init_r_dead)
            self._speed_res_history.append(init_speed_res)

        # CUSUM 初始化
        self._cusum = np.zeros((self.N, 1), dtype=np.float32)
        # print(self.attack_writeback)
        # 攻击场景管理
        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to sample attack scenario: {e}")
                    self.atk_scenario = None

        # 事件级状态复位
        self._y_prevprev[:] = 0
        self._event_open[:] = False
        self._detected_in_event[:] = False
        self._event_t0[:] = 0

        self._neg_hold_counter[:] = 0
        self._last_fp_step[:] = -10**9

        # 构造初始观测
        x = self._build_det_obs_method_b()
        info: dict = {
            "obs_dim_breakdown": {
                "own": self.det_dim_own,
                "neighbor": self.det_dim_neighbor,
                "formation": self.det_dim_formation,
                "total": self.det_dim,
            }
        }
        return x, info

    def step(self, a_det) -> tuple:
        """
        推进检测环境一步 - 【方案A修复版】因果对齐
        
        执行顺序：
        1. 处理动作输入
        2. 【关键】获取当前时刻 t 的攻击状态（在环境推进前）
        3. 用 a_t 检测 y_t（因果对齐）
        4. 推进环境到 t+1
        5. 构造下一时刻观测 x_{t+1}
        """
        # ========== 1. 处理输入动作 ==========
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

        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] == 2:
            a = a.argmax(axis=1).astype(np.int32)
        elif a.ndim == 1:
            a = a.astype(np.int32)
        else:
            raise ValueError(f"Expected action shape (N,) or (N, 2), got {a.shape}")
        a = a.reshape(self.N)

        # ========== 2. 【修复关键】获取当前时刻 t 的攻击状态 ==========
        current_t = int(getattr(self.unwrapped_env, "t", 0))
        
        # 2a. 应用攻击到测量通道（在环境推进之前）
        _attack_info_current: dict = {}
        # print(self.attack_writeback)
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None):
            try:
                # 获取当前真值
                p = np.asarray(
                    getattr(self.unwrapped_env, "p", self.unwrapped_env.p),
                    dtype=np.float32
                )
                true_pos = np.asarray(
                    getattr(self.unwrapped_env, "true_p", self.unwrapped_env.p),
                    dtype=np.float32
                )
                heading_now = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
                self._hist_true_pos.append(true_pos.copy())

                # 应用攻击
                z, active = self.atk_manager.apply(
                    t=current_t,
                    gps=p,
                    hist_true_pos=self._hist_true_pos,
                    hist_gps_pos=self._hist_gps_pos,
                    scenario=self.atk_scenario,
                    dt=self.dt,
                    heading_now=heading_now,
                )

                # 写回到测量通道
                if hasattr(self.unwrapped_env, "gps"):
                    setattr(self.unwrapped_env, "gps", z.astype(np.float32))
                else:
                    setattr(self.unwrapped_env, "p", z.astype(np.float32))

                setattr(self.unwrapped_env, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))

                # 构造当前时刻的攻击信息
                _attack_info_current = {
                    "attack_active": bool(
                        np.any(active) and (self.atk_scenario.t0 <= current_t <= self.atk_scenario.t1)
                    ),
                    "attack_mask": np.asarray(active, dtype=bool),
                    "attack_type": str(getattr(self.atk_scenario, "atk_type", "")),
                    "attack_t0_t1": (int(self.atk_scenario.t0), int(self.atk_scenario.t1)),
                }
            except Exception as e:
                import warnings
                warnings.warn(f"Attack application failed: {e}")
        
        # 2b. 获取当前时刻的真值标签 y_t
        y_current = self._as_attack_mask(_attack_info_current)
        gps = self._get_gps()
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)

        # 自身特征（14）
        pred = self._prev_gps + self.v0 * self._prev_h * self.dt

        r_dead = (gps - pred) / self.R
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)

        self._cusum = np.clip(
            self._cusum + (r_norm - self.cusum_threshold - self.cusum_drift),
            0.0,
            self.cusum_max
        )
        # ========== 3. 【修复关键】用 a_t 检测 y_t（因果对齐）==========
        r = self._compute_detection_reward(a, y_current, _attack_info_current)

        # 统计混淆矩阵（用于日志）
        tp = ((a == 1) & (y_current == 1))
        fp = ((a == 1) & (y_current == 0))
        fn = ((a == 0) & (y_current == 1))
        tn = ((a == 0) & (y_current == 0))

        # ========== 4. 导航策略推进底层环境（从 t → t+1）==========
        stage_one_obs = self.unwrapped_env._get_obs()
        obs_vec = np.expand_dims(
            np.concatenate([stage_one_obs['obs'], stage_one_obs['nbr_mask']], axis=1),
            axis=0
        )
        nav_actions = self.nav.act(obs_vec)

        _, _, terminated, truncated, info_base = self.unwrapped_env.step(nav_actions[0])

        # ========== 5. 构造返回信息 ==========
        # 合并攻击信息到 info
        info = dict(_attack_info_current)
        info.update(info_base)
        info["det_tp_fp_fn_tn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum()))

        # ========== 6. 构造下一时刻观测 x_{t+1} ==========
        x_next = self._build_det_obs_method_b()

        # 返回 (x_{t+1}, r_t, done_t, trunc_t, info_t)
        return x_next, r.astype(np.float32), bool(terminated), bool(truncated), info

    # -----------------------------
    # 观测构造（保持不变）
    # -----------------------------
    def _build_det_obs_method_b(self) -> np.ndarray:
        """构造方案B检测特征 (N, 28)"""
        gps = self._get_gps()
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)

        # 自身特征（14）
        pred = self._prev_gps + self.v0 * self._prev_h * self.dt
        r_dead = (gps - pred) / self.R
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)

        speed_res = (
            np.linalg.norm(gps - self._prev_gps, axis=1, keepdims=True)
            - self.v0 * self.dt
        ) / self.R

        dh = np.linalg.norm(h - self._prev_h, axis=1, keepdims=True)

        # 方向一致性
        actual_disp = gps - self._prev_gps
        expected_disp = self._prev_h * self.v0 * self.dt
        actual_norm = np.linalg.norm(actual_disp, axis=1, keepdims=True)
        expected_norm = np.linalg.norm(expected_disp, axis=1, keepdims=True)
        valid_mask = (actual_norm > 1e-3) & (expected_norm > 1e-3)
        direction_cosine = np.where(
            valid_mask,
            np.sum(actual_disp * expected_disp, axis=1, keepdims=True) /
            (actual_norm * expected_norm + 1e-9),
            np.ones((self.N, 1), dtype=np.float32)
        )
        direction_cosine = np.clip(direction_cosine, -1.0, 1.0)

        # 横向偏移
        r_along_heading = np.sum(r_dead * self._prev_h, axis=1, keepdims=True)
        r_lateral = np.sqrt(np.maximum(0.0, r_norm**2 - r_along_heading**2))

        # 时序特征（6）
        self._r_norm_history.append(r_norm.copy())
        self._r_dead_history.append(r_dead.copy())
        self._speed_res_history.append(speed_res.copy())

        r_norm_change = r_norm - self._r_norm_history[-2]
        r_norm_accel = (r_norm - 2*self._r_norm_history[-2] + self._r_norm_history[-3])

        hist_array = np.concatenate(list(self._r_norm_history), axis=1)
        r_norm_std = np.std(hist_array, axis=1, keepdims=True)
        r_norm_trend = (hist_array[:, -1:] - hist_array[:, :1])

        # CUSUM
        # self._cusum = np.clip(
        #     self._cusum + (r_norm - self.cusum_threshold - self.cusum_drift),
        #     0.0,
        #     self.cusum_max
        # )
        cusum_normalized = self._cusum / self.cusum_max

        # 方向变化
        prev_r_dead = self._r_dead_history[-2]
        r_dead_norm_curr = np.linalg.norm(r_dead, axis=1, keepdims=True)
        r_dead_norm_prev = np.linalg.norm(prev_r_dead, axis=1, keepdims=True)
        valid_angle_mask = (r_dead_norm_curr > 1e-6) & (r_dead_norm_prev > 1e-6)
        r_dead_angle_change = np.where(
            valid_angle_mask,
            np.sum(r_dead * prev_r_dead, axis=1, keepdims=True) /
            (r_dead_norm_curr * r_dead_norm_prev + 1e-9),
            np.ones((self.N, 1), dtype=np.float32)
        )
        r_dead_angle_change = np.clip(r_dead_angle_change, -1.0, 1.0)

        temporal_features = np.concatenate([
            r_norm_change, r_norm_accel, r_norm_std,
            r_norm_trend, cusum_normalized, r_dead_angle_change
        ], axis=1)

        own_features = np.concatenate([
            r_dead, r_norm, speed_res, dh,
            direction_cosine, r_lateral, temporal_features
        ], axis=1)

        # 邻域特征（10）
        neighbor_features = self._compute_neighbor_features_method_b(r_dead, r_norm)

        # 编队特征（4）
        formation_features = self._compute_formation_features_method_b(r_dead, r_norm)

        x = np.concatenate([own_features, neighbor_features, formation_features], axis=1)

        assert x.shape == (self.N, 28), f"特征维度错误: 期望 ({self.N}, 28), 实际 {x.shape}"
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            import warnings
            warnings.warn(f"检测到异常特征值: NaN={np.sum(np.isnan(x))}, Inf={np.sum(np.isinf(x))}")
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x.astype(np.float32)

    def _compute_neighbor_features_method_b(self, r_dead:np.ndarray, r_norm: np.ndarray):
        N = self.N
        k = self.k_neighbors

        gps = self._get_gps()
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)

        median = np.median(r_norm)
        mad = np.median(np.abs(r_norm - median))
        global_threshold = median + 3 * max(mad, 1e-6)

        neighbor_feats = []

        for i in range(N):
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            neighbor_idx = neighbor_idx[dist_matrix[i][neighbor_idx] < np.inf]

            if len(neighbor_idx) == 0:
                neighbor_feats.append(np.zeros(10, dtype=np.float32))
                continue

            nbr_r_dead = r_dead[neighbor_idx]
            nbr_r_norm = r_norm[neighbor_idx].flatten()

            nbr_mean_r_dead = np.mean(nbr_r_dead, axis=0)
            nbr_std_r_dead = np.std(nbr_r_dead, axis=0) if len(neighbor_idx) > 1 else np.zeros(3, dtype=np.float32)
            nbr_mean_r_norm = np.mean(nbr_r_norm)

            own_r_dead = r_dead[i]
            own_norm = np.linalg.norm(own_r_dead)
            nbr_norm_val = np.linalg.norm(nbr_mean_r_dead)
            if own_norm > 1e-6 and nbr_norm_val > 1e-6:
                consistency = np.clip(
                    np.dot(own_r_dead, nbr_mean_r_dead) / (own_norm * nbr_norm_val + 1e-9),
                    -1.0, 1.0
                )
            else:
                consistency = 1.0

            nbr_anomaly_ratio = np.sum(nbr_r_norm > global_threshold) / len(neighbor_idx)

            if len(neighbor_idx) > 1:
                nbr_norms = np.linalg.norm(nbr_r_dead, axis=1, keepdims=True)
                valid_directions = nbr_norms.flatten() > 1e-6
                if np.any(valid_directions):
                    nbr_directions = nbr_r_dead[valid_directions] / nbr_norms[valid_directions]
                    direction_variance = np.var(nbr_directions, axis=0).sum()
                else:
                    direction_variance = 0.0
            else:
                direction_variance = 0.0

            feat_i = np.concatenate([
                nbr_mean_r_dead,
                nbr_std_r_dead,
                [nbr_mean_r_norm],
                [consistency],
                [nbr_anomaly_ratio],
                [direction_variance]
            ])
            neighbor_feats.append(feat_i)

        return np.stack(neighbor_feats, axis=0)

    def _compute_formation_features_method_b(self, r_dead:np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """编队特征（4维）"""
        N = self.N
        mean_r_norm = np.mean(r_norm)
        std_r_norm = np.std(r_norm) if N > 1 else 1e-6
        std_r_norm = max(std_r_norm, 1e-6)
        median_r_norm = np.median(r_norm)

        formation_feats = []
        mad_all = np.median(np.abs(r_norm - median_r_norm))
        threshold = median_r_norm + 3 * max(mad_all, 1e-6)

        for i in range(N):
            z_score = (r_norm[i, 0] - mean_r_norm) / std_r_norm
            z_score = np.clip(z_score, -10.0, 10.0)
            anomaly_ratio = np.mean((r_norm.flatten() > threshold).astype(np.float32))
            if N > 2 and std_r_norm > 1e-6:
                skewness = np.mean(((r_norm - mean_r_norm) / std_r_norm) ** 3)
                skewness = np.clip(skewness, -10.0, 10.0)
            else:
                skewness = 0.0
            percentile_rank = np.sum(r_norm < r_norm[i, 0]) / max(N, 1)

            formation_feats.append([z_score, anomaly_ratio, skewness, percentile_rank])

        return np.array(formation_feats, dtype=np.float32)

    # -----------------------------
    # 事件级奖励（保持不变，但现在是因果对齐的）
    # -----------------------------
    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray, info: dict) -> np.ndarray:
        """
        事件级计分 - 【现在是因果对齐的】
        a_t 检测 y_t（而非 y_{t-1}）
        """
        a = np.asarray(a, dtype=np.int32).reshape(self.N)
        y = np.asarray(y, dtype=np.int32).reshape(self.N)
        r = np.zeros(self.N, dtype=np.float32)

        # 边沿检测
        rise_true = (self._y_prevprev == 0) & (y == 1)
        fall_true = (self._y_prevprev == 1) & (y == 0)
        rise_pred = (self._a_prev == 0) & (a == 1)

        current_t = int(getattr(self.unwrapped_env, "t", 0))

        # 1) 真事件开始
        if np.any(rise_true):
            self._event_open[rise_true] = True
            self._detected_in_event[rise_true] = False
            t0_info = info.get("attack_t0_t1", (current_t, current_t))[0]
            self._event_t0[rise_true] = int(t0_info)

            # 即时命中（边沿同时触发）
            instant_hit = rise_true & (a == 1) & (~self._detected_in_event)
            if np.any(instant_hit):
                r[instant_hit] += self.R_TP
                self._detected_in_event[instant_hit] = True

        # 2) 上升沿：TP（时延折扣）
        hit_mask = rise_pred & (y == 1) & (~self._detected_in_event)
        if np.any(hit_mask):
            elapsed_steps = np.maximum(0, current_t - self._event_t0[hit_mask])
            elapsed_sec = elapsed_steps * self.dt
            delay_weight = np.exp(-self.delay_k * np.clip(elapsed_sec, 0.0, self.delay_full_seconds)
                                  / max(self.delay_full_seconds, 1e-6))
            r[hit_mask] += (self.R_TP * delay_weight).astype(np.float32)
            self._detected_in_event[hit_mask] = True

        # FP（冷却机制）
        fp_edge = rise_pred & (y == 0)
        if np.any(fp_edge):
            cooldown_ok = (current_t - self._last_fp_step[fp_edge]) >= self.fp_cooldown_steps
            idx = np.where(fp_edge)[0]
            penalize_idx = idx[cooldown_ok]
            if penalize_idx.size > 0:
                r[penalize_idx] -= self.R_FP
                self._last_fp_step[penalize_idx] = current_t

        # 3) 负类常开惩罚（宽限+豁免）
        hold_neg = (y == 0) & (a == 1) & (~rise_pred)
        self._neg_hold_counter[hold_neg] += 1
        self._neg_hold_counter[~hold_neg] = 0

        cusum_flat = np.asarray(self._cusum).reshape(-1)
        waive = (self._neg_hold_counter <= self.neg_hold_grace_steps) | (cusum_flat >= self.cusum_waive_threshold)
        apply_cost = hold_neg & (~waive)
        if np.any(apply_cost):
            r[apply_cost] -= self.hold_cost

        # 4) 正类持续跟踪
        if self.event_scoring_mode in ("hybrid", "dense"):
            stay = (y == 1) & (a == 1)
            if np.any(stay):
                r[stay] += self.stay_reward

        # 5) TN 基线
        tn = (y == 0) & (a == 0)
        if np.any(tn):
            r[tn] += self.R_TN

        # 6) 事件结束未命中 → FN
        miss_mask = fall_true & self._event_open & (~self._detected_in_event)
        if np.any(miss_mask):
            r[miss_mask] -= self.R_FN
        if np.any(fall_true):
            self._event_open[fall_true] = False
            self._detected_in_event[fall_true] = False

        # 7) 翻转惩罚
        if self.flip_penalty > 0:
            r -= self.flip_penalty * (a != self._a_prev).astype(np.float32)

        # 推进缓存
        self._a_prev = a.copy()
        self._y_prevprev = y.copy()
        return r

    # -----------------------------
    # 其他辅助
    # -----------------------------
    def set_attack_scenario(self, scenario):
        """手动设置攻击场景"""
        self.atk_scenario = scenario

    def clear_attack_scenario(self):
        """清空攻击场景"""
        self.atk_scenario = None

    def get_feature_names(self):
        """返回所有28个特征的名称"""
        own_names = [
            "r_dead_x", "r_dead_y", "r_dead_z",
            "r_norm", "speed_res", "dh",
            "direction_cosine", "r_lateral",
            "r_norm_change", "r_norm_accel", "r_norm_std",
            "r_norm_trend", "cusum", "r_dead_angle_change"
        ]
        neighbor_names = [
            "nbr_mean_r_dead_x", "nbr_mean_r_dead_y", "nbr_mean_r_dead_z",
            "nbr_std_r_dead_x", "nbr_std_r_dead_y", "nbr_std_r_dead_z",
            "nbr_mean_r_norm", "consistency",
            "nbr_anomaly_ratio", "direction_variance"
        ]
        formation_names = [
            "z_score", "anomaly_ratio", "skewness", "percentile_rank"
        ]
        return own_names + neighbor_names + formation_names

    def get_reward_summary(self):
        """返回奖励参数总结（用于调试）"""
        return {
            "R_TP": self.R_TP,
            "R_FP": self.R_FP,
            "R_FN": self.R_FN,
            "R_TN": self.R_TN,
            "hold_cost": self.hold_cost,
            "flip_penalty": self.flip_penalty,
            "collaborative_bonus": self.collaborative_bonus,
            "cusum_max": self.cusum_max,
            "delay_k": self.delay_k,
            "delay_full_seconds": self.delay_full_seconds,
        }

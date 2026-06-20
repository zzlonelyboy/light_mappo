# stage2/stage2_detect_env_enhanced.py
from __future__ import annotations
from typing import Callable, Optional, Any
from collections import deque
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete
from envs.NavPolicyAdapter import NavPolicyAdapter

class Stage2DetectEnv(gym.Env):
    """
    方案B：GPS欺骗检测环境（28维特征）- 已修复版本
    
    特征组成：
    - 自身特征（14维）：基础几何(8) + 时序动态(6)
    - 邻域特征（10维）：基础统计(9) + 协同检测(1)
    - 编队特征（4维）：全局统计 + 分布形态
    

    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        nav_adapter,
        R: Optional[float] = None,
        dt: Optional[float] = None,
        gps_getter: Optional[Callable] = None,
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = False,
        resample_attack_on_reset: bool = True,
        # 邻域参数
        k_neighbors: int = 3,
        history_window: int = 20,  # 时序窗口大小

        reward_tp: float = 10.0,      # 
        reward_fp: float = 2.4,      # 
        reward_fn: float = 4.0,     # 
        flip_penalty: float = 0, # 
        delay_full_seconds: float = 3.0,
        collaborative_bonus: float = 0,
        # 🔧 新增：CUSUM参数
        cusum_threshold: float = 0.1,
        cusum_drift: float = 0.05,
        cusum_max: float = 2.0,
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
        
        # 邻域与时序参数
        self.k_neighbors = min(int(k_neighbors), self.N - 1)
        self.history_window = int(history_window)
        
        # 特征维度：自身(14) + 邻域(10) + 编队(4) = 28
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

        # 🔧 CUSUM参数
        self.cusum_threshold = float(cusum_threshold)
        self.cusum_drift = float(cusum_drift)
        self.cusum_max = float(cusum_max)

        # ========== 时序历史缓存 ==========
        self._r_norm_history = deque(maxlen=self.history_window)
        self._r_dead_history = deque(maxlen=self.history_window)
        self._speed_res_history = deque(maxlen=self.history_window)
        self._cusum = None  # 初始化在reset中

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
        
        # ========== 🔧 修复1：预填充时序历史，避免冷启动问题 ==========
        self._r_norm_history.clear()
        self._r_dead_history.clear()
        self._speed_res_history.clear()
        
        # 用小随机扰动填充历史（模拟初始正常状态）
        for _ in range(self.history_window):
            # 正常状态下的残差应该很小，用均值0.02、标准差0.01的分布
            init_r_norm = np.random.normal(0.02, 0.01, size=(self.N, 1)).astype(np.float32)
            init_r_norm = np.clip(init_r_norm, 0.0, 0.1)  # 限制在合理范围
            
            init_r_dead = np.random.normal(0.0, 0.01, size=(self.N, 3)).astype(np.float32)
            init_r_dead = np.clip(init_r_dead, -0.05, 0.05)
            
            init_speed_res = np.random.normal(0.0, 0.01, size=(self.N, 1)).astype(np.float32)
            init_speed_res = np.clip(init_speed_res, -0.05, 0.05)
            
            self._r_norm_history.append(init_r_norm)
            self._r_dead_history.append(init_r_dead)
            self._speed_res_history.append(init_speed_res)
        
        # ========== 🔧 修复2：CUSUM初始化 ==========
        self._cusum = np.zeros((self.N, 1), dtype=np.float32)
        
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
        self.prev_infobase={}.copy()
        self.prev_y=self._as_attack_mask(self.prev_infobase)
        # 构造检测观测
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
        """推进检测环境一步"""
        # 处理输入动作格式
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
        
        # 处理动作格式
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] == 2:
            a = a.argmax(axis=1).astype(np.int32)
        elif a.ndim == 1:
            a = a.astype(np.int32)
        else:
            raise ValueError(f"Expected action shape (N,) or (N, 2), got {a.shape}")
        a = a.reshape(self.N)


        # 计算检测奖励，(检测的是上一个时刻是否出现攻击)
        r = self._compute_detection_reward(a, self.prev_y, self.prev_infobase)

        info = dict(self.prev_infobase)
        tp = ((a == 1) & (self.prev_y == 1))
        fp = ((a == 1) & (self.prev_y == 0))
        fn = ((a == 0) & (self.prev_y  == 1))
        tn = ((a == 0) & (self.prev_y  == 0))
        info["det_tp_fp_fn_tn"]  = (int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum()))

        # 攻击写回
        _attack_info_to_add: dict = {}
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None):
            try:
                true_pos = np.asarray(self.unwrapped_env.true_p, dtype=np.float32)
                heading_now = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
                self._hist_true_pos.append(true_pos.copy())
                
                # 安全获取时间步
                current_t = int(getattr(self.unwrapped_env, "t", len(self._hist_true_pos) - 1))
                
                z, active = self.atk_manager.apply(
                    t=current_t, # 这是 t
                    true_pos=true_pos,
                    hist_true_pos=self._hist_true_pos,
                    hist_gps_pos=self._hist_gps_pos,
                    scenario=self.atk_scenario,
                    dt=self.dt,
                    heading_now=heading_now,
                )
                
                # 在 step 发生 *之前* 修改环境状态
                setattr(self.unwrapped_env, "p", z)
                setattr(self.unwrapped_env, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))
                
                # 🔧 关键修复：暂存要添加到 info_base 的信息
                _attack_info_to_add = {
                    "attack_active": bool(
                        np.any(active) and (self.atk_scenario.t0 <= current_t <= self.atk_scenario.t1)
                    ),
                    "attack_mask": np.asarray(active, dtype=bool),
                    "attack_type": str(getattr(self.atk_scenario, "atk_type", "")),
                    "attack_t0_t1": (int(self.atk_scenario.t0), int(self.atk_scenario.t1))
                }
                
            except Exception as e:
                import warnings
                warnings.warn(f"Attack application failed: {e}")

        # 3b. 用导航策略在攻击场景下推进底层环境
        stage_one_obs = self.unwrapped_env._get_obs()
        obs_vec = np.expand_dims(
            np.concatenate([stage_one_obs['obs'], stage_one_obs['nbr_mask']], axis=1),
            axis=0
        )
        nav_actions = self.nav.act(obs_vec)
        
        # 执行 step，环境从 t 变为 t+1
        _, _, terminated, truncated, info_base = self.unwrapped_env.step(nav_actions[0])
        
        # 3c. 🔧 关键修复：*现在*才更新 info_base
        if _attack_info_to_add:
            info_base.update(_attack_info_to_add)

        # ========== 4. 获取下一时刻的状态 (t+1) ==========
        
        # 获取 t+1 时刻的真值
        y_next = self._as_attack_mask(info_base)
        
        # 存储 t+1 的状态，用于 *下一次* step 调用
        self.prev_y = y_next
        self.prev_infobase = info_base.copy()

        # 构造 t+1 时刻的检测观测
        x_next = self._build_det_obs_method_b()
        
        # ========== 5. 返回 (x_t+1, r_t, done_t, trunc_t, info_t) ==========
        return x_next, r.astype(np.float32), bool(terminated), bool(truncated), info

    # ========== 方案B观测构造 ==========
    def _build_det_obs_method_b(self) -> np.ndarray:
        """
        构造方案B检测特征 (N, 28)
        已修复：时序冷启动、CUSUM饱和、数值稳定性
        """
        gps = self._get_gps()  # (N, 3)
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)  # (N, 3)

        # ========== 1. 自身特征（14维）==========
        # 基础几何特征（8维）
        pred = self._prev_gps + self.v0 * self._prev_h * self.dt
        r_dead = (gps - pred) / self.R  # (N, 3)
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)  # (N, 1)
        
        speed_res = (
            np.linalg.norm(gps - self._prev_gps, axis=1, keepdims=True) 
            - self.v0 * self.dt
        ) / self.R  # (N, 1)
        
        dh = np.linalg.norm(h - self._prev_h, axis=1, keepdims=True)  # (N, 1)
        
        # ========== 方向一致性计算 ==========
        actual_disp = gps - self._prev_gps
        expected_disp = self._prev_h * self.v0 * self.dt
        
        actual_norm = np.linalg.norm(actual_disp, axis=1, keepdims=True)
        expected_norm = np.linalg.norm(expected_disp, axis=1, keepdims=True)
        
        # 只有当两个向量都足够大时才计算余弦，否则默认为1（认为一致）
        valid_mask = (actual_norm > 1e-3) & (expected_norm > 1e-3)
        direction_cosine = np.where(
            valid_mask,
            np.sum(actual_disp * expected_disp, axis=1, keepdims=True) / 
            (actual_norm * expected_norm + 1e-9),
            np.ones((self.N, 1), dtype=np.float32)
        )  # (N, 1)
        direction_cosine = np.clip(direction_cosine, -1.0, 1.0)  # 防止数值误差
        
        # 横向偏移（数值稳定性处理）
        r_along_heading = np.sum(r_dead * self._prev_h, axis=1, keepdims=True)
        r_lateral = np.sqrt(np.maximum(0.0, r_norm**2 - r_along_heading**2))  # (N, 1)
        
        # ========== 时序特征（6维 ==========
        self._r_norm_history.append(r_norm.copy())
        self._r_dead_history.append(r_dead.copy())
        self._speed_res_history.append(speed_res.copy())
        
        # 现在历史已预填充，直接计算
        # 一阶差分（变化率）
        r_norm_change = r_norm - self._r_norm_history[-2]  # (N, 1)
        
        # 二阶差分（加速度）
        r_norm_accel = (r_norm - 2*self._r_norm_history[-2] + self._r_norm_history[-3])  # (N, 1)
        
        # 滑动统计
        hist_array = np.concatenate(list(self._r_norm_history), axis=1)  # (N, window)
        r_norm_std = np.std(hist_array, axis=1, keepdims=True)  # (N, 1)
        r_norm_trend = (hist_array[:, -1:] - hist_array[:, :1])  # (N, 1)
        
        # ========== 🔧 修复2：CUSUM有界累积 + 自适应归一化 ==========
        self._cusum = np.clip(
            self._cusum + (r_norm - self.cusum_threshold - self.cusum_drift),
            0.0,
            self.cusum_max
        )
        cusum_normalized = self._cusum / self.cusum_max  # 归一化到 [0, 1]
        
        # 方向变化（数值稳定性保护）
        prev_r_dead = self._r_dead_history[-2]
        r_dead_norm_curr = np.linalg.norm(r_dead, axis=1, keepdims=True)
        r_dead_norm_prev = np.linalg.norm(prev_r_dead, axis=1, keepdims=True)
        
        valid_angle_mask = (r_dead_norm_curr > 1e-6) & (r_dead_norm_prev > 1e-6)
        r_dead_angle_change = np.where(
            valid_angle_mask,
            np.sum(r_dead * prev_r_dead, axis=1, keepdims=True) / 
            (r_dead_norm_curr * r_dead_norm_prev + 1e-9),
            np.ones((self.N, 1), dtype=np.float32)
        )  # (N, 1)
        r_dead_angle_change = np.clip(r_dead_angle_change, -1.0, 1.0)
        
        temporal_features = np.concatenate([
            r_norm_change, r_norm_accel, r_norm_std,
            r_norm_trend, cusum_normalized, r_dead_angle_change
        ], axis=1)  # (N, 6)
        
        own_features = np.concatenate([
            r_dead, r_norm, speed_res, dh,
            direction_cosine, r_lateral, temporal_features
        ], axis=1)  # (N, 14)

        # ========== 2. 邻域特征（10维）==========
        neighbor_features = self._compute_neighbor_features_method_b(r_dead, r_norm)

        # ========== 3. 编队特征（4维）==========
        formation_features = self._compute_formation_features_method_b(r_dead, r_norm)

        x = np.concatenate([own_features, neighbor_features, formation_features], axis=1)
        
        # ========== 🔧 添加断言检查特征维度 ==========
        assert x.shape == (self.N, 28), f"特征维度错误: 期望 ({self.N}, 28), 实际 {x.shape}"
        
        # 🔧 检查NaN/Inf
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            import warnings
            warnings.warn(f"检测到异常特征值: NaN={np.sum(np.isnan(x))}, Inf={np.sum(np.isinf(x))}")
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 更新历史
        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x.astype(np.float32)

    def _compute_neighbor_features_method_b(self,r_dead: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        邻域特征（10维）
        已修复：MAD=0保护、余弦数值稳定性
        """
        N = self.N
        k = self.k_neighbors
        
        # 计算位置距离矩阵
        gps = self._get_gps()
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # ========== 🔧 修复4：MAD=0时的除零保护 ==========
        median = np.median(r_norm)
        mad = np.median(np.abs(r_norm - median))
        global_threshold = median + 3 * max(mad, 1e-6)
        
        neighbor_feats = []
        
        for i in range(N):
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            
            # 🔧 过滤无效邻居
            neighbor_idx = neighbor_idx[dist_matrix[i][neighbor_idx] < np.inf]
            
            if len(neighbor_idx) == 0:
                neighbor_feats.append(np.zeros(10, dtype=np.float32))
                continue
            
            nbr_r_dead = r_dead[neighbor_idx]  # (k, 3)
            nbr_r_norm = r_norm[neighbor_idx].flatten()  # (k,)
            
            # 特征1-3：邻居残差均值
            nbr_mean_r_dead = np.mean(nbr_r_dead, axis=0)  # (3,)
            
            # 特征4-6：邻居残差标准差
            nbr_std_r_dead = np.std(nbr_r_dead, axis=0) if len(neighbor_idx) > 1 else np.zeros(3, dtype=np.float32)
            
            # 特征7：邻居残差模长均值
            nbr_mean_r_norm = np.mean(nbr_r_norm)
            
            # ========== 🔧 修复：余弦一致性数值稳定性 ==========
            own_r_dead = r_dead[i]
            own_norm = np.linalg.norm(own_r_dead)
            nbr_norm = np.linalg.norm(nbr_mean_r_dead)
            
            if own_norm > 1e-6 and nbr_norm > 1e-6:
                consistency = np.clip(
                    np.dot(own_r_dead, nbr_mean_r_dead) / (own_norm * nbr_norm + 1e-9),
                    -1.0, 1.0
                )
            else:
                consistency = 1.0  # 默认一致
            
            # 特征9：邻居异常比例
            nbr_anomaly_ratio = np.sum(nbr_r_norm > global_threshold) / len(neighbor_idx)
            
            # 特征10：方向分散度
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
                nbr_mean_r_dead,       # (3)
                nbr_std_r_dead,        # (3)
                [nbr_mean_r_norm],     # (1)
                [consistency],         # (1)
                [nbr_anomaly_ratio],   # (1)
                [direction_variance]   # (1)
            ])
            
            neighbor_feats.append(feat_i)
        
        return np.stack(neighbor_feats, axis=0)  # (N, 10)

    def _compute_formation_features_method_b(self, r_dead:np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        编队特征（4维）
        已修复：std=0保护、偏度极端值限制
        """
        N = self.N
        
        mean_r_norm = np.mean(r_norm)
        # ========== 🔧 修复：std=0时的除零保护 ==========
        std_r_norm = np.std(r_norm) if N > 1 else 1e-6
        std_r_norm = max(std_r_norm, 1e-6)
        
        median_r_norm = np.median(r_norm)
        
        formation_feats = []
        
        for i in range(N):
            # 特征1：z-score
            z_score = (r_norm[i, 0] - mean_r_norm) / std_r_norm
            z_score = np.clip(z_score, -10.0, 10.0)  # 限制极端值
            
            # 特征2：MAD异常比例
            mad = np.median(np.abs(r_norm - median_r_norm))
            threshold = median_r_norm + 3 * max(mad, 1e-6)
            anomaly_ratio = np.mean((r_norm.flatten() > threshold).astype(np.float32))
            
            # 特征3：偏度
            if N > 2 and std_r_norm > 1e-6:
                skewness = np.mean(((r_norm - mean_r_norm) / std_r_norm) ** 3)
                skewness = np.clip(skewness, -10.0, 10.0)
            else:
                skewness = 0.0
            
            # 特征4：分位数排名
            percentile_rank = np.sum(r_norm < r_norm[i, 0]) / max(N, 1)
            
            formation_feats.append([z_score, anomaly_ratio, skewness, percentile_rank])
        
        return np.array(formation_feats, dtype=np.float32)  # (N, 4)

    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray, info: dict) -> np.ndarray:
        """
        计算检测奖励
        已修复：安全获取时间步
        """
        # 基础奖励：TP/FP/FN/TN
        tp = ((a == 1) & (y == 1))
        fp = ((a == 1) & (y == 0))
        fn = ((a == 0) & (y == 1))
        tn = ((a == 0) & (y == 0))
        
        # ========== 🔧 修复：时延计算的安全性 ==========
        if info.get("attack_active", False):
            t0, _ = info.get("attack_t0_t1", (0, 0))
            current_t = int(getattr(self.unwrapped_env, "t", 0))
            elapsed = max(0, current_t - int(t0))
        else:
            elapsed = 0
        w_delay = min(1.0, elapsed * self.dt / max(self.delay_full_seconds, 1e-6))
        
        # 基础奖励
        r = (
            + self.R_TP * tp.astype(np.float32)
            - self.R_FP * fp.astype(np.float32)
            - (self.R_FN * w_delay) * fn.astype(np.float32)
            + 1 * tn.astype(np.float32)  # 🔧 新增：正确否定的小奖励
        )
        
        # 翻转惩罚
        r -= self.flip_penalty * (a != self._a_prev).astype(np.float32)
        
        # 协同奖励
        if self.collaborative_bonus > 0:
            neighbor_bonus = self._compute_collaborative_bonus(a, y)
            r += neighbor_bonus
        
        self._a_prev = a.copy()
        return r

    def _compute_collaborative_bonus(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """协同检测奖励"""
        N = self.N
        k = self.k_neighbors
        
        gps = self._get_gps()
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        
        bonus = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            neighbor_idx = neighbor_idx[dist_matrix[i][neighbor_idx] < np.inf]
            
            if len(neighbor_idx) == 0:
                continue
            
            nbr_actions = a[neighbor_idx]
            nbr_labels = y[neighbor_idx]
            
            # 奖励1：邻居都正确
            nbr_correct = ((nbr_actions == 1) & (nbr_labels == 1)) | ((nbr_actions == 0) & (nbr_labels == 0))
            if np.all(nbr_correct):
                bonus[i] += self.collaborative_bonus * 0.3
            
            # 奖励2：协同检测到攻击
            if (a[i] == 1) and (y[i] == 1):
                nbr_also_detected = np.sum((nbr_actions == 1) & (nbr_labels == 1))
                if nbr_also_detected > 0:
                    bonus[i] += self.collaborative_bonus * 0.5 * (nbr_also_detected / len(neighbor_idx))
            
            # 奖励3：协同正确否定
            if (a[i] == 0) and (y[i] == 0):
                nbr_also_normal = np.sum((nbr_actions == 0) & (nbr_labels == 0))
                if nbr_also_normal == len(neighbor_idx):
                    bonus[i] += self.collaborative_bonus * 0.2
        
        return bonus

    def success_from_info(self, info) -> bool:
        """判断任务是否成功"""
        if "term_reason" in info:
            return info["term_reason"] == "success"
        
        for k in ["success", "episode_success", "all_reached"]:
            if k in info:
                v = info[k]
                return bool(v.item() if hasattr(v, "item") else v)
        
        if hasattr(self.unwrapped_env, "true_c") and hasattr(self.unwrapped_env, "g"):
            if hasattr(self.unwrapped_env, "goal_radius"):
                dc = np.linalg.norm(self.unwrapped_env.true_c - self.unwrapped_env.g)
                return bool(dc <= float(self.unwrapped_env.goal_radius))
        
        return False
    
    def set_attack_scenario(self, scenario):
        """手动设置攻击场景"""
        self.atk_scenario = scenario
    
    def clear_attack_scenario(self):
        """清空攻击场景"""
        self.atk_scenario = None
    
    def get_feature_names(self):
        """返回所有28个特征的名称"""
        own_names = [
            # 基础几何特征（8）
            "r_dead_x", "r_dead_y", "r_dead_z",
            "r_norm", "speed_res", "dh",
            "direction_cosine", "r_lateral",
            # 时序特征（6）
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
            "flip_penalty": self.flip_penalty,
            "collaborative_bonus": self.collaborative_bonus,
            "cusum_max": self.cusum_max,
        }

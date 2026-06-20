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
    方案B：推荐版GPS欺骗检测环境（28维特征）
    
    特征组成：
    - 自身特征（14维）：基础几何(8) + 时序动态(6)
    - 邻域特征（10维）：基础统计(9) + 协同检测(1)
    - 编队特征（4维）：全局统计 + 分布形态
    
    参考文献：
    [1] Psiaki & Humphreys (2016) GNSS Spoofing and Detection, Proc. IEEE
    [23] Page (1954) Continuous Inspection Schemes, Biometrika
    [36] Rousseeuw & Croux (1993) Alternatives to MAD, JASA
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
        history_window: int = 5,  # 时序窗口大小
        # 奖励参数
        reward_tp: float = 1.0,
        reward_fp: float = 1.8,
        reward_fn: float = 3.5,
        flip_penalty: float = 0.01,
        delay_full_seconds: float = 3.0,
        collaborative_bonus: float = 0.2,
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
        self._prev_gps = self._get_gps().copy()
        self._prev_h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32).copy()
        self._a_prev[:] = 0
        
        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()
        
        # ========== 清空时序历史 ==========
        self._r_norm_history.clear()
        self._r_dead_history.clear()
        self._speed_res_history.clear()
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

        # 用导航策略推进底层环境
        stage_one_obs = self.unwrapped_env._get_obs()
        obs_vec = np.expand_dims(
            np.concatenate([stage_one_obs['obs'], stage_one_obs['nbr_mask']], axis=1),
            axis=0
        )
        nav_actions = self.nav.act(obs_vec)
        obs_base, rew_base, terminated, truncated, info_base = self.unwrapped_env.step(nav_actions[0])

        # 攻击写回
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
                
                setattr(self.unwrapped_env, "gps", z)
                setattr(self.unwrapped_env, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))
                
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

        # 计算检测奖励
        y = self._as_attack_mask(info_base)
        r = self._compute_detection_reward(a, y, info_base)

        # 构造检测观测
        x = self._build_det_obs_method_b()

        info = dict(info_base)
        tp = ((a == 1) & (y == 1))
        fp = ((a == 1) & (y == 0))
        fn = ((a == 0) & (y == 1))
        info["det_tp_fp_fn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
        
        return x, r.astype(np.float32), bool(terminated), bool(truncated), info

    # ========== 方案B观测构造 ==========
    def _build_det_obs_method_b(self) -> np.ndarray:
        """
        构造方案B检测特征 (N, 28)
        
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
        
        # 方向一致性
        actual_disp = gps - self._prev_gps
        expected_disp = self._prev_h * self.v0 * self.dt
        direction_cosine = np.sum(actual_disp * expected_disp, axis=1, keepdims=True) / (
            np.linalg.norm(actual_disp, axis=1, keepdims=True) * 
            np.linalg.norm(expected_disp, axis=1, keepdims=True) + 1e-9
        )  # (N, 1)
        
        # 横向偏移
        r_along_heading = np.sum(r_dead * self._prev_h, axis=1, keepdims=True)
        r_lateral = np.sqrt(np.maximum(0, r_norm**2 - r_along_heading**2))  # (N, 1)
        
        # 时序特征（6维）
        self._r_norm_history.append(r_norm.copy())
        self._r_dead_history.append(r_dead.copy())
        self._speed_res_history.append(speed_res.copy())
        
        if len(self._r_norm_history) >= 3:
            # 一阶差分（变化率）
            r_norm_change = r_norm - self._r_norm_history[-2]  # (N, 1)
            
            # 二阶差分（加速度）
            r_norm_accel = (r_norm - 2*self._r_norm_history[-2] + self._r_norm_history[-3])  # (N, 1)
            
            # 滑动统计
            hist_array = np.concatenate(list(self._r_norm_history), axis=1)  # (N, window)
            r_norm_std = np.std(hist_array, axis=1, keepdims=True)  # (N, 1)
            r_norm_trend = (hist_array[:, -1:] - hist_array[:, :1])  # (N, 1)
            
            
            threshold_cusum = 0.1  # 正常残差期望
            drift = 0.05  # 容忍漂移
            self._cusum = np.maximum(0, self._cusum + (r_norm - threshold_cusum - drift))
            cusum_normalized = np.clip(self._cusum / 1.0, 0, 1)  # (N, 1)
            
            # 方向变化
            prev_r_dead = self._r_dead_history[-2]
            r_dead_angle_change = np.sum(r_dead * prev_r_dead, axis=1, keepdims=True) / (
                np.linalg.norm(r_dead, axis=1, keepdims=True) * 
                np.linalg.norm(prev_r_dead, axis=1, keepdims=True) + 1e-9
            )  # (N, 1)
            
            temporal_features = np.concatenate([
                r_norm_change, r_norm_accel, r_norm_std,
                r_norm_trend, cusum_normalized, r_dead_angle_change
            ], axis=1)  # (N, 6)
        else:
            temporal_features = np.zeros((self.N, 6), dtype=np.float32)
        
        own_features = np.concatenate([
            r_dead, r_norm, speed_res, dh,
            direction_cosine, r_lateral, temporal_features
        ], axis=1)  # (N, 14)

        # ========== 2. 邻域特征（10维）==========
        neighbor_features = self._compute_neighbor_features_method_b(r_dead, r_norm)

        # ========== 3. 编队特征（4维）==========
        formation_features = self._compute_formation_features_method_b(r_dead, r_norm)

        x = np.concatenate([own_features, neighbor_features, formation_features], axis=1)
        
        # 更新历史
        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x.astype(np.float32)

    def _compute_neighbor_features_method_b(self, r_dead:np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        邻域特征（10维）
        
        参考文献：
        - 分布式检测：Chair & Varshney (1986)
        - MAD鲁棒统计：Rousseeuw & Croux (1993)
        """
        N = self.N
        k = self.k_neighbors
        
        # 计算位置距离矩阵
        gps = self._get_gps()
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # MAD鲁棒阈值（Rousseeuw & Croux, 1993）
        median = np.median(r_norm)
        mad = np.median(np.abs(r_norm - median))
        global_threshold = median + 3 * mad
        
        neighbor_feats = []
        
        for i in range(N):
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            
            if len(neighbor_idx) == 0:
                neighbor_feats.append(np.zeros(10, dtype=np.float32))
                continue
            
            nbr_r_dead = r_dead[neighbor_idx]  # (k, 3)
            nbr_r_norm = r_norm[neighbor_idx].flatten()  # (k,)
            
            # 特征1-3：邻居残差均值
            nbr_mean_r_dead = np.mean(nbr_r_dead, axis=0)  # (3,)
            
            # 特征4-6：邻居残差标准差
            nbr_std_r_dead = np.std(nbr_r_dead, axis=0) if len(neighbor_idx) > 1 else np.zeros(3)  # (3,)
            
            # 特征7：邻居残差模长均值
            nbr_mean_r_norm = np.mean(nbr_r_norm)  # scalar
            
            # 特征8：余弦一致性
            own_r_dead = r_dead[i]
            if np.linalg.norm(own_r_dead) > 1e-6 and np.linalg.norm(nbr_mean_r_dead) > 1e-6:
                consistency = np.dot(own_r_dead, nbr_mean_r_dead) / (
                    np.linalg.norm(own_r_dead) * np.linalg.norm(nbr_mean_r_dead) + 1e-9
                )
            else:
                consistency = 0.0
            
            # 特征9：邻居异常比例
            nbr_anomaly_ratio = np.sum(nbr_r_norm > global_threshold) / len(neighbor_idx)
            
            # 特征10：方向分散度
            if len(neighbor_idx) > 1:
                nbr_directions = nbr_r_dead / (np.linalg.norm(nbr_r_dead, axis=1, keepdims=True) + 1e-9)
                direction_variance = np.var(nbr_directions, axis=0).sum()
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

    def _compute_formation_features_method_b(self,r_dead: np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        编队特征（4维）
        
        参考文献：
        - 统计假设检验：Lehmann & Romano (2005)
        - 偏度检验：D'Agostino (1970)
        """
        N = self.N
        
        mean_r_norm = np.mean(r_norm)
        std_r_norm = np.std(r_norm) if N > 1 else 1e-6
        median_r_norm = np.median(r_norm)
        
        formation_feats = []
        
        for i in range(N):
            # 特征1：z-score（Lehmann & Romano, 2005）
            z_score = (r_norm[i, 0] - mean_r_norm) / (std_r_norm + 1e-9)
            
            # 特征2：MAD异常比例
            mad = np.median(np.abs(r_norm - median_r_norm))
            threshold = median_r_norm + 3 * mad
            anomaly_ratio = np.mean((r_norm.flatten() > threshold).astype(np.float32))
            
            # 特征3：偏度（D'Agostino, 1970）
            if N > 2:
                skewness = np.mean(((r_norm - mean_r_norm) / (std_r_norm + 1e-9)) ** 3)
            else:
                skewness = 0.0
            
            # 特征4：分位数排名（Wilcoxon, 1945）
            percentile_rank = np.sum(r_norm < r_norm[i, 0]) / N
            
            formation_feats.append([z_score, anomaly_ratio, skewness, percentile_rank])
        
        return np.array(formation_feats, dtype=np.float32)  # (N, 4)

    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray, info: dict) -> np.ndarray:
        """计算检测奖励"""
        # 基础奖励：TP/FP/FN
        tp = ((a == 1) & (y == 1)) | ((a == 0) & (y == 0))
        fp = (a == 1) & (y == 0)
        fn = (a == 0) & (y == 1)
        
        # 时延加权
        if info.get("attack_active", False):
            t0, _ = info.get("attack_t0_t1", (self.unwrapped_env.t, self.unwrapped_env.t))
            elapsed = max(0, int(self.unwrapped_env.t) - int(t0))
        else:
            elapsed = 0
        w_delay = min(1.0, elapsed * self.dt / max(self.delay_full_seconds, 1e-6))
        
        r = (
            + self.R_TP * tp.astype(np.float32)
            - self.R_FP * fp.astype(np.float32)
            - (self.R_FN * w_delay) * fn.astype(np.float32)
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
            if len(neighbor_idx) == 0:
                continue
            
            nbr_actions = a[neighbor_idx]
            nbr_labels = y[neighbor_idx]
            
            # 奖励1：邻居都正确
            nbr_correct = ((nbr_actions == 1) & (nbr_labels == 1)) | ((nbr_actions == 0) & (nbr_labels == 0))
            if np.all(nbr_correct):
                bonus[i] += self.collaborative_bonus * 0.3
            
            # 奖励2：自己和邻居都检测到攻击
            if (a[i] == 1) and (y[i] == 1):
                nbr_also_detected = np.sum((nbr_actions == 1) & (nbr_labels == 1))
                if nbr_also_detected > 0:
                    bonus[i] += self.collaborative_bonus * 0.5 * (nbr_also_detected / len(neighbor_idx))
            
            # 奖励3：自己和邻居都正确否定
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
            # 基础特征（8）
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

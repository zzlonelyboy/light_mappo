# stage2/stage2_detect_env_enhanced.py
from __future__ import annotations
from typing import Callable, Optional, Any
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete
from envs.NavPolicyAdapter import NavPolicyAdapter

class Stage2DetectEnv(gym.Env):
    """
    Stage-2 增强版检测环境：集中式观测 + 邻域协同特征
    
    每个 UAV 的观测包含：
    - 自身特征（8维）：航位推算残差等
    - 邻域特征（9维）：最近邻的统计信息
    - 编队特征（2维）：全局异常度指标
    
    总计：19 维/UAV
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
        resample_attack_on_reset: bool = True,
        # 🆕 邻域参数
        k_neighbors: int = 3,  # 邻域大小
        # 奖励参数
        reward_tp: float = 1.0,
        reward_fp: float = 1.8,
        reward_fn: float = 3.5,
        flip_penalty: float = 0.01,
        delay_full_seconds: float = 3.0,
        # 🆕 协同奖励
        collaborative_bonus: float = 0.2,  # 邻域一致性奖励
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
        
        # 🆕 邻域参数
        self.k_neighbors = min(int(k_neighbors), self.N - 1)  # 最多 N-1 个邻居
        
        # 特征维度：自身(8) + 邻域(9) + 编队(2) = 19
        self.det_dim_own = 8
        self.det_dim_neighbor = 9
        self.det_dim_formation = 2
        self.det_dim = self.det_dim_own + self.det_dim_neighbor + self.det_dim_formation

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
        x = self._build_det_obs_enhanced()
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

        # 计算检测奖励（增强版）
        y = self._as_attack_mask(info_base)
        r = self._compute_detection_reward(a, y, info_base)

        # 构造检测观测
        x = self._build_det_obs_enhanced()

        info = dict(info_base)
        tp = ((a == 1) & (y == 1) | (a==0)&(y==0))
        fp = ((a == 1) & (y == 0))
        fn = ((a == 0) & (y == 1))
        info["det_tp_fp_fn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()))
        
        return x, r.astype(np.float32), bool(terminated), bool(truncated), info

    # ========== 🆕 增强版观测构造 ==========
    def _build_det_obs_enhanced(self) -> np.ndarray:
        """
        构造增强版检测特征 (N, 19)
        
        特征组成：
        1. 自身特征（8维）
        2. 邻域特征（9维）
        3. 编队特征（2维）
        """
        gps = self._get_gps()  # (N, 3)
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)  # (N, 3)

        # ========== 1. 自身特征（8维）==========
        pred = self._prev_gps + float(self.unwrapped_env.v0) * self._prev_h * self.dt
        r_dead = (gps - pred) / self.R  # (N, 3)
        r_norm = np.linalg.norm(r_dead, axis=1, keepdims=True)  # (N, 1)
        
        speed_res = (
            np.linalg.norm(gps - self._prev_gps, axis=1, keepdims=True) 
            - float(self.unwrapped_env.v0) * self.dt
        )  # (N, 1)
        
        dh = np.linalg.norm(h - self._prev_h, axis=1, keepdims=True)  # (N, 1)
        m = np.mean(r_dead, axis=1, keepdims=True)  # (N, 1)
        mx = np.max(np.abs(r_dead), axis=1, keepdims=True)  # (N, 1)
        
        own_features = np.concatenate([r_dead, r_norm, speed_res, dh, m, mx], axis=1)  # (N, 8)

        # ========== 2. 邻域特征（9维）==========
        neighbor_features = self._compute_neighbor_features(r_dead, r_norm)  # (N, 9)

        # ========== 3. 编队特征（2维）==========
        formation_features = self._compute_formation_features(r_dead, r_norm)  # (N, 2)

        # ========== 拼接 ==========
        x = np.concatenate([own_features, neighbor_features, formation_features], axis=1).astype(np.float32)
        
        # 更新历史
        self._prev_gps = gps.copy()
        self._prev_h = h.copy()
        return x

    def _compute_neighbor_features(self, r_dead:np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        计算邻域协同特征（9维）
        
        Args:
            (N, 3) 残差向量
            r_norm: (N, 1) 残差模长
        
        Returns:
            (N, 9) 邻域特征矩阵
        """
        N = self.N
        k = self.k_neighbors
        
        # 计算位置距离矩阵（用于找最近邻）
        gps = self._get_gps()  # (N, 3)
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)  # (N, N)
        np.fill_diagonal(dist_matrix, np.inf)  # 排除自己
        
        neighbor_feats = []
        
        for i in range(N):
            # 找到 k 个最近邻
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            
            if len(neighbor_idx) == 0:
                # 无邻居（N=1 的极端情况）
                neighbor_feats.append(np.zeros(9, dtype=np.float32))
                continue
            
            # 提取邻居的残差
            nbr_r_dead = r_dead[neighbor_idx]  # (k, 3)
            nbr_r_norm = r_norm[neighbor_idx].flatten()  # (k,)
            
            # 特征 1-3：邻居残差向量的均值（3维）
            nbr_mean_r_dead = np.mean(nbr_r_dead, axis=0)  # (3,)
            
            # 特征 4-6：邻居残差向量的标准差（3维）
            nbr_std_r_dead = np.std(nbr_r_dead, axis=0) if len(neighbor_idx) > 1 else np.zeros(3)  # (3,)
            
            # 特征 7：邻居残差模长的均值（1维）
            nbr_mean_r_norm = np.mean(nbr_r_norm)  # scalar
            
            # 特征 8：自己与邻居的残差一致性（余弦相似度）（1维）
            own_r_dead = r_dead[i]  # (3,)
            if np.linalg.norm(own_r_dead) > 1e-6 and nbr_mean_r_dead.any():
                consistency = np.dot(own_r_dead, nbr_mean_r_dead) / (
                    np.linalg.norm(own_r_dead) * np.linalg.norm(nbr_mean_r_dead) + 1e-9
                )
            else:
                consistency = 0.0
            
            # 特征 9：邻居数量归一化（1维）
            neighbor_count_normalized = len(neighbor_idx) / k
            
            # 拼接
            feat_i = np.concatenate([
                nbr_mean_r_dead,           # (3,)
                nbr_std_r_dead,            # (3,)
                [nbr_mean_r_norm],         # (1,)
                [consistency],             # (1,)
                [neighbor_count_normalized] # (1,)
            ])
            
            neighbor_feats.append(feat_i)
        
        return np.stack(neighbor_feats, axis=0)  # (N, 9)

    def _compute_formation_features(self, r_dead:np.ndarray, r_norm: np.ndarray) -> np.ndarray:
        """
        计算编队级特征（2维）
        
        Args:
            (N, 3) 残差向量
            r_norm: (N, 1) 残差模长
        
        Returns:
            (N, 2) 编队特征矩阵
        """
        N = self.N
        
        # 全局统计
        global_mean_r_norm = np.mean(r_norm)
        global_std_r_norm = np.std(r_norm) if N > 1 else 1e-6
        
        formation_feats = []
        
        for i in range(N):
            # 特征 1：自己相对全局的 z-score（异常度）
            z_score = (r_norm[i, 0] - global_mean_r_norm) / (global_std_r_norm + 1e-9)
            
            # 特征 2：局部异常 UAV 比例（基于阈值）
            # 定义"异常"为残差模长 > 全局均值 + 1.5 * 标准差
            threshold = global_mean_r_norm + 1.5 * global_std_r_norm
            anomaly_mask = (r_norm.flatten() > threshold)
            local_anomaly_ratio = np.mean(anomaly_mask.astype(np.float32))
            
            formation_feats.append([z_score, local_anomaly_ratio])
        
        return np.array(formation_feats, dtype=np.float32)  # (N, 2)

    # ========== 🆕 增强版奖励计算 ==========
    def _compute_detection_reward(
        self, 
        a: np.ndarray, 
        y: np.ndarray, 
        info: dict
    ) -> np.ndarray:
        """
        计算检测奖励（增强版，加入协同奖励）
        
        Args:
            a: (N,) 检测动作
            y: (N,) 真实标签
            info: 环境信息字典
        
        Returns:
            (N,) 每个 UAV 的奖励
        """
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
        
        # 🆕 协同奖励：邻域一致性
        if self.collaborative_bonus > 0:
            neighbor_bonus = self._compute_collaborative_bonus(a, y)
            r += neighbor_bonus
        
        self._a_prev = a.copy()
        return r

    def _compute_collaborative_bonus(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        计算协同检测奖励
        
        逻辑：
        - 如果邻居的检测结果与自己一致，且都正确 → 小额奖励
        - 如果检测到协同攻击（多机同时受攻击且都检测到）→ 额外奖励
        
        Args:
            a: (N,) 检测动作
            y: (N,) 真实标签
        
        Returns:
            (N,) 协同奖励
        """
        N = self.N
        k = self.k_neighbors
        
        # 计算位置距离矩阵
        gps = self._get_gps()
        dist_matrix = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=-1)
        np.fill_diagonal(dist_matrix, np.inf)
        
        bonus = np.zeros(N, dtype=np.float32)
        
        for i in range(N):
            neighbor_idx = np.argsort(dist_matrix[i])[:k]
            if len(neighbor_idx) == 0:
                continue
            
            # 检查邻居的检测一致性
            nbr_actions = a[neighbor_idx]
            nbr_labels = y[neighbor_idx]
            
            # 奖励 1：邻居都正确检测（与自己结果无关）
            nbr_correct = ((nbr_actions == 1) & (nbr_labels == 1)) | ((nbr_actions == 0) & (nbr_labels == 0))
            if np.all(nbr_correct):
                bonus[i] += self.collaborative_bonus * 0.3
            
            # 奖励 2：自己和邻居都检测到攻击，且都正确
            if (a[i] == 1) and (y[i] == 1):
                nbr_also_detected = np.sum((nbr_actions == 1) & (nbr_labels == 1))
                if nbr_also_detected > 0:
                    bonus[i] += self.collaborative_bonus * 0.5 * (nbr_also_detected / len(neighbor_idx))
            
            # 奖励 3：自己正确否定，邻居也都正确否定（无攻击的一致性）
            if (a[i] == 0) and (y[i] == 0):
                nbr_also_normal = np.sum((nbr_actions == 0) & (nbr_labels == 0))
                if nbr_also_normal == len(neighbor_idx):
                    bonus[i] += self.collaborative_bonus * 0.2
        
        return bonus

    # ========== 辅助方法 ==========
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
        """返回特征名称（用于可视化/调试）"""
        own_names = [
            "r_dead_x", "r_dead_y", "r_dead_z",
            "r_norm", "speed_res", "dh",
            "mean_r_dead", "max_abs_r_dead"
        ]
        
        neighbor_names = [
            "nbr_mean_r_dead_x", "nbr_mean_r_dead_y", "nbr_mean_r_dead_z",
            "nbr_std_r_dead_x", "nbr_std_r_dead_y", "nbr_std_r_dead_z",
            "nbr_mean_r_norm", "consistency", "neighbor_count_ratio"
        ]
        
        formation_names = [
            "global_z_score", "local_anomaly_ratio"
        ]
        
        return own_names + neighbor_names + formation_names

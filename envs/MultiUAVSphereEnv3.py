from __future__ import annotations
import math
from typing import List, Tuple, Optional
import numpy as np
import gym
from gym.spaces import Box, Dict as SpaceDict


# ------------------------- 基础工具 -------------------------

def wrap_rad(theta: np.ndarray) -> np.ndarray:
    """保证角度在 (-pi, pi] 范围"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def heading_from_angles(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """
    由偏航 yaw 与俯仰 pitch 计算单位朝向向量 h=[cos(yaw)cos(pitch), sin(yaw)cos(pitch), sin(pitch)]
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    h = np.stack([cp * cy, cp * sy, sp], axis=-1)
    return h.astype(np.float32)


def fibonacci_dirs(N: int) -> np.ndarray:
    """在单位球面上生成 N 个近似均匀的方向（Fibonacci 采样）。"""
    gamma = math.pi * (3.0 - math.sqrt(5.0))
    k = np.arange(N, dtype=np.float64)
    z = 1.0 - 2.0 * (k + 0.5) / N
    phi = k * gamma
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    U = np.stack([x, y, z], axis=-1)
    return U.astype(np.float32)


def rotate_to(ez_to: np.ndarray) -> np.ndarray:
    """Rodrigues 旋转：把 e_z=[0,0,1] 旋到单位向量 ez_to。"""
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.asarray(ez_to, dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-9)
    v = np.cross(ez, n)
    s = np.linalg.norm(v)
    c = float(ez @ n)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    theta = math.acos(c)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R.astype(np.float32)


# ------------------------- AABB（方体障碍）工具 -------------------------

def aabb_closest_point(P: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    返回点集 P 到 AABB (center=c, half-size=h) 的最近点。
    P: (N,3)   c: (3,)   h: (3,)
    返回: (N,3)
    """
    return np.minimum(np.maximum(P, c[None, :] - h[None, :]), c[None, :] + h[None, :])


def aabb_signed_distance(P: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    点到 AABB 的有符号距离：盒外为正、盒内为负、贴面为 0。
    P: (N,3)   c: (3,)   h: (3,)
    返回: (N,)
    参考：Inigo Quilez 的 AABB SDF 写法。
    """
    q = np.abs(P - c[None, :]) - h[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.maximum.reduce(q, axis=1), 0.0)
    return outside + inside


# ------------------------- 环境定义 -------------------------

class MultiUAVSphereEnvWithObstacle(gym.Env):
    """Stage-1 多无人机球面编队环境（含可控数量的方形障碍物）。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        N: int = 8,                 # 无人机数
        R: float = 20.0,            # 球面编队半径
        v0: float = 4.0,            # 固定速度 m/s
        dt: float = 0.1,            # 时间步长 s
        d_safe: float = 2.0,        # 机-机安全间距阈值（剐蹭判据）
        yaw_rate_max: float = np.deg2rad(60.0),
        pitch_rate_max: float = np.deg2rad(45.0),
        pitch_abs_max: float = np.deg2rad(60.0),
        K_hist: int = 4,            # heading 历史帧数（形成 K 个差分）
        k_nbr: int = 4,             # 观测中包含的最近邻数量
        episode_seconds: float = 40.0,
        goal_radius: float = 20.0,  # 成功半径（重心到目标）
        align_reward: bool = True,
        seed: Optional[int] = None,

        # ---- 障碍物相关（新增） ----
        num_obstacles: int = 4,     # 障碍物数量（0 表示无障碍）
        obs_k: int = 2,             # 观测中纳入最近障碍个数（每个障碍拼 4 维：方向(3)+净空(1)）；0 表示不加入
        d_safe_obs: float = 4.0,    # 与障碍表面的安全净空（软惩罚阈）
        w_obs: float = 1,         # 障碍净空惩罚权重
        # 障碍生成参数（随机生成时使用）
        obstacle_halfsize_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = ((3, 8), (3, 8), (3, 8)),  # 每轴半尺寸范围
        obstacle_path_span: Tuple[float, float] = (0.3, 0.8),  # 沿（起点->目标）的参数 t 取值范围
        obstacle_lateral_jitter: float = 15.0,                 # 相对路径的横向抖动（米）
        obstacle_clear_margin: float = 5.0,                    # 障碍生成时与初始机位的最小净空约束

        # ---- 奖励参数 ----
        kp: float = 3.0,
        w_form: float = 0.03,
        w_sep: float = 1,
        w_smooth: float = 0.05,
        k_align_weight: float = 0.5,  # 若 align_reward=False 则置 0
        R_succ: float = 400.0,
        alive_penalty_per_step: float = 1.0,   # 每步小额负奖励（按人均分）
        no_progress_window: int = 20,          # 无进展窗口（步）
        no_progress_eps: float = 0.5,          # dc 改善阈值（米）
        no_prog_penalty: float = 4.0,          # 触发时每步额外惩罚（按人均分）
        # 机-机“剐蹭”惩罚（不终止）
        R_scrape_ind: float = 60.0,       # 参与者个体惩罚
        R_scrape_shared: float = 0.0,     # 小额团队连带（可 0）
        # 障碍物碰撞（终止）惩罚
        R_obs_coll_ind: float = 300.0,    # 进入盒体的个体惩罚
        R_obs_coll_shared: float = 0.0,   # 团队连带（可 0）
        R_tout: float = 20.0,              # 可选超时惩罚（默认未启用）
    ) -> None:
        super().__init__()

        # ---- 基本参数 ----
        self.N = int(N)
        self.R = float(R)
        self.v0 = float(v0)
        self.dt = float(dt)
        self.d_safe = float(d_safe)
        self.yaw_rate_max = float(yaw_rate_max)
        self.pitch_rate_max = float(pitch_rate_max)
        self.pitch_abs_max = float(pitch_abs_max)
        self.K_hist = int(K_hist)
        self.k_nbr = int(k_nbr)
        self.T_max = int(round(episode_seconds / dt))
        self.goal_radius = float(goal_radius)
        self.align_reward = bool(align_reward)
        self._rng = np.random.RandomState(42 if seed is None else seed)

        # ---- 奖励权重 ----
        self.kp = float(kp)
        self.w_form = float(w_form)
        self.w_sep = float(w_sep)
        self.w_smooth = float(w_smooth)
        self.k_align = float(k_align_weight if align_reward else 0.0)
        self.R_succ = float(R_succ)
        self.R_tout = float(R_tout)
        

        self.alive_penalty_per_step = float(alive_penalty_per_step) ## 每步小额负奖励（按人均分）
        self.no_progress_window = int(no_progress_window)
        self.no_progress_eps = float(no_progress_eps)
        self.no_prog_penalty = float(no_prog_penalty)

        # 剐蹭与障碍碰撞罚
        self.R_scrape_ind = float(R_scrape_ind)
        self.R_scrape_shared = float(R_scrape_shared)
        self.R_obs_coll_ind = float(R_obs_coll_ind)
        self.R_obs_coll_shared = float(R_obs_coll_shared)

        # ---- 障碍参数 ----
        self.num_obstacles = int(num_obstacles)
        self.obs_k = int(obs_k)
        self.d_safe_obs = float(d_safe_obs)
        self.w_obs = float(w_obs)
        self.obstacle_halfsize_range = obstacle_halfsize_range
        self.obstacle_path_span = obstacle_path_span
        self.obstacle_lateral_jitter = float(obstacle_lateral_jitter)
        self.obstacle_clear_margin = float(obstacle_clear_margin)

        # AABB 列表：每个元素 (center[np.float32(3,)], half[np.float32(3,)])
        self.obstacles: List[Tuple[np.ndarray, np.ndarray]] = []
        self._obstacles_fixed: bool = False  # True 表示用户手动设置，reset 不再随机重置
        # ---- 空间定义 ----
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.N, 2), dtype=np.float32)
        self.per_agent_obs_dim = self._compute_per_agent_obs_dim()
        self.observation_space = SpaceDict(
            {
                "obs": Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.N, self.per_agent_obs_dim),
                    dtype=np.float32,
                ),
                "nbr_mask": Box(low=0.0, high=1.0, shape=(self.N, self.k_nbr), dtype=np.float32),
            }
        )

        # ---- 状态缓存 ----
        self.reset()

    # ------------------------- 外部 API -------------------------

    def add_box_obstacles(
        self,
        centers: List[Tuple[float, float, float]],
        half_sizes: List[Tuple[float, float, float]],
    ):
        """手动设置障碍（设置后在 reset 期间不再随机生成）。"""
        assert len(centers) == len(half_sizes)
        self.obstacles = []
        for c, h in zip(centers, half_sizes):
            c = np.array(c, dtype=np.float32)
            h = np.array(h, dtype=np.float32)
            self.obstacles.append((c, h))
        self.num_obstacles = len(self.obstacles)
        self._obstacles_fixed = True

    def clear_obstacles(self):
        """清空障碍，下次 reset 会按 num_obstacles 随机生成。"""
        self.obstacles = []
        self._obstacles_fixed = False

    def set_num_obstacles(self, n: int):
        """设置随机障碍物数量（仅在 _obstacles_fixed=False 时生效，reset 时生成）。"""
        self.num_obstacles = int(n)
        if self._obstacles_fixed:
            # 用户手工设置过则优先手工；如果想恢复随机请 clear_obstacles()
            pass

    # ------------------------- Gym API -------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng.seed(seed)
        options = options or {}
        self.t = 0

        # ---------- 起点重心 c：默认原点；options 可覆盖 ----------
        start_opt = options.get("start", None)
        if start_opt is not None:
            self.c = np.asarray(start_opt, dtype=np.float32).copy()
        else:
            self.c = np.zeros(3, dtype=np.float32)
        self.true_c = self.c.copy()

        # ---------- 目标点 g：默认随机；options 可覆盖 ----------
        goal_opt = options.get("goal", None)
        if goal_opt is not None:
            self.g = np.asarray(goal_opt, dtype=np.float32).copy()
        else:
            goal_dist = self._rng.uniform(80.0, 120.0)
            dir_xyz = self._rng.randn(3).astype(np.float32)
            dir_xyz /= (np.linalg.norm(dir_xyz) + 1e-9)
            self.g = dir_xyz * goal_dist

        # 目标槽位方向（对齐目标方向）
        U0 = fibonacci_dirs(self.N)
        # Rn = rotate_to((self.g - self.c) / (np.linalg.norm(self.g - self.c) + 1e-9))
        # self.U = (Rn @ U0.T).T.astype(np.float32)
        self.U=U0.astype(np.float32)
        # 初始位置：放在球面槽位（你当前版本不加 jitter）
        self.p = (self.c[None, :] + self.R * self.U).astype(np.float32)
        self.true_p = self.p.copy()
        # 生成/保持障碍
        if not self._obstacles_fixed:
            self._generate_random_obstacles()

        # 初始姿态：朝向目标并加微噪
        to_goal = (self.g[None, :] - self.p)
        to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)
        yaw = np.arctan2(to_goal[:, 1], to_goal[:, 0])
        pitch = np.arcsin(to_goal[:, 2])
        yaw += self._rng.normal(0.0, np.deg2rad(5.0), size=self.N)
        pitch += self._rng.normal(0.0, np.deg2rad(5.0), size=self.N)
        self.yaw = wrap_rad(yaw).astype(np.float32)
        self.pitch = np.clip(pitch, -self.pitch_abs_max, self.pitch_abs_max).astype(np.float32)

        # heading 历史（K_hist+1 帧，用于差分）
        h = heading_from_angles(self.yaw, self.pitch)
        self.h_hist = np.repeat(h[None, :, :], self.K_hist + 1, axis=0)

        # 前一时刻动作
        self.prev_action = np.zeros((self.N, 2), dtype=np.float32)

        # 缓存距离
        self._update_center()
        self._prev_dc = np.linalg.norm(self.c - self.g)
        self._dc_window = [self._prev_dc]

        obs, info = self._get_obs(), self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self.t += 1
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (self.N, 2):
            raise ValueError(f"action shape must be (N,2), got {a.shape}")
        a = np.clip(a, -1.0, 1.0)

        # 角速度积分
        d_yaw = a[:, 0] * (self.yaw_rate_max * self.dt)
        d_pitch = a[:, 1] * (self.pitch_rate_max * self.dt)
        self.yaw = wrap_rad(self.yaw + d_yaw)
        self.pitch = np.clip(self.pitch + d_pitch, -self.pitch_abs_max, self.pitch_abs_max)

        # 更新 heading 历史
        new_h = heading_from_angles(self.yaw, self.pitch)
        self.h_hist = np.roll(self.h_hist, shift=-1, axis=0)
        self.h_hist[-1] = new_h

        # 固定速度推进
        self.p = self.p + self.v0 * new_h * self.dt
        self.true_p =self.true_p+self.v0 * new_h * self.dt
        # 重心
        self._update_center()

        # 奖励与终止
        rew = self._compute_reward(a)
        terminated, term_reason = self._check_terminated()
        truncated = self.t >= self.T_max

        if self.alive_penalty_per_step != 0.0:
            rew -= (self.alive_penalty_per_step / self.N)
        self._dc_window.append(np.linalg.norm(self.c - self.g))
        if len(self._dc_window) > max(2, self.no_progress_window):
            self._dc_window.pop(0)
            if self.no_prog_penalty != 0.0:
                dc_improve = self._dc_window[0] - self._dc_window[-1]
                if dc_improve < self.no_progress_eps:
                    rew -= (self.no_prog_penalty / self.N)
        # 可选超时惩罚（如需启用，可在终止步时生效）
        if truncated and not terminated and self.R_tout > 0:
            rew -= (self.R_tout / self.N)
        

        info = self._get_info()
        if terminated or truncated:
            if terminated:
                info["term_reason"] = term_reason
            else:
                info["term_reason"] = "timeout"
            # print(info['ep_done'])
        self.prev_action = a.copy()
        obs = self._get_obs()
        return obs, rew, terminated, truncated, info

    def render(self):
        print(f"t={self.t}, center={self.c}, goal={self.g}, d_c={np.linalg.norm(self.c-self.g):.2f}")

    # ------------------------- 内部方法 -------------------------

    def _update_center(self):
        self.c = self.p.mean(axis=0)
        self.true_c = self.true_p.mean(axis=0)


    def _compute_per_agent_obs_dim(self) -> int:
        # (p-c)/R (3) + e_i/R (3) + (g-p)/||·|| (3) + h_i (3) + Δh K (3K)
        base = 3 + 3 + 3 + 3 + 3 * self.K_hist
        # 邻居：Δp_ij/R(3) + Δv_ij(3) + d_ij/R(1) + mask(1) = 8/邻居
        nbr = 8 * self.k_nbr
        # 障碍特征：每个障碍拼 dir(3) + clearance/R(1) = 4
        obs_extra = 4 * int(self.obs_k)
        return base + nbr + obs_extra

    def _get_obs(self):
        # 自身特征
        e_i = self.p - (self.c[None, :] + self.R * self.U)  # (N,3)
        e_i_norm = e_i / self.R
        pc_rel = (self.p - self.c[None, :]) / self.R
        to_goal = self.g[None, :] - self.p
        to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)
        h_now = self.h_hist[-1]
        dh_list = (self.h_hist[1:] - self.h_hist[:-1])  # (K, N, 3)
        dh_flat = dh_list.transpose(1, 0, 2).reshape(self.N, -1)

        # 邻居特征
        nbr_mask = np.zeros((self.N, self.k_nbr), dtype=np.float32)
        nbr_feats = np.zeros((self.N, 8 * self.k_nbr), dtype=np.float32)
        P = self.p
        dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1) + np.eye(self.N) * 1e9
        idx_sorted = np.argsort(dmat, axis=1)
        for i in range(self.N):
            nbr_idx = [j for j in idx_sorted[i, : self.k_nbr + 1] if j != i][: self.k_nbr]
            for k, j in enumerate(nbr_idx):
                dp = (P[j] - P[i]) / self.R
                dv = (self.h_hist[-1, j] - self.h_hist[-1, i])
                dij = np.linalg.norm(P[j] - P[i]) / self.R
                nbr_mask[i, k] = 1.0
                feat = np.concatenate([dp, dv, [dij], [1.0]], axis=0)
                nbr_feats[i, k * 8 : (k + 1) * 8] = feat

        obs_agent = np.concatenate([pc_rel, e_i_norm, to_goal, h_now, dh_flat, nbr_feats], axis=1)

        # 障碍物最近 obs_k 个特征（可选）
        if self.obs_k > 0:
            if len(self.obstacles) > 0:
                sd_all_list = []
                dir_all_list = []
                for (oc, oh) in self.obstacles:
                    cp = aabb_closest_point(P, oc, oh)        # (N,3)
                    dvec = cp - P                              # (N,3)
                    dist = np.linalg.norm(dvec, axis=1)        # (N,)
                    dir_unit = np.divide(dvec, dist[:, None] + 1e-9)
                    sd = aabb_signed_distance(P, oc, oh)       # (N,)
                    sd_all_list.append(sd)
                    dir_all_list.append(dir_unit)

                sd_all = np.stack(sd_all_list, axis=1)         # (N, Mobs)
                dir_all = np.stack(dir_all_list, axis=2)       # (N,3,Mobs)
                idx = np.argsort(sd_all, axis=1)               # (N, Mobs)

                feats = []
                Mobs = sd_all.shape[1]
                take = min(self.obs_k, Mobs)
                for i in range(self.N):
                    fi = []
                    for k in range(take):
                        j = idx[i, k]
                        dir_ijk = dir_all[i, :, j]             # (3,)
                        sd_ij = sd_all[i, j]
                        fi.append(np.concatenate([dir_ijk, [sd_ij / self.R]], axis=0))  # 4
                    while len(fi) < self.obs_k:
                        fi.append(np.zeros(4, dtype=np.float32))
                    feats.append(np.concatenate(fi, axis=0))   # 4*obs_k
                feats = np.stack(feats, axis=0)
            else:
                # 无障碍：直接零填充
                feats = np.zeros((self.N, 4 * self.obs_k), dtype=np.float32)

            obs_agent = np.concatenate([obs_agent, feats], axis=1)
        
        obs = {
            "obs": obs_agent.astype(np.float32),
            "nbr_mask": nbr_mask.astype(np.float32),
        }
        return obs

    def _compute_reward(self, action: np.ndarray) -> np.ndarray:
        # ---- 共享：重心进展（望远镜和式）----
        dc = np.linalg.norm(self.c - self.g)
        r_prog = self.kp * (self._prev_dc - dc)
        self._prev_dc = dc

        # ---- per-agent：编队误差 ----
        e = self.true_p - (self.c[None, :] + self.R * self.U)
        r_form_i = - self.w_form * (np.linalg.norm(e, axis=1) / self.R)

        # ---- per-agent：机-机安全（软铰链）----
        P = self.true_p
        dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        dmin_i = dmat.min(axis=1)
        hinge_i = np.maximum(0.0, (self.d_safe - dmin_i) / self.d_safe)
        r_sep_i = - self.w_sep * (hinge_i ** 2)

        # ---- per-agent：动作平滑 ----
        da = action - self.prev_action
        r_smooth_i = - self.w_smooth * np.sum(da * da, axis=1)

        # ---- per-agent：对齐（可选）----
        if self.k_align > 0.0:
            to_goal = self.g[None, :] - self.p
            to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)
            h = self.h_hist[-1]
            r_align_i = self.k_align * np.sum(h * to_goal, axis=1)
        else:
            r_align_i = np.zeros(self.N, dtype=np.float32)

        # ---- per-agent：障碍净空（软惩罚，可关）----
        if self.w_obs > 0.0 and len(self.obstacles) > 0:
            sd_min = None
            for (oc, oh) in self.obstacles:
                sd = aabb_signed_distance(self.p, oc, oh)  # (N,)
                sd_min = sd if sd_min is None else np.minimum(sd_min, sd)
            hinge_obs = np.maximum(0.0, (self.d_safe_obs - sd_min) / self.d_safe_obs)
            r_obs_i = - self.w_obs * (hinge_obs ** 2)
        else:
            r_obs_i = np.zeros(self.N, dtype=np.float32)

        # 合成 per-agent 基本奖励
        r_i = r_prog + r_form_i + r_sep_i + r_smooth_i + r_align_i + r_obs_i  # (N,)

        # ---- 机-机剐蹭（不终止）：最近邻 < d_safe 的个体个罚 ----
        scrape_involved = (dmin_i < self.d_safe)  # (N,)
        if scrape_involved.any():
            r_i[scrape_involved] -= self.R_scrape_ind
            if self.R_scrape_shared > 0.0:
                r_i -= self.R_scrape_shared / self.N

        # ---- 终止事件 ----
        term, reason = self._check_terminated()
        if term:
            if reason == "success":
                r_i += self.R_succ / self.N
            elif reason == "collision":
                involved = np.zeros(self.N, dtype=bool)
                for (oc, oh) in self.obstacles:
                    sd = aabb_signed_distance(self.p, oc, oh)
                    involved |= (sd <= 0.0)
                r_i[involved] -= self.R_obs_coll_ind
                if self.R_obs_coll_shared > 0.0:
                    r_i -= self.R_obs_coll_shared / self.N

        return r_i.astype(np.float32)

    def _check_terminated(self):
        # 成功
        # if np.linalg.norm(self.c - self.g) <= self.goal_radius:
        if np.linalg.norm(self.true_c-self.g)<=self.goal_radius:
            # print("success")
            return True, "success"

        # 仅障碍物碰撞才终止
        if len(self.obstacles) > 0:
            for (oc, oh) in self.obstacles:
                sd = aabb_signed_distance(self.true_p, oc, oh)  # (N,)
                if (sd <= 0.0).any():
                    return True, "collision"
        return False, ""

    def _get_info(self):
        P = self.p
        dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1) + np.eye(self.N) * 1e9
        dmin = dmat.min(axis=1)
        e_i = self.p - (self.c[None, :] + self.R * self.U)
        info = {
            "t": self.t,
            "center": self.c.copy(),
            "goal": self.g.copy(),
            "dc": float(np.linalg.norm(self.c - self.g)),
            "min_dist_p5_p50_p95": np.percentile(dmin, [5, 50, 95]).astype(np.float32),
            "mean_form_err": float(np.mean(np.linalg.norm(e_i, axis=1))),
        }
        if len(self.obstacles) > 0:
            clears = []
            for (oc, oh) in self.obstacles:
                clears.append(aabb_signed_distance(self.p, oc, oh))
            clears = np.stack(clears, axis=1)
            info["min_obs_clearance"] = float(clears.min())
        return info

    # ------------------------- 障碍生成（内部） -------------------------

    # def _generate_random_obstacles(self):
    #     """根据 num_obstacles 随机生成 AABB，沿路径附近分布。"""
    #     self.obstacles = []
    #     if self.num_obstacles <= 0:
    #         return

    #     # 路径：c(0)→g(1) 的直线
    #     c0 = self.c.copy()
    #     g0 = self.g.copy()
    #     dir_path = g0 - c0
    #     L = float(np.linalg.norm(dir_path))
    #     if L < 1e-6:
    #         dir_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    #     else:
    #         dir_unit = dir_path / L

    #     # 构造两个与路径正交的基
    #     tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    #     if abs(np.dot(tmp, dir_unit)) > 0.9:
    #         tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    #     e1 = np.cross(dir_unit, tmp); e1 /= (np.linalg.norm(e1) + 1e-9)
    #     e2 = np.cross(dir_unit, e1);  e2 /= (np.linalg.norm(e2) + 1e-9)

    #     tmin, tmax = self.obstacle_path_span
    #     for _ in range(self.num_obstacles):
    #         # 沿路径的参数 t（不放太靠近起点/终点）
    #         t = float(self._rng.uniform(tmin, tmax))
    #         base = c0 + t * dir_unit * L
    #         # 横向抖动
    #         jitter = (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e1 \
    #                + (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e2
    #         center = base + jitter

    #         # 半尺寸
    #         hx = self._rng.uniform(*self.obstacle_halfsize_range[0])
    #         hy = self._rng.uniform(*self.obstacle_halfsize_range[1])
    #         hz = self._rng.uniform(*self.obstacle_halfsize_range[2])
    #         half = np.array([hx, hy, hz], dtype=np.float32)

    #         # 与所有初始机位保持最小净空（避免一开场就卡在障碍里）
    #         sd0 = aabb_signed_distance(self.p, center, half)  # (N,)
    #         if (sd0 < self.obstacle_clear_margin).any():
    #             # 若不满足净空则再抖一次（简单重采样 3 次）
    #             ok = False
    #             for _retry in range(3):
    #                 jitter = (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e1 \
    #                        + (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e2
    #                 center = base + jitter
    #                 sd0 = aabb_signed_distance(self.p, center, half)
    #                 if not (sd0 < self.obstacle_clear_margin).any():
    #                     ok = True
    #                     break
    #             if not ok:
    #                 # 放弃这一障碍，继续下一个，确保不会把初始机位困住
    #                 continue

    #         self.obstacles.append((center.astype(np.float32), half.astype(np.float32)))
    def _generate_random_obstacles(self):
        """随机生成 AABB，沿 c→g 附近分布；写死一条可达通路（航线走廊）约束。"""
        self.obstacles = []
        M = int(self.num_obstacles)
        if M <= 0:
            return

        # 固定常量（写死）
        PATH_CLEAR_MARGIN = 8.0          # 航线走廊半宽（米）
        MAX_TRIES_PER_OBS = 10           # 每个障碍的最大放置尝试
        GLOBAL_MAX_TRIES = max(50, 10*M) # 全局尝试上限
        SHRINK_FACTOR = 0.85             # 放置失败时缩小盒体
        SECOND_STAGE_TRIES = 6           # 缩小后再试次数

        # 路径：c(0)→g(1)
        c0 = self.c.copy()
        g0 = self.g.copy()
        dir_path = g0 - c0
        L = float(np.linalg.norm(dir_path))
        if L < 1e-6:
            dir_unit = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            dir_unit = dir_path / L

        # 正交基（横向偏移用）
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(np.dot(tmp, dir_unit)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        e1 = np.cross(dir_unit, tmp); e1 /= (np.linalg.norm(e1) + 1e-9)
        e2 = np.cross(dir_unit, e1);  e2 /= (np.linalg.norm(e2) + 1e-9)

        tmin, tmax = self.obstacle_path_span

        placed = 0
        global_tries = 0
        while placed < M and global_tries < GLOBAL_MAX_TRIES:
            global_tries += 1

            # 路径上的基准点
            t = float(self._rng.uniform(tmin, tmax))
            base = c0 + t * dir_unit * L

            # 初始尺寸
            hx = self._rng.uniform(*self.obstacle_halfsize_range[0])
            hy = self._rng.uniform(*self.obstacle_halfsize_range[1])
            hz = self._rng.uniform(*self.obstacle_halfsize_range[2])
            half = np.array([hx, hy, hz], dtype=np.float32)

            ok = False
            # 第一阶段：原尺寸尝试
            for _ in range(MAX_TRIES_PER_OBS):
                # 横向抖动
                jitter = (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e1 \
                    + (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter) * e2
                center = base + jitter

                # ① 航线走廊约束：障碍中心到路径距离 >= PATH_CLEAR_MARGIN
                #    d_line = || (center - c0) × dir_unit ||
                d_line = np.linalg.norm(np.cross(center - c0, dir_unit))
                if d_line < PATH_CLEAR_MARGIN:
                    continue

                # ② 初始机位净空（不把无人机一出生就卡住）
                sd0 = aabb_signed_distance(self.p, center, half)  # (N,)
                if (sd0 < self.obstacle_clear_margin).any():
                    continue

                ok = True
                break

            # 第二阶段：若还不行，缩小尺寸再试几次
            if not ok:
                half2 = half * SHRINK_FACTOR
                for _ in range(SECOND_STAGE_TRIES):
                    jitter = (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter * 0.7) * e1 \
                        + (self._rng.uniform(-1.0, 1.0) * self.obstacle_lateral_jitter * 0.7) * e2
                    center = base + jitter

                    d_line = np.linalg.norm(np.cross(center - c0, dir_unit))
                    if d_line < PATH_CLEAR_MARGIN:
                        continue

                    sd0 = aabb_signed_distance(self.p, center, half2)
                    if (sd0 < self.obstacle_clear_margin).any():
                        continue

                    half = half2  # 采用缩小后的尺寸
                    ok = True
                    break

            if not ok:
                # 这一次放置失败，换一个候选（继续 while 循环）
                continue

            # 成功放置
            self.obstacles.append((center.astype(np.float32), half.astype(np.float32)))
            placed += 1

from __future__ import annotations
import math
import numpy as np
import gym
from gym.spaces import Box, Dict as SpaceDict


def wrap_rad(theta: np.ndarray) -> np.ndarray:
    """保证旋转角角度在(-pi, pi]范围内"""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def heading_from_angles(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """
    从偏航角yaw和俯仰角pitch计算单位 heading 向量
    h=[ cos(yaw)*cos(pitch), sin(yaw)*cos(pitch), sin(pitch) ]
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # [x, y, z]
    hx = cp * cy
    hy = cp * sy
    hz = sp
    h = np.stack([hx, hy, hz], axis=-1)
    # already unit (cp^2*(cy^2+sy^2) + sp^2 = 1)
    return h.astype(np.float32)

# 生成N个单位向量，分布在单位球面上（Fibonacci）
def fibonacci_dirs(N: int) -> np.ndarray:
    """Generate N nearly-uniform directions on the unit sphere (Fibonacci)."""
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
    """Rodrigues rotation matrix that rotates e_z to ez_to (unit vector)."""
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


class MultiUAVSphereEnv(gym.Env):
    """Stage-1 multi-UAV spherical formation environment (Gym-style)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        N: int = 8, #无人机梳理
        R: float = 20.0, #球面半径
        v0: float = 4.0, #初始速度
        dt: float = 0.1, #时间步长
        d_safe: float = 2.0, #安全距离
        yaw_rate_max: float = np.deg2rad(60.0),   # rad/s
        pitch_rate_max: float = np.deg2rad(45.0), # rad/s
        pitch_abs_max: float = np.deg2rad(60.0),  # |pitch| ≤ 60°
        # 记录的历史轨迹的长度
        K_hist: int = 4,  # heading history length
        k_nbr: int = 4,   # number of nearest neighbors to include
        episode_seconds: float = 40.0,
        goal_radius: float = 20.0,  # success when center within this radius
        align_reward: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()

        self.N = N
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

        self._rng = np.random.RandomState(seed if seed is not None else 42)

        # Spaces
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
                # 由于每个无人机不一定都有k个相邻的无人机，所以这里用一个mask来表示每个无人机是否有k个相邻的无人机，真实存在设置为1否则设置为0
                "nbr_mask": Box(low=0.0, high=1.0, shape=(self.N, self.k_nbr), dtype=np.float32),
            }
        )

        # Reward weights (defaults from our discussion)
        self.kp = 5.0
        self.w_form = 0.03
        self.w_sep = 0.8
        self.w_smooth = 0.01
        self.k_align = 0.5 if self.align_reward else 0.0

        self.R_succ = 100.0
        self.R_coll = 100.0
        self.R_tout = 20.0
        self.R_coll_ind = 100.0  #发生碰撞的无人机个体惩罚
        self.R_coll_shared=40

        # State buffers
        self.reset()

    # ---------- Gym API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng.seed(seed)
        self.t = 0

        # Goal and center
        self.c = np.zeros(3, dtype=np.float32)  # center of formation (state-computed)
        # Sample a goal direction and distance
        goal_dist = self._rng.uniform(80.0, 120.0)  # meters
        dir_xyz = self._rng.randn(3).astype(np.float32)
        dir_xyz /= (np.linalg.norm(dir_xyz) + 1e-9)
        # 目标点位
        self.g = dir_xyz * goal_dist

        # 初始化各个无人机在球面的目标位置
        U0 = fibonacci_dirs(self.N)  # (N,3)
        Rn = rotate_to((self.g - self.c) / (np.linalg.norm(self.g - self.c) + 1e-9))
        self.U = (Rn @ U0.T).T.astype(np.float32)  # unit directions u_i

        # # 初始化各个球面位置
        # jitter = self._rng.normal(0.0, 0.1, size=(self.N, 3)).astype(np.float32)  # small
        # jitter -= (self.U * (self.U * jitter).sum(axis=1, keepdims=True))  # tangential jitter
        # 初始化各个无人机在球面的位置
        self.p = (self.c[None, :] + self.R * self.U).astype(np.float32)

        # Initialize yaw/pitch to face the goal with noise
        to_goal = (self.g[None, :] - self.p)
        to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)
        yaw = np.arctan2(to_goal[:, 1], to_goal[:, 0])
        pitch = np.arcsin(to_goal[:, 2])
        yaw += self._rng.normal(0.0, np.deg2rad(5.0), size=self.N)
        pitch += self._rng.normal(0.0, np.deg2rad(5.0), size=self.N)
        self.yaw = wrap_rad(yaw).astype(np.float32)
        self.pitch = np.clip(pitch, -self.pitch_abs_max, self.pitch_abs_max).astype(np.float32)

        # Heading buffers (K_hist+1 frames to form K diffs)
        h = heading_from_angles(self.yaw, self.pitch)
        self.h_hist = np.repeat(h[None, :, :], self.K_hist + 1, axis=0)  # (K+1, N, 3)

        # Previous action (for smoothness)
        self.prev_action = np.zeros((self.N, 2), dtype=np.float32)

        # 更新重心并记录上一帧的重心距离
        self._update_center()
        self._prev_dc = np.linalg.norm(self.c - self.g)

        obs, info = self._get_obs(), self._get_info()
        return obs, info

    def step(self, action: np.ndarray):
        self.t += 1
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (self.N, 2):
            raise ValueError(f"action shape must be (N,2), got {a.shape}")
        a = np.clip(a, -1.0, 1.0)

        # Map normalized actions to angle increments (bounded rates)
        d_yaw = a[:, 0] * (self.yaw_rate_max * self.dt)
        d_pitch = a[:, 1] * (self.pitch_rate_max * self.dt)

        # Integrate angles with wrap/clip
        self.yaw = wrap_rad(self.yaw + d_yaw)
        self.pitch = np.clip(self.pitch + d_pitch, -self.pitch_abs_max, self.pitch_abs_max)

        # Update heading history
        new_h = heading_from_angles(self.yaw, self.pitch)  # (N,3)
        self.h_hist = np.roll(self.h_hist, shift=-1, axis=0)
        self.h_hist[-1] = new_h

        # Kinematics: fixed speed
        self.p = self.p + self.v0 * new_h * self.dt

        # Update center
        self._update_center()

        # Compute reward and termination
        rew = self._compute_reward(a)
        terminated, term_reason = self._check_terminated()
        truncated = self.t >= self.T_max

        info = self._get_info()
        info["term_reason"] = term_reason

        self.prev_action = a.copy()
        obs = self._get_obs()
        return obs, rew, terminated, truncated, info

    def render(self):
        # Minimal textual render (for quick debugging)
        print(
            f"t={self.t}, center={self.c}, goal={self.g}, d_c={np.linalg.norm(self.c-self.g):.2f}"
        )

    # ---------- Internals ----------

    def _update_center(self):
        self.c = self.p.mean(axis=0)

    def _compute_per_agent_obs_dim(self) -> int:
        # (p-c)/R (3) + e_i/R (3) + (g-p)/||·|| (3) + h_i (3) + Δh K (3K)
        base = 3 + 3 + 3 + 3 + 3 * self.K_hist
        # neighbors per j: Δp_ij/R (3) + Δv_ij/v0 (3) + d_ij/R (1) + mask (1) = 8
        nbr = 8 * self.k_nbr
        return base + nbr

    def _get_obs(self):
        # Self features
        e_i = self.p - (self.c[None, :] + self.R * self.U)  # formation error (N,3)
        e_i_norm = e_i / self.R

        pc_rel = (self.p - self.c[None, :]) / self.R  # (N,3)
        to_goal = self.g[None, :] - self.p
        to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)  # unit (N,3)

        h_now = self.h_hist[-1]  # (N,3)
        dh_list = (self.h_hist[1:] - self.h_hist[:-1])  # (K, N, 3)
        dh_flat = dh_list.transpose(1, 0, 2).reshape(self.N, -1)  # (N, 3K)

        # Neighbor features
        nbr_mask = np.zeros((self.N, self.k_nbr), dtype=np.float32)
        nbr_feats = np.zeros((self.N, 8 * self.k_nbr), dtype=np.float32)

        # Pairwise distances
        P = self.p
        # 计算距离，并在对角线上加上1e9，确保自己与自己的距离极大,不会被选中
        dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1) + np.eye(self.N) * 1e9
        # 排序后的索引数组
        idx_sorted = np.argsort(dmat, axis=1)

        for i in range(self.N):
            # k nearest neighbors (excluding self)
            nbr_idx = [j for j in idx_sorted[i, : self.k_nbr + 1] if j != i][: self.k_nbr]
            for k, j in enumerate(nbr_idx):
                dp = (P[j] - P[i]) / self.R
                dv = (self.h_hist[-1, j] - self.h_hist[-1, i])  # unit diff
                dij = np.linalg.norm(P[j] - P[i]) / self.R
                nbr_mask[i, k] = 1.0
                feat = np.concatenate([dp, dv, [dij], [1.0]], axis=0)  # 3+3+1+1=8
                nbr_feats[i, k * 8 : (k + 1) * 8] = feat

        obs_agent = np.concatenate([pc_rel, e_i_norm, to_goal, h_now, dh_flat, nbr_feats], axis=1)
        obs = {
            "obs": obs_agent.astype(np.float32),
            "nbr_mask": nbr_mask.astype(np.float32),
        }
        return obs

    def _compute_reward(self, action: np.ndarray) -> np.ndarray:
        # ---------- 共享：进展 ----------
        dc = np.linalg.norm(self.c - self.g)
        r_prog = self.kp * (self._prev_dc - dc)  # scalar
        self._prev_dc = dc

        # ---------- per-agent：编队误差 ----------
        e = self.p - (self.c[None, :] + self.R * self.U)  # (N,3)
        r_form_i = - self.w_form * (np.linalg.norm(e, axis=1) / self.R)  # (N,)

        # ---------- per-agent：邻距安全（软铰链） ----------
        P = self.p
        dmat = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)  # (N,N)
        np.fill_diagonal(dmat, np.inf)  # 忽略自身
        dmin_i = dmat.min(axis=1)  # (N,)
        hinge_i = np.maximum(0.0, (self.d_safe - dmin_i) / self.d_safe)
        r_sep_i = - self.w_sep * (hinge_i ** 2)  # (N,)

        # ---------- per-agent：动作平滑 ----------
        da = action - self.prev_action  # (N,2)
        r_smooth_i = - self.w_smooth * np.sum(da * da, axis=1)  # (N,)

        # ---------- per-agent：朝向对齐（可选） ----------
        if self.k_align > 0.0:
            to_goal = self.g[None, :] - self.p
            to_goal /= (np.linalg.norm(to_goal, axis=1, keepdims=True) + 1e-9)
            h = self.h_hist[-1]
            r_align_i = self.k_align * np.sum(h * to_goal, axis=1)  # (N,)
        else:
            r_align_i = np.zeros(self.N, dtype=np.float32)

        # ---------- 合成：共享进展 + 个体 shaping ----------
        r_i = r_prog + r_form_i + r_sep_i + r_smooth_i + r_align_i  # (N,)

        # ---------- 事件分数 ----------
        term, reason = self._check_terminated()
        if term:
            if reason == "success":
                # 成功均分，避免随 N 爆量
                r_i += self.R_succ / self.N
            elif reason == "collision":
                # 参与碰撞的个体（与任意他人距离 < d_safe）
                involved = (dmat < self.d_safe).any(axis=1)  # (N,) bool
                # 个体重罚（默认等于原来的 R_coll，可在 __init__ 里单独设置 R_coll_ind）
                r_i[involved] -= self.R_coll_ind
                # 可选：小额团队惩罚（表达任务失败感知），默认 0
                if self.R_coll_shared > 0.0:
                    r_i -= self.R_coll_shared / self.N

        return r_i.astype(np.float32)

    def _check_terminated(self):
        # success
        if np.linalg.norm(self.c - self.g) <= self.goal_radius:
            return True, "success"
        # collision
        dmat = np.linalg.norm(self.p[:, None, :] - self.p[None, :, :], axis=-1) + np.eye(self.N) * 1e9
        if (dmat.min(axis=1) < self.d_safe).any():
            return True, "collision"
        return False, ""

    def _get_info(self):
        # metrics for logging
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
        return info

# Save this module path for download
print("Saved MultiUAVSphereEnv in-memory. You can import this cell output as a .py by saving it if needed.")

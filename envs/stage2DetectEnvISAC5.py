from __future__ import annotations

from typing import Callable, Optional, Any, Dict, Tuple
from collections import deque
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete

from envs.NavPolicyAdapter import NavPolicyAdapter


# ============================================================
# Utils
# ============================================================
def _unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.clip(dot / (na * nb + eps), -1.0, 1.0)


# ============================================================
# ISAC measurement generator
# SNR is computed from a monostatic radar link-budget model:
#   rho = Pt * sigma_A * lambda^2 / [ (4*pi)^3 * r^4 * N0 * B ]
#
# Measurement uncertainty:
#   sigma_r^2 = (c / B)^2 / rho
#   sigma_d^2 = (lambda / T_coh)^2 / rho
#   sigma_u^2 = (lambda / D_eff)^2 / rho
# ============================================================
class ISACGenerator:
    """
    Output:
      - r:        (N,M)  noisy range
      - u:        (N,M,3) noisy LOS unit vector
      - dop:      (N,M) noisy radial velocity / Doppler
      - mask:     (N,M) 1 valid / 0 invalid
      - rho:      (N,M) linear SNR
      - rho_db:   (N,M) dB SNR
      - sigma_r2: (N,M)
      - sigma_d2: (N,M)
      - sigma_u2: (N,M)
      - anchors:  (M,3)
    """

    def __init__(
            self,
            anchors: np.ndarray,  # (M,3)

            # ----- OFDM / link-budget related -----
            fc_hz: float = 3e9,
            bw_hz: float = 30e6,

            pt_dbm: float = 27.0,
            sigma_A: float = 22.5,
            # ----- SNR clipping / validity -----
            rho_clip_db_min: float = -10.0,
            rho_clip_db_max: float = 40.0,
            rho_valid_threshold_db: float = 0.0,

            T_coh=0.05,  # 50 ms
            D_eff=0.35,

            # ----- optional random invalidation -----
            p_dropout: float = 0.0,
            rng: Optional[np.random.RandomState] = None,
    ):
        self.anchors = np.asarray(anchors, dtype=np.float32)
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 3:
            raise ValueError("anchors must be of shape (M,3)")
        self.M = int(self.anchors.shape[0])

        # OFDM / link budget
        self.fc_hz = float(fc_hz)
        self.bw_hz = float(bw_hz)

        self.pt_dbm = float(pt_dbm)
        self.sigma_A = float(sigma_A)

        # SNR clipping / validity
        self.rho_clip_db_min = float(rho_clip_db_min)
        self.rho_clip_db_max = float(rho_clip_db_max)
        self.rho_valid_threshold_db = float(rho_valid_threshold_db)

        self.T_coh = float(T_coh)
        self.D_eff = float(D_eff)

        self.p_dropout = float(p_dropout)
        self.rng = np.random.RandomState(0) if rng is None else rng

    def _compute_rho(self, true_p: np.ndarray):
        """
        双程回波 SNR，忽略 Gt、Gr、NF、misc loss
        rho = Pt * sigma_A * lambda^2 / [ (4*pi)^3 * r^4 * N0 * B ]
        """
        A = self.anchors[None, :, :]  # (1, M, 3)
        P = true_p[:, None, :]  # (N, 1, 3)
        dvec = A - P
        dist = np.linalg.norm(dvec, axis=-1).astype(np.float32)
        dist = np.maximum(dist, 1.0)  # 防止距离过小数值爆炸

        # 常数
        c = 3.0e8
        lam = c / self.fc_hz

        # 发射功率：dBm -> W
        pt_w = 10.0 ** ((self.pt_dbm) / 10.0)

        # 热噪声功率谱密度：-174 dBm/Hz -> W/Hz
        n0_w_hz = 10.0 ** ((-174.0) / 10.0)

        # 总噪声功率
        noise_w = n0_w_hz * self.bw_hz

        # 双程回波 SNR
        rho = (
                pt_w
                * self.sigma_A
                * (lam ** 2)
                / (((4.0 * np.pi) ** 3) * (dist ** 4) * noise_w)
        )

        rho = np.maximum(rho, 1e-30)
        rho_db = 10.0 * np.log10(rho)

        # 保持与你现有框架一致
        rho_db = np.clip(rho_db, self.rho_clip_db_min, self.rho_clip_db_max)
        rho = np.power(10.0, rho_db / 10.0).astype(np.float32)

        return rho.astype(np.float32), rho_db.astype(np.float32)

    def measure(self, true_p: np.ndarray, h: np.ndarray, v0: float) -> Dict[str, np.ndarray]:
        true_p = np.asarray(true_p, dtype=np.float32)  # (N,3)
        h = np.asarray(h, dtype=np.float32)  # (N,3)
        N = int(true_p.shape[0])

        # -------------------------
        # geometry
        # -------------------------
        A = self.anchors[None, :, :]  # (1,M,3)
        P = true_p[:, None, :]  # (N,1,3)
        dvec = A - P  # (N,M,3)

        r_true = np.linalg.norm(dvec, axis=-1)  # (N,M)
        u_true = _unit(dvec)  # (N,M,3)

        v = (float(v0) * h)[:, None, :]  # (N,1,3)
        dop_true = np.sum(v * u_true, axis=-1)  # (N,M)

        # -------------------------
        # SNR
        # -------------------------
        rho, rho_db = self._compute_rho(true_p)  # (N,M)
        rho_safe = np.maximum(rho, 1e-9).astype(np.float32)

        # -------------------------
        # physical constants
        # -------------------------
        c = 299792458.0
        lam = c / float(self.fc_hz)

        # sigma^2 from formulas
        # sigma_r ≈ c / (B_eff * sqrt(rho))
        # sigma_d ≈ lam / (T_coh * sqrt(rho))
        # sigma_u ≈ lam / (D_eff * sqrt(rho))
        sigma_r2 = ((c / self.bw_hz) ** 2) / rho_safe
        sigma_d2 = ((lam / self.T_coh) ** 2) / rho_safe
        sigma_u2 = ((lam / self.D_eff) ** 2) / rho_safe

        sigma_r = np.sqrt(sigma_r2).astype(np.float32)
        sigma_d = np.sqrt(sigma_d2).astype(np.float32)
        sigma_u = np.sqrt(sigma_u2).astype(np.float32)

        # -------------------------
        # add measurement noisep
        # -------------------------
        # range noise
        r_meas = r_true + self.rng.normal(0.0, 1.0, size=r_true.shape).astype(np.float32) * sigma_r

        # doppler / radial velocity noise
        dop_meas = dop_true + self.rng.normal(0.0, 1.0, size=dop_true.shape).astype(np.float32) * sigma_d

        # direction noise: tangent-plane perturbation
        raw_noise = self.rng.normal(0.0, 1.0, size=u_true.shape).astype(np.float32)
        proj = np.sum(raw_noise * u_true, axis=-1, keepdims=True) * u_true
        tangent_noise = raw_noise - proj
        tangent_noise = tangent_noise * sigma_u[:, :, None]
        u_meas = _unit(u_true + tangent_noise)

        # -------------------------
        # valid anchor mask
        # -------------------------
        rho_valid_threshold = 10.0 ** (self.rho_valid_threshold_db / 10.0)
        mask = (rho >= rho_valid_threshold).astype(np.float32)

        if self.p_dropout > 0.0:
            drop = (self.rng.rand(N, self.M) > self.p_dropout).astype(np.float32)
            mask *= drop

        r_meas *= mask
        dop_meas *= mask
        u_meas *= mask[:, :, None]

        return {
            "r": r_meas.astype(np.float32),
            "u": u_meas.astype(np.float32),
            "dop": dop_meas.astype(np.float32),
            "mask": mask.astype(np.float32),
            "rho": rho.astype(np.float32),
            "rho_db": rho_db.astype(np.float32),
            "sigma_r2": sigma_r2.astype(np.float32),
            "sigma_d2": sigma_d2.astype(np.float32),
            "sigma_u2": sigma_u2.astype(np.float32),
            "anchors": self.anchors.astype(np.float32),
        }


# ============================================================
# Main environment
# ============================================================
class Stage2ISACDetectEnv(gym.Env):
    """
    27-dim observation:
      - GPS-ISAC residual consistency features: 21
      - self features: 2
      - coordinated neighborhood spatial features: 4
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            base_env: gym.Env,
            gps_getter: Optional[Callable] = None,
            anchors: Optional[np.ndarray] = None,

            # OFDM-ISAC / link budget
            fc_hz: float = 3e9,
            # bw_hz: float = 100e6,
            bw_hz: float = 30e6,

            pt_dbm: float = 27.0,

            sigma_A: float = 25,

            rho_clip_db_min: float = -10.0,
            rho_clip_db_max: float = 40.0,
            rho_valid_threshold_db: float = 0.0,

            T_coh=0.05,  # 50 ms
            D_eff=0.35,

            p_dropout: float = 0.0,

            # attack
            atk_manager: Optional[Any] = None,
            atk_scenario: Optional[Any] = None,
            attack_writeback: bool = True,
            resample_attack_on_reset: bool = True,

            # normalization
            pos_scale: float = 2.0,
            k_neighbors: int = 3,
            history_window=24,
            cusum_beta=0.9,

            # reward
            reward_tp=75.0,
            reward_fp=30.0,
            reward_tn=0.8,
            stay_reward=0.3,
            delay_full_seconds=2.0,
            fn_step_penalty: float = 0.25,
            delay_k: float = 1.0,

            # WLS position proxy
            p_isac_tol = 1e-5,
            p_isac_lam=1e-5,
            p_isac_step_clip=10,
            p_isac_iters=50,
            min_anchors_for_p_isac: int = 4,
            # Stage1 stepper
            nav_stepper: Optional[NavPolicyAdapter] = None,
    ):
        super().__init__()

        self.base = base_env
        self.nav_stepper = nav_stepper
        self.gps_getter = gps_getter
        # unwrap
        self.T_coh = T_coh
        self.D_eff = D_eff

        self.unwrapped_env = base_env
        while hasattr(self.unwrapped_env, "env") or hasattr(self.unwrapped_env, "_env"):
            if hasattr(self.unwrapped_env, "env"):
                self.unwrapped_env = self.unwrapped_env.env
            else:
                self.unwrapped_env = self.unwrapped_env._env

        self.N = int(getattr(self.unwrapped_env, "N", 0))
        if self.N <= 0:
            raise ValueError("base_env must expose positive integer attribute N")

        self.dt = float(getattr(self.unwrapped_env, "dt", 0.1))
        self.v0 = float(getattr(self.unwrapped_env, "v0", 4.0))

        if anchors is None:
            anchors = np.array(
                [[20, 40, 5], [20, 80, 10], [40, 40, 20], [80, 40, 15]],
                dtype=np.float32
            )

        self.isac_gen = ISACGenerator(
            anchors=anchors,

            fc_hz=fc_hz,
            bw_hz=bw_hz,

            sigma_A=sigma_A,

            pt_dbm=pt_dbm,

            rho_clip_db_min=rho_clip_db_min,
            rho_clip_db_max=rho_clip_db_max,
            rho_valid_threshold_db=rho_valid_threshold_db,

            T_coh=self.T_coh,
            D_eff=self.D_eff,

            p_dropout=p_dropout,
            rng=getattr(self.unwrapped_env, "_rng", None),
        )
        self.M = self.isac_gen.M

        # attack
        self.atk_manager = atk_manager
        self.atk_scenario = atk_scenario
        self.attack_writeback = bool(attack_writeback)
        self.resample_attack_on_reset = bool(resample_attack_on_reset)
        self._hist_true_pos = []
        self._hist_gps_pos = []

        # normalization
        self.pos_scale = float(pos_scale)

        # graph
        self.k_neighbors = max(0, min(int(k_neighbors), self.N - 1))

        # temporal stats
        self.history_window = max(1, int(history_window))
        self._residual_hist = deque(maxlen=self.history_window)
        self._neighbor_geo_mean_hist = deque(maxlen=self.history_window)
        self.cusum_beta = float(cusum_beta)
        self._residual_ewma = None
        self._residual_cusum = None

        # reward
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_TN = float(reward_tn)
        self.fn_step_penalty = float(fn_step_penalty)
        self.stay_reward = float(stay_reward)
        self.delay_full_seconds = float(delay_full_seconds)
        self.delay_k = float(delay_k)

        # position proxy
        self.p_isac_iters = int(p_isac_iters)
        self.p_isac_lam = float(p_isac_lam)
        self.p_isac_step_clip = float(p_isac_step_clip)
        self.p_isac_tol = float(p_isac_tol)
        self.min_anchors_for_p_isac = int(min_anchors_for_p_isac)
        self._p_isac_prev = None

        # history / internal states
        self._y_prevprev = np.zeros((self.N,), dtype=np.int32)
        self._detected_in_event = np.zeros((self.N,), dtype=bool)
        self._event_t0 = np.zeros((self.N,), dtype=np.int32)
        self._delay_prev_y = np.zeros((self.N,), dtype=np.int32)
        self._delay_event_open = np.zeros((self.N,), dtype=bool)
        self._delay_event_detected = np.zeros((self.N,), dtype=bool)
        self._delay_event_t0 = np.zeros((self.N,), dtype=np.int32)

        # spaces
        self.action_space = MultiDiscrete(np.array([2] * self.N, dtype=np.int64))
        self.feature_names = self.get_feature_names()
        self.obs_dim = len(self.feature_names)
        # assert self.obs_dim == 27, f"Expected obs_dim=27, got {self.obs_dim}"
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.N, self.obs_dim),
            dtype=np.float32
        )
        # print("TP:",self.R_TP)

        self._attack_info_current: Dict[str, Any] = {}
        self._y_current = np.zeros((self.N,), dtype=np.int32)

    # ========================================================
    # IO helpers
    # ========================================================
    def _get_gps(self) -> np.ndarray:
        if callable(self.gps_getter):
            return np.asarray(self.gps_getter(self.unwrapped_env), dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            return np.asarray(self.unwrapped_env.gps, dtype=np.float32)
        if hasattr(self.unwrapped_env, "p"):
            return np.asarray(self.unwrapped_env.p, dtype=np.float32)
        raise RuntimeError("Cannot get GPS: provide gps_getter or expose .gps/.p on base_env")

    def _set_gps(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            setattr(self.unwrapped_env, "gps", z)
        else:
            setattr(self.unwrapped_env, "p", z)

    def _get_heading_now(self) -> np.ndarray:
        if hasattr(self.unwrapped_env, "h_hist"):
            hh = getattr(self.unwrapped_env, "h_hist")
            if isinstance(hh, (list, tuple)) and len(hh) > 0:
                return np.asarray(hh[-1], dtype=np.float32)
        if hasattr(self.unwrapped_env, "h"):
            return np.asarray(getattr(self.unwrapped_env, "h"), dtype=np.float32)

        h = np.zeros((self.N, 3), dtype=np.float32)
        h[:, 0] = 1.0
        return h

    def _as_attack_mask(self, info: dict) -> np.ndarray:
        mask = info.get("attack_mask", None)
        if mask is None:
            return np.zeros((self.N,), dtype=np.int32)
        return np.asarray(mask, dtype=bool).astype(np.int32)

    def _apply_attack_writeback(self) -> Dict[str, Any]:
        if not (self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None)):
            try:
                setattr(self.unwrapped_env, "_atk_active_mask", np.zeros((self.N,), dtype=bool))
            except Exception:
                pass
            return {}

        t = int(getattr(self.unwrapped_env, "t", 0))

        try:
            p_meas = self._get_gps()
            true_pos = np.asarray(getattr(self.unwrapped_env, "true_p", p_meas), dtype=np.float32)
            heading_now = self._get_heading_now()

            self._hist_true_pos.append(true_pos.copy())

            z, active = self.atk_manager.apply(
                t=t,
                gps=p_meas,
                hist_true_pos=self._hist_true_pos,
                hist_gps_pos=self._hist_gps_pos,
                scenario=self.atk_scenario,
                dt=self.dt,
                heading_now=heading_now,
            )

            z = np.asarray(z, dtype=np.float32)
            active = np.asarray(active, dtype=bool)

            self._set_gps(z)
            try:
                setattr(self.unwrapped_env, "_atk_active_mask", active)
            except Exception:
                pass

            self._hist_gps_pos.append(z.copy())

            t0 = int(getattr(self.atk_scenario, "t0", t))
            t1 = int(getattr(self.atk_scenario, "t1", t))
            atk_type = str(getattr(self.atk_scenario, "atk_type", ""))

            return {
                "attack_t": t,
                "attack_active": bool(np.any(active) and (t0 <= t <= t1)),
                "attack_mask": active,
                "attack_type": atk_type,
                "attack_t0_t1": (t0, t1),
            }

        except Exception:
            try:
                setattr(self.unwrapped_env, "_atk_active_mask", np.zeros((self.N,), dtype=bool))
            except Exception:
                pass
            return {}


    def _estimate_p_from_range_dir(
            self,
            anchors: np.ndarray,
            r: np.ndarray,
            u: np.ndarray,
            mask: np.ndarray,
            p_init: np.ndarray,
    ) -> np.ndarray:
        """
        使用无权几何非线性最小二乘，根据距离和LoS方向测量估计UAV位置。

        对于第 i 个UAV，求解：

            min_p sum_k [
                (||a_k - p||_2 - r_ik)^2
                + ||P_ik (a_k - p)||_2^2
            ]

        其中：

            P_ik = I_3 - u_ik u_ik^T

        第一项为距离残差平方；
        第二项为候选位置到测量LoS直线的横向距离平方。

        Parameters
        ----------
        anchors : np.ndarray
            ISAC基站位置，形状为 (M, 3)。

        r : np.ndarray
            UAV到各ISAC基站的距离测量，形状为 (N, M)。

        u : np.ndarray
            从UAV指向各ISAC基站的LoS方向测量，形状为 (N, M, 3)。

        mask : np.ndarray
            有效测量掩码，形状为 (N, M)。

        p_init : np.ndarray
            UAV位置初始值，形状为 (N, 3)。

        Returns
        -------
        np.ndarray
            估计得到的UAV位置，形状为 (N, 3)。
        """
        N, M = r.shape

        if anchors.shape != (M, 3):
            raise ValueError(
                f"anchors应具有形状 ({M}, 3)，当前为 {anchors.shape}"
            )

        if u.shape != (N, M, 3):
            raise ValueError(
                f"u应具有形状 ({N}, {M}, 3)，当前为 {u.shape}"
            )

        if mask.shape != (N, M):
            raise ValueError(
                f"mask应具有形状 ({N}, {M})，当前为 {mask.shape}"
            )

        if p_init.shape != (N, 3):
            raise ValueError(
                f"p_init应具有形状 ({N}, 3)，当前为 {p_init.shape}"
            )

        A = anchors.astype(np.float32)
        ranges = r.astype(np.float32)
        directions = u.astype(np.float32)

        p = p_init.astype(np.float32).copy()

        I3 = np.eye(3, dtype=np.float32)
        converged = np.zeros((N,), dtype=bool)

        for _ in range(self.p_isac_iters):
            all_done = True

            for i in range(N):
                if converged[i]:
                    continue

                # ---------------------------------------------------------
                # 选择有效测量
                # ---------------------------------------------------------
                mi = mask[i].astype(bool).copy()
                mi &= np.isfinite(ranges[i])
                mi &= ranges[i] > 0.0
                mi &= np.all(np.isfinite(directions[i]), axis=1)
                mi &= np.all(np.isfinite(A), axis=1)

                if int(np.sum(mi)) < self.min_anchors_for_p_isac:
                    # 有效基站不足，保留上一位置估计。
                    converged[i] = True
                    continue

                Ai = A[mi]  # (K, 3)
                ri = ranges[i, mi]  # (K,)
                ui = directions[i, mi].copy()  # (K, 3)

                # 将方向测量归一化，保证 P = I - uu^T 为正交投影矩阵。
                ui_norm = np.linalg.norm(ui, axis=1, keepdims=True)
                ui = ui / np.maximum(ui_norm, 1e-9)

                pi = p[i].copy()

                # 阻尼Gauss-Newton矩阵。
                H = self.p_isac_lam * I3.copy()
                g = np.zeros((3,), dtype=np.float32)

                cost0 = 0.0

                # ---------------------------------------------------------
                # 构造Gauss-Newton正规方程
                # ---------------------------------------------------------
                for k in range(Ai.shape[0]):
                    ak = Ai[k]
                    rk = ri[k]
                    uk = ui[k]

                    # 从候选UAV位置指向ISAC基站的向量。
                    q = ak - pi
                    dk = float(np.linalg.norm(q))
                    dk = max(dk, 1e-9)

                    # =====================================================
                    # 1. 距离残差
                    #
                    # er = ||a_k - p|| - r_k
                    # =====================================================
                    er = dk - rk

                    # d||a_k-p||/dp = -(a_k-p)/||a_k-p||
                    Jr = -q / dk  # (3,)

                    H += np.outer(Jr, Jr)
                    g += Jr * er
                    cost0 += float(er * er)

                    # =====================================================
                    # 2. LoS横向距离残差
                    #
                    # eu = P_k (a_k - p)
                    # P_k = I - u_k u_k^T
                    # =====================================================
                    Pk = I3 - np.outer(uk, uk)

                    eu = Pk @ q  # (3,)

                    # 因为 q = a_k - p，且Pk由测量方向确定，
                    # 所以 d[Pk(a_k-p)]/dp = -Pk。
                    Ju = -Pk  # (3, 3)

                    H += Ju.T @ Ju
                    g += Ju.T @ eu
                    cost0 += float(eu @ eu)

                try:
                    dx = -np.linalg.solve(H, g).astype(np.float32)
                except np.linalg.LinAlgError:
                    all_done = False
                    continue

                # ---------------------------------------------------------
                # 限制单次更新幅度
                # ---------------------------------------------------------
                step_norm = float(np.linalg.norm(dx))

                if step_norm > self.p_isac_step_clip:
                    dx *= self.p_isac_step_clip / (step_norm + 1e-9)

                # ---------------------------------------------------------
                # 回溯搜索，保证目标函数不增大
                # ---------------------------------------------------------
                accepted = False
                step_scale = 1.0

                p_new = pi
                cost1 = cost0

                for _ in range(8):
                    candidate = pi + step_scale * dx
                    candidate_cost = 0.0

                    for k in range(Ai.shape[0]):
                        ak = Ai[k]
                        rk = ri[k]
                        uk = ui[k]

                        q_new = ak - candidate
                        d_new = float(np.linalg.norm(q_new))
                        d_new = max(d_new, 1e-9)

                        # 距离残差。
                        er_new = d_new - rk
                        candidate_cost += float(er_new * er_new)

                        # LoS横向距离残差。
                        Pk = I3 - np.outer(uk, uk)
                        eu_new = Pk @ q_new
                        candidate_cost += float(eu_new @ eu_new)

                    if candidate_cost <= cost0:
                        p_new = candidate
                        cost1 = candidate_cost
                        accepted = True
                        break

                    step_scale *= 0.5

                if not accepted:
                    # 当前更新方向无法进一步降低目标函数。
                    converged[i] = True
                    continue

                p[i] = p_new.astype(np.float32)

                actual_step_norm = float(np.linalg.norm(p_new - pi))

                # ---------------------------------------------------------
                # 收敛条件1：位置更新足够小
                # ---------------------------------------------------------
                if actual_step_norm < self.p_isac_tol:
                    converged[i] = True
                    continue

                # ---------------------------------------------------------
                # 收敛条件2：目标函数改善足够小
                # ---------------------------------------------------------
                relative_improvement = (
                                               cost0 - cost1
                                       ) / (abs(cost0) + 1e-9)

                if relative_improvement < 1e-3:
                    converged[i] = True
                    continue

                all_done = False

            if all_done:
                break

        return p.astype(np.float32)


    def _build_knn(self, p_ref: np.ndarray) -> np.ndarray:
        N = int(p_ref.shape[0])
        k = min(self.k_neighbors, max(N - 1, 0))
        if k <= 0:
            return np.zeros((N, 0), dtype=np.int64)

        diff = p_ref[:, None, :] - p_ref[None, :, :]
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        np.fill_diagonal(dist, np.inf)

        kth = max(k - 1, 0)
        return np.argpartition(dist, kth=kth, axis=1)[:, :k].astype(np.int64)


    def _compute_neighbor_features(
            self,
            gps: np.ndarray,
            p_isac: np.ndarray,
            dp: np.ndarray,
    ) -> np.ndarray:
        N = int(gps.shape[0])
        nbr_idx = self._build_knn(gps)

        if nbr_idx.shape[1] == 0:
            current_geo_mean = np.zeros((N, 1), dtype=np.float32)
            self._neighbor_geo_mean_hist.append(current_geo_mean.copy())
            return np.zeros((N, 4), dtype=np.float32)

        pos_scale = max(self.pos_scale, 1e-6)

        d_gps = np.linalg.norm(gps[:, None, :] - gps[None, :, :], axis=2)
        d_isac = np.linalg.norm(p_isac[:, None, :] - p_isac[None, :, :], axis=2)

        pair_geo = np.abs(d_gps - d_isac) / pos_scale
        pair_geo = np.clip(pair_geo, 0.0, 10.0).astype(np.float32)
        np.fill_diagonal(pair_geo, 0.0)

        rows = np.arange(N)[:, None]

        dp_med = np.median(dp[nbr_idx], axis=1).astype(np.float32)
        nbr_rel_bias_norm = np.linalg.norm(
            dp - dp_med,
            axis=1,
            keepdims=True
        ).astype(np.float32)

        nbr_pair_geo = pair_geo[rows, nbr_idx]
        nbr_pair_geo_mean = np.mean(
            nbr_pair_geo,
            axis=1,
            keepdims=True
        ).astype(np.float32)
        nbr_pair_geo_max = np.max(
            nbr_pair_geo,
            axis=1,
            keepdims=True
        ).astype(np.float32)

        if len(self._neighbor_geo_mean_hist) == 0:
            nbr_pair_geo_drift = np.zeros_like(nbr_pair_geo_mean, dtype=np.float32)
        else:
            geo_ref = self._neighbor_geo_mean_hist[0]
            nbr_pair_geo_drift = np.abs(nbr_pair_geo_mean - geo_ref).astype(np.float32)

        self._neighbor_geo_mean_hist.append(nbr_pair_geo_mean.copy())

        return np.concatenate(
            [
                nbr_rel_bias_norm,
                nbr_pair_geo_mean,
                nbr_pair_geo_max,
                nbr_pair_geo_drift,
            ],
            axis=1,
        ).astype(np.float32)




    # ========================================================
    # Gym API
    # ========================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            _ = self.base.reset(seed=seed, options=options)
        else:
            _ = self.base.reset(options={"goal": [100, 100, 50]})

        self._residual_hist.clear()
        self._neighbor_geo_mean_hist.clear()
        self._residual_ewma = None
        self._residual_cusum = None

        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()

        self._p_isac_prev = None
        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception:
                    self.atk_scenario = None

        self._y_prevprev[:] = 0
        self._detected_in_event[:] = False
        self._event_t0[:] = 0
        self._delay_prev_y[:] = 0
        self._delay_event_open[:] = False
        self._delay_event_detected[:] = False
        self._delay_event_t0[:] = 0

        self._attack_info_current = self._apply_attack_writeback()
        self._y_current = self._as_attack_mask(self._attack_info_current)

        x0, info0 = self._build_obs_and_info()
        info0 = dict(info0)
        info0.update(self._attack_info_current)
        info0["note"] = "reset"
        return x0, info0

    def step(self, a_det):
        if isinstance(a_det, dict):
            # print("a:",a)
            a = a_det.get("a", None)
            if a is None:
                for k in ("action", "actions", "act"):
                    if k in a_det:
                        a = a_det[k]
                        break
            if a is None:
                raise ValueError("No action found in dict")
        else:
            a = a_det
        # print("a:",a_det)
        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] == 2:
            a = a.argmax(axis=1).astype(np.int32)
        elif a.ndim == 1:
            a = a.astype(np.int32)
        else:
            raise ValueError(f"Expected action shape (N,) or (N,2), got {a.shape}")
        a = a.reshape(self.N)

        attack_info_current = dict(self._attack_info_current) if isinstance(self._attack_info_current, dict) else {}
        y_current = self._y_current.copy()
        delay_info = self._update_detection_delay_info(a, y_current, attack_info_current)

        r = self._compute_detection_reward(a, y_current)

        if callable(self.nav_stepper):
            base_action = self.nav_stepper(self.unwrapped_env)
        else:
            base_action = np.zeros((self.N, 2), dtype=np.float32)

        step_ret = self.unwrapped_env.step(base_action)
        if isinstance(step_ret, (list, tuple)) and len(step_ret) == 5:
            _, _, terminated, truncated, info_base = step_ret
        elif isinstance(step_ret, (list, tuple)) and len(step_ret) == 4:
            _, _, done, info_base = step_ret
            terminated, truncated = bool(done), False
        else:
            raise RuntimeError("Unexpected base env step return format")

        self._attack_info_current = self._apply_attack_writeback()
        self._y_current = self._as_attack_mask(self._attack_info_current)

        x_next, info_next = self._build_obs_and_info()
        # print("isattack:",y_current)
        info = {}
        if isinstance(attack_info_current, dict):
            info.update(attack_info_current)
        if isinstance(info_base, dict):
            info.update(info_base)
        if isinstance(info_next, dict):
            info.update(info_next)
        info.update(delay_info)
        # print("a:",a)
        tp = ((a == 1) & (y_current == 1))
        fp = ((a == 1) & (y_current == 0))
        fn = ((a == 0) & (y_current == 1))
        tn = ((a == 0) & (y_current == 0))
        # print("tp,fp,fn,tn:",tp.sum(),fp.sum(),fn.sum(),tn.sum())
        info["det_tp_fp_fn_tn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum()))

        info["attack_mask_next"] = self._attack_info_current.get("attack_mask", np.zeros((self.N,), dtype=bool))
        info["attack_active_next"] = self._attack_info_current.get("attack_active", False)
        info["attack_t_next"] = self._attack_info_current.get("attack_t", None)
        return x_next, r.astype(np.float32), bool(terminated), bool(truncated), info

    def _update_detection_delay_info(self, a: np.ndarray, y: np.ndarray, info: dict) -> Dict[str, Any]:
        """Track per-agent attack events and first-detection delay for logging only."""
        a = np.asarray(a, dtype=np.int32).reshape(self.N)
        y = np.asarray(y, dtype=np.int32).reshape(self.N)
        current_t = int(info.get("attack_t", getattr(self.unwrapped_env, "t", 0)))

        rise = (self._delay_prev_y == 0) & (y == 1)
        fall = (self._delay_prev_y == 1) & (y == 0)

        if np.any(rise):
            t0_info = int(info.get("attack_t0_t1", (current_t, current_t))[0])
            # If the scenario t0 is stale or unavailable, fall back to the observed rising edge.
            t0 = t0_info if t0_info <= current_t else current_t
            self._delay_event_open[rise] = True
            self._delay_event_detected[rise] = False
            self._delay_event_t0[rise] = t0

        first_hit = self._delay_event_open & (~self._delay_event_detected) & (y == 1) & (a == 1)
        hit_agents = np.where(first_hit)[0]
        delay_steps = []
        delay_seconds = []
        if hit_agents.size > 0:
            delays = np.maximum(0, current_t - self._delay_event_t0[hit_agents]).astype(np.int32)
            delay_steps = [int(x) for x in delays]
            delay_seconds = [float(x * self.dt) for x in delays]
            self._delay_event_detected[hit_agents] = True

        missed = fall & self._delay_event_open & (~self._delay_event_detected)
        missed_count = int(missed.sum())

        if np.any(fall):
            self._delay_event_open[fall] = False
            self._delay_event_detected[fall] = False

        self._delay_prev_y = y.copy()

        return {
            "det_event_started": int(rise.sum()),
            "det_event_detected": int(hit_agents.size),
            "det_event_missed": missed_count,
            "det_event_open": int(self._delay_event_open.sum()),
            "det_delay_steps": delay_steps,
            "det_delay_seconds": delay_seconds,
            "det_delay_mean_steps": float(np.mean(delay_steps)) if delay_steps else float("nan"),
            "det_delay_mean_seconds": float(np.mean(delay_seconds)) if delay_seconds else float("nan"),
        }



    # ========================================================
    # Observation builder
    # ========================================================
    def _build_obs_and_info(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        gps = self._get_gps()

        true_p = np.asarray(getattr(self.unwrapped_env, "true_p", gps), dtype=np.float32)
        h = self._get_heading_now().astype(np.float32)
        v_model = (self.v0 * h).astype(np.float32)

        isac = self.isac_gen.measure(true_p=true_p, h=h, v0=self.v0)
        r_isac = isac["r"]
        u_isac = isac["u"]
        d_isac = isac["dop"]
        mask = isac["mask"]
        rho = isac["rho"]
        rho_db = isac["rho_db"]
        sigma_r2 = isac["sigma_r2"]
        sigma_d2 = isac["sigma_d2"]
        sigma_u2 = isac["sigma_u2"]
        anchors = isac["anchors"]

        # GPS-derived range, direction, and radial velocity.
        dvec_gps = anchors[None, :, :] - gps[:, None, :]
        r_gps = np.linalg.norm(dvec_gps, axis=-1).astype(np.float32)
        u_gps = _unit(dvec_gps).astype(np.float32)
        d_gps = np.sum(v_model[:, None, :] * u_gps, axis=-1).astype(np.float32)

        dr = (r_isac - r_gps).astype(np.float32)
        dd = (d_isac - d_gps).astype(np.float32)
        dir_err = np.clip(1.0 - _cos_sim(u_isac, u_gps), 0.0, 2.0).astype(np.float32)

        residual_link = np.stack(
            [
                np.abs(dr),
                np.abs(dd),
                dir_err,
            ],
            axis=2,
        ).astype(np.float32)

        valid = (mask > 0.5).astype(np.float32)
        valid3 = valid[:, :, None]
        valid_count = np.maximum(valid.sum(axis=1, keepdims=True), 1.0)
        residual_mean = (
            np.sum(residual_link * valid3, axis=1) / valid_count
        ).astype(np.float32)
        residual_max = np.max(
            np.where(valid3 > 0.0, residual_link, 0.0),
            axis=1,
        ).astype(np.float32)

        if len(self._residual_hist) >= 1:
            residual_change = (residual_mean - self._residual_hist[-1]).astype(np.float32)
        else:
            residual_change = np.zeros_like(residual_mean, dtype=np.float32)

        if len(self._residual_hist) >= 2:
            residual_accel = (
                residual_mean
                - 2.0 * self._residual_hist[-1]
                + self._residual_hist[-2]
            ).astype(np.float32)
        else:
            residual_accel = np.zeros_like(residual_mean, dtype=np.float32)

        self._residual_hist.append(residual_mean.copy())
        residual_window = np.stack(tuple(self._residual_hist), axis=0).astype(np.float32)
        residual_window_std = np.std(residual_window, axis=0).astype(np.float32)
        residual_window_drift = np.abs(residual_mean - residual_window[0]).astype(np.float32)

        if self._residual_ewma is None or self._residual_ewma.shape != residual_mean.shape:
            residual_ema_error = np.zeros_like(residual_mean, dtype=np.float32)
            self._residual_ewma = residual_mean.copy()
            self._residual_cusum = np.zeros_like(residual_mean, dtype=np.float32)
        else:
            residual_ema_error = (residual_mean - self._residual_ewma).astype(np.float32)
            if self._residual_cusum is None or self._residual_cusum.shape != residual_mean.shape:
                self._residual_cusum = np.zeros_like(residual_mean, dtype=np.float32)
            self._residual_cusum = (
                self._residual_cusum + residual_ema_error
            ).astype(np.float32)
            self._residual_ewma = (
                self.cusum_beta * self._residual_ewma
                + (1.0 - self.cusum_beta) * residual_mean
            ).astype(np.float32)

        residual_cusum = self._residual_cusum.astype(np.float32, copy=True)

        x_cons21 = np.concatenate(
            [
                residual_mean,
                residual_max,
                residual_change,
                residual_accel,
                residual_window_std,
                residual_window_drift,
                residual_cusum,
            ],
            axis=1,
        ).astype(np.float32)

        if self._p_isac_prev is None:
            self._p_isac_prev = self._init_p_isac_from_ranges(
                anchors=anchors,
                r=r_isac,
                mask=mask,
            )

        p_isac = self._estimate_p_from_range_dir(
            anchors=anchors,
            r=r_isac,
            u=u_isac,
            mask=mask,
            p_init=self._p_isac_prev,
        )
        self._p_isac_prev = p_isac.copy()

        dp = (gps - p_isac) / max(self.pos_scale, 1e-6)
        dp_norm = np.linalg.norm(dp, axis=1, keepdims=True).astype(np.float32)
        dp_proj = np.sum(dp * h, axis=1, keepdims=True).astype(np.float32)
        dp_proj = np.clip(dp_proj, -5.0, 5.0).astype(np.float32)
        x_self2 = np.concatenate([dp_norm, dp_proj], axis=1).astype(np.float32)

        neighbor_features = self._compute_neighbor_features(
            gps=gps,
            p_isac=p_isac,
            dp=dp,
        )

        x = np.concatenate([x_cons21, x_self2, neighbor_features], axis=1).astype(np.float32)
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        info = {
            "isac_M": int(self.M),
            "residual_link": residual_link.copy(),
            "residual_mean": residual_mean.copy(),
            "residual_max": residual_max.copy(),
            "residual_change": residual_change.copy(),
            "residual_change_rate": residual_accel.copy(),
            "residual_window_std": residual_window_std.copy(),
            "residual_window_drift": residual_window_drift.copy(),
            "residual_ema_error": residual_ema_error.copy(),
            "residual_cusum": residual_cusum.copy(),
            "residual_dr": dr.copy(),
            "residual_dd": dd.copy(),
            "residual_dir_err": dir_err.copy(),
            "residual_mask": mask.astype(bool, copy=True),
            "pair_geo_mean": neighbor_features[:, 1:2].copy(),
            "pair_geo_max": neighbor_features[:, 2:3].copy(),
            "pair_geo_drift": neighbor_features[:, 3:4].copy(),
            "isac_residual_mean_norm": float(np.linalg.norm(residual_mean, axis=1).mean()),
            "residual_cusum_mean": float(residual_cusum.mean()),
            "isac_rho_mean": float(rho.mean()),
            "isac_rho_db_mean": float(rho_db.mean()),
            "isac_sigma_r2_mean": float(sigma_r2.mean()),
            "isac_sigma_d2_mean": float(sigma_d2.mean()),
            "isac_sigma_u2_mean": float(sigma_u2.mean()),
            "p_isac_valid_ratio": float((mask.sum(axis=1) >= self.min_anchors_for_p_isac).mean()),
            "obs_mode": "features_txt_27d",
        }
        return x, info

    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Simplified detection reward.

        Reward structure:
            R = R_cls + R_early

        where:
            R_cls:
                TP step reward      -> self.stay_reward
                TN step reward      -> self.R_TN
                FP step penalty     -> self.R_FP
                FN step penalty     -> self.fn_step_penalty

            R_early:
                First detection bonus in each attack event
                -> self.R_TP with delay decay
        """

        a = np.asarray(a, dtype=np.int32).reshape(self.N)
        y = np.asarray(y, dtype=np.int32).reshape(self.N)

        R = np.zeros((self.N,), dtype=np.float32)

        current_t = int(getattr(self.unwrapped_env, "t", 0))

        # ===== 1) Detect attack event transitions =====
        rise_true = (self._y_prevprev == 0) & (y == 1)
        fall_true = (self._y_prevprev == 1) & (y == 0)

        # ===== 2) Initialize newly opened attack events =====
        if np.any(rise_true):
            self._detected_in_event[rise_true] = False
            self._event_t0[rise_true] = current_t

        # ===== 3) Step-level classification reward =====
        tp = (y == 1) & (a == 1)
        tn = (y == 0) & (a == 0)
        fp = (y == 0) & (a == 1)
        fn = (y == 1) & (a == 0)

        if np.any(tp):
            R[tp] += self.stay_reward

        if np.any(tn):
            R[tn] += self.R_TN

        if np.any(fp):
            R[fp] -= self.R_FP

        if np.any(fn):
            R[fn] -= self.fn_step_penalty

        # ===== 4) Event-level first-detection bonus =====
        first_hit = (y == 1) & (a == 1) & (~self._detected_in_event)

        if np.any(first_hit):
            elapsed_steps = np.maximum(
                0,
                current_t - self._event_t0[first_hit]
            )

            Td_steps = max(
                1,
                int(round(self.delay_full_seconds / max(self.dt, 1e-9)))
            )

            delay_weight = np.exp(
                -self.delay_k
                * np.clip(elapsed_steps, 0, Td_steps)
                / float(Td_steps)
            ).astype(np.float32)

            R[first_hit] += self.R_TP * delay_weight

            self._detected_in_event[first_hit] = True

        # ===== 5) Close finished attack events =====
        if np.any(fall_true):
            self._detected_in_event[fall_true] = False

        # ===== 6) Update previous states =====
        self._y_prevprev = y.copy()

        return R.astype(np.float32)

    # def _init_p_isac_from_ranges(
    #         self,
    #         anchors: np.ndarray,
    #         r: np.ndarray,
    #         mask: np.ndarray,
    #         sigma_r2: np.ndarray,
    # ) -> np.ndarray:
    #     N, M = r.shape
    #     A = anchors.astype(np.float32)
    #     p0 = np.zeros((N, 3), dtype=np.float32)
    #
    #     global_center = np.mean(A, axis=0).astype(np.float32)
    #
    #     for i in range(N):
    #         idx = np.where(mask[i] > 0.5)[0]
    #         if idx.size < self.min_anchors_for_p_isac:
    #             p0[i] = global_center
    #             continue
    #
    #         # 选一个参考锚点：用测距最小的那个更稳一点
    #         ref = idx[np.argmin(r[i, idx])]
    #         others = idx[idx != ref]
    #
    #         # 3D 至少需要 3 条独立方程
    #         if others.size < 3:
    #             p0[i] = np.mean(A[idx], axis=0).astype(np.float32)
    #             continue
    #
    #         Aref = A[ref]
    #         rref = float(r[i, ref])
    #
    #         H_list = []
    #         b_list = []
    #         w_list = []
    #
    #         for k in others:
    #             Ak = A[k]
    #             rk = float(r[i, k])
    #
    #             # 2 (Ak - Aref)^T p = rref^2 - rk^2 + ||Ak||^2 - ||Aref||^2
    #             H_list.append(2.0 * (Ak - Aref))
    #             b_list.append(
    #                 rref * rref - rk * rk
    #                 + float(np.dot(Ak, Ak) - np.dot(Aref, Aref))
    #             )
    #
    #             # 简单加权
    #             wk = 1.0 / float(sigma_r2[i, k] + sigma_r2[i, ref] + 1e-6)
    #             w_list.append(wk)
    #
    #         H = np.asarray(H_list, dtype=np.float32)  # (K,3)
    #         b = np.asarray(b_list, dtype=np.float32)  # (K,)
    #         W = np.diag(np.asarray(w_list, dtype=np.float32))
    #
    #         try:
    #             lhs = H.T @ W @ H + 1e-3 * np.eye(3, dtype=np.float32)
    #             rhs = H.T @ W @ b
    #             p0[i] = np.linalg.solve(lhs, rhs).astype(np.float32)
    #         except np.linalg.LinAlgError:
    #             p0[i] = np.mean(A[idx], axis=0).astype(np.float32)
    #
    #     return p0
    def _init_p_isac_from_ranges(
            self,
            anchors: np.ndarray,
            r: np.ndarray,
            mask: np.ndarray,
    ) -> np.ndarray:
        """
        Obtain an initial 3D position estimate using unweighted
        linearized multilateration.

        Parameters
        ----------
        anchors : ndarray, shape (M, 3)
            Anchor/ISAC-BS positions.
        r : ndarray, shape (N, M)
            Measured ranges from N UAVs to M anchors.
        mask : ndarray, shape (N, M)
            Validity mask of the range measurements.

        Returns
        -------
        p0 : ndarray, shape (N, 3)
            Initial position estimates.
        """
        N, M = r.shape
        A = anchors.astype(np.float32)
        p0 = np.zeros((N, 3), dtype=np.float32)

        global_center = np.mean(A, axis=0).astype(np.float32)

        for i in range(N):
            idx = np.where(mask[i] > 0.5)[0]

            # At least four anchors are required to form
            # three linear equations in 3D.
            if idx.size < self.min_anchors_for_p_isac:
                p0[i] = global_center
                continue

            # Use the nearest measured anchor as the reference.
            ref = idx[np.argmin(r[i, idx])]
            others = idx[idx != ref]

            if others.size < 3:
                p0[i] = np.mean(A[idx], axis=0).astype(np.float32)
                continue

            A_ref = A[ref]
            r_ref = float(r[i, ref])

            H_list = []
            b_list = []

            for k in others:
                A_k = A[k]
                r_k = float(r[i, k])

                H_list.append(2.0 * (A_k - A_ref))
                b_list.append(
                    r_ref ** 2
                    - r_k ** 2
                    + float(np.dot(A_k, A_k) - np.dot(A_ref, A_ref))
                )

            H = np.asarray(H_list, dtype=np.float32)
            b = np.asarray(b_list, dtype=np.float32)

            try:
                regularization = 1e-3 * np.eye(3, dtype=np.float32)
                lhs = H.T @ H + regularization
                rhs = H.T @ b

                p0[i] = np.linalg.solve(lhs, rhs).astype(np.float32)

            except np.linalg.LinAlgError:
                p0[i] = np.mean(A[idx], axis=0).astype(np.float32)

        return p0

    # ========================================================
    # Feature names
    # ========================================================
    def get_feature_names(self):
        isac_names = [
            "residual_mean_range",
            "residual_mean_radial_velocity",
            "residual_mean_direction",
            "residual_max_range",
            "residual_max_radial_velocity",
            "residual_max_direction",
            "residual_change_range",
            "residual_change_radial_velocity",
            "residual_change_direction",
            "residual_accel_range",
            "residual_accel_radial_velocity",
            "residual_accel_direction",
            "residual_window_std_range",
            "residual_window_std_radial_velocity",
            "residual_window_std_direction",
            "residual_window_drift_range",
            "residual_window_drift_radial_velocity",
            "residual_window_drift_direction",
            "residual_cusum_range",
            "residual_cusum_radial_velocity",
            "residual_cusum_direction",
        ]

        self_names = [
            "dp_norm",
            "dp_proj",
        ]


        neighbor_names = [
            "nbr_rel_bias_norm",
            "nbr_pair_geo_mean",
            "nbr_pair_geo_max",
            "nbr_pair_geo_drift",
        ]

        return isac_names + self_names + neighbor_names

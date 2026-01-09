#   - ISAC local (12): same slots, but improved stability:
#       * cusum becomes dual-sided (cpos/cneg) still one scalar
#       * score_delta slot uses mix(change, accel) for smoother boundary behavior
#       * score_std slot uses mix(std, |trend|) for robustness
#   - self dp (2): dp_norm, dp_z_score (same)
#   - neighbor (10): optional EMA smoothing (same 10 dims)
#   - formation (4): optional EMA smoothing of median/MAD threshold (same 4 dims)
#
# Reward:
#   - add hold-cost waiver when CUSUM is strong (cusum_waive_threshold)
#   - add hysteresis penalty for turning off when evidence still strong

from __future__ import annotations

from typing import Callable, Optional, Any, Dict, Tuple
from collections import deque
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete

from envs.NavPolicyAdapter import NavPolicyAdapter


# ----------------------------
# Utils
# ----------------------------
def _unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (n + eps)


def _cos_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    denom = na * nb + eps
    dot = np.sum(a * b, axis=-1)
    return np.clip(dot / denom, -1.0, 1.0)


# ----------------------------
# ISAC measurement generator (keeps Stage1 unchanged)
# ----------------------------
class ISACGenerator:
    """
    Stage2 internal ISAC measurement:
      - range r:   (N,M)
      - dir   u:   (N,M,3)  unit vector from UAV -> anchor
      - doppler d: (N,M)    radial speed approx dot(v, u)
      - mask:      (N,M)    1 valid / 0 dropout
    """

    def __init__(
        self,
        anchors: np.ndarray,  # (M,3)
        sigma_r: float = 0.3,
        sigma_dir: float = 0.02,
        sigma_dop: float = 0.2,
        p_dropout: float = 0.05,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.anchors = np.asarray(anchors, dtype=np.float32)
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 3:
            raise ValueError("anchors must be (M,3)")
        self.M = int(self.anchors.shape[0])
        self.sigma_r = float(sigma_r)
        self.sigma_dir = float(sigma_dir)
        self.sigma_dop = float(sigma_dop)
        self.p_dropout = float(p_dropout)
        self.rng = np.random.RandomState(0) if rng is None else rng

    def measure(self, true_p: np.ndarray, h: np.ndarray, v0: float) -> Dict[str, np.ndarray]:
        true_p = np.asarray(true_p, dtype=np.float32)  # (N,3)
        h = np.asarray(h, dtype=np.float32)            # (N,3)
        N = true_p.shape[0]
        A = self.anchors[None, :, :]   # (1,M,3)
        P = true_p[:, None, :]         # (N,1,3)

        dvec = A - P                   # (N,M,3)
        r = np.linalg.norm(dvec, axis=-1)  # (N,M)
        u = _unit(dvec)                    # (N,M,3)

        v = (float(v0) * h)[:, None, :]    # (N,1,3)
        dop = np.sum(v * u, axis=-1)       # (N,M)

        # noise
        r_meas = r + self.rng.normal(0.0, self.sigma_r, size=r.shape).astype(np.float32)
        u_meas = _unit(u + self.rng.normal(0.0, self.sigma_dir, size=u.shape).astype(np.float32))
        dop_meas = dop + self.rng.normal(0.0, self.sigma_dop, size=dop.shape).astype(np.float32)

        # dropout (kept as all-ones for stability; you can turn on if needed)
        # mask = (self.rng.rand(N, self.M) > self.p_dropout).astype(np.float32)
        mask = np.ones((N, self.M), dtype=np.float32)

        r_meas *= mask
        u_meas *= mask[:, :, None]
        dop_meas *= mask

        return {"r": r_meas, "u": u_meas, "dop": dop_meas, "mask": mask, "anchors": self.anchors}


# ----------------------------
# Main environment (28-dim enhanced)
# ----------------------------
class Stage2ISACDetectEnv(gym.Env):
    """
    28-dim ISAC-only spoofing detector with collaborative+formation features, enhanced stability.

    Obs per agent (fixed at 28):
      - 12 local ISAC consistency features
      - 2 self dp features
      - 10 neighbor features (collab)
      - 4 formation features
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        gps_getter: Optional[Callable] = None,
        anchors: Optional[np.ndarray] = None,
        sigma_r: float = 0.3,
        sigma_dir: float = 0.02,
        sigma_dop: float = 0.2,
        p_dropout: float = 0.05,
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = True,
        resample_attack_on_reset: bool = True,
        range_scale: float = 50.0,
        doppler_scale: float = 10.0,
        pos_scale: float = 20.0,
        k_neighbors: int = 3,

        # history for stable temporal stats (still only uses 2 slots: score_delta/score_std)
        history_window: int = 5,

        # Dual CUSUM on local score (still outputs 1 slot: cusum)
        cusum_threshold: float = 0.15,
        cusum_drift: float = 0.01,
        cusum_max: float = 1.0,
        cusum_beta: float = 0.99,
        cusum_waive_threshold: float = 0.12,   # normalized evidence threshold for hold-cost waiver

        # rewards
        reward_tp: float = 50.0,
        reward_fp: float = 2.0,
        reward_fn: float = 30.0,
        reward_tn: float = 1.0,
        fn_step_penalty: float = 0.5,
        stay_reward: float = 2.0,
        flip_penalty: float = 0.01,
        delay_full_seconds: float = 2.0,
        hold_cost: float = 1.5,
        neg_hold_grace_steps: int = 3,
        fp_cooldown_steps: int = 3,
        delay_k: float = 0.08,
        hysteresis_penalty: float = 0.2,

        # smoothing (no dim change)
        nbr_ema_beta: float = 0.90,
        form_ema_beta: float = 0.90,

        # p_isac solver
        p_isac_iters: int = 100,
        p_isac_lam: float = 1e-3,
        p_isac_step_clip: float = 5.0,
        min_anchors_for_p_isac: int = 4,

        # Stage1 stepper
        nav_stepper: NavPolicyAdapter = None,
    ):
        print("ISAC")
        super().__init__()
        self.base = base_env
        self.nav_stepper = nav_stepper

        # unwrap to base env
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

        self.gps_getter = gps_getter

        if anchors is None:
            anchors = np.array([[80, 80, 0], [80, -80, 0], [-80, 80, 0], [-80, -80, 0]], dtype=np.float32)

        self.isac_gen = ISACGenerator(
            anchors=anchors,
            sigma_r=sigma_r,
            sigma_dir=sigma_dir,
            sigma_dop=sigma_dop,
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

        # scales
        self.range_scale = float(range_scale)
        self.doppler_scale = float(doppler_scale)
        self.pos_scale = float(pos_scale)

        # neighbors
        self.k_neighbors = min(int(k_neighbors), max(self.N - 1, 1))

        # history window
        self.history_window = int(history_window)
        self._score_hist = deque(maxlen=self.history_window)

        # Dual CUSUM states
        self.cusum_threshold = float(cusum_threshold)
        self.cusum_drift = float(cusum_drift)
        self.cusum_max = float(cusum_max)
        self.cusum_beta = float(cusum_beta)
        self.cusum_waive_threshold = float(cusum_waive_threshold)
        self._ema_mu = None
        self._cpos = None
        self._cneg = None
        self._cusum = None

        # smoothing
        self.nbr_ema_beta = float(nbr_ema_beta)
        self._nbr_ema = None
        self.form_ema_beta = float(form_ema_beta)
        self._form_med = None
        self._form_mad = None

        # rewards
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_FN = float(reward_fn)
        self.R_TN = float(reward_tn)
        self.stay_reward = float(stay_reward)
        self.flip_penalty = float(flip_penalty)
        self.delay_full_seconds = float(delay_full_seconds)
        self.delay_k = float(delay_k)
        self.hold_cost = float(hold_cost)
        self.neg_hold_grace_steps = int(neg_hold_grace_steps)
        self.fp_cooldown_steps = int(fp_cooldown_steps)
        self.hysteresis_penalty = float(hysteresis_penalty)
        self.fn_step_penalty = float(fn_step_penalty)

        # p_isac solver params
        self.p_isac_iters = int(p_isac_iters)
        self.p_isac_lam = float(p_isac_lam)
        self.p_isac_step_clip = float(p_isac_step_clip)
        self.min_anchors_for_p_isac = int(min_anchors_for_p_isac)
        self._p_isac_prev = None

        # history
        self._prev_gps = None

        # event states
        self._a_prev = np.zeros((self.N,), np.int32)
        self._y_prevprev = np.zeros(self.N, dtype=np.int32)
        self._event_open = np.zeros(self.N, dtype=bool)
        self._detected_in_event = np.zeros(self.N, dtype=bool)
        self._event_t0 = np.zeros(self.N, dtype=np.int32)
        self._neg_hold_counter = np.zeros(self.N, dtype=np.int32)
        self._last_fp_step = np.full(self.N, -10**9, dtype=np.int32)

        # spaces (fixed 28)
        self.action_space = MultiDiscrete(np.array([2] * self.N, dtype=np.int64))
        self.feature_names = self.get_feature_names()
        self.obs_dim = len(self.feature_names)
        # assert self.obs_dim == 28, f"Expected obs_dim=28, got {self.obs_dim}"
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.N, self.obs_dim), dtype=np.float32)
        self._attack_info_current: Dict[str, Any] = {}
        self._y_current = np.zeros(self.N, dtype=np.int32)
    # -----------------------------
    # IO helpers
    # -----------------------------
    def _get_gps(self) -> np.ndarray:
        if callable(self.gps_getter):
            return np.asarray(self.gps_getter(self.unwrapped_env), dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            return np.asarray(self.unwrapped_env.gps, dtype=np.float32)
        if hasattr(self.unwrapped_env, "p"):
            return np.asarray(self.unwrapped_env.p, dtype=np.float32)
        raise RuntimeError("Cannot get GPS: provide gps_getter or expose .gps/.p on base_env")

    def _as_attack_mask(self, info: dict) -> np.ndarray:
        mask = info.get("attack_mask", None)
        if mask is None:
            return np.zeros(self.N, dtype=np.int32)
        return np.asarray(mask, dtype=bool).astype(np.int32)
    def _set_gps(self, z: np.ndarray) -> None:
        """Write spoofed GPS measurement back into env (gps preferred, else p)."""
        z = np.asarray(z, dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            setattr(self.unwrapped_env, "gps", z)
        else:
            setattr(self.unwrapped_env, "p", z)

    def _get_heading_now(self) -> np.ndarray:
        """Robust heading fetch (avoid crash if h_hist missing)."""
        if hasattr(self.unwrapped_env, "h_hist"):
            hh = getattr(self.unwrapped_env, "h_hist")
            if isinstance(hh, (list, tuple)) and len(hh) > 0:
                return np.asarray(hh[-1], dtype=np.float32)
        if hasattr(self.unwrapped_env, "h"):
            return np.asarray(getattr(self.unwrapped_env, "h"), dtype=np.float32)

        # fallback
        h = np.zeros((self.N, 3), dtype=np.float32)
        h[:, 0] = 1.0
        return h

    def _apply_attack_writeback(self) -> Dict[str, Any]:
        """
        Apply attack to CURRENT env time t and write spoofed GPS back to env.p/gps.

        ✅ IMPORTANT:
        - 这个函数应当在“对外返回 obs 之前”调用，这样：
        1) obs_t 使用的是 attacked p/gps
        2) base_env.step() 也会用 attacked p/gps 去推进下一步的 p/gps（符合你描述的 base_env 机制）
        """
        if not (self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None)):
            try:
                setattr(self.unwrapped_env, "_atk_active_mask", np.zeros(self.N, dtype=bool))
            except Exception:
                pass
            return {}

        t = int(getattr(self.unwrapped_env, "t", 0))

        try:
            # current measurement (this is the state used by base_env to propagate p/gps)
            p_meas = self._get_gps()

            # true position (you confirmed it exists and is NOT modified by attack)
            true_pos = np.asarray(getattr(self.unwrapped_env, "true_p", p_meas), dtype=np.float32)
            heading_now = self._get_heading_now()

            # update histories for attacker
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

            # write spoofed measurement back
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
                setattr(self.unwrapped_env, "_atk_active_mask", np.zeros(self.N, dtype=bool))
            except Exception:
                pass
            return {}

    # -----------------------------
    # ISAC range -> p_isac estimation
    # -----------------------------
    def _estimate_p_from_ranges(
            self,
            anchors: np.ndarray,
            r: np.ndarray,
            mask: np.ndarray,
            p_init: np.ndarray,
    ) -> np.ndarray:
        N, M = r.shape
        p = p_init.astype(np.float32).copy()
        A = anchors.astype(np.float32)

        # 记录每个点是否收敛（收敛后就不再更新它）
        converged = np.zeros(N, dtype=bool)

        for _ in range(self.p_isac_iters):
            all_done = True  # 如果这一轮所有点都收敛，则提前结束

            for i in range(N):
                if converged[i]:
                    continue

                mi = mask[i].astype(bool)
                if int(mi.sum()) < self.min_anchors_for_p_isac:
                    # 锚点不足，直接视为无法优化
                    converged[i] = True
                    continue

                Ai = A[mi]  # (K,3)
                ri = r[i, mi]  # (K,)
                pi = p[i]  # (3,)

                # ---------- cost before update ----------
                diff = pi[None, :] - Ai
                di = np.linalg.norm(diff, axis=1) + 1e-9
                res = di - ri
                cost0 = float(np.dot(res, res))  # sum(res^2)

                # ---------- GN/LM step ----------
                J = diff / di[:, None]  # (K,3)
                H = (J.T @ J).astype(np.float32) + self.p_isac_lam * np.eye(3, dtype=np.float32)
                g = (J.T @ res).astype(np.float32)

                try:
                    dx = -np.linalg.solve(H, g).astype(np.float32)
                except np.linalg.LinAlgError:
                    # 这次求解失败，跳过该点本轮更新
                    all_done = False
                    continue

                n = float(np.linalg.norm(dx))
                if n > self.p_isac_step_clip:
                    dx *= (self.p_isac_step_clip / (n + 1e-9))

                p_new = pi + dx

                # ---------- cost after update ----------
                diff2 = p_new[None, :] - Ai
                di2 = np.linalg.norm(diff2, axis=1) + 1e-9
                res2 = di2 - ri
                cost1 = float(np.dot(res2, res2))

                # 接受更新
                p[i] = p_new

                # ---------- Early stopping conditions ----------
                # 1) step small
                if n < 0.1:
                    converged[i] = True
                    continue

                # 2) cost improvement small (relative)
                # relative improvement = |cost0 - cost1| / (cost0 + eps)
                rel_improve = abs(cost0 - cost1) / (cost0 + 1e-9)
                if rel_improve < 0.1:
                    converged[i] = True
                    continue

                # 如果走到这里，说明该点还没收敛
                all_done = False

            # 所有点都收敛了，外层也提前结束
            if all_done:
                break

        return p

    # -----------------------------
    # Neighbor & formation features
    # -----------------------------
    def _compute_neighbor_features(self, p_isac: np.ndarray, dp: np.ndarray, dp_norm: np.ndarray) -> np.ndarray:
        N = self.N
        k = self.k_neighbors
        if k <= 0:
            return np.zeros((N, 10), dtype=np.float32)

        diff = p_isac[:, None, :] - p_isac[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2).astype(np.float32)
        np.fill_diagonal(dist_matrix, np.inf)
        neighbor_indices = np.argpartition(dist_matrix, k, axis=1)[:, :k]  # (N,k)

        nbr_dp_all = dp[neighbor_indices]                         # (N,k,3)
        nbr_dp_norm_all = dp_norm[neighbor_indices].squeeze(-1)   # (N,k)

        nbr_mean_dp = np.mean(nbr_dp_all, axis=1)  # (N,3)
        nbr_mean_dp_norm = np.mean(nbr_dp_norm_all, axis=1, keepdims=True)  # (N,1)

        if k > 1:
            nbr_std_dp = np.std(nbr_dp_all, axis=1)  # (N,3)
        else:
            nbr_std_dp = np.zeros((N, 3), dtype=np.float32)

        own_norm = np.linalg.norm(dp, axis=1)
        nbr_norm_val = np.linalg.norm(nbr_mean_dp, axis=1)
        denom = own_norm * nbr_norm_val + 1e-9
        dot_prod = np.sum(dp * nbr_mean_dp, axis=1)
        consistency = np.clip(dot_prod / denom, -1.0, 1.0).reshape(N, 1).astype(np.float32)
        invalid = (own_norm < 1e-6) | (nbr_norm_val < 1e-6)
        consistency[invalid] = 1.0

        med = float(np.median(dp_norm))
        mad = float(np.median(np.abs(dp_norm - med)))
        global_threshold = med + 3.0 * max(mad, 1e-6)

        nbr_anomaly_ratio = np.mean((nbr_dp_norm_all > global_threshold).astype(np.float32), axis=1, keepdims=True)

        if k > 1:
            nbr_norms_indiv = np.linalg.norm(nbr_dp_all, axis=2, keepdims=True)  # (N,k,1)
            valid = nbr_norms_indiv > 1e-6
            nbr_dirs = np.divide(nbr_dp_all, nbr_norms_indiv, where=valid, out=np.zeros_like(nbr_dp_all))
            direction_variance = np.sum(np.var(nbr_dirs, axis=1), axis=1, keepdims=True).astype(np.float32)
        else:
            direction_variance = np.zeros((N, 1), dtype=np.float32)

        feats = np.concatenate(
            [nbr_mean_dp, nbr_std_dp, nbr_mean_dp_norm, consistency, nbr_anomaly_ratio, direction_variance],
            axis=1
        ).astype(np.float32)

        # EMA smoothing (no dim change)
        if self._nbr_ema is None:
            self._nbr_ema = np.zeros_like(feats)
        self._nbr_ema = self.nbr_ema_beta * self._nbr_ema + (1.0 - self.nbr_ema_beta) * feats
        return self._nbr_ema.astype(np.float32)

    def _compute_formation_features(self, dp_norm: np.ndarray) -> np.ndarray:
        N = self.N
        x = dp_norm.flatten().astype(np.float32)

        mean_x = float(np.mean(x))
        std_x = float(np.std(x))
        std_x = max(std_x, 1e-6)

        # EMA-smoothed median/MAD threshold (no dim change)
        med_now = float(np.median(x))
        mad_now = float(np.median(np.abs(x - med_now)))
        mad_now = max(mad_now, 1e-6)

        if self._form_med is None:
            self._form_med = med_now
            self._form_mad = mad_now
        else:
            self._form_med = self.form_ema_beta * self._form_med + (1.0 - self.form_ema_beta) * med_now
            self._form_mad = self.form_ema_beta * self._form_mad + (1.0 - self.form_ema_beta) * mad_now

        thr = float(self._form_med + 3.0 * self._form_mad)

        z_score = np.clip((dp_norm - mean_x) / std_x, -10.0, 10.0).astype(np.float32)  # (N,1)

        anomaly_ratio = float(np.mean((x > thr).astype(np.float32)))
        anomaly_ratio_vec = np.full((N, 1), anomaly_ratio, dtype=np.float32)

        if N > 2 and std_x > 1e-6:
            skew_val = float(np.mean(((x - mean_x) / std_x) ** 3))
            skew_val = float(np.clip(skew_val, -10.0, 10.0))
        else:
            skew_val = 0.0
        skewness_vec = np.full((N, 1), skew_val, dtype=np.float32)

        ranks = np.argsort(np.argsort(x))
        percentile_rank = (ranks / max(N, 1)).reshape(N, 1).astype(np.float32)

        return np.concatenate([z_score, anomaly_ratio_vec, skewness_vec, percentile_rank], axis=1).astype(np.float32)

    # -----------------------------
    # Gym API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            _ = self.base.reset(seed=seed, options=options)
        else:
            _ = self.base.reset(options={"goal":[100,100,50]})
        print("start_point:",self.unwrapped_env.c)
        print("end_point:",self.unwrapped_env.g)
        # init buffers
        gps0 = self._get_gps().copy()
        print("gps0:",gps0)
        self._prev_gps = gps0.copy()
        self._prev_v_gps = np.zeros_like(gps0, dtype=np.float32)

        self._score_hist.clear()

        # Dual-CUSUM init
        self._ema_mu = np.zeros((self.N, 1), dtype=np.float32)
        self._cpos = np.zeros((self.N, 1), dtype=np.float32)
        self._cneg = np.zeros((self.N, 1), dtype=np.float32)
        self._cusum = np.zeros((self.N, 1), dtype=np.float32)

        # smoothing buffers
        self._nbr_ema = np.zeros((self.N, 10), dtype=np.float32)
        self._form_med = None
        self._form_mad = None

        # histories
        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()

        self._p_isac_prev = gps0.copy()

        # sample attack scenario
        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception:
                    self.atk_scenario = None

        # event states reset
        self._a_prev[:] = 0
        self._y_prevprev[:] = 0
        self._event_open[:] = False
        self._detected_in_event[:] = False
        self._event_t0[:] = 0
        self._neg_hold_counter[:] = 0
        self._last_fp_step[:] = -10**9

        # ✅ IMPORTANT: apply attack NOW (t=0) BEFORE returning obs
        self._attack_info_current = self._apply_attack_writeback()
        self._y_current = self._as_attack_mask(self._attack_info_current)

        # re-init prev buffers using attacked measurement
        gps_att = self._get_gps().copy()
        self._prev_gps = gps_att.copy()
        self._prev_v_gps = np.zeros_like(gps_att, dtype=np.float32)
        self._p_isac_prev = gps_att.copy()

        x0, info0 = self._build_obs_and_info(self._attack_info_current)

        info0 = dict(info0)
        info0.update(self._attack_info_current)
        info0["note"] = "reset"
        return x0, info0


    def step(self, a_det):
        # parse action (same as your original)
        if isinstance(a_det, dict):
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

        a = np.asarray(a, dtype=np.float32)
        if a.ndim == 2 and a.shape[1] == 2:
            a = a.argmax(axis=1).astype(np.int32)
        elif a.ndim == 1:
            a = a.astype(np.int32)
        else:
            raise ValueError(f"Expected action shape (N,) or (N,2), got {a.shape}")
        a = a.reshape(self.N)

        # ✅ CURRENT attack info/label (aligned with current observation)
        attack_info_current = dict(self._attack_info_current) if isinstance(self._attack_info_current, dict) else {}
        y_current = self._y_current.copy()

        # reward at time t
        r = self._compute_detection_reward(a, y_current, attack_info_current)

        # ✅ optional: per-step FN penalty (helps step-wise recall)
        if getattr(self, "fn_step_penalty", 0.0) > 0:
            fn_step = (y_current == 1) & (a == 0)
            if np.any(fn_step):
                r[fn_step] -= float(self.fn_step_penalty)

        # base env step t -> t+1 (base_env will use current attacked p/gps as you described)
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

        # ✅ After stepping to new time, apply attack for NEW time (t+1) before returning next obs
        attack_info_next = self._apply_attack_writeback()
        self._attack_info_current = attack_info_next
        self._y_current = self._as_attack_mask(attack_info_next)

        x_next, info_next = self._build_obs_and_info(attack_info_next)

        # info pack
        info = {}
        if isinstance(attack_info_current, dict):
            info.update(attack_info_current)  # label used for current reward/metrics
        if isinstance(info_base, dict):
            info.update(info_base)
        if isinstance(info_next, dict):
            info.update(info_next)

        # metrics for current decision
        tp = ((a == 1) & (y_current == 1))
        fp = ((a == 1) & (y_current == 0))
        fn = ((a == 0) & (y_current == 1))
        tn = ((a == 0) & (y_current == 0))
        info["det_tp_fp_fn_tn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum()))

        # (debug) next label
        info["attack_mask_next"] = attack_info_next.get("attack_mask", np.zeros(self.N, dtype=bool))
        info["attack_active_next"] = attack_info_next.get("attack_active", False)
        info["attack_t_next"] = attack_info_next.get("attack_t", None)

        return x_next, r.astype(np.float32), bool(terminated), bool(truncated), info


    # -----------------------------
    # Observation builder (28-dim)
    # -----------------------------
    def _build_obs_and_info(self, attack_info_current: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        gps = self._get_gps()
        v_gps = (gps - self._prev_gps) / max(self.dt, 1e-6)

        true_p = np.asarray(getattr(self.unwrapped_env, "true_p", gps), dtype=np.float32)
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)

        isac = self.isac_gen.measure(true_p=true_p, h=h, v0=self.v0)
        r_isac = isac["r"]
        u_isac = isac["u"]
        d_isac = isac["dop"]
        mask = isac["mask"]
        anchors = isac["anchors"]

        # GPS derived geometry
        A = anchors[None, :, :]
        P = gps[:, None, :]
        dvec_gps = A - P
        r_gps = np.linalg.norm(dvec_gps, axis=-1)
        u_gps = _unit(dvec_gps)
        d_gps = np.sum(v_gps[:, None, :] * u_gps, axis=-1)

        # residuals
        dr = (r_isac - r_gps) / max(self.range_scale, 1e-6)
        dd = (d_isac - d_gps) / max(self.doppler_scale, 1e-6)
        cos_ud = _cos_sim(u_isac, u_gps)
        dir_err = 1.0 - cos_ud

        dr *= mask
        dd *= mask
        dir_err *= mask

        mcount = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
        abs_dr = np.abs(dr)
        abs_dd = np.abs(dd)

        f_dr_mean = abs_dr.sum(axis=1, keepdims=True) / mcount
        f_dr_max = abs_dr.max(axis=1, keepdims=True)
        f_dd_mean = abs_dd.sum(axis=1, keepdims=True) / mcount
        f_dd_max = abs_dd.max(axis=1, keepdims=True)
        f_dir_mean = dir_err.sum(axis=1, keepdims=True) / mcount
        f_dir_max = dir_err.max(axis=1, keepdims=True)
        meas_ratio = mask.mean(axis=1, keepdims=True)
        mask_min = mask.min(axis=1, keepdims=True)

        # local score (same as v1)
        # score = f_dr_mean + 0.5 * f_dd_mean + 0.5 * f_dir_mean  # (N,1)
        score = 0.5 * f_dr_mean + 0.25 * f_dd_mean + 0.25 * f_dir_mean + 0.25 * (f_dr_max + f_dd_max + f_dir_max) / 3.0

        # --- stable temporal stats but keep 2 slots only (score_delta, score_std) ---
        self._score_hist.append(score.copy())

        if len(self._score_hist) >= 2:
            change = self._score_hist[-1] - self._score_hist[-2]
        else:
            change = np.zeros_like(score)

        if len(self._score_hist) >= 3:
            accel = self._score_hist[-1] - 2 * self._score_hist[-2] + self._score_hist[-3]
            hist = np.concatenate(list(self._score_hist), axis=1)  # (N,W)
            std = np.std(hist, axis=1, keepdims=True)
            trend = hist[:, -1:] - hist[:, :1]
        else:
            accel = np.zeros_like(score)
            std = np.zeros_like(score)
            trend = np.zeros_like(score)

        # pack into the same two slots:
        # - score_delta slot: mix(change, accel) reduces one-step noise sensitivity
        # - score_std slot: mix(std, |trend|) adds persistence awareness
        score_delta = (0.7 * change + 0.3 * accel).astype(np.float32)
        score_std = (0.7 * std + 0.3 * np.abs(trend)).astype(np.float32)

        # --- Dual-sided CUSUM on score (still output 1 cusum slot) ---
        if self._ema_mu is None:
            self._ema_mu = np.zeros((self.N, 1), dtype=np.float32)
        if self._cpos is None:
            self._cpos = np.zeros((self.N, 1), dtype=np.float32)
        if self._cneg is None:
            self._cneg = np.zeros((self.N, 1), dtype=np.float32)
        if self._cusum is None:
            self._cusum = np.zeros((self.N, 1), dtype=np.float32)

        mu = self._ema_mu = self.cusum_beta * self._ema_mu + (1.0 - self.cusum_beta) * score
        e = score - mu

        thr = self.cusum_threshold
        drift = self.cusum_drift
        cmax = self.cusum_max

        self._cpos = np.clip(self._cpos + (e - thr - drift), 0.0, cmax)
        self._cneg = np.clip(self._cneg + (-e - thr - drift), 0.0, cmax)

        self._cusum = np.maximum(self._cpos, self._cneg)
        cus = self._cusum / max(cmax, 1e-6)

        # 12 local ISAC features (same count/order as v1)
        x_isac12 = np.concatenate(
            [
                f_dr_mean, f_dr_max,
                f_dd_mean, f_dd_max,
                f_dir_mean, f_dir_max,
                meas_ratio,
                score,
                cus,
                score_delta,   # same slot name
                score_std,     # same slot name
                mask_min,
            ],
            axis=1,
        ).astype(np.float32)

        # ---- ISAC range -> p_isac, and dp features (same as v1) ----
        if self._p_isac_prev is None:
            self._p_isac_prev = gps.copy()

        p_isac = self._estimate_p_from_ranges(anchors=anchors, r=r_isac, mask=mask, p_init=self._p_isac_prev)
        self._p_isac_prev = p_isac.copy()

        dp = (gps - p_isac) / max(self.pos_scale, 1e-6)  # (N,3)
        dp_norm = np.linalg.norm(dp, axis=1, keepdims=True)  # (N,1)

        formation_features = self._compute_formation_features(dp_norm)  # (N,4) includes z_score
        # dp_z = formation_features[:, 0:1]  # (N,1)
        dp_proj = np.sum(dp * h, axis=1, keepdims=True).astype(np.float32)  #
        dp_proj = np.clip(dp_proj, -5.0, 5.0).astype(np.float32)
        x_self2 = np.concatenate([dp_norm, dp_proj], axis=1).astype(np.float32)  # (N,2)

        neighbor_features = self._compute_neighbor_features(p_isac=p_isac, dp=dp, dp_norm=dp_norm)  # (N,10)

        # final concat: 12 + 2 + 10 + 4 = 28
        x = np.concatenate([x_isac12, x_self2, neighbor_features, formation_features], axis=1).astype(np.float32)
        # x= np.concatenate([x_isac12,x_self2],axis=1).astype(np.float32)
        # if x.shape[1] != 28:
        #     raise RuntimeError(f"obs_dim mismatch: expected 28, got {x.shape[1]}")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        self._prev_gps = gps.copy()

        info = {
            "isac_M": int(self.M),
            "isac_meas_ratio_mean": float(meas_ratio.mean()),
            "isac_score_mean": float(score.mean()),
            "isac_cusum_mean": float(cus.mean()),
            "p_isac_valid_ratio": float((mask.sum(axis=1) >= self.min_anchors_for_p_isac).mean()),
        }
        return x, info

    # -----------------------------
    # Event-based reward (enhanced but obs unchanged)
    # -----------------------------
    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray, info: dict) -> np.ndarray:
        a = np.asarray(a, dtype=np.int32).reshape(self.N)
        y = np.asarray(y, dtype=np.int32).reshape(self.N)
        r = np.zeros(self.N, dtype=np.float32)

        rise_true = (self._y_prevprev == 0) & (y == 1)
        fall_true = (self._y_prevprev == 1) & (y == 0)
        rise_pred = (self._a_prev == 0) & (a == 1)
        fall_pred = (self._a_prev == 1) & (a == 0)
        current_t = int(getattr(self.unwrapped_env, "t", 0))

        if np.any(rise_true):
            self._event_open[rise_true] = True
            self._detected_in_event[rise_true] = False
            t0_info = info.get("attack_t0_t1", (current_t, current_t))[0]
            self._event_t0[rise_true] = int(t0_info)

            instant_hit = rise_true & (a == 1) & (~self._detected_in_event)
            if np.any(instant_hit):
                r[instant_hit] += self.R_TP
                self._detected_in_event[instant_hit] = True

        hit_mask = rise_pred & (y == 1) & (~self._detected_in_event)
        if np.any(hit_mask):
            elapsed_steps = np.maximum(0, current_t - self._event_t0[hit_mask])
            elapsed_sec = elapsed_steps * self.dt
            delay_weight = np.exp(
                -self.delay_k * np.clip(elapsed_sec, 0.0, self.delay_full_seconds) / max(self.delay_full_seconds, 1e-6)
            )
            r[hit_mask] += (self.R_TP * delay_weight).astype(np.float32)
            self._detected_in_event[hit_mask] = True

        # FP cooldown on rising edge
        fp_edge = rise_pred & (y == 0)
        if np.any(fp_edge):
            cooldown_ok = (current_t - self._last_fp_step[fp_edge]) >= self.fp_cooldown_steps
            idx = np.where(fp_edge)[0]
            penalize_idx = idx[cooldown_ok]
            if penalize_idx.size > 0:
                r[penalize_idx] -= self.R_FP
                self._last_fp_step[penalize_idx] = current_t

        # negative hold cost with grace + CUSUM waiver
        hold_neg = (y == 0) & (a == 1) & (~rise_pred)
        self._neg_hold_counter[hold_neg] += 1
        self._neg_hold_counter[~hold_neg] = 0

        cusum_flat = np.asarray(self._cusum).reshape(-1) if self._cusum is not None else np.zeros((self.N,), np.float32)
        waive = (self._neg_hold_counter <= self.neg_hold_grace_steps) | (cusum_flat >= self.cusum_waive_threshold)
        apply_cost = hold_neg & (~waive)
        if np.any(apply_cost):
            r[apply_cost] -= self.hold_cost

        # stay reward
        stay = (y == 1) & (a == 1)
        if np.any(stay):
            r[stay] += self.stay_reward

        # TN baseline
        tn = (y == 0) & (a == 0)
        if np.any(tn):
            r[tn] += self.R_TN

        # event ends without detection -> FN
        miss_mask = fall_true & self._event_open & (~self._detected_in_event)
        if np.any(miss_mask):
            r[miss_mask] -= self.R_FN

        if np.any(fall_true):
            self._event_open[fall_true] = False
            self._detected_in_event[fall_true] = False

        # hysteresis: discourage turning OFF when evidence still strong
        if self.hysteresis_penalty > 0 and np.any(fall_pred):
            strong = cusum_flat >= self.cusum_waive_threshold
            idx = np.where(fall_pred & strong)[0]
            if idx.size > 0:
                r[idx] -= self.hysteresis_penalty

        # flip penalty
        if self.flip_penalty > 0:
            r -= self.flip_penalty * (a != self._a_prev).astype(np.float32)

        self._a_prev = a.copy()
        self._y_prevprev = y.copy()
        return r

    # -----------------------------
    # Feature names (28)
    # -----------------------------
    def get_feature_names(self):
        isac_names = [
            "abs_dr_mean", "abs_dr_max",
            "abs_dd_mean", "abs_dd_max",
            "dir_err_mean", "dir_err_max",
            "meas_ratio",
            "score",
            "cusum",
            "score_delta",
            "score_std",
            "mask_min",
        ]
        self_names = [
            "dp_norm",
            "dp_proj",
        ]
        neighbor_names = [
            "nbr_mean_dp_x", "nbr_mean_dp_y", "nbr_mean_dp_z",
            "nbr_std_dp_x", "nbr_std_dp_y", "nbr_std_dp_z",
            "nbr_mean_dp_norm",
            "consistency",
            "nbr_anomaly_ratio",
            "direction_variance",
        ]
        formation_names = [
            "z_score",
            "anomaly_ratio",
            "skewness",
            "percentile_rank",
        ]
        # return isac_names + self_names
        return isac_names + self_names + neighbor_names + formation_names

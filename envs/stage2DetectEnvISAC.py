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
    dot = np.sum(a * b,e axis=-1)
    return np.clip(dot / (na * nb + eps), -1.0, 1.0)


# ============================================================
# ISAC measurement generator
# SNR is computed from a link-budget model:
#   Pn(dBm)   = -174 + 10 log10(B) + NF
#   PL(dB)    = a + 10 alpha log10(d)
#   Pr(dBm)   = Pt + Gt + Gr - PL - Lmisc
#   SNR(dB)   = Pr - Pn
#   rho       = 10^(SNR(dB)/10)
#
# Measurement uncertainty:
#   sigma_r^2 = kappa_r / rho
#   sigma_d^2 = kappa_d / rho
#   sigma_u^2 = kappa_u / rho
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
        anchors: np.ndarray,                   # (M,3)

        # ----- OFDM / link-budget related -----
        fc_hz: float = 3.5e9,
        bw_hz: float = 40e6,

        pt_dbm: float = 27.0,
        gt_dbi: float = 0.0,
        gr_dbi: float = 6.0,
        nf_db: float = 7.0,
        misc_loss_db: float = 5.0,

        # ----- path loss -----
        pl_a_db: float = 30.0,
        pl_alpha_los: float = 2.0,
        pl_alpha_nlos: float = 3.75,
        use_nlos: bool = False,

        # ----- SNR clipping / validity -----
        rho_clip_db_min: float = -10.0,
        rho_clip_db_max: float = 40.0,
        rho_valid_threshold_db: float = 0.0,

        # ----- fixed kappas -----
        kappa_r: float = 8.55,
        kappa_d: float = 4.45,
        kappa_u: float = 0.00563,

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
        self.gt_dbi = float(gt_dbi)
        self.gr_dbi = float(gr_dbi)
        self.nf_db = float(nf_db)
        # self.misc_loss_db = float(misc_loss_db)

        # path loss
        self.pl_a_db = float(pl_a_db)
        self.pl_alpha_los = float(pl_alpha_los)
        self.pl_alpha_nlos = float(pl_alpha_nlos)
        self.use_nlos = bool(use_nlos)

        # SNR clipping / validity
        self.rho_clip_db_min = float(rho_clip_db_min)
        self.rho_clip_db_max = float(rho_clip_db_max)
        self.rho_valid_threshold_db = float(rho_valid_threshold_db)

        # fixed kappas
        self.kappa_r = float(kappa_r)
        self.kappa_d = float(kappa_d)
        self.kappa_u = float(kappa_u)

        self.p_dropout = float(p_dropout)
        self.rng = np.random.RandomState(0) if rng is None else rng

    def _pathloss_db(self, dist: np.ndarray) -> np.ndarray:
        d = np.maximum(dist, 1.0)
        alpha = self.pl_alpha_nlos if self.use_nlos else self.pl_alpha_los
        return self.pl_a_db + 10.0 * alpha * np.log10(d)

    def _compute_rho(self, true_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          rho: linear SNR
          rho_db: dB SNR
        """
        A = self.anchors[None, :, :]      # (1,M,3)
        P = true_p[:, None, :]            # (N,1,3)
        dvec = A - P
        dist = np.linalg.norm(dvec, axis=-1) + 1e-6

        # Path loss
        pl_db = self._pathloss_db(dist)

        # Noise power: Pn = -174 + 10log10(B) + NF
        noise_dbm = -174.0 + 10.0 * np.log10(self.bw_hz) + self.nf_db

        # Received power
        # pr_dbm = self.pt_dbm + self.gt_dbi + self.gr_dbi - pl_db - self.misc_loss_db
        pr_dbm = self.pt_dbm + self.gt_dbi + self.gr_dbi - pl_db
        # SNR
        rho_db = pr_dbm - noise_dbm
        rho_db = np.clip(rho_db, self.rho_clip_db_min, self.rho_clip_db_max)
        rho = np.power(10.0, rho_db / 10.0).astype(np.float32)

        return rho.astype(np.float32), rho_db.astype(np.float32)

    def measure(self, true_p: np.ndarray, h: np.ndarray, v0: float) -> Dict[str, np.ndarray]:
        true_p = np.asarray(true_p, dtype=np.float32)   # (N,3)
        h = np.asarray(h, dtype=np.float32)             # (N,3)
        N = int(true_p.shape[0])

        A = self.anchors[None, :, :]    # (1,M,3)
        P = true_p[:, None, :]          # (N,1,3)
        dvec = A - P                    # (N,M,3)

        r_true = np.linalg.norm(dvec, axis=-1)          # (N,M)
        u_true = _unit(dvec)                            # (N,M,3)
        v = (float(v0) * h)[:, None, :]                 # (N,1,3)
        dop_true = np.sum(v * u_true, axis=-1)          # (N,M)

        # SNR model
        rho, rho_db = self._compute_rho(true_p)         # (N,M)

        # CRLB-like uncertainty without floors
        sigma_r2 = (self.kappa_r / np.maximum(rho, 1e-9)).astype(np.float32)
        sigma_d2 = (self.kappa_d / np.maximum(rho, 1e-9)).astype(np.float32)
        sigma_u2 = (self.kappa_u / np.maximum(rho, 1e-9)).astype(np.float32)

        sigma_r = np.sqrt(sigma_r2).astype(np.float32)
        sigma_d = np.sqrt(sigma_d2).astype(np.float32)
        sigma_u = np.sqrt(sigma_u2).astype(np.float32)

        # Range noise
        r_meas = r_true + self.rng.normal(0.0, 1.0, size=r_true.shape).astype(np.float32) * sigma_r

        # Doppler noise
        dop_meas = dop_true + self.rng.normal(0.0, 1.0, size=dop_true.shape).astype(np.float32) * sigma_d

        # Direction noise: tangent-plane perturbation
        raw_noise = self.rng.normal(0.0, 1.0, size=u_true.shape).astype(np.float32)
        proj = np.sum(raw_noise * u_true, axis=-1, keepdims=True) * u_true
        tangent_noise = raw_noise - proj
        tangent_noise = tangent_noise * sigma_u[:, :, None]
        u_meas = _unit(u_true + tangent_noise)

        # Valid anchor set
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
    28-dim observation:
      - local ISAC features: 12
      - self features: 2
      - neighbor features: 10
      - formation features: 4
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        gps_getter: Optional[Callable] = None,
        anchors: Optional[np.ndarray] = None,

        # OFDM-ISAC / link budget
        fc_hz: float = 3.5e9,
        bw_hz: float = 40e6,

        pt_dbm: float = 27.0,
        gt_dbi: float = 0.0,
        gr_dbi: float = 6.0,
        nf_db: float = 7.0,
        misc_loss_db: float = 5.0,

        pl_a_db: float = 30.0,
        pl_alpha_los: float = 2.0,
        pl_alpha_nlos: float = 3.75,
        use_nlos: bool = False,

        rho_clip_db_min: float = -10.0,
        rho_clip_db_max: float = 40.0,
        rho_valid_threshold_db: float = 0.0,

        kappa_r: float = 8.55,
        kappa_d: float = 4.45,
        kappa_u: float = 0.00563,
        p_dropout: float = 0.0,

        # attack
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = True,
        resample_attack_on_reset: bool = True,

        # normalization
        range_scale_floor: float = 0.001,
        doppler_scale_floor: float = 0.091,
        pos_scale: float = 5.0,
        k_neighbors: int = 3,

        # temporal stats
        # history_window: int = 5,
        # cusum_drift: float = 0.01,
        # cusum_max: float = 1.0,
        # cusum_beta: float = 0.99,
        # cusum_waive_threshold: float = 0.3,   # normalized cusum threshold for hold-cost waiver / hysteresis
        # score_delta_mix: float = 0.7,
        # score_std_mix: float = 0.7,
        # history_window = 10,
        # cusum_drift = 0.03,
        # cusum_beta = 0.97,
        # cusum_max = 1.0,
        # cusum_waive_threshold = 0.5,
        # score_delta_mix = 0.5,
        # score_std_mix = 0.5,
        history_window = 20,
        cusum_drift = 0.06,
        cusum_beta = 0.95,
        cusum_max = 1.0,
        cusum_waive_threshold = 0.72,
        score_delta_mix = 0.5,
        score_std_mix = 0.5,

        # reward
        reward_tp = 75.0,
        reward_fp = 30.0,
        reward_fn = 25.0,
        reward_tn = 0.8,
        stay_reward = 0.3,
        flip_penalty = 0.06,
        delay_full_seconds = 2.0,
        fn_step_penalty: float = 0.25,
        #run11
        delay_k:float=1.0,
        # delay_k: float = 0.08,
        # hold_cost=1.5
        hold_cost: float = 4.0,
        #neg_hold_grace_stps:3
        # neg_hold_grace_steps: int = 2,
        # fp_cooldown_steps: int = 3,
        neg_hold_grace_steps = 1,
        fp_cooldown_steps = 3,
        hysteresis_penalty = 0.3,

        # smoothing
        nbr_ema_beta: float = 0.90,
        form_ema_beta: float = 0.90,

        # WLS position proxy
        p_isac_iters: int = 100,
        p_isac_lam: float = 1e-3,
        p_isac_step_clip: float = 5.0,
        p_isac_tol: float = 0.01,
        min_anchors_for_p_isac: int = 4,

        # Stage1 stepper
        nav_stepper: Optional[NavPolicyAdapter] = None,
    ):
        super().__init__()

        self.base = base_env
        self.nav_stepper = nav_stepper
        self.gps_getter = gps_getter

        # unwrap
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
                [[80, 80, 0], [80, -80, 0], [-80, 80, 0], [-80, -80, 0]],
                dtype=np.float32
            )

        self.isac_gen = ISACGenerator(
            anchors=anchors,

            fc_hz=fc_hz,
            bw_hz=bw_hz,

            pt_dbm=pt_dbm,
            gt_dbi=gt_dbi,
            gr_dbi=gr_dbi,
            nf_db=nf_db,
            misc_loss_db=misc_loss_db,

            pl_a_db=pl_a_db,
            pl_alpha_los=pl_alpha_los,
            pl_alpha_nlos=pl_alpha_nlos,
            use_nlos=use_nlos,

            rho_clip_db_min=rho_clip_db_min,
            rho_clip_db_max=rho_clip_db_max,
            rho_valid_threshold_db=rho_valid_threshold_db,

            kappa_r=kappa_r,
            kappa_d=kappa_d,
            kappa_u=kappa_u,

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
        self.range_scale_floor = float(range_scale_floor)
        self.doppler_scale_floor = float(doppler_scale_floor)
        self.pos_scale = float(pos_scale)

        # graph
        self.k_neighbors = min(int(k_neighbors), max(self.N - 1, 1))

        # temporal stats
        self.history_window = int(history_window)
        self._score_hist = deque(maxlen=self.history_window)

        self.cusum_drift = float(cusum_drift)
        self.cusum_max = float(cusum_max)
        self.cusum_beta = float(cusum_beta)
        self.cusum_waive_threshold = float(cusum_waive_threshold)
        self.score_delta_mix = float(score_delta_mix)
        self.score_std_mix = float(score_std_mix)

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

        # reward
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_FN = float(reward_fn)
        self.R_TN = float(reward_tn)
        self.fn_step_penalty = float(fn_step_penalty)
        self.stay_reward = float(stay_reward)
        self.flip_penalty = float(flip_penalty)
        self.delay_full_seconds = float(delay_full_seconds)
        self.delay_k = float(delay_k)
        self.hold_cost = float(hold_cost)
        self.neg_hold_grace_steps = int(neg_hold_grace_steps)
        self.fp_cooldown_steps = int(fp_cooldown_steps)
        self.hysteresis_penalty = float(hysteresis_penalty)

        # position proxy
        self.p_isac_iters = int(p_isac_iters)
        self.p_isac_lam = float(p_isac_lam)
        self.p_isac_step_clip = float(p_isac_step_clip)
        self.p_isac_tol = float(p_isac_tol)
        self.min_anchors_for_p_isac = int(min_anchors_for_p_isac)
        self._p_isac_prev = None

        # history / internal states
        self._prev_gps = None

        self._a_prev = np.zeros((self.N,), np.int32)
        self._y_prevprev = np.zeros((self.N,), dtype=np.int32)
        self._event_open = np.zeros((self.N,), dtype=bool)
        self._detected_in_event = np.zeros((self.N,), dtype=bool)
        self._event_t0 = np.zeros((self.N,), dtype=np.int32)
        self._neg_hold_counter = np.zeros((self.N,), dtype=np.int32)
        self._last_fp_step = np.full((self.N,), -10**9, dtype=np.int32)

        # spaces
        self.action_space = MultiDiscrete(np.array([2] * self.N, dtype=np.int64))
        self.feature_names = self.get_feature_names()
        self.obs_dim = len(self.feature_names)
        assert self.obs_dim == 28, f"Expected obs_dim=28, got {self.obs_dim}"
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

    # ========================================================
    # Range-based weighted least-squares position proxy
    # ========================================================
    def _estimate_p_from_ranges(
        self,
        anchors: np.ndarray,
        r: np.ndarray,
        mask: np.ndarray,
        sigma_r2: np.ndarray,
        p_init: np.ndarray,
    ) -> np.ndarray:
        N, M = r.shape
        p = p_init.astype(np.float32).copy()
        A = anchors.astype(np.float32)

        converged = np.zeros((N,), dtype=bool)

        for _ in range(self.p_isac_iters):
            all_done = True

            for i in range(N):
                if converged[i]:
                    continue

                mi = mask[i].astype(bool)
                if int(mi.sum()) < self.min_anchors_for_p_isac:
                    converged[i] = True
                    continue

                Ai = A[mi]                           # (K,3)
                ri = r[i, mi].astype(np.float32)     # (K,)
                wi = 1.0 / np.maximum(sigma_r2[i, mi].astype(np.float32), 1e-6)  # (K,)
                pi = p[i].astype(np.float32)         # (3,)

                diff = pi[None, :] - Ai              # (K,3)
                di = np.linalg.norm(diff, axis=1) + 1e-9
                res = di - ri                        # (K,)
                J = diff / di[:, None]               # (K,3)

                WJ = wi[:, None] * J                 # (K,3)
                H = (J.T @ WJ).astype(np.float32) + self.p_isac_lam * np.eye(3, dtype=np.float32)
                g = (J.T @ (wi * res)).astype(np.float32)

                try:
                    dx = -np.linalg.solve(H, g).astype(np.float32)
                except np.linalg.LinAlgError:
                    all_done = False
                    continue

                n = float(np.linalg.norm(dx))
                if n > self.p_isac_step_clip:
                    dx *= self.p_isac_step_clip / (n + 1e-9)

                p_new = pi + dx

                # evaluate weighted cost decrease
                diff2 = p_new[None, :] - Ai
                di2 = np.linalg.norm(diff2, axis=1) + 1e-9
                res2 = di2 - ri

                cost0 = float(np.dot(wi * res, res))
                cost1 = float(np.dot(wi * res2, res2))

                p[i] = p_new

                if n < self.p_isac_tol:
                    converged[i] = True
                    continue

                rel_improve = abs(cost0 - cost1) / (abs(cost0) + 1e-9)
                if rel_improve < 1e-4:
                    converged[i] = True
                    continue

                all_done = False

            if all_done:
                break

        return p.astype(np.float32)

    # ========================================================
    # Neighbor & formation features
    # ========================================================
    def _compute_neighbor_features(
        self,
        p_isac: np.ndarray,
        dp: np.ndarray,
        dp_norm: np.ndarray
    ) -> np.ndarray:
        N = self.N
        k = self.k_neighbors
        if k <= 0:
            return np.zeros((N, 10), dtype=np.float32)

        diff = p_isac[:, None, :] - p_isac[None, :, :]
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        np.fill_diagonal(dist, np.inf)

        neighbor_indices = np.argpartition(dist, kth=k, axis=1)[:, :k]  # (N,k)

        nbr_dp = dp[neighbor_indices]                                  # (N,k,3)
        nbr_dp_norm = dp_norm[neighbor_indices].squeeze(-1)            # (N,k)

        mu_d = np.mean(nbr_dp, axis=1).astype(np.float32)              # (N,3)
        sigma_d = np.std(nbr_dp, axis=1).astype(np.float32)            # (N,3)
        mean_norm = np.mean(nbr_dp_norm, axis=1, keepdims=True).astype(np.float32)

        own_norm = np.linalg.norm(dp, axis=1)
        nbr_norm = np.linalg.norm(mu_d, axis=1)
        denom = own_norm * nbr_norm + 1e-9
        rho_spatial = (np.sum(dp * mu_d, axis=1) / denom).reshape(N, 1).astype(np.float32)
        invalid = (own_norm < 1e-6) | (nbr_norm < 1e-6)
        rho_spatial[invalid] = 1.0

        # local outlier threshold tau_loc: use robust formation threshold from dp_norm
        x = dp_norm.flatten()
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        tau_loc = med + 3.0 * max(mad, 1e-6)

        r_anom = np.mean((nbr_dp_norm > tau_loc).astype(np.float32), axis=1, keepdims=True)

        if k > 1:
            nbr_vec_norm = np.linalg.norm(nbr_dp, axis=2, keepdims=True)
            nbr_dir = np.divide(nbr_dp, nbr_vec_norm + 1e-9)
            v_dir = np.sum(np.var(nbr_dir, axis=1), axis=1, keepdims=True).astype(np.float32)
        else:
            v_dir = np.zeros((N, 1), dtype=np.float32)

        feats = np.concatenate(
            [mu_d, sigma_d, mean_norm, rho_spatial, r_anom, v_dir],
            axis=1
        ).astype(np.float32)  # (N,10)

        if self._nbr_ema is None:
            self._nbr_ema = np.zeros_like(feats)
        self._nbr_ema = self.nbr_ema_beta * self._nbr_ema + (1.0 - self.nbr_ema_beta) * feats
        return self._nbr_ema.astype(np.float32)

    def _compute_formation_features(self, dp_norm: np.ndarray) -> np.ndarray:
        N = self.N
        x = dp_norm.flatten().astype(np.float32)

        mu = float(np.mean(x))
        sigma = max(float(np.std(x)), 1e-6)

        med_now = float(np.median(x))
        mad_now = float(np.median(np.abs(x - med_now)))
        mad_now = max(mad_now, 1e-6)

        if self._form_med is None:
            self._form_med = med_now
            self._form_mad = mad_now
        else:
            self._form_med = self.form_ema_beta * self._form_med + (1.0 - self.form_ema_beta) * med_now
            self._form_mad = self.form_ema_beta * self._form_mad + (1.0 - self.form_ema_beta) * mad_now

        tau = float(self._form_med + 3.0 * self._form_mad)

        z = np.clip((x - mu) / sigma, -10.0, 10.0).reshape(N, 1).astype(np.float32)
        r_frm = float(np.mean((x > tau).astype(np.float32)))
        r_frm_vec = np.full((N, 1), r_frm, dtype=np.float32)

        if N > 2:
            kappa = float(np.mean(((x - mu) / sigma) ** 3))
            kappa = float(np.clip(kappa, -10.0, 10.0))
        else:
            kappa = 0.0
        kappa_vec = np.full((N, 1), kappa, dtype=np.float32)

        # q_i = 1/(N-1) sum_{j!=i} I(x_j <= x_i)
        q = np.zeros((N, 1), dtype=np.float32)
        if N > 1:
            all_idx = np.arange(N)
            for i in range(N):
                q[i, 0] = np.sum(x[all_idx != i] <= x[i]) / float(N - 1)

        return np.concatenate([z, r_frm_vec, kappa_vec, q], axis=1).astype(np.float32)

    # ========================================================
    # Gym API
    # ========================================================
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            _ = self.base.reset(seed=seed, options=options)
        else:
            _ = self.base.reset(options={"goal": [100, 100, 50]})

        gps0 = self._get_gps().copy()
        self._prev_gps = gps0.copy()

        self._score_hist.clear()
        self._ema_mu = np.zeros((self.N, 1), dtype=np.float32)
        self._cpos = np.zeros((self.N, 1), dtype=np.float32)
        self._cneg = np.zeros((self.N, 1), dtype=np.float32)
        self._cusum = np.zeros((self.N, 1), dtype=np.float32)

        self._nbr_ema = np.zeros((self.N, 10), dtype=np.float32)
        self._form_med = None
        self._form_mad = None

        self._hist_true_pos.clear()
        self._hist_gps_pos.clear()

        self._p_isac_prev = gps0.copy()

        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception:
                    self.atk_scenario = None

        self._a_prev[:] = 0
        self._y_prevprev[:] = 0
        self._event_open[:] = False
        self._detected_in_event[:] = False
        self._event_t0[:] = 0
        self._neg_hold_counter[:] = 0
        self._last_fp_step[:] = -10**9

        self._attack_info_current = self._apply_attack_writeback()
        self._y_current = self._as_attack_mask(self._attack_info_current)

        gps_att = self._get_gps().copy()
        self._prev_gps = gps_att.copy()
        self._p_isac_prev = gps_att.copy()

        x0, info0 = self._build_obs_and_info(self._attack_info_current)
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

        r = self._compute_detection_reward(a, y_current, attack_info_current)

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

        x_next, info_next = self._build_obs_and_info(self._attack_info_current)
        # print("isattack:",y_current)
        info = {}
        if isinstance(attack_info_current, dict):
            info.update(attack_info_current)
        if isinstance(info_base, dict):
            info.update(info_base)
        if isinstance(info_next, dict):
            info.update(info_next)
        # print("a:",a)
        tp = ((a == 1) & (y_current == 1))
        fp = ((a == 1) & (y_current == 0))
        fn = ((a == 0) & (y_current == 1))
        tn = ((a == 0) & (y_current == 0))
        info["det_tp_fp_fn_tn"] = (int(tp.sum()), int(fp.sum()), int(fn.sum()), int(tn.sum()))

        info["attack_mask_next"] = self._attack_info_current.get("attack_mask", np.zeros((self.N,), dtype=bool))
        info["attack_active_next"] = self._attack_info_current.get("attack_active", False)
        info["attack_t_next"] = self._attack_info_current.get("attack_t", None)
        return x_next, r.astype(np.float32), bool(terminated), bool(truncated), info

    # ========================================================
    # Observation builder
    # ========================================================
    def _build_obs_and_info(self, attack_info_current: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        gps = self._get_gps()  # \hat{p}_t

        true_p = np.asarray(getattr(self.unwrapped_env, "true_p", gps), dtype=np.float32)
        h = self._get_heading_now().astype(np.float32)
        v_model = (self.v0 * h).astype(np.float32)  # trusted heading-based velocity model

        isac = self.isac_gen.measure(true_p=true_p, h=h, v0=self.v0)
        r_isac = isac["r"]               # (N,M)
        u_isac = isac["u"]               # (N,M,3)
        d_isac = isac["dop"]             # (N,M)
        mask = isac["mask"]              # (N,M)
        rho = isac["rho"]                # (N,M)
        rho_db = isac["rho_db"]          # (N,M)
        sigma_r2 = isac["sigma_r2"]      # (N,M)
        sigma_d2 = isac["sigma_d2"]      # (N,M)
        sigma_u2 = isac["sigma_u2"]      # (N,M)
        anchors = isac["anchors"]        # (M,3)

        # GPS-derived range / direction / Doppler
        A = anchors[None, :, :]
        P = gps[:, None, :]
        dvec_gps = A - P

        r_gps = np.linalg.norm(dvec_gps, axis=-1)                     # (N,M)
        u_gps = _unit(dvec_gps)                                       # (N,M,3)
        d_gps = np.sum(v_model[:, None, :] * u_gps, axis=-1)          # (N,M)

        # Paper-consistent residuals:
        # delta = [ |r-rhat|/s_r, |d-dhat|/s_d, 1-u·uhat ]
        # Here s_r, s_d are implemented using uncertainty-aware scales derived from sigma_r2 / sigma_d2.
        # s_r = np.sqrt(np.maximum(sigma_r2, self.range_scale_floor)).astype(np.float32)
        # s_d = np.sqrt(np.maximum(sigma_d2, self.doppler_scale_floor)).astype(np.float32)

        # dr = (r_isac - r_gps) / s_r
        # dd = (d_isac - d_gps) / s_d
        dr=r_isac-r_gps
        dd=d_isac-d_gps
        dir_err = 1.0 - _cos_sim(u_isac, u_gps)

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

        # Scalar anomaly score S_i^t = w^T [mean residuals, max residuals]
        score = 0.5 * f_dr_mean + 0.25 * f_dd_mean + 0.25 * f_dir_mean + 0.25 * (f_dr_max + f_dd_max + f_dir_max) / 3.0
        # Temporal evidence
        self._score_hist.append(score.copy())

        if len(self._score_hist) >= 2:
            change = self._score_hist[-1] - self._score_hist[-2]
        else:
            change = np.zeros_like(score)

        if len(self._score_hist) >= 3:
            accel = self._score_hist[-1] - 2.0 * self._score_hist[-2] + self._score_hist[-3]
            hist = np.concatenate(list(self._score_hist), axis=1)
            std = np.std(hist, axis=1, keepdims=True).astype(np.float32)
            trend = (hist[:, -1:] - hist[:, :1]).astype(np.float32)
        else:
            accel = np.zeros_like(score)
            std = np.zeros_like(score)
            trend = np.zeros_like(score)

        score_delta = (self.score_delta_mix * change + (1.0 - self.score_delta_mix) * accel).astype(np.float32)
        score_std = (self.score_std_mix * std + (1.0 - self.score_std_mix) * np.abs(trend)).astype(np.float32)

        # Dual-sided CUSUM
        if self._ema_mu is None:
            self._ema_mu = np.zeros((self.N, 1), dtype=np.float32)
        if self._cpos is None:
            self._cpos = np.zeros((self.N, 1), dtype=np.float32)
        if self._cneg is None:
            self._cneg = np.zeros((self.N, 1), dtype=np.float32)
        if self._cusum is None:
            self._cusum = np.zeros((self.N, 1), dtype=np.float32)

        self._ema_mu = self.cusum_beta * self._ema_mu + (1.0 - self.cusum_beta) * score
        e = score - self._ema_mu

        self._cpos = np.clip(self._cpos + e - self.cusum_drift, 0.0, self.cusum_max)
        self._cneg = np.clip(self._cneg - e - self.cusum_drift, 0.0, self.cusum_max)
        self._cusum = np.maximum(self._cpos, self._cneg)
        cus = self._cusum / max(self.cusum_max, 1e-6)

        # 12 local features
        x_isac12 = np.concatenate(
            [
                f_dr_mean, f_dr_max,
                f_dd_mean, f_dd_max,
                f_dir_mean, f_dir_max,
                meas_ratio,
                score,
                cus,
                score_delta,
                score_std,
                mask_min,
            ],
            axis=1,
        ).astype(np.float32)

        # ISAC-based position proxy from weighted least squares on range
        if self._p_isac_prev is None:
            self._p_isac_prev = gps.copy()

        p_isac = self._estimate_p_from_ranges(
            anchors=anchors,
            r=r_isac,
            mask=mask,
            sigma_r2=sigma_r2,
            p_init=self._p_isac_prev,
        )
        self._p_isac_prev = p_isac.copy()

        # self features
        dp = (gps - p_isac) / max(self.pos_scale, 1e-6)           # (N,3)
        # print("gps:",gps)
        # print("p_isac:",p_isac)
        # print("gps-p_isac:",gps-p_isac)
        
        dp_norm = np.linalg.norm(dp, axis=1, keepdims=True)       # (N,1)
        dp_proj = np.sum(dp * h, axis=1, keepdims=True).astype(np.float32)
        dp_proj = np.clip(dp_proj, -5.0, 5.0).astype(np.float32)
        x_self2 = np.concatenate([dp_norm, dp_proj], axis=1).astype(np.float32)

        # neighbor + formation features
        neighbor_features = self._compute_neighbor_features(
            p_isac=p_isac,
            dp=dp,
            dp_norm=dp_norm
        )
        formation_features = self._compute_formation_features(dp_norm)

        x = np.concatenate([x_isac12, x_self2, neighbor_features, formation_features], axis=1).astype(np.float32)

        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        self._prev_gps = gps.copy()

        info = {
            "isac_M": int(self.M),
            "isac_meas_ratio_mean": float(meas_ratio.mean()),
            "isac_score_mean": float(score.mean()),
            "isac_cusum_mean": float(cus.mean()),
            "isac_rho_mean": float(rho.mean()),
            "isac_rho_db_mean": float(rho_db.mean()),
            "isac_sigma_r2_mean": float(sigma_r2.mean()),
            "isac_sigma_d2_mean": float(sigma_d2.mean()),
            "isac_sigma_u2_mean": float(sigma_u2.mean()),
            "p_isac_valid_ratio": float((mask.sum(axis=1) >= self.min_anchors_for_p_isac).mean()),
            "obs_mode": "full_isac_28d_ofdm_isac",
        }
        return x, info

    # ========================================================
    # Event-aware reward
    # ========================================================
    def _compute_detection_reward(self, a: np.ndarray, y: np.ndarray, info: dict) -> np.ndarray:
        a = np.asarray(a, dtype=np.int32).reshape(self.N)
        y = np.asarray(y, dtype=np.int32).reshape(self.N)

        R = np.zeros((self.N,), dtype=np.float32)

        rise_true = (self._y_prevprev == 0) & (y == 1)   # Δy
        fall_true = (self._y_prevprev == 1) & (y == 0)   # ∇y
        rise_pred = (self._a_prev == 0) & (a == 1)       # Δa
        fall_pred = (self._a_prev == 1) & (a == 0)       # ∇a

        current_t = int(getattr(self.unwrapped_env, "t", 0))

        # event start
        if np.any(rise_true):
            self._event_open[rise_true] = True
            self._detected_in_event[rise_true] = False
            t0_info = int(info.get("attack_t0_t1", (current_t, current_t))[0])
            self._event_t0[rise_true] = t0_info

        # first true positive reward (once per event)
        first_hit = (y == 1) & (a == 1) & (~self._detected_in_event)
        if np.any(first_hit):
            elapsed_steps = np.maximum(0, current_t - self._event_t0[first_hit])
            Td_steps = max(1, int(round(self.delay_full_seconds / max(self.dt, 1e-9))))
            delay_weight = np.exp(
                -self.delay_k * np.clip(elapsed_steps, 0, Td_steps) / float(Td_steps)
            ).astype(np.float32)

            R[first_hit] += self.R_TP * delay_weight
            self._detected_in_event[first_hit] = True

        # stay reward
        stay = (y == 1) & (a == 1)
        if np.any(stay):
            R[stay] += self.stay_reward

        # TN reward
        tn = (y == 0) & (a == 0)
        if np.any(tn):
            R[tn] += self.R_TN

        # FP edge penalty with cooldown
        fp_edge = rise_pred & (y == 0)
        if np.any(fp_edge):
            idx = np.where(fp_edge)[0]
            cooldown_ok = (current_t - self._last_fp_step[idx]) >= self.fp_cooldown_steps
            penalize_idx = idx[cooldown_ok]
            if penalize_idx.size > 0:
                R[penalize_idx] -= self.R_FP
                self._last_fp_step[penalize_idx] = current_t

        # false-alarm hold cost after grace, waived by strong CUSUM
        hold_neg = (y == 0) & (a == 1) & (~rise_pred)
        self._neg_hold_counter[hold_neg] += 1
        self._neg_hold_counter[~hold_neg] = 0

        cus_norm = (
            np.asarray(self._cusum).reshape(-1) / max(self.cusum_max, 1e-6)
            if self._cusum is not None
            else np.zeros((self.N,), dtype=np.float32)
        )

        waive = (self._neg_hold_counter <= self.neg_hold_grace_steps) | (cus_norm >= self.cusum_waive_threshold)
        apply_cost = hold_neg & (~waive)
        if np.any(apply_cost):
            R[apply_cost] -= self.hold_cost

        # step miss penalty
        miss = (y == 1) & (a == 0)
        if np.any(miss):
            R[miss] -= self.fn_step_penalty

        # event-level FN penalty
        miss_event = fall_true & self._event_open & (~self._detected_in_event)
        if np.any(miss_event):
            R[miss_event] -= self.R_FN

        # event close
        if np.any(fall_true):
            self._event_open[fall_true] = False
            self._detected_in_event[fall_true] = False

        # hysteresis penalty
        if self.hysteresis_penalty > 0.0:
            bad_off = fall_pred & (cus_norm >= self.cusum_waive_threshold)
            if np.any(bad_off):
                R[bad_off] -= self.hysteresis_penalty

        # flip penalty
        if self.flip_penalty > 0.0:
            R -= self.flip_penalty * (a != self._a_prev).astype(np.float32)

        self._a_prev = a.copy()
        self._y_prevprev = y.copy()

        # shared team reward
        # r_team = float(np.mean(R))
        # return np.full((self.N,), R, dtype=np.float32)
        return R.astype(np.float32)
    # ========================================================
    # Feature names
    # ========================================================
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
        return isac_names + self_names + neighbor_names + formation_names

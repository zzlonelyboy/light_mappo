from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional



# ============================================================
# helpers
# ============================================================
def _normalize_atk_types(x) -> Tuple[str, ...]:
    """Robustly normalize attack types from tuple/list/set/str."""
    if isinstance(x, (list, tuple, set)):
        return tuple(str(s).strip() for s in x if str(s).strip())
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",")]
        parts = [p for p in parts if p]
        return tuple(parts) if parts else tuple()
    raise TypeError(f"Unsupported atk_types type: {type(x)}")


def _unit_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise unit normalization with safe fallbacks."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def make_attack_injector(config=None):
    if config is None:
        exit("error_attack_config")
    atk_types = config["atk_types"]
    # atk_types = _normalize_atk_types(config.get("atk_types", ("random", "replay", "stealth")))
    # if not atk_types:
    #     raise ValueError("atk_types must not be empty.")

    atk_mgr = GPSAttackManager(
        rng=config["rng"],
        N=config["N"],
        atk_types=atk_types,
        t_min_ratio=config["t_min_ratio"],
        t_max_ratio=config["t_max_ratio"],
        dur_ratio=(config["dur_lo_ratio"], config["dur_hi_ratio"]),
        random_sigma=config["random_sigma"],
        stealth_rate=config["stealth_rate"],
        replay_delay_range=(config["replay_delay_lo"], config["replay_delay_hi"]),
        p_global=config.get("p_global", False),
    )
    return atk_mgr, atk_types


# ============================================================
# scenario
# ============================================================
@dataclass
class GPSAttackScenario:
    t0: int
    t1: int
    mask: np.ndarray  # (N,) bool
    atk_type: str     # "random" | "replay" | "stealth"

    # replay frozen params
    replay_delay_steps: Optional[np.ndarray] = None         # (N,) int
    global_replay_delay_steps: Optional[int] = None         # scalar

    # stealth frozen params
    stealth_dir: Optional[np.ndarray] = None                # (N,D)

    # bookkeeping
    _dt_cached: Optional[float] = None


# ============================================================
# manager
# ============================================================
class GPSAttackManager:
    """
    Attack models follow the paper formulation:
      1) random: true position plus Gaussian bias
      2) replay: delayed historical true/GPS-valid trajectory
      3) stealth: fixed drift direction with bounded linear growth
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        N: int,
        atk_types: Tuple[str, ...] = ("random", "replay", "stealth"),
        t_min_ratio: float = 0.05,
        t_max_ratio: float = 0.25,
        dur_ratio: Tuple[float, float] = (0.4, 0.8),

        random_sigma: float = 15.0,
        stealth_rate: float = 1.2,
        replay_delay_range: Tuple[float, float] = (1.0, 5.0),
        p_global: bool = False,
    ) -> None:
        self.rng = rng
        self.N = int(N)
        self.atk_types = tuple(atk_types)
        self.t_min_ratio = float(t_min_ratio)
        self.t_max_ratio = float(t_max_ratio)
        self.dur_ratio = (float(dur_ratio[0]), float(dur_ratio[1]))
        self.params: Dict[str, object] = dict(
            random_sigma=float(random_sigma),
            stealth_rate=float(stealth_rate),
            replay_delay_range=(float(replay_delay_range[0]), float(replay_delay_range[1])),
        )
        self.p_global = bool(p_global)

    # ------------------------
    # sampling
    # ------------------------
    def sample_scenario(self, T_max: int, dt: float) -> GPSAttackScenario:
        if T_max <= 1:
            t0, t1 = 0, 0
        else:
            t0 = int(self.rng.uniform(self.t_min_ratio, self.t_max_ratio) * T_max)
            dur = int(self.rng.uniform(*self.dur_ratio) * T_max)
            t1 = min(T_max - 1, t0 + max(1, dur))

        # 保持你当前逻辑：不优先小规模攻击
        if self.p_global:
            mask = np.ones(self.N, dtype=bool)
        else:
            k = self.rng.randint(1, max(2, int(self.N / 2)))
            idx = self.rng.choice(self.N, size=k, replace=False)
            mask = np.zeros(self.N, dtype=bool)
            mask[idx] = True

        atk_types = list(self.atk_types)

        # 给 stealth 更高概率
        weights = np.ones(len(atk_types), dtype=np.float32)
        for i, t in enumerate(atk_types):
            if str(t).lower()=='random':
                weights[i]=1.0
            elif str(t).lower()=='replay':
                weights[i]=1.0
            elif str(t).lower() == "stealth":
                weights[i] = 1.0  # 这里越大，stealth 越常出现
        weights = weights / weights.sum()
        atk_type = str(self.rng.choice(atk_types, p=weights))

        return GPSAttackScenario(
            t0=t0,
            t1=t1,
            mask=mask,
            atk_type=atk_type,
        )

    # ------------------------
    # prepare frozen runtime params
    # ------------------------
    def prepare_scenario(
        self,
        scenario: GPSAttackScenario,
        dt: float,
        D: int,
        heading_at_t0: Optional[np.ndarray] = None,
    ) -> None:
        scenario._dt_cached = float(dt)
        active = scenario.mask

        # replay: a common delay tau_atk for all attacked UAVs.
        if scenario.atk_type == "replay" and scenario.replay_delay_steps is None:
            lo_s, hi_s = self.params["replay_delay_range"]
            delay_s = float(self.rng.uniform(float(lo_s), float(hi_s)))
            delay_steps_scalar = int(max(1, np.round(delay_s / dt)))

            delay_steps = np.full((self.N,), delay_steps_scalar, dtype=np.int32)
            delay_steps[~active] = 0

            scenario.global_replay_delay_steps = int(delay_steps_scalar)
            scenario.replay_delay_steps = delay_steps.astype(np.int32)

        # stealth: each attacked UAV keeps a fixed drift direction during the attack.
        if scenario.atk_type == "stealth" and scenario.stealth_dir is None:
            dir_all = np.zeros((self.N, D), dtype=np.float32)

            if heading_at_t0 is not None:
                heading = _unit_rows(np.asarray(heading_at_t0, dtype=np.float32))
            else:
                heading = None

            for i in range(self.N):
                if not active[i]:
                    continue
                if heading is not None:
                    dir_all[i] = heading[i]
                else:
                    d_i = self.rng.normal(size=(1, D)).astype(np.float32)
                    dir_all[i] = _unit_rows(d_i)[0]

            scenario.stealth_dir = dir_all.astype(np.float32)
    # ------------------------
    # apply
    # ------------------------
    def apply(
        self,
        t: int,
        gps: np.ndarray,                  # (N,D)
        hist_true_pos: list[np.ndarray],  # list of (N,D)
        hist_gps_pos: list[np.ndarray],   # reserved
        scenario: GPSAttackScenario,
        dt: float,
        heading_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not (scenario.t0 <= t <= scenario.t1):
            return gps.astype(np.float32, copy=True), np.zeros(self.N, dtype=bool)

        z = gps.astype(np.float32, copy=True)
        active = scenario.mask.copy()
        typ = scenario.atk_type
        N, D = gps.shape

        if N != self.N:
            raise ValueError(f"N mismatch: gps has {N}, manager N={self.N}")
        if active.dtype != bool or active.shape != (self.N,):
            raise ValueError("scenario.mask must be (N,) bool")

        # lazy prepare
        if typ == "replay" and scenario.replay_delay_steps is None:
            self.prepare_scenario(scenario, dt, D)
        if typ == "stealth" and scenario.stealth_dir is None:
            self.prepare_scenario(scenario, dt, D, heading_at_t0=heading_now)

        # --------------------
        # random
        # --------------------
        if typ == "random":
            sigma = float(self.params["random_sigma"])
            m = int(active.sum())
            if m > 0:
                true_now = np.asarray(hist_true_pos[-1], dtype=np.float32)
                noise = self.rng.normal(0.0, sigma, size=(m, D)).astype(np.float32)
                z[active] = true_now[active] + noise

        # --------------------
        # replay
        # --------------------
        elif typ == "replay":
            delay_steps = scenario.replay_delay_steps
            if delay_steps is None:
                delay_steps = np.zeros(self.N, dtype=np.int32)

            idxs = np.nonzero(active)[0]
            if len(idxs) > 0:
                rows = []
                max_hist_idx = len(hist_true_pos) - 1
                for i in idxs:
                    k = t - int(delay_steps[i])
                    k = 0 if k < 0 else (max_hist_idx if k > max_hist_idx else k)
                    rows.append(hist_true_pos[k][i].astype(np.float32))
                z[active] = np.stack(rows, axis=0)

        # --------------------
        # stealthy drift: fixed direction, bounded linear growth
        # --------------------
        elif typ == "stealth":
            rate = float(self.params["stealth_rate"])
            dirs = scenario.stealth_dir
            if dirs is None:
                dirs = np.zeros((self.N, D), dtype=np.float32)
                dirs[:, 0] = 1.0

            elapsed_steps = max(0, int(t) - int(scenario.t0))
            drift_norm = rate * float(dt) * float(elapsed_steps)

            if active.any():
                true_now = np.asarray(hist_true_pos[-1], dtype=np.float32)
                z[active] = true_now[active] + drift_norm * dirs[active]
        elif typ == "none":
                return gps, np.zeros(self.N, dtype=bool)

        return z, active

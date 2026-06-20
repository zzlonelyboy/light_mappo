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


def _pull_hold_curve(progress: float, pull_ratio: float = 0.65) -> float:
    """
    progress in [0,1]
    First pull_ratio: smooth pull-off
    Remaining: hold near 1
    """
    s = float(np.clip(progress, 0.0, 1.0))
    pr = float(np.clip(pull_ratio, 0.2, 0.95))

    if s <= pr:
        x = s / pr
        # smoothstep
        return x * x * (3.0 - 2.0 * x)
    else:
        return 1.0


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

        # new params
        replay_jitter_seconds=config.get("replay_jitter_seconds", 0.15),
        stealth_dir_jitter=config.get("stealth_dir_jitter", 0.10),
        use_heading_as_main_dir=config.get("use_heading_as_main_dir", True),
        stealth_pull_ratio=config.get("stealth_pull_ratio", 0.65),
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
    replay_delay_jitter_steps: Optional[np.ndarray] = None  # (N,) int

    # stealth frozen params
    stealth_dir: Optional[np.ndarray] = None                # (N,D)
    global_stealth_dir: Optional[np.ndarray] = None         # (D,)
    stealth_pull_ratio: Optional[float] = None             # scalar

    # bookkeeping
    _dt_cached: Optional[float] = None


# ============================================================
# manager
# ============================================================
class GPSAttackManager:
    """
    Refactor goals:
      1) replay = global delay + per-node small jitter
      2) stealth = global main direction + per-node small jitter
      3) stealth uses pull_hold curve
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

        # new params
        replay_jitter_seconds: float = 0.15,
        stealth_dir_jitter: float = 0.10,
        use_heading_as_main_dir: bool = True,
        stealth_pull_ratio: float = 0.65,
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

            replay_jitter_seconds=float(replay_jitter_seconds),
            stealth_dir_jitter=float(stealth_dir_jitter),
            use_heading_as_main_dir=bool(use_heading_as_main_dir),
            stealth_pull_ratio=float(stealth_pull_ratio),
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

        # replay: global delay + small jitter
        if scenario.atk_type == "replay" and scenario.replay_delay_steps is None:
            lo_s, hi_s = self.params["replay_delay_range"]
            replay_jitter_seconds = float(self.params["replay_jitter_seconds"])

            global_delay_s = float(self.rng.uniform(float(lo_s), float(hi_s)))
            global_delay_steps = int(max(0, np.round(global_delay_s / dt)))

            jitter_s = self.rng.normal(
                loc=0.0,
                scale=replay_jitter_seconds,
                size=self.N
            ).astype(np.float32)
            jitter_steps = np.round(jitter_s / dt).astype(np.int32)

            delay_steps = np.full((self.N,), global_delay_steps, dtype=np.int32) + jitter_steps
            delay_steps = np.maximum(delay_steps, 0)
            delay_steps[~active] = 0

            scenario.global_replay_delay_steps = int(global_delay_steps)
            scenario.replay_delay_jitter_steps = jitter_steps.astype(np.int32)
            scenario.replay_delay_steps = delay_steps.astype(np.int32)

        # stealth: global direction + small per-node jitter
        if scenario.atk_type == "stealth" and scenario.stealth_dir is None:
            use_heading_as_main_dir = bool(self.params["use_heading_as_main_dir"])
            stealth_dir_jitter = float(self.params["stealth_dir_jitter"])
            stealth_pull_ratio = float(self.params["stealth_pull_ratio"])

            # choose global main direction
            if use_heading_as_main_dir and heading_at_t0 is not None and np.any(active):
                h_active = np.asarray(heading_at_t0[active], dtype=np.float32)
                main_dir = np.mean(h_active, axis=0, keepdims=True)
                main_dir = _unit_rows(main_dir)[0]
            else:
                main_dir = self.rng.normal(size=(1, D)).astype(np.float32)
                main_dir = _unit_rows(main_dir)[0]

            dir_all = np.zeros((self.N, D), dtype=np.float32)
            for i in range(self.N):
                if not active[i]:
                    continue
                jitter = self.rng.normal(0.0, stealth_dir_jitter, size=(D,)).astype(np.float32)
                d_i = main_dir + jitter
                dir_all[i] = _unit_rows(d_i[None, :])[0]

            scenario.global_stealth_dir = main_dir.astype(np.float32)
            scenario.stealth_dir = dir_all.astype(np.float32)
            scenario.stealth_pull_ratio = float(stealth_pull_ratio)

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
                noise = self.rng.normal(0.0, sigma, size=(m, D)).astype(np.float32)
                z[active] += noise

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
        # stealth with pull_hold
        # --------------------
        elif typ == "stealth":
            rate = float(self.params["stealth_rate"])
            dirs = scenario.stealth_dir
            if dirs is None:
                dirs = np.zeros((self.N, D), dtype=np.float32)
                dirs[:, 0] = 1.0

            denom = max(1, scenario.t1 - scenario.t0)
            progress = float(t - scenario.t0) / float(denom)
            pull_ratio = 0.65 if scenario.stealth_pull_ratio is None else float(scenario.stealth_pull_ratio)
            amp = _pull_hold_curve(progress, pull_ratio=pull_ratio)

            # 最大拖拽幅值 = rate * 持续总时间
            total_drift = rate * dt * float(denom) * amp

            if active.any():
                z[active] = gps[active] + total_drift * dirs[active]

        elif typ == "none":
                return gps, np.zeros(self.N, dtype=bool)

        return z, active

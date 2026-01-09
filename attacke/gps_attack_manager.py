from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


# --------- helpers ---------
def _normalize_atk_types(x) -> Tuple[str, ...]:
    """Robustly normalize attack types from tuple/list/set/str."""
    if isinstance(x, (list, tuple, set)):
        return tuple(str(s).strip() for s in x if str(s).strip())
    if isinstance(x, str):
        # allow "random,replay" or "random"
        parts = [p.strip() for p in x.split(",")]
        parts = [p for p in parts if p]  # remove empties
        return tuple(parts) if parts else tuple()
    raise TypeError(f"Unsupported atk_types type: {type(x)}")


def _unit_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise unit normalization with safe fallbacks (keep zero rows as zero)."""
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n


def make_attack_injector(config=None):
    if config is None:
        config = attack_config

    # atk_types = _normalize_atk_types(config.get("atk_types", ("random", "replay", "stealth")))
    atk_types = _normalize_atk_types(config.get("atk_types", ("random", "replay")))
    # print(atk_types)
    if not atk_types:
        raise ValueError("atk_types must not be empty.")

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
        p_global=config.get("p_global", False)
    )
    # print(atk_mgr.params)
    return atk_mgr, atk_types


@dataclass
class GPSAttackScenario:
    t0: int
    t1: int
    mask: np.ndarray  # (N,) bool
    atk_type: str     # "random" | "replay" | "stealth"

    # ---- 冻结后的运行态（惰性填充/可选） ----
    # replay：每个节点独立的 delay（单位：step）
    replay_delay_steps: Optional[np.ndarray] = None  # (N,) int
    # stealth：每个节点固定的方向单位向量（(N,D)）
    stealth_dir: Optional[np.ndarray] = None         # (N,D)
    # 记录用于冻结参数的 dt（方便跨帧检查）
    _dt_cached: Optional[float] = None


class GPSAttackManager:
    """
    负责随机采样攻击场景，并在 step 中对 GPS 量测进行篡改。
    支持：random / replay / stealth（慢漂移）。

    关键优化：
    - 场景级冻结参数（replay 的延时、stealth 的方向），保证窗口内一致性与可复现性。
    - 历史索引钳制；窗口外零开销早返回。
    - 强健的 atk_types 解析。
    """

    def __init__(
        self,
        rng: np.random.RandomState,
        N: int,
        atk_types: Tuple[str, ...] = ("random", "replay", "stealth"),
        t_min_ratio: float = 0.05,
        t_max_ratio: float = 0.25,
        dur_ratio: Tuple[float, float] = (0.4, 0.8),
        random_sigma: float = 15.0,            # m，高斯偏移强度
        stealth_rate: float = 1.2,             # m/s，沿方向慢漂移
        replay_delay_range: Tuple[float, float] = (5.0, 20.0),  # s
        p_global: bool = False,                 # 是否总是攻击所有节点
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
    # 场景采样 & 预备
    # ------------------------
    def sample_scenario(self, T_max: int, dt: float) -> GPSAttackScenario:
        """
        随机生成一次攻击场景：时间窗口 [t0, t1]、受影响掩码 mask、攻击类型 atk_type。
        注：dt 在这里不用于数值转换（replay/stealth 的冻结在首次 apply 时完成）。
        """
        if T_max <= 1:
            # 退化：单步 episode，给出空窗口
            t0, t1 = 0, 0
        else:
            t0 = int(self.rng.uniform(self.t_min_ratio, self.t_max_ratio) * T_max)
            dur = int(self.rng.uniform(*self.dur_ratio) * T_max)
            t1 = min(T_max - 1, t0 + max(1, dur))
        if  self.p_global:
            mask = np.ones(self.N, dtype=bool)
        else:
            k = self.rng.randint(1, self.N+1)  # 从1到N（含）
            idx = self.rng.choice(self.N, size=k, replace=False)
            mask = np.zeros(self.N, dtype=bool)
            mask[idx] = True
        atk_type = str(self.rng.choice(self.atk_types))
        return GPSAttackScenario(t0=t0, t1=t1, mask=mask, atk_type=atk_type)

    def prepare_scenario(
        self,
        scenario: GPSAttackScenario,
        dt: float,
        D: int,
        heading_at_t0: Optional[np.ndarray] = None,
    ) -> None:
        """
        显式冻结场景的运行态参数（可选）。
        - replay：为每个 active 节点采样固定的 delay_steps
        - stealth：为每个 active 节点确定固定方向（优先使用 heading_at_t0）
        """
        scenario._dt_cached = float(dt)

        active = scenario.mask
        if scenario.atk_type == "replay" and scenario.replay_delay_steps is None:
            lo_s, hi_s = self.params["replay_delay_range"]
            delay_s = self.rng.uniform(float(lo_s), float(hi_s), size=self.N)
            delay_steps = np.maximum(0, np.round(delay_s / dt).astype(np.int32))
            delay_steps[~active] = 0  # 非活动节点不使用
            scenario.replay_delay_steps = delay_steps

        if scenario.atk_type == "stealth" and scenario.stealth_dir is None:
            # 确定每个节点的方向单位向量
            if heading_at_t0 is not None:
                dir_all = _unit_rows(heading_at_t0.astype(np.float32))
            else:
                dir_all = np.zeros((self.N, D), dtype=np.float32)
                dir_all[:, 0] = 1.0  # 退化：+x
            dir_all[~active] = 0.0
            scenario.stealth_dir = dir_all

    # ------------------------
    # 应用
    # ------------------------
    def apply(
        self,
        t: int,
        gps: np.ndarray,                  # (N,D)
        hist_true_pos: list[np.ndarray],       # list of (N,D)
        hist_gps_pos: list[np.ndarray],        # 未用，但保留接口
        scenario: GPSAttackScenario,
        dt: float,
        heading_now: Optional[np.ndarray] = None,  # (N,D)，仅在首次冻结方向时使用
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回被篡改的 GPS 量测与当前被攻击掩码。
        若 t 不在攻击窗口，则直接返回真值副本。
        冻结参数为惰性：首次进入窗口时自动准备（也可先手动 prepare_scenario）。
        """
        # ---- 零开销路径：窗口外 ----
        if not (scenario.t0 <= t <= scenario.t1):
            # 拷贝为 float32，保持与窗口内一致的 dtype
            return gps.astype(np.float32, copy=True), np.zeros(self.N, dtype=bool)

        z = gps.astype(np.float32, copy=True)
        active = scenario.mask.copy()
        typ = scenario.atk_type
        N, D = gps.shape

        # 基本健壮性检查（不会抛错，只在开发期调试可开启 assert）
        if N != self.N:
            raise ValueError(f"N mismatch: gps has {N}, manager N={self.N}")
        if active.dtype != bool or active.shape != (self.N,):
            raise ValueError("scenario.mask must be (N,) bool")

        # ---- 惰性冻结运行态（仅当需要且未准备）----
        if typ == "replay" and scenario.replay_delay_steps is None:
            self.prepare_scenario(scenario, dt, D)  # 无 heading 也可冻结
        if typ == "stealth" and scenario.stealth_dir is None:
            # 尝试使用当前 heading 作为“第一次看到”的方向
            self.prepare_scenario(scenario, dt, D, heading_at_t0=heading_now)

        # ---- 应用攻击 ----
        if typ == "random":
            sigma = float(self.params["random_sigma"])  # m
            # 每个 step 独立高斯扰动
            m = int(active.sum())
            if m > 0:
                noise = self.rng.normal(0.0, sigma, size=(m, D)).astype(np.float32)
                z[active] += noise

        elif typ == "replay":
            # 场景固定的每节点 delay_steps
            delay_steps = scenario.replay_delay_steps
            if delay_steps is None:
                # 极端防御：若未成功冻结，退化为 0 延时
                delay_steps = np.zeros(self.N, dtype=np.int32)

            idxs = np.nonzero(active)[0]
            if len(idxs) > 0:
                # 逐节点按各自 delay 从历史取值（N 通常较小，逐节点开销可接受）
                rows = []
                max_hist_idx = len(hist_true_pos) - 1
                for i in idxs:
                    k = t - int(delay_steps[i])
                    k = 0 if k < 0 else (max_hist_idx if k > max_hist_idx else k)
                    rows.append(hist_true_pos[k][i].astype(np.float32))
                z[active] = np.stack(rows, axis=0)

        elif typ == "stealth":
            rate = float(self.params["stealth_rate"])  # m/s
            steps_since = max(0, t - scenario.t0)
            drift = rate * dt * steps_since
            # print("drift:", drift)
            # 场景固定方向；若仍为空则最后兜底为 +x
            dirs = scenario.stealth_dir
            if dirs is None:
                dirs = np.zeros((self.N, D), dtype=np.float32)
                dirs[:, 0] = 1.0
            if active.any():
                z[active] = gps[active] + drift * dirs[active]

        else:
            # 未知类型（保持原值）
            pass

        return z, active

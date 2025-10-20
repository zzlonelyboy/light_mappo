from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

attack_config={
    "rng":np.random.RandomState(1),
    "N":8,
    "atk_types":("random", "replay", "stealth"),
    "p_global":0.7,
    "t_min_ratio":0.2,
    "t_max_ratio":0.7,
    "dur_lo_ratio":0.2,
    "dur_hi_ratio":0.6,
    "random_sigma":15.0,
    "stealth_rate":0.4,
    "replay_delay_lo":5.0,
    "replay_delay_hi":20.0,
}


def make_attack_injector(config=None):
    if config is None:
        config = attack_config
    atk_types = tuple([s.strip() for s in str(config['atk_types']).split(",") if s.strip()])
    atk_mgr = GPSAttackManager(
        rng=config['rng'], N=config['N'], p_global=config['p_global'], atk_types=atk_types,
        t_min_ratio=config['t_min_ratio'], t_max_ratio=config['t_max_ratio'],
        dur_ratio=(config['dur_lo_ratio'], config['dur_hi_ratio']),
        random_sigma=config['random_sigma'], stealth_rate=config['stealth_rate'],
        replay_delay_range=(config['replay_delay_lo'], config['replay_delay_hi']),
    )
    return atk_mgr,atk_types


@dataclass
class GPSAttackScenario:
    t0: int
    t1: int
    mask: np.ndarray # (N,) bool
    atk_type: str # "random" | "replay" | "stealth"




class GPSAttackManager:
    """
        负责随机采样攻击场景，并在 step 中对 GPS 量测进行篡改。
        支持三类：random / replay / stealth（慢漂移）。
    """
    def __init__(
        self,
        rng: np.random.RandomState,
        N: int,
        p_global: float = 0.7,
        atk_types: Tuple[str, ...] = ("random", "replay", "stealth"),
        # 发生时机与持续时间（占整个 episode 的比例）
        t_min_ratio: float = 0.2,
        t_max_ratio: float = 0.7,
        dur_ratio: Tuple[float, float] = (0.2, 0.6),
        # 各攻击类型强度参数
        random_sigma: float = 15.0,  # m，高斯偏移强度
        stealth_rate: float = 0.4,   # m/s，沿方向慢漂移
        replay_delay_range: Tuple[float, float] = (5.0, 20.0),  # s
    ) -> None:
        self.rng = rng
        self.N = int(N)
        self.p_global = float(p_global)
        self.atk_types = atk_types
        self.t_min_ratio = float(t_min_ratio)
        self.t_max_ratio = float(t_max_ratio)
        self.dur_ratio = (float(dur_ratio[0]), float(dur_ratio[1]))
        self.params: Dict[str, object] = dict(
            random_sigma=float(random_sigma),
            stealth_rate=float(stealth_rate),
            replay_delay_range=(float(replay_delay_range[0]), float(replay_delay_range[1])),
        )


    # ------------------------
    # 场景采样 & 应用
    # ------------------------
    def sample_scenario(self, T_max: int, dt: float) -> GPSAttackScenario:
        """
        随机生成一次攻击场景：时间窗口 [t0, t1]、受影响掩码 mask、攻击类型 atk_type。
        注：目前未使用 dt；dt 更常用于 replay/stealth 的步长换算。
        """
        t0 = int(self.rng.uniform(self.t_min_ratio, self.t_max_ratio) * T_max)
        dur = int(self.rng.uniform(*self.dur_ratio) * T_max)
        t1 = min(T_max - 1, t0 + max(1, dur))

        # 受影响节点：全局 或 随机选若干
        if self.rng.rand() < self.p_global:
            mask = np.ones(self.N, dtype=bool)
        else:
            # 随机选择至少 1 个，至多 ~N/2 个
            upper = max(2, self.N // 2 + 1)
            k = self.rng.randint(1, min(upper, self.N + 1))
            idx = self.rng.choice(self.N, size=k, replace=False)
            mask = np.zeros(self.N, dtype=bool)
            mask[idx] = True

        atk_type = str(self.rng.choice(self.atk_types))
        return GPSAttackScenario(t0=t0, t1=t1, mask=mask, atk_type=atk_type)


    def apply(
        self,
        t: int,
        true_pos: np.ndarray,  # (N,D) 常见为 (N,3)
        hist_true_pos: list[np.ndarray],  # list of (N,D)
        hist_gps_pos: list[np.ndarray],   # list of (N,D)（当前未用）
        scenario: GPSAttackScenario,
        dt: float,
        heading_now: np.ndarray | None = None,  # (N,D) 单位向量；stealth 时若提供将沿各自前向漂移
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        返回被篡改的 GPS 量测与当前被攻击掩码。
        若 t 不在攻击窗口，则直接返回真值副本。
        """
        z = true_pos.copy().astype(np.float32)
        active = np.zeros(self.N, dtype=bool)
        if not (scenario.t0 <= t <= scenario.t1):
            return z, active

        active = scenario.mask
        typ = scenario.atk_type
        D = true_pos.shape[1]

        if typ == "random":
            sigma = float(self.params["random_sigma"])  # m
            noise = self.rng.normal(0.0, sigma, size=(int(active.sum()), D)).astype(np.float32)
            z[active] += noise

        elif typ == "replay":
            lo, hi = self.params["replay_delay_range"]  # seconds
            delay = float(self.rng.uniform(lo, hi))
            delay_steps = max(0, int(round(delay / dt)))
            k = max(0, t - delay_steps)
            src = hist_true_pos[k] if k < len(hist_true_pos) else true_pos
            z[active] = src[active]

        elif typ == "stealth":
            rate = float(self.params["stealth_rate"])  # m/s
            # 累计漂移距离（从攻击开始计）
            drift = rate * dt * max(0, t - scenario.t0)
            if heading_now is not None:
                z[active] = true_pos[active] + (drift * heading_now[active])
            else:
                # 退化：统一沿 +x
                v = np.zeros((D,), dtype=np.float32)
                v[0] = 1.0
                z[active] = true_pos[active] + (drift * v)

        # 统一返回
        return z, active


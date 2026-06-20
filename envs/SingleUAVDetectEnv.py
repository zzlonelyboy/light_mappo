from __future__ import annotations
import numpy as np
import gym
from gym.spaces import Box, Discrete
from typing import Optional, Callable, Any, Union
from collections import deque
from envs.NavPolicyAdapter import NavPolicyAdapter

class SingleUAVDetectEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        base_env: gym.Env,
        nav_adapter: NavPolicyAdapter,
        agent_index: Union[int, str] = 2,
        R: Optional[float] = None,
        dt: Optional[float] = None,
        gps_getter: Optional[Callable] = None,
        atk_manager: Optional[Any] = None,
        atk_scenario: Optional[Any] = None,
        attack_writeback: bool = False,
        resample_attack_on_reset: bool = True,
        history_window: int = 5,
        reward_tp: float = 16.0,
        stay_reward:float=0.5,
        reward_fp: float = 1.8,
        reward_fn: float = 6.0,
        flip_penalty: float = 0.005,
        delay_full_seconds: float = 3.0,
        cusum_threshold: float = 0.005,
        cusum_drift: float = 0.002,
        cusum_max: float = 0.15,
    ):
        super().__init__()
        self.base = base_env
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
            raise ValueError("base_env 必须包含整数属性 N。")
        self.agent_index_conf = agent_index
        self.agent_index = agent_index
        self.R = float(R if R is not None else getattr(self.unwrapped_env, "R", 20.0))
        self.dt = float(dt if dt is not None else getattr(self.unwrapped_env, "dt", 0.1))
        self.v0 = float(getattr(self.unwrapped_env, "v0", 4.0))
        self.det_dim_own = 14
        self.action_space = Discrete(2)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.det_dim_own,), dtype=np.float32)
        self.gps_getter = gps_getter
        self.atk_manager = atk_manager
        self.atk_scenario = atk_scenario
        self.attack_writeback = bool(attack_writeback)
        self.resample_attack_on_reset = bool(resample_attack_on_reset)
        self._hist_true_pos = []
        self._hist_gps_pos = []
        self.R_TP = float(reward_tp)
        self.R_FP = float(reward_fp)
        self.R_FN = float(reward_fn)
        self.R_TN = 0.15
        self.hold_cost = 0.02
        self.flip_penalty = float(flip_penalty)
        self.delay_full_seconds = float(delay_full_seconds)
        self.delay_k = 0.08
        self.cusum_threshold = float(cusum_threshold)
        self.cusum_drift = float(cusum_drift)
        self.cusum_max = float(cusum_max)
        self.cusum_beta = 0.99
        self._ema_mu = 0.0
        self._cpos = 0.0
        self._cneg = 0.0
        self._cusum = 0.0
        self._rnorm_hist = deque(maxlen=int(history_window))
        self._r_dead_prev = np.zeros(3, dtype=np.float32)
        self._prev_gps_i = np.zeros(3, dtype=np.float32)
        self._prev_h_i = np.zeros(3, dtype=np.float32)
        self._a_prev_i = 0
        self._y_prevprev_i = 0
        self._event_open_i = False
        self._detected_in_event_i = False
        self._event_t0_i = 0
        self.stay_reward = stay_reward
        self.neg_hold_grace_steps = 3
        self.cusum_waive_threshold = 0.6
        self.fp_cooldown_steps = 3
        self._neg_hold_counter_i = 0
        self._last_fp_step_i = -10**9

    def _pick_agent_index(self):
        if isinstance(self.agent_index_conf, str) and self.agent_index_conf == "random":
            self.agent_index = int(np.random.randint(0, self.N))
        else:
            self.agent_index = int(self.agent_index_conf)
            if not (0 <= self.agent_index < self.N):
                raise ValueError("agent_index 越界。")

    def _get_gps(self) -> np.ndarray:
        if callable(self.gps_getter):
            z = self.gps_getter(self.unwrapped_env)
            return np.asarray(z, dtype=np.float32)
        if hasattr(self.unwrapped_env, "gps"):
            return np.asarray(self.unwrapped_env.gps, dtype=np.float32)
        if hasattr(self.unwrapped_env, "p"):
            return np.asarray(self.unwrapped_env.p, dtype=np.float32)
        raise RuntimeError("无法获取 GPS。")

    def _as_attack_mask(self, info: dict) -> np.ndarray:
        mask = info.get("attack_mask", None)
        if mask is None:
            return np.zeros(self.N, dtype=np.int32)
        return np.asarray(mask, dtype=bool).astype(np.int32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self._pick_agent_index()
        if seed is not None:
            ret = self.base.reset(seed=seed, options=options)
        else:
            ret = self.base.reset()
        _ = ret[0] if isinstance(ret, (list, tuple)) else ret
        gps = self._get_gps()
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
        i = self.agent_index
        self._prev_gps_i = gps[i].astype(np.float32).copy()
        self._prev_h_i = h[i].astype(np.float32).copy()
        self._a_prev_i = 0
        self._y_prevprev_i = 0
        self._event_open_i = False
        self._detected_in_event_i = False
        self._event_t0_i = 0
        self._neg_hold_counter_i = 0
        self._last_fp_step_i = -10**9
        self._rnorm_hist.clear()
        for _k in range(max(3, self._rnorm_hist.maxlen or 5)):
            self._rnorm_hist.append(float(np.clip(np.random.normal(0.02, 0.01), 0.0, 0.1)))
        self._r_dead_prev = np.zeros(3, dtype=np.float32)
        self._ema_mu = 0.0
        self._cpos = 0.0
        self._cneg = 0.0
        self._cusum = 0.0
        if self.attack_writeback and (self.atk_manager is not None):
            if self.resample_attack_on_reset or (self.atk_scenario is None):
                T_max = int(getattr(self.unwrapped_env, "T_max", 0) or 1000)
                try:
                    self.atk_scenario = self.atk_manager.sample_scenario(T_max=T_max, dt=self.dt)
                except Exception:
                    self.atk_scenario = None
        obs = self._build_obs_i()
        return obs.astype(np.float32), {}

    def step(self, a_det):
        a = int(a_det)
        a = 1 if a != 0 else 0
        t = int(getattr(self.unwrapped_env, "t", 0))
        attack_info = {}
        if self.attack_writeback and (self.atk_manager is not None) and (self.atk_scenario is not None):
            try:
                p = np.asarray(getattr(self.unwrapped_env, "p", self.unwrapped_env.p), dtype=np.float32)
                true_pos = np.asarray(getattr(self.unwrapped_env, "true_p", self.unwrapped_env.p), dtype=np.float32)
                heading_now = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
                self._hist_true_pos.append(true_pos.copy())
                z, active = self.atk_manager.apply(
                    t=t,
                    gps=p,
                    hist_true_pos=self._hist_true_pos,
                    hist_gps_pos=self._hist_gps_pos,
                    scenario=self.atk_scenario,
                    dt=self.dt,
                    heading_now=heading_now,
                )
                if hasattr(self.unwrapped_env, "gps"):
                    setattr(self.unwrapped_env, "gps", z.astype(np.float32))
                else:
                    setattr(self.unwrapped_env, "p", z.astype(np.float32))
                setattr(self.unwrapped_env, "_atk_active_mask", np.asarray(active, dtype=bool))
                self._hist_gps_pos.append(np.asarray(z, dtype=np.float32))
                attack_info = {
                    "attack_active": bool(np.any(active) and (self.atk_scenario.t0 <= t <= self.atk_scenario.t1)),
                    "attack_mask": np.asarray(active, dtype=bool),
                    "attack_type": str(getattr(self.atk_scenario, "atk_type", "")),
                    "attack_t0_t1": (int(self.atk_scenario.t0), int(self.atk_scenario.t1)),
                }
            except Exception:
                attack_info = {}
        y_vec = self._as_attack_mask(attack_info)
        y_i = int(y_vec[self.agent_index]) if y_vec.size == self.N else 0
        r = self._compute_detection_reward_single(a, y_i, attack_info)
        stage_one_obs = self.unwrapped_env._get_obs()
        obs_vec = np.expand_dims(np.concatenate([stage_one_obs['obs'], stage_one_obs['nbr_mask']], axis=1), axis=0)
        nav_actions = self.nav.act(obs_vec)
        _, _, terminated, truncated, info_base = self.unwrapped_env.step(nav_actions[0])
        obs_next = self._build_obs_i().astype(np.float32)
        tp = ((a == 1) & (y_i == 1))
        fp = ((a == 1) & (y_i == 0))
        fn = ((a == 0) & (y_i == 1))
        tn = ((a == 0) & (y_i == 0))
        info = dict()
        info["det_tp_fp_fn_tn"] = (int(tp), int(fp), int(fn), int(tn))
        return obs_next, float(r), bool(terminated), bool(truncated), info

    def _build_obs_i(self) -> np.ndarray:
        gps = self._get_gps()
        h = np.asarray(self.unwrapped_env.h_hist[-1], dtype=np.float32)
        i = self.agent_index
        pred = self._prev_gps_i + self.v0 * self._prev_h_i * self.dt
        r_dead = (gps[i] - pred) / self.R
        r_norm = float(np.linalg.norm(r_dead))
        speed_res = (np.linalg.norm(gps[i] - self._prev_gps_i) - self.v0 * self.dt) / self.R
        dh = float(np.linalg.norm(h[i] - self._prev_h_i))
        actual_disp = gps[i] - self._prev_gps_i
        expected_disp = self._prev_h_i * self.v0 * self.dt
        an = float(np.linalg.norm(actual_disp))
        en = float(np.linalg.norm(expected_disp))
        if an > 1e-3 and en > 1e-3:
            direction_cosine = float(np.clip(np.dot(actual_disp, expected_disp) / (an * en + 1e-9), -1.0, 1.0))
        else:
            direction_cosine = 1.0
        r_along = float(np.dot(r_dead, self._prev_h_i))
        r_lateral = float(np.sqrt(max(0.0, r_norm**2 - r_along**2)))
        self._rnorm_hist.append(r_norm)
        r_norm_change = float(r_norm - self._rnorm_hist[-2])
        r_norm_accel = float(r_norm - 2*self._rnorm_hist[-2] + self._rnorm_hist[-3])
        hist_arr = np.array(self._rnorm_hist, dtype=np.float32)
        r_norm_std = float(np.std(hist_arr))
        r_norm_trend = float(hist_arr[-1] - hist_arr[0])
        mu = self._ema_mu = self.cusum_beta * self._ema_mu + (1.0 - self.cusum_beta) * r_norm
        e = r_norm - mu
        self._cpos = float(np.clip(self._cpos + (e - self.cusum_threshold - self.cusum_drift), 0.0, self.cusum_max))
        self._cneg = float(np.clip(self._cneg + (-e - self.cusum_threshold - self.cusum_drift), 0.0, self.cusum_max))
        self._cusum = float(max(self._cpos, self._cneg))
        cusum_normalized = float(self._cusum / self.cusum_max)
        rdn = float(np.linalg.norm(r_dead))
        rdp = float(np.linalg.norm(self._r_dead_prev))
        if rdn > 1e-6 and rdp > 1e-6:
            r_dead_angle_change = float(np.clip(np.dot(r_dead, self._r_dead_prev) / (rdn * rdp + 1e-9), -1.0, 1.0))
        else:
            r_dead_angle_change = 1.0
        obs = np.array([
            r_dead[0], r_dead[1], r_dead[2],
            r_norm, speed_res, dh,
            direction_cosine, r_lateral,
            r_norm_change, r_norm_accel, r_norm_std,
            r_norm_trend, cusum_normalized, r_dead_angle_change
        ], dtype=np.float32)
        self._prev_gps_i = gps[i].astype(np.float32).copy()
        self._prev_h_i = h[i].astype(np.float32).copy()
        self._r_dead_prev = r_dead.astype(np.float32).copy()
        return obs

    def _compute_detection_reward_single(self, a: int, y: int, info: dict) -> float:
        r = 0.0
        t = int(getattr(self.unwrapped_env, "t", 0))
        rise_true = (self._y_prevprev_i == 0) and (y == 1)
        fall_true = (self._y_prevprev_i == 1) and (y == 0)
        rise_pred = (self._a_prev_i == 0) and (a == 1)
        if rise_true:
            self._event_open_i = True
            self._detected_in_event_i = False
            t0_info = info.get("attack_t0_t1", (t, t))[0] if isinstance(info, dict) else t
            self._event_t0_i = int(t0_info)
            if a == 1 and (not self._detected_in_event_i):
                r += self.R_TP
                self._detected_in_event_i = True
        if rise_pred and (y == 1) and (not self._detected_in_event_i):
            elapsed_steps = max(0, t - self._event_t0_i)
            elapsed_sec = elapsed_steps * self.dt
            delay_weight = float(np.exp(-self.delay_k * np.clip(elapsed_sec, 0.0, self.delay_full_seconds) / max(self.delay_full_seconds, 1e-6)))
            r += self.R_TP * delay_weight
            self._detected_in_event_i = True
        if rise_pred and (y == 0):
            if (t - self._last_fp_step_i) >= self.fp_cooldown_steps:
                r -= self.R_FP
                self._last_fp_step_i = t
        if (y == 0) and (a == 1) and (not rise_pred):
            self._neg_hold_counter_i += 1
        else:
            self._neg_hold_counter_i = 0
        waive = (self._neg_hold_counter_i <= self.neg_hold_grace_steps) or (self._cusum >= self.cusum_waive_threshold)
        if (y == 0) and (a == 1) and (not rise_pred) and (not waive):
            r -= self.hold_cost
        if (y == 1) and (a == 1):
            r += self.stay_reward
        if (y == 0) and (a == 0):
            r += self.R_TN
        if fall_true and self._event_open_i and (not self._detected_in_event_i):
            r -= self.R_FN
        if fall_true:
            self._event_open_i = False
            self._detected_in_event_i = False
        if self.flip_penalty > 0 and (a != self._a_prev_i):
            r -= self.flip_penalty
        self._a_prev_i = a
        self._y_prevprev_i = y
        return float(r)

    def set_attack_scenario(self, scenario):
        self.atk_scenario = scenario

    def clear_attack_scenario(self):
        self.atk_scenario = None

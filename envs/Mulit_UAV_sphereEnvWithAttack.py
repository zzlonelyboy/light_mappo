from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np
import gym

from envs.MultiUAVSphereEnv2 import MultiUAVSphereEnvWithObstacle as BaseEnv


class MultiUAVSphereEnvWithAttack(BaseEnv):
		"""
		在基础环境 MultiUAVSphereEnvWithObstacle 上，增加攻击状态的 info 输出。

		说明：
		- 本类不直接实现攻击（如 GPS 篡改）；建议由外部攻击管理器在 step 前/后修改观测，
			并把攻击状态写回 env 的属性（attack_active/mask/type/t0_t1），
			以便通过 _get_info 在日志中记录。
		- 这样可以最小化对原环境动力学与奖励逻辑的侵入。
		"""

		def __init__(self, *args, **kwargs) -> None:
				super().__init__(*args, **kwargs)
				# 攻击状态占位（外部控制）
				self.attack_active: bool = False
				self.attack_mask = np.zeros(self.N, dtype=bool)
				self.attack_type: str = "none"
				self.attack_t0_t1: Tuple[int, int] = (0, 0)

		def _get_info(self) -> Dict[str, Any]:
				info = super()._get_info()
				# 附加攻击信息（用于记录与可视化）
				info["attack_active"] = bool(self.attack_active)
				info["attack_mask"] = self.attack_mask.copy()
				info["attack_type"] = str(self.attack_type)
				info["attack_t0_t1"] = tuple(self.attack_t0_t1)
				return info
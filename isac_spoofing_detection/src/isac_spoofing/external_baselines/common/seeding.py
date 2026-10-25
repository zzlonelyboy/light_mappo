"""Deterministic random seeds for offline baseline derivations."""

from __future__ import annotations

import hashlib


def stable_seed(base_seed: int, episode_id: str, namespace: str) -> int:
    """Return a process-independent 32-bit seed."""
    payload = f"{int(base_seed)}|{episode_id}|{namespace}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

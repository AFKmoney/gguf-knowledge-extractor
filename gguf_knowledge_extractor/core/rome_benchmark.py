"""Model-level behavioral metrics for ROME edits.

This module intentionally measures behavior rather than treating causal
localization or an equation test as edit success. Callers provide the
pre-edit and post-edit logits for the same prompt plus the target token.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RomeBehaviorMetrics:
    target_token_id: int
    pre_target_probability: float
    post_target_probability: float
    pre_rank: int
    post_rank: int
    target_logit_change: float
    kl_pre_to_post: float
    edit_success: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z)
    p = np.exp(z)
    return p / np.sum(p)


def measure_edit_behavior(pre_logits: np.ndarray, post_logits: np.ndarray, target_token_id: int) -> RomeBehaviorMetrics:
    """Measure target movement, rank, distribution drift, and top-1 success."""
    pre = np.asarray(pre_logits, dtype=np.float64).reshape(-1)
    post = np.asarray(post_logits, dtype=np.float64).reshape(-1)
    if pre.shape != post.shape:
        raise ValueError("pre/post logits must have identical shapes")
    if target_token_id < 0 or target_token_id >= pre.size:
        raise ValueError("target token id out of range")
    p_pre = _probabilities(pre)
    p_post = _probabilities(post)
    pre_rank = 1 + int(np.sum(pre > pre[target_token_id]))
    post_rank = 1 + int(np.sum(post > post[target_token_id]))
    kl = float(np.sum(p_pre * (np.log(np.maximum(p_pre, 1e-30)) - np.log(np.maximum(p_post, 1e-30)))))
    return RomeBehaviorMetrics(
        target_token_id=int(target_token_id),
        pre_target_probability=float(p_pre[target_token_id]),
        post_target_probability=float(p_post[target_token_id]),
        pre_rank=pre_rank,
        post_rank=post_rank,
        target_logit_change=float(post[target_token_id] - pre[target_token_id]),
        kl_pre_to_post=kl,
        edit_success=bool(int(np.argmax(post)) == target_token_id),
    )
